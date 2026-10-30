"""R101 private R81 one-shot wrapper. Default/offline command never contacts hardware.

CPU tests use explicit temporary fake inventory/ledgers. Production uses ONLY the
R53 ledger and R15 authenticated bridge. No stage/build/automatic retry exists.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import tarfile
import time
import types

if not __debug__:
    raise RuntimeError('Python optimization forbidden: historical contracts use assertions')
ROOT = Path(__file__).resolve().parent
RUNS = ROOT.parent
R81 = RUNS / 'host-input-rust-worker-session-r81-20261027'
R96 = RUNS / 'r81-strict-validator-link-r96-20261028'
R53 = RUNS / 'sm120-worker-fresh-identities-r53'
R15 = RUNS / 'gpu-crossnode-ssh-20261024-r15'
LEDGER = R53 / 'namespace-ledger'
R96_MANIFEST_SHA = '7360a6d920ce949d59f17f43f019eb1ef84c38e91713bfaa6f360325928997ad'
PINS_SHA = '5d9792d0816846984ad9255dfd8e27c73b6e44382bccf9a5ccf51976cee8c7e3'
CAP = 8 * 1024 * 1024


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def strict(raw):
    def pairs(items):
        out = {}
        for k, v in items:
            if k in out:
                raise ValueError('duplicate JSON key')
            out[k] = v
        return out
    return json.loads(raw, object_pairs_hook=pairs)


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def read(path, expected=None, limit=CAP):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('not a regular file')
        raw = bytearray()
        while True:
            b = os.read(fd, min(65536, limit + 1 - len(raw)))
            if not b:
                break
            raw.extend(b)
            if len(raw) > limit:
                raise ValueError('read cap exceeded')
    finally:
        os.close(fd)
    raw = bytes(raw)
    if expected is not None and digest(raw) != expected:
        raise ValueError('digest mismatch: '+str(path))
    return raw


def exclusive(path, raw):
    """Persist file AND directory before a later submit. Partial writes stay consumed."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def module(path, raw):
    m = types.ModuleType('r101_'+path.stem)
    m.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), m.__dict__)
    return m


def dependencies():
    pins = strict(read(ROOT/'PINS.json', PINS_SHA))
    snapshots = {name: read(RUNS/name, h) for name, h in pins.items()}
    def get(root, name):
        return snapshots[str((root/name).relative_to(RUNS))]
    manifest = strict(get(R96, 'MANIFEST.json'))
    assert digest(get(R96, 'MANIFEST.json')) == R96_MANIFEST_SHA
    assert digest(get(R96, 'pinned_validator.py')) == manifest['files']['pinned_validator.py']['sha256']
    adapter = module(R96/'pinned_validator.py', get(R96, 'pinned_validator.py'))
    _, oracle_pins = adapter.load_pinned()  # Validate transitively pinned bytes before readiness.
    adapter.read_pinned(adapter.GOLDEN, adapter.GOLDEN_SHA)
    L = module(R81/'r81_contract.py', get(R81, 'r81_contract.py'))
    ledger = module(R53/'namespace_ledger.py', get(R53, 'namespace_ledger.py'))
    runtime = {n: digest(get(R81, n)) for n in ['r81_contract.py', 'r81_outer.py']}
    runtime_raw = encoded(runtime)
    assert digest(runtime_raw) == 'dc4ae43a21d00c051bf16110ed079a706605ee05e291df2269b9a313dea7b80a'
    staged = strict(get(R81, 'evidence/controller-runtime-staged.json'))
    assert staged['status'] == 'R81_RUNTIME_STAGED_NO_GPU_NO_RESERVATION'
    assert staged['files']['RUNTIME.json'] == digest(runtime_raw)
    assert all(staged['files'][n] == h for n, h in runtime.items())
    readback = strict(get(R81, 'evidence/controller-runtime-readback.json'))
    assert readback['launch_once_absent'] is True
    assert all(readback['files'][n]['sha256'] == h for n, h in staged['files'].items())
    assert digest(get(R81, 'CASE_PLAN.json')) == L.PLAN_SHA
    assert digest(get(R81, 'PACKAGE.json')) == L.PLAN['package_sha256']
    assert L.PLAN['session_seconds'] == 30 and L.PLAN['guard_seconds'] == 90 and L.PLAN['outer_seconds'] == 105
    return dict(L=L, ledger=ledger, adapter=adapter, runtime=runtime, runtime_raw=runtime_raw,
                snapshots=snapshots, pins=pins, oracle_pins=oracle_pins)


def capture(argv, timeout, cwd, cap=CAP):
    """Actual pipes, bounded capture/time; uncertainty never produces an auto retry."""
    p = subprocess.Popen(argv, cwd=cwd, env={k:v for k,v in os.environ.items()
        if not k.startswith(('PYTHON','LD_'))}, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    sel = selectors.DefaultSelector()
    buffers = [bytearray(), bytearray()]
    deadline = time.monotonic()+timeout
    try:
        for index, stream in enumerate([p.stdout, p.stderr]):
            os.set_blocking(stream.fileno(), False)
            sel.register(stream, selectors.EVENT_READ, index)
        while sel.get_map():
            if time.monotonic() >= deadline:
                raise TimeoutError('bridge deadline exceeded; UNKNOWN_NO_REPLAY')
            for key, _ in sel.select(min(.05, max(0, deadline-time.monotonic()))):
                b = os.read(key.fileobj.fileno(), 65536)
                if not b:
                    sel.unregister(key.fileobj)
                else:
                    buffers[key.data].extend(b)
                    if sum(map(len, buffers)) > cap:
                        raise ValueError('bridge output cap exceeded; UNKNOWN_NO_REPLAY')
        code = p.wait(timeout=max(.001, deadline-time.monotonic()))
        return code, bytes(buffers[0]), bytes(buffers[1])
    finally:
        if p.poll() is None:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(timeout=3)
        sel.close()
        p.stdout.close()
        p.stderr.close()


def command_for(body):
    return shlex.join(['/usr/bin/env','-u','LD_PRELOAD','-u','LD_AUDIT','-u','LD_LIBRARY_PATH',
                       '/usr/bin/python3','-B','-c',body]).encode()


def bridge(name, body, timeout, deps, root=ROOT):
    """Same R15 authenticated SSH bridge; exclusive local command/intent receipts."""
    compile(body, name, 'exec')
    for name_pin in ['MANIFEST.json','ssh_bridge.py','vm_bootstrap.py','askpass.py']:
        key = str((R15/name_pin).relative_to(RUNS))
        read(R15/name_pin, deps['pins'][key])
    command = command_for(body)
    path = root/(name+'.command')
    exclusive(path, command)
    exclusive(root/(name+'.intent.json'), encoded(dict(status='UNKNOWN_NO_REPLAY', automatic_retry=False,
              command_sha256=digest(command), start_ns=time.monotonic_ns())))
    argv = ['/usr/bin/python3','-B',str(R15/'ssh_bridge.py'),'--command-file',str(path),
            '--receipt',str(root/(name+'.bridge.json')),'--timeout',str(timeout)]
    code, out, err = capture(argv, timeout+5, root)
    exclusive(root/(name+'.stdout'), out)
    exclusive(root/(name+'.stderr'), err)
    exclusive(root/(name+'.terminal.json'), encoded(dict(exit_code=code, end_ns=time.monotonic_ns(),
              stdout_sha256=digest(out), stderr_sha256=digest(err), process_terminal=True)))
    if code != 0:
        raise RuntimeError('bridge nonzero; UNKNOWN_NO_REPLAY; reconcile without retry')
    return out


def readiness_body(d):
    """Only the authenticated R76 inventory body, with R81 status and staged hashes."""
    L = d['L']
    key = 'host-input-native-owner-r76-20261027/fresh_gpu_readiness.py'
    tree = ast.parse(d['snapshots'][key])
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id == 'body' for t in n.targets)]
    assert len(nodes) == 1
    expected = {**d['runtime'], 'CASE_PLAN.json':L.PLAN_SHA, 'PACKAGE.json':L.PLAN['package_sha256'],
                'worker_guard.py':L.GUARD_SHA, 'RUNTIME.json':digest(d['runtime_raw'])}
    scope = {'L':L, 'runtime':{n:{'sha256':h} for n,h in expected.items()}}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'r101-readiness-template', 'exec'), scope)
    return scope['body'].replace('R76_NATIVE_GPU_ADMISSION_BLOCKED_NO_COMPUTE','R81_BLOCKED_NO_COMPUTE').replace(
        'R76_NATIVE_GPU_ADMISSION_READY_NO_COMPUTE','R81_READY_NO_COMPUTE')


def readiness(d, root=ROOT):
    raw = bridge('readiness-once', readiness_body(d), 40, d, root)
    result = strict(raw)
    ready = result['readiness']
    # The R76 body produces canonical bytes with the same trailing newline.
    assert digest(encoded(ready)) == result['readiness_sha256']
    exclusive(root/'READINESS.json', encoded(ready))
    exclusive(root/'READINESS_LINK.json', encoded(result))
    assert ready['status'] == 'R81_READY_NO_COMPUTE'
    assert ready['GPU_compute'] is False and ready['namespace_reserved'] is False
    assert ready['lease_free_tested'] is True and ready['lease_held_after_return'] is False
    d['L'].expected_config(read(R81/'CASE_PLAN.json',d['L'].PLAN_SHA),
                          read(R81/'PACKAGE.json',d['L'].PLAN['package_sha256']),ready,d['runtime'])
    return ready


def reserve(d, ready, external_plan_sha, root=ROOT, ledger_root=LEDGER):
    L = d['L']
    if external_plan_sha != L.PLAN_SHA:
        raise ValueError('external plan SHA mismatch')
    # Readiness time is VM monotonic. Fresh local bridge receipt bounds transit age.
    terminal = strict(read(root/'readiness-once.terminal.json'))
    assert terminal['exit_code'] == 0 and terminal['process_terminal'] is True
    assert 0 <= time.monotonic_ns()-terminal['end_ns'] < 5_000_000_000
    cfg = L.expected_config(read(R81/'CASE_PLAN.json',L.PLAN_SHA),
                            read(R81/'PACKAGE.json',L.PLAN['package_sha256']),ready,d['runtime'])
    assert ready['GPU_compute'] is False and ready['namespace_reserved'] is False
    assert ready['lease_free_tested'] is True and ready['lease_held_after_return'] is False
    context = dict(phase='R81_R101', case_plan_sha256=external_plan_sha, requests=L.PLAN['requests'],
                   hardware_invocation_limit=1, adapter_manifest_sha256=R96_MANIFEST_SHA)
    ids = d['ledger'].reserve(ledger_root,L.NAMESPACE,context)
    reservation = dict(effective_ids=ids,case_plan_sha256=external_plan_sha,reserved_before_submit=True,
                       ledger_root=str(ledger_root),entries={n:strict(read(ledger_root/(n+'.json'))) for n in ids})
    raw = encoded(reservation)
    exclusive(root/'RESERVATION.json',raw)
    case = dict(status='R81_SUBMITTED_UNKNOWN_NO_REPLAY',namespace=L.NAMESPACE,
                case_plan_sha256=external_plan_sha,reservation_sha256=digest(raw),
                readiness_sha256=digest(encoded(ready)),runtime_sha256=digest(d['runtime_raw']),
                adapter_manifest_sha256=R96_MANIFEST_SHA,hardware_invocations_used=1,
                automatic_retry=False,config=cfg)
    # Must be durable before invoking any transport, even when transport fails.
    exclusive(root/'CASE_SUBMISSION.json',encoded(case))
    return case, raw


def submit_body(d, case, reservation_raw):
    L = d['L']
    return f'''import pathlib,hashlib,json,base64,os,sys,time,types
assert __debug__
r=pathlib.Path({L.TARGET!r});expected={d['runtime']!r}
raws={{n:(r/n).read_bytes() for n in expected}}
assert all(hashlib.sha256(raws[n]).hexdigest()==h for n,h in expected.items())
runtime=(r/'RUNTIME.json').read_bytes();assert hashlib.sha256(runtime).hexdigest()=={case['runtime_sha256']!r}
assert json.loads(runtime)==expected
ready=(r/'READINESS.json').read_bytes();assert hashlib.sha256(ready).hexdigest()=={case['readiness_sha256']!r}
assert 0<=time.monotonic_ns()-json.loads(ready)['timestamp_monotonic_ns']<60*10**9
assert hashlib.sha256((r/'CASE_PLAN.json').read_bytes()).hexdigest()=={case['case_plan_sha256']!r}
raw=base64.b64decode({base64.b64encode(reservation_raw).decode()!r},validate=True)
assert hashlib.sha256(raw).hexdigest()=={case['reservation_sha256']!r}
with (r/'RESERVATION.json').open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o400)
fd=os.open(r,os.O_DIRECTORY|os.O_RDONLY);os.fsync(fd);os.close(fd)
os.chdir(r);os.environ['VF_R81_ALLOW_GPU']='1'
# Execute exactly the bytes hashed above; avoid path reopen/pyc for the contract.
m=types.ModuleType('r81_contract');m.__file__=str(r/'r81_contract.py')
exec(compile(raws['r81_contract.py'],m.__file__,'exec'),m.__dict__);sys.modules['r81_contract']=m
sys.argv=[str(r/'r81_outer.py'),{case['case_plan_sha256']!r},{case['reservation_sha256']!r},{case['readiness_sha256']!r},{case['runtime_sha256']!r}]
exec(compile(raws['r81_outer.py'],str(r/'r81_outer.py'),'exec'),{{'__name__':'__main__','__file__':str(r/'r81_outer.py')}})
'''


def submit(d, external_plan_sha, root=ROOT):
    if external_plan_sha != d['L'].PLAN_SHA:
        raise ValueError('external plan SHA mismatch')
    # Prevent a second readiness too, including failed/uncertain transport attempts.
    if (root/'CASE_SUBMISSION.json').exists() or (root/'readiness-once.command').exists():
        raise FileExistsError('one-shot already attempted; no retry')
    ready = readiness(d,root)
    case, raw = reserve(d,ready,external_plan_sha,root)
    return bridge('submit-once',submit_body(d,case,raw),115,d,root)


def collect_body(d):
    L = d['L']
    return f'''import pathlib,json,hashlib,os,stat,subprocess,fcntl,tarfile,io,sys
r=pathlib.Path({L.TARGET!r});c=r/'control'
op=json.loads((c/'operation.json').read_bytes());guard=json.loads((c/'guard.json').read_bytes());ready=json.loads((r/'READINESS.json').read_bytes())
pids=[op['outer_pid'],op['guard_pid'],guard['worker_pid']]+[w['pid'] for w in op['native_processes']]
assert len(pids)==4 and len(set(pids))==4
post=dict(pid_present={{str(p):pathlib.Path('/proc/'+str(p)).exists() for p in pids}},groups_present={{}},source_hashes={{}},config_hashes={{}})
for p in [op['guard_pid'],guard['worker_pid']]:
 try:os.killpg(p,0);post['groups_present'][str(p)]=True
 except ProcessLookupError:post['groups_present'][str(p)]=False
for n,h in {d['runtime']!r}.items():post['source_hashes'][n]=hashlib.sha256((r/n).read_bytes()).hexdigest();assert post['source_hashes'][n]==h
for p,h in op['config']['hashes'].items():post['config_hashes'][p]=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();assert post['config_hashes'][p]==h
lease=pathlib.Path({L.LEASE!r});fd=os.open(lease,os.O_RDWR|os.O_NOFOLLOW)
try:
 s=os.fstat(fd);t=lease.lstat();pin={{'device':s.st_dev,'inode':s.st_ino,'uid':s.st_uid}}
 assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and (s.st_dev,s.st_ino,s.st_uid)==(t.st_dev,t.st_ino,t.st_uid)
 assert pin==op['config']['lease_identity']
 fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB);post.update(lease_free=True,lease_identity=pin)
 q=subprocess.run(['/usr/bin/nvidia-smi','-i',{L.UUID!r},'--query-compute-apps=pid','--format=csv,noheader'],capture_output=True,timeout=10)
 assert q.returncode==0 and len(q.stdout)+len(q.stderr)<1024*1024;post['gpu_apps']=q.stdout.decode().strip()
 q=subprocess.run(['/usr/bin/nvidia-smi','-i',{L.UUID!r},'--query-gpu=uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,timeout=10)
 assert q.returncode==0 and len(q.stdout)+len(q.stderr)<1024*1024;post['gpu_state']=q.stdout.decode().strip()
 t=lease.lstat();assert (s.st_dev,s.st_ino,s.st_uid)==(t.st_dev,t.st_ino,t.st_uid)
finally:os.close(fd)
assert not any(post['pid_present'].values()) and not any(post['groups_present'].values()) and not post['gpu_apps']
files={{n:(c/n).read_bytes() for n in ['operation.json','guard.json','guard.stdout','guard.stderr']}}
files.update({{n:(r/n).read_bytes() for n in ['READINESS.json','RESERVATION.json','RUNTIME.json']}})
files['postflight.json']=(json.dumps(post,sort_keys=True,separators=(',',':'))+'\\n').encode()
assert all(len(b)<=2*1024**2 for b in files.values()) and sum(map(len,files.values()))<=8*1024**2
b=io.BytesIO()
with tarfile.open(fileobj=b,mode='w:',format=tarfile.USTAR_FORMAT)as t:
 for n,raw in sorted(files.items()):m=tarfile.TarInfo(n);m.size=len(raw);m.mode=0o400;t.addfile(m,io.BytesIO(raw))
sys.stdout.buffer.write(b.getvalue())
'''


EVIDENCE_NAMES = {'operation.json','guard.json','guard.stdout','guard.stderr',
                  'READINESS.json','RESERVATION.json','RUNTIME.json','postflight.json'}


def collect(d,root=ROOT):
    strict(read(root/'CASE_SUBMISSION.json'))  # Collection cannot create a new submission.
    raw = bridge('postflight-once',collect_body(d),35,d,root)
    exclusive(root/'COLLECTED.tar',raw)
    out = root/'gpu-result'
    out.mkdir(mode=0o700)
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as t:
        members = t.getmembers()
        assert len(members)==len(EVIDENCE_NAMES) and {m.name for m in members}==EVIDENCE_NAMES
        assert all(m.isfile() and m.size<=2*1024**2 for m in members)
        for m in members:
            exclusive(out/m.name,t.extractfile(m).read())


def verify_transport(root, name, body):
    command = read(root/(name+'.command'))
    assert command == command_for(body)
    intent = strict(read(root/(name+'.intent.json')))
    receipt = strict(read(root/(name+'.bridge.json')))
    terminal = strict(read(root/(name+'.terminal.json')))
    assert intent['command_sha256']==receipt['command_sha256']==digest(command)
    assert intent['automatic_retry'] is False and receipt['automatic_retry'] is False
    assert receipt['status']=='TERMINAL_TRANSPORT' and receipt['auth_ready'] is True and receipt['errors']==[]
    assert all(type(receipt[k]) is int and receipt[k]==0 for k in ['exit_code','ssh_exit_code'])
    assert type(terminal['exit_code']) is int and terminal['exit_code']==0 and terminal['process_terminal'] is True
    out=read(root/(name+'.stdout'),terminal['stdout_sha256'])
    err=read(root/(name+'.stderr'),terminal['stderr_sha256'])
    assert len(out)==receipt['output_bytes'] and len(err)==receipt['stderr_bytes']
    return out


def verify(d,root=ROOT,ledger_root=LEDGER):
    """All row and process facts required; empty/incomplete postflight is an error."""
    L = d['L']
    raws = {n:read(root/'gpu-result'/n,limit=2*1024**2) for n in EVIDENCE_NAMES}
    op, guard, ready, res, post = [strict(raws[n]) for n in
        ['operation.json','guard.json','READINESS.json','RESERVATION.json','postflight.json']]
    case = strict(read(root/'CASE_SUBMISSION.json'))
    ready_stdout=verify_transport(root,'readiness-once',readiness_body(d))
    assert strict(ready_stdout)==dict(readiness=ready,readiness_sha256=digest(raws['READINESS.json']))
    submit_stdout=verify_transport(root,'submit-once',submit_body(d,case,raws['RESERVATION.json']))
    assert strict(submit_stdout)==op
    collected=verify_transport(root,'postflight-once',collect_body(d))
    assert collected==read(root/'COLLECTED.tar')
    with tarfile.open(fileobj=io.BytesIO(collected),mode='r:') as archive:
        members=archive.getmembers()
        assert len(members)==len(EVIDENCE_NAMES) and {m.name for m in members}==EVIDENCE_NAMES
        assert all(m.isfile() and m.size<=2*1024**2 for m in members)
        assert {m.name:archive.extractfile(m).read() for m in members}==raws
    assert case['status']=='R81_SUBMITTED_UNKNOWN_NO_REPLAY' and case['automatic_retry'] is False
    assert case['namespace']==op['namespace']==L.NAMESPACE
    assert op['automatic_retry'] is False
    assert case['hardware_invocations_used']==1 and type(case['hardware_invocations_used']) is int
    assert case['adapter_manifest_sha256']==R96_MANIFEST_SHA
    cfg = L.expected_config(read(R81/'CASE_PLAN.json',L.PLAN_SHA),read(R81/'PACKAGE.json',L.PLAN['package_sha256']),ready,d['runtime'])
    for n,key in [('READINESS.json','readiness_sha256'),('RESERVATION.json','reservation_sha256'),('RUNTIME.json','runtime_sha256')]:
        assert digest(raws[n])==case[key]==op[key]
    assert raws['READINESS.json']==read(root/'READINESS.json')
    assert raws['RESERVATION.json']==read(root/'RESERVATION.json')
    assert raws['RUNTIME.json']==d['runtime_raw']
    assert case['case_plan_sha256']==op['case_plan_sha256']==res['case_plan_sha256']==L.PLAN_SHA
    assert L.encoded(cfg)==L.encoded(op['config'])==L.encoded(guard['config'])==L.encoded(case['config'])
    assert op['guard_result']==guard and guard['lease_identity']==cfg['lease_identity']
    assert op['status']=='R81_SESSION_PROCESS_TERMINAL_OFFLINE_ORACLE_AND_POSTFLIGHT_PENDING'
    assert op['complete_GPU_qualification'] is False and op['GPU_mode'] is True
    assert type(op['guard_seconds']) is int and type(op['outer_seconds']) is int
    assert op['guard_seconds']==90 and op['outer_seconds']==105 and cfg['mode']=='gpu'
    assert guard['status']=='WORKER_EXITED' and op['guard_terminal'] is True
    for obj,keys in [(op,['guard_exit_code','guard_returncode','normal_guard_exit','normal_test_runner_exit']),
                     (guard,['exit_code','worker_exit_code'])]:
        assert all(type(obj[k]) is int and obj[k]==0 for k in keys)
    assert digest(raws['guard.stdout'])==op['stdout_sha256'] and digest(raws['guard.stderr'])==op['stderr_sha256']
    assert b'test result: ok. 1 passed; 0 failed; 0 ignored;' in raws['guard.stdout']
    runner = op['runner_observation']; workers = op['native_processes']
    assert len(workers)==1
    w = workers[0]
    assert runner['pid']==op['runner_pid']==guard['worker_pid']==w['runner_pid']
    assert runner['ppid']==op['guard_pid'] and runner['pgid']==runner['sid']==runner['pid']
    assert runner['uid']==cfg['lease_identity']['uid'] and type(runner['starttime']) is int and runner['starttime']>0
    L.admit_runner(runner,cfg)
    assert w['r70_worker_hello'] is True and w['rank']==7 and w['pid']==w['hello']['pid']
    assert w['exe']==L.PLAN['worker'] and w['exe_sha256']==L.PLAN['worker_sha256']
    a=w['stat_before'].rsplit(') ',1)[1].split();b=w['stat_after'].rsplit(') ',1)[1].split()
    assert a[19]==b[19] and int(a[19])>0 and a[1]==b[1]==str(runner['pid'])
    assert a[2]==a[3]==b[2]==b[3]==str(runner['pid'])
    pids=[op['outer_pid'],op['guard_pid'],runner['pid'],w['pid']]
    assert all(type(p) is int and p>0 for p in pids) and len(set(pids))==4
    assert post['pid_present']=={str(p):False for p in pids}
    assert all(v is False for v in post['pid_present'].values())
    assert post['groups_present']=={str(p):False for p in [op['guard_pid'],runner['pid']]}
    assert all(v is False for v in post['groups_present'].values())
    assert post['lease_free'] is True and post['lease_identity']==cfg['lease_identity']
    assert all(type(v) is int for v in post['lease_identity'].values())
    assert post['gpu_apps']=='' and post['source_hashes']==d['runtime'] and post['config_hashes']==cfg['hashes']
    cols=[x.strip() for x in post['gpu_state'].split(',')]
    assert len(cols)==3 and cols[0]==L.UUID and 0<=int(cols[1])<64 and int(cols[2])==0
    assert op['owned_processes_at_return']=={str(p):None for p in pids[1:]}
    assert res['effective_ids']==[L.NAMESPACE,L.NAMESPACE+'-duplicate-slot'] and res['reserved_before_submit'] is True
    assert res['ledger_root']==str(ledger_root)
    for n in res['effective_ids']:
        entry=strict(read(ledger_root/(n+'.json')))
        assert entry==res['entries'][n] and entry['status']=='RESERVED_UNKNOWN_NO_REPLAY'
        assert entry['base_id']==L.NAMESPACE and entry['effective_run_id']==n
        assert entry['context']==dict(phase='R81_R101',case_plan_sha256=L.PLAN_SHA,requests=L.PLAN['requests'],
                    hardware_invocation_limit=1,adapter_manifest_sha256=R96_MANIFEST_SHA)
    rows, proof = d['adapter'].validate_rows_log(raws['guard.stdout'])
    assert rows['exact_Collector_values']==2944
    # R96 validates stamp/binding/audit payloads; bind its HELLO to observed native PID too.
    validator, _ = d['adapter'].load_pinned()
    parsed = validator.rows_from_log(raws['guard.stdout'])
    hellos = [row for row in parsed if row.get('r70_worker_hello') is True]
    assert hellos == [w]
    return dict(status='R81_PRIVATE_SESSION_VERIFIED',namespace=L.NAMESPACE,hardware_invocations_used=1,
                row_result=rows,oracle_pins=proof,source_pins_sha256=PINS_SHA,
                adapter_manifest_sha256=R96_MANIFEST_SHA,performance_claim=False,
                R77_still_UNKNOWN=True,full_model=False,proof_sha256={n:digest(raw) for n,raw in raws.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['preflight','submit','collect','verify'])
    p.add_argument('--plan-sha256')
    p.add_argument('--allow-gpu',action='store_true')
    args=p.parse_args()
    d=dependencies()
    if args.action=='preflight':
        print(json.dumps(dict(status='CPU_OFFLINE_WRAPPER_PINS_VALID',GPU=False,namespace_reserved_by_this_command=False,
                              ledger_entries_present=[n for n in [d['L'].NAMESPACE,d['L'].NAMESPACE+'-duplicate-slot'] if os.path.lexists(LEDGER/(n+'.json'))],
                              plan_sha256=d['L'].PLAN_SHA,adapter_manifest_sha256=R96_MANIFEST_SHA)))
    elif args.action=='submit':
        assert args.allow_gpu and os.environ.get('VF_R81_ALLOW_GPU')=='1','parent HARDWARE_GATE and explicit R81 admission required'
        submit(d,args.plan_sha256)
        print('SUBMITTED_ONCE; run collect then verify; no automatic retry')
    elif args.action=='collect':
        collect(d)
        print('COLLECTED; offline verify required')
    else:
        result=verify(d)
        exclusive(ROOT/'final-result.json',encoded(result))
        print(json.dumps(result))


if __name__=='__main__':
    main()
