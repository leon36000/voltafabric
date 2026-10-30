"""CPU-only contracts: synthetic rows, fake GPU inventory, private ledger/lease.
Real Python pipes/processes/fsync/pins; no SSH, CUDA, model submission or shared lease.
"""
import copy
import io
import tarfile
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock
import wrapper as W

D = W.dependencies()
L = D['L']


def synthetic_rows():
    validator, _ = D['adapter'].load_pinned()
    path = D['adapter'].R88.parent/'test_nominal_contract.py'
    with mock.patch.dict(sys.modules,{'nominal_contract':validator}):
        m = W.module(path,W.read(path))
    return m.synthetic()[0]


def write(path,value):
    path.write_bytes(W.encoded(value))


def ready():
    return dict(status='R81_READY_NO_COMPUTE',UUID=L.UUID,lease_device=1,lease_inode=3,
                lease_uid=os.getuid(),nvidia_smi_sha256='a'*64,timestamp_monotonic_ns=100,
                GPU_compute=False,namespace_reserved=False,lease_free_tested=True,lease_held_after_return=False)


def case_fixture(root):
    """Every process fact is explicitly fabricated CPU fixture data, never GPU evidence."""
    ledger=root/'CPU_FAKE_LEDGER';ledger.mkdir()
    r=ready()
    write(root/'READINESS.json',r)
    write(root/'readiness-once.terminal.json',dict(exit_code=0,process_terminal=True,end_ns=time.monotonic_ns()))
    case,res=W.reserve(D,r,L.PLAN_SHA,root,ledger)
    cfg=case['config'];rows=synthetic_rows();worker=rows[0]
    output=b'\n'.join(W.encoded(row).rstrip(b'\n') for row in rows)+b'\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'
    runner=dict(pid=101,ppid=100,pgid=101,sid=101,uid=os.getuid(),starttime=99,
                exe=cfg['argv'][0],exe_sha256=L.PLAN['harness_sha256'],
                cmdline_sha256=W.digest(b'\0'.join(x.encode() for x in cfg['argv'])+b'\0'))
    guard=dict(config=cfg,lease_identity=cfg['lease_identity'],status='WORKER_EXITED',exit_code=0,worker_exit_code=0,worker_pid=101)
    op=dict(case,status='R81_SESSION_PROCESS_TERMINAL_OFFLINE_ORACLE_AND_POSTFLIGHT_PENDING',
            guard_result=guard,complete_GPU_qualification=False,GPU_mode=True,guard_seconds=90,
            outer_seconds=105,guard_terminal=True,guard_exit_code=0,guard_returncode=0,
            normal_guard_exit=0,normal_test_runner_exit=0,stdout_sha256=W.digest(output),
            stderr_sha256=W.digest(b''),runner_observation=runner,native_processes=[worker],
            runner_pid=101,guard_pid=100,outer_pid=99,owned_processes_at_return={'100':None,'101':None,'102':None})
    post=dict(pid_present={str(p):False for p in [99,100,101,102]},groups_present={'100':False,'101':False},
              lease_free=True,lease_identity=cfg['lease_identity'],gpu_apps='',source_hashes=D['runtime'],
              config_hashes=cfg['hashes'],gpu_state=L.UUID+', 0, 0')
    dest=root/'gpu-result';dest.mkdir()
    for name,value in [('operation.json',op),('guard.json',guard),('postflight.json',post),('READINESS.json',r)]:
        write(dest/name,value)
    (dest/'RESERVATION.json').write_bytes(res)
    (dest/'RUNTIME.json').write_bytes(D['runtime_raw'])
    (dest/'guard.stdout').write_bytes(output)
    (dest/'guard.stderr').write_bytes(b'')
    refresh_fake_transport(root)
    return ledger


def refresh_fake_transport(root):
    dest=root/'gpu-result'
    op=W.strict(W.read(dest/'operation.json'));case=W.strict(W.read(root/'CASE_SUBMISSION.json'))
    b=io.BytesIO()
    with tarfile.open(fileobj=b,mode='w:') as archive:
        for name in sorted(W.EVIDENCE_NAMES):
            raw=W.read(dest/name);member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
    raw=b.getvalue();(root/'COLLECTED.tar').write_bytes(raw)
    r=W.strict(W.read(dest/'READINESS.json'))
    for name,body,out in [
        ('readiness-once',W.readiness_body(D),W.encoded(dict(readiness=r,readiness_sha256=W.digest(W.read(dest/'READINESS.json'))))),
        ('submit-once',W.submit_body(D,case,W.read(dest/'RESERVATION.json')),W.encoded(op)),
        ('postflight-once',W.collect_body(D),raw)]:
        command=W.command_for(body);(root/(name+'.command')).write_bytes(command)
        write(root/(name+'.intent.json'),dict(command_sha256=W.digest(command),automatic_retry=False))
        write(root/(name+'.bridge.json'),dict(command_sha256=W.digest(command),automatic_retry=False,status='TERMINAL_TRANSPORT',auth_ready=True,errors=[],exit_code=0,ssh_exit_code=0,output_bytes=len(out),stderr_bytes=0))
        write(root/(name+'.terminal.json'),dict(exit_code=0,process_terminal=True,stdout_sha256=W.digest(out),stderr_sha256=W.digest(b'')))
        (root/(name+'.stdout')).write_bytes(out);(root/(name+'.stderr')).write_bytes(b'')


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=W.ROOT,prefix='CPU_ONLY_')
        self.root=Path(self.tmp.name)
    def tearDown(self):
        self.tmp.cleanup()

    def test_pinned_dependencies_and_templates_compile_offline(self):
        self.assertEqual(D['adapter'].R88_SHA,'c810319dea715523906ac76a1c2bc2aef71b217a313ec28df4b0e22678d31b3c')
        compile(W.readiness_body(D),'CPU inspect readiness','exec')
        compile(W.collect_body(D),'CPU inspect postflight','exec')
        ledger=case_fixture(self.root)
        case=W.strict(W.read(self.root/'CASE_SUBMISSION.json'))
        body=W.submit_body(D,case,W.read(self.root/'RESERVATION.json'))
        compile(body,'CPU inspect submit','exec')
        self.assertIn("os.environ['VF_R81_ALLOW_GPU']='1'",body)
        self.assertIn(case['case_plan_sha256'],body)
        self.assertEqual(case['config']['mode'],'gpu')

    def test_real_pipe_success_nonzero_cap_timeout(self):
        for code in [0,7]:
            result=W.capture([sys.executable,'-B','-c',f'import sys;print("CPU pipe");sys.stderr.write("CPU stderr");sys.exit({code})'],2,self.root)
            self.assertEqual(result,(code,b'CPU pipe\n',b'CPU stderr'))
        with self.assertRaisesRegex(ValueError,'cap'):
            W.capture([sys.executable,'-B','-c','print("x"*4096)'],2,self.root,cap=100)
        with self.assertRaises(TimeoutError):
            W.capture([sys.executable,'-B','-c','import time;time.sleep(1)'],.05,self.root)

    def test_real_fsync_exclusive_and_partial_ledger_consumption(self):
        p=self.root/'receipt'
        seen=[];real=os.fsync
        def fsync(fd):
            seen.append(os.fstat(fd).st_mode)
            return real(fd)
        with mock.patch.object(W.os,'fsync',side_effect=fsync):
            W.exclusive(p,b'CPU persisted')
        self.assertEqual(len(seen),2)
        with self.assertRaises(FileExistsError):W.exclusive(p,b'changed')
        ledger=self.root/'ledger';ledger.mkdir()
        base='r53-CPU-private-partial'
        (ledger/(base+'-duplicate-slot.json')).write_bytes(b'PREEXISTING')
        with self.assertRaises(FileExistsError):D['ledger'].reserve(ledger,base,{'CPU_ONLY':True})
        self.assertTrue((ledger/(base+'.json')).exists())
        with self.assertRaises(FileExistsError):D['ledger'].reserve(ledger,base,{'CPU_ONLY':True})

    def test_at_most_once_unknown_before_transport(self):
        ledger=self.root/'ledger';ledger.mkdir()
        write(self.root/'readiness-once.terminal.json',dict(exit_code=0,process_terminal=True,end_ns=time.monotonic_ns()))
        r=ready()
        with self.assertRaisesRegex(ValueError,'external'):
            W.reserve(D,r,'0'*64,self.root,ledger)
        self.assertEqual(list(ledger.iterdir()),[])
        case,raw=W.reserve(D,r,L.PLAN_SHA,self.root,ledger)
        self.assertEqual(len(list(ledger.iterdir())),2)
        self.assertEqual(W.strict(W.read(self.root/'CASE_SUBMISSION.json'))['status'],'R81_SUBMITTED_UNKNOWN_NO_REPLAY')
        with mock.patch.object(W,'bridge',side_effect=AssertionError('must never contact SSH')):
            with self.assertRaises(FileExistsError):W.submit(D,L.PLAN_SHA,self.root)
        with self.assertRaises(FileExistsError):W.reserve(D,r,L.PLAN_SHA,self.root,ledger)

    def test_submit_transport_failure_keeps_consumed_unknown_without_retry(self):
        ledger=self.root/'CPU_ONLY_LEDGER';ledger.mkdir()
        write(self.root/'readiness-once.terminal.json',dict(exit_code=0,process_terminal=True,end_ns=time.monotonic_ns()))
        actual_reserve=W.reserve
        def reserve_private(d,r,h,root):
            return actual_reserve(d,r,h,root,ledger)
        calls=[]
        def failed_transport(name,body,timeout,d,root):
            self.assertEqual(len(list(ledger.iterdir())),2)
            self.assertEqual(W.strict(W.read(root/'CASE_SUBMISSION.json'))['status'],'R81_SUBMITTED_UNKNOWN_NO_REPLAY')
            calls.append(name)
            code,out,err=W.capture([sys.executable,'-B','-c','raise SystemExit(7)'],2,root)
            self.assertEqual(code,7)
            raise RuntimeError('CPU pipe simulates uncertain SSH exit; no network')
        with mock.patch.object(W,'readiness',return_value=ready()), mock.patch.object(W,'reserve',side_effect=reserve_private), mock.patch.object(W,'bridge',side_effect=failed_transport):
            with self.assertRaises(RuntimeError):W.submit(D,L.PLAN_SHA,self.root)
            with self.assertRaises(FileExistsError):W.submit(D,L.PLAN_SHA,self.root)
        self.assertEqual(calls,['submit-once'])

    def test_unchanged_snapshots_execute_not_reopened_path(self):
        p=self.root/'module.py';raw=b'VALUE=17\n';p.write_bytes(raw)
        snapshot=W.read(p,W.digest(raw));p.write_text('raise RuntimeError("changed path")')
        self.assertEqual(W.module(p,snapshot).VALUE,17)
        with self.assertRaisesRegex(ValueError,'digest'):W.read(p,W.digest(raw))
        link=self.root/'link';link.symlink_to(p)
        with self.assertRaises(OSError):W.read(link)
        with self.assertRaisesRegex(ValueError,'duplicate'):W.strict(b'{"x":1,"x":2}')

    def test_full_synthetic_chain_and_incomplete_postflight(self):
        ledger=case_fixture(self.root)
        result=W.verify(D,self.root,ledger)
        self.assertEqual(result['row_result']['exact_Collector_values'],2944)
        p=self.root/'gpu-result/postflight.json';raw=p.read_bytes();post=W.strict(raw)
        for field,value in [('pid_present',{}),('groups_present',{}),('lease_free',False),('gpu_apps','123'),
                            ('config_hashes',{}),('source_hashes',{}),('gpu_state',L.UUID+', 100, 0')]:
            with self.subTest(field=field):
                write(p,{**post,field:value});refresh_fake_transport(self.root)
                with self.assertRaises(AssertionError):W.verify(D,self.root,ledger)
        p.write_bytes(raw);refresh_fake_transport(self.root)
        self.assertEqual(W.verify(D,self.root,ledger)['status'],'R81_PRIVATE_SESSION_VERIFIED')

    def test_altered_binding_rejected_after_consistent_log_hash_update(self):
        ledger=case_fixture(self.root)
        out=self.root/'gpu-result/guard.stdout'
        rows=synthetic_rows()
        audits=[json.loads(x) for x in rows[5]['close']['stderr'].splitlines()]
        audits[1]['binding']['artifact_sha256']='0'*64
        rows[5]['close']['stderr']='\n'.join(json.dumps(x) for x in audits)
        raw=b'\n'.join(W.encoded(r).strip() for r in rows)+b'\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n'
        out.write_bytes(raw)
        p=self.root/'gpu-result/operation.json';op=W.strict(p.read_bytes());op['stdout_sha256']=W.digest(raw);write(p,op);refresh_fake_transport(self.root)
        with self.assertRaises(AssertionError):W.verify(D,self.root,ledger)

    def test_exit_stamp_and_config_corruptions(self):
        ledger=case_fixture(self.root)
        p=self.root/'gpu-result/operation.json';raw=p.read_bytes();op=W.strict(raw)
        for key,value in [('guard_exit_code',False),('guard_returncode',7),('normal_test_runner_exit',None),
                          ('GPU_mode',False),('outer_seconds',120),('guard_terminal',False),
                          ('case_plan_sha256','0'*64),('native_processes',[]),('owned_processes_at_return',{})]:
            with self.subTest(key=key):
                write(p,{**op,key:value});refresh_fake_transport(self.root)
                with self.assertRaises(AssertionError):W.verify(D,self.root,ledger)
        p.write_bytes(raw)
        mutated=copy.deepcopy(op);mutated['config']['mode']='cpu';write(p,mutated);refresh_fake_transport(self.root)
        with self.assertRaises(AssertionError):W.verify(D,self.root,ledger)

    def test_real_readiness_template_fake_GPU_no_shared_lease(self):
        # Execute the actual generated template, changing only explicit CPU paths and inventory program.
        target=self.root/'CPU_FAKE_VM';target.mkdir();lease=self.root/'CPU_PRIVATE.lease';lease.write_bytes(b'CPU_ONLY')
        for name in ['r81_contract.py','r81_outer.py','CASE_PLAN.json','PACKAGE.json']:
            (target/name).write_bytes(W.read(W.R81/name))
        guard=W.RUNS/'host-input-native-owner-r76-20261027/guard-private/worker_guard_r76.py'
        (target/'worker_guard.py').write_bytes(W.read(guard,L.GUARD_SHA))
        (target/'RUNTIME.json').write_bytes(D['runtime_raw'])
        smi=self.root/'CPU_FAKE_nvidia-smi'
        smi.write_text('#!/usr/bin/python3\nimport sys\n'
            'a=" ".join(sys.argv)\n'
            f'print({L.UUID!r}+", CPU FAKE V100, 0000:00:00, 7.0, 16384, 0, 0, CPU_FAKE" if "query-gpu" in a else "CPU FAKE TOPOLOGY" if "topo" in a else "")\n')
        smi.chmod(0o700)
        body=W.readiness_body(D).replace(L.TARGET,str(target)).replace(L.LEASE,str(lease)).replace('/usr/bin/nvidia-smi',str(smi)).replace('==1000','=='+str(os.getuid()))
        code,out,err=W.capture([sys.executable,'-B','-c',body],5,self.root)
        self.assertEqual(code,0,err)
        result=W.strict(out);self.assertEqual(result['readiness']['status'],'R81_READY_NO_COMPUTE')
        self.assertFalse(result['readiness']['namespace_reserved'])
        self.assertFalse(result['readiness']['lease_held_after_return'])
        self.assertEqual(result['readiness']['lease_inode'],lease.stat().st_ino)
        self.assertEqual(W.digest(W.read(target/'READINESS.json')),result['readiness_sha256'])
        self.assertFalse((target/'control').exists())

    def test_R76_inode_pin_real_guard_fake_GPU_pipeline(self):
        # Reuse existing R81 real guard CPU test seam; redirect all writes to this sibling.
        contract=D['L'];outerpath=W.R81/'r81_outer.py'
        with mock.patch.dict(sys.modules,{'r81_contract':contract}):
            outer=W.module(outerpath,D['snapshots'][str(outerpath.relative_to(W.RUNS))])
        with mock.patch.dict(sys.modules,{'r81_contract':contract,'r81_outer':outer}):
            source=W.R81/'test_r81_pipeline.py';pipeline=W.module(source,W.read(source))
        (self.root/'tmp').mkdir()
        # Case requires the original source bytes in its redirected ROOT.
        for name in ['r81_contract.py','r81_outer.py']:(self.root/name).write_bytes(W.read(W.R81/name))
        pipeline.ROOT=self.root
        for name in ['test_real_guard_and_pipeline_normal_CPU_double','test_nonzero_child_keeps_unknown_CPU_double','test_stale_lease_pin_refused_before_control_launch']:
            test=pipeline.FullPipeline(name)
            result=unittest.TestResult();test.run(result)
            self.assertTrue(result.wasSuccessful(),str(result.errors)+str(result.failures))

    def test_missing_or_uncertain_transport_exits_cannot_qualify(self):
        ledger=case_fixture(self.root)
        for name in ['readiness-once','submit-once','postflight-once']:
            p=self.root/(name+'.bridge.json');raw=p.read_bytes();receipt=W.strict(raw)
            for key,value in [('ssh_exit_code',7),('exit_code',False),('auth_ready',False),('errors',['CPU fake disconnect']),('command_sha256','0'*64)]:
                with self.subTest(name=name,key=key):
                    write(p,{**receipt,key:value})
                    with self.assertRaises(AssertionError):W.verify(D,self.root,ledger)
            p.write_bytes(raw)
        (self.root/'submit-once.terminal.json').unlink()
        with self.assertRaises(FileNotFoundError):W.verify(D,self.root,ledger)

    def test_optimized_python_refused(self):
        q=subprocess.run([sys.executable,'-O','-B',str(W.ROOT/'wrapper.py'),'preflight'],capture_output=True,timeout=10)
        self.assertNotEqual(q.returncode,0)
        self.assertIn(b'optimization forbidden',q.stderr)


if __name__=='__main__':
    unittest.main(verbosity=2)
