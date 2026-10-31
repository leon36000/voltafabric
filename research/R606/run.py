"""One fresh, bounded PC1-only layer transaction; never retry a submitted identity."""
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path

work = Path('/home/pc1/NAOS/vf-r606-work')
project = work / 'project'
packdir = Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s')
model = Path('/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf')
layer = work / 'target/release/vf-r606-scalar-layer'
worker = work / 'target/release/vf-rank-session'
ledger = work / 'ledger'
fixtures = project / 'fixtures'
res0 = Path('/home/pc1/NAOS/vf-r428-work/residual0.f32')
res1 = Path('/home/pc1/NAOS/vf-r428-work/residual1.f32')
args = [fixtures/'r418/layer2.json', packdir/'dense.bin', model, res0, res1,
        packdir/'experts.bin', packdir/'native_experts.txt', worker, ledger,
        fixtures/'r422/layer2.json', fixtures/'r424/layer2.json']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def exclusive(path, data):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

assert sha(res0) == 'ed1d74a7f93128f6b6a3b30855687a660cba93eca8e1c76c72086dc942fa7061'
assert sha(res1) == '9a69b12aec635edd0a042269e830779b49be0db018f6812479f617867371d96d'
assert sha(packdir/'native_experts.txt') == 'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24'
for p, digest in [(args[0], 'd822695e4fceee7b285b510e9ec7f97cc0a1507188b6ca698de5b6b216073348'),
                  (args[9], '8959d5ff4bbac4c6be08e6a978c5d27ffa1f09f8e2f3541add9395368121b0f1'),
                  (args[10], '2a9a91e11381fb9ef59636437c7f4649682cc1a89877e44818cd950c097e5d0c')]:
    assert sha(p) == digest
mem = {}
for line in Path('/proc/meminfo').read_text().splitlines():
    k, v = line.split(':', 1)
    if v.strip(): mem[k] = int(v.split()[0])
assert mem['MemAvailable'] >= 8 * 1024 * 1024 and mem['SwapFree'] >= 1024 * 1024
assert os.statvfs(work).f_bavail * os.statvfs(work).f_frsize >= 10 * 1024**3
assert not ledger.exists() and not (work/'SUBMITTED').exists()
ledger.mkdir(mode=0o700)
source_paths = sorted(p for d in (project/'src', project/'crates', project/'adapters/native_ggml')
                      for p in d.rglob('*') if p.is_file())
source_hash = hashlib.sha256()
for p in source_paths:
    source_hash.update(str(p.relative_to(project)).encode()+b'\0'+bytes.fromhex(sha(p)))
run_id = 'r606-' + uuid.uuid4().hex
plan = {'run_id':run_id, 'layer_binary_sha256':sha(layer), 'worker_sha256':sha(worker),
        'runner_sha256':sha(Path(__file__)), 'layer_source_sha256':sha(project/'research/R606/layer.rs'),
        'source_tree_sha256':source_hash.hexdigest(), 'native_scalar_library_sha256':sha(work/'native/libvf_native_scalar_reference.a'),
        'ggml_base_sha256':sha(work/'native/ggml/src/libggml-base.a'), 'ggml_cpu_sha256':sha(work/'native/ggml/src/libggml-cpu.a'),
        'args':[str(p) for p in args], 'checkpoint_stat':{str(p):{'device':p.stat().st_dev,'inode':p.stat().st_ino,'size':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in (args[1],args[2],args[5])},
        'scope':{'memory_max_bytes':4*1024**3,'swap_max_bytes':0,'cpu_quota_percent':200,'tasks_max':64},
        'gpu':False,'network_worker':False,'claims':'UNKNOWN never replay'}
raw=(json.dumps(plan,sort_keys=True,indent=2)+'\n').encode(); digest=hashlib.sha256(raw).hexdigest()
exclusive(work/'PLAN.json',raw)
exclusive(work/'SUBMITTED',(digest+'\n').encode())
env=os.environ.copy(); env.update(TMPDIR=str(work/'tmp'), OMP_NUM_THREADS='1',
    VF_R606_RUN_ID=run_id, VF_R606_PLAN_SHA256=digest, VF_R606_WORKER_SHA256=plan['worker_sha256'])
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=4G','-p','MemorySwapMax=0',
     '-p','TasksMax=64','-p','CPUQuota=200%','/usr/bin/ionice','-c2','-n7','/usr/bin/nice','-n','10',str(layer),*[str(p) for p in args]]
with (work/'stdout').open('xb') as out, (work/'stderr').open('xb') as err:
    process=subprocess.run(cmd, env=env, cwd=project, stdout=out, stderr=err)
    out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
exclusive(work/'EXIT_STATUS',(str(process.returncode)+'\n').encode())
print('exit',process.returncode,'plan',digest)
raise SystemExit(process.returncode)
