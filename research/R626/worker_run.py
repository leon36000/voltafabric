"""One-shot planned local synthetic CPU worker qualification; no replay."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
work = root / '.runs/vf-r626-worker'
raw = (root / 'research/R626/WORKER-PLAN.json').read_bytes()
plan = json.loads(raw)
plan_hash = hashlib.sha256(raw).hexdigest()
assert not any((work / 'ledger').iterdir())
for name, key in [(plan['worker'], 'worker_sha256'), (plan['test'], 'test_sha256'),
                  ('src/session.rs', 'session_source_sha256'), ('src/sc117_expert_source.rs', 'expert_source_sha256'), ('src/native_ggml_cpu.rs', 'native_source_sha256'), ('src/sc117_layer_chain.rs', 'layer_source_sha256'), ('research/R626/worker_run.py', 'runner_sha256'), ('fixtures/strata_sc117_native_experts_v3.txt', 'index_sha256')]:
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == plan[key]
for name, expected in plan['integration_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
for name, expected in plan['input_sha256'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
assert os.statvfs(work).f_bavail * os.statvfs(work).f_frsize >= 10 * 1024**3
with open('/proc/meminfo') as f:
    memory = {line.split(':')[0]: int(line.split()[1]) for line in f if len(line.split()) >= 2 and line.split()[1].isdigit()}
assert memory['MemAvailable'] >= 4 * 1024 * 1024
fd = os.open(work / 'SUBMITTED', os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
os.write(fd, (plan_hash + '\n').encode()); os.fsync(fd); os.close(fd)
fd = os.open(work, os.O_RDONLY | os.O_DIRECTORY); os.fsync(fd); os.close(fd)
env = os.environ.copy()
env.update(TMPDIR=str(work / 'tmp'), VF_R603_WORKER=str(root / plan['worker']),
           VF_R603_WORKER_SHA256=plan['worker_sha256'], VF_R603_PLAN=plan_hash,
           VF_R603_RUN=plan['run_id'], VF_R603_LEDGER=str(work / 'ledger'), VF_R603_OUTDIR=str(work))
cmd = ['systemd-run','--user','--scope','--quiet','-p','MemoryMax=1G','-p','MemorySwapMax=0',
       '-p','TasksMax=32','-p','CPUQuota=100%','/usr/bin/ionice','-c2','-n7','/usr/bin/nice','-n','10',
       str(root / plan['test']),'--ignored','--exact',plan['test_name'],'--nocapture']
with open(work / 'stdout', 'xb') as out, open(work / 'stderr', 'xb') as err:
    p = subprocess.run(cmd, cwd=root, env=env, stdout=out, stderr=err)
    out.flush(); err.flush(); os.fsync(out.fileno()); os.fsync(err.fileno())
with open(work / 'exit-status', 'x') as f:
    f.write(str(p.returncode) + '\n'); f.flush(); os.fsync(f.fileno())
print('exit', p.returncode)
raise SystemExit(p.returncode)
