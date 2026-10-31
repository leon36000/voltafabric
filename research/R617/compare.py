"""Compare one retained PC1 corpus with PC6's existing scalar expert ABI.

This runs no worker and creates no distributed claims. A mismatch is evidence,
not grounds to retry a previously submitted operation.
"""
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[2]
work = root / '.runs/vf-r617-work'
source = json.loads((root / 'research/R615/PLAN.json').read_text())
formats = json.loads((root / 'research/R615/PLAN-formats.json').read_text())
binary = root / formats['binary']
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
assert digest(binary) == formats['binary_sha256']
activation = work / 'vf-r612-witness/activation.f32'
assert digest(activation) == source['activation_sha256']
for expert in source['experts']:
    blob = work / 'vf-r612-witness' / Path(expert['blob']).name
    capture = work / 'vf-r612-pairs' / Path(expert['capture']).name
    assert digest(blob) == expert['blob_sha256']
    assert digest(capture) == expert['capture_sha256']
marker = work / 'SUBMITTED'
fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
os.write(fd, b'R617 local cross-host corpus compare, no worker\n')
os.fsync(fd)
os.close(fd)
results = []
for expert in source['experts']:
    blob = work / 'vf-r612-witness' / Path(expert['blob']).name
    capture = work / 'vf-r612-pairs' / Path(expert['capture']).name
    raw = capture.read_bytes()
    assert len(raw) == 30720
    expected = work / ('expected-' + str(expert['id']) + '.f64')
    with expected.open('xb') as f:
        f.write(raw[10240:])
    cmd = ['systemd-run', '--user', '--scope', '--quiet', '-p', 'MemoryMax=512M', '-p', 'MemorySwapMax=0', '-p', 'TasksMax=16', '-p', 'CPUQuota=50%', '/usr/bin/nice', '-n', '10', str(binary), str(blob), str(activation), str(expected), '18', '20']
    completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    results.append({'expert': expert['id'], 'exit_code': completed.returncode, 'stderr': completed.stderr.decode(errors='replace')[:500]})
    if completed.returncode not in (0, 5) or completed.stdout or completed.stderr:
        break
out = {'cases': results, 'binary_sha256': formats['binary_sha256'], 'input_sha256': source['activation_sha256'], 'exact': sum(r['exit_code'] == 0 for r in results), 'mismatch': sum(r['exit_code'] == 5 for r in results), 'gpu': False, 'network_worker': False}
with (work / 'RESULT.json').open('x') as f:
    json.dump(out, f, indent=2)
    f.write('\n')
    f.flush()
    os.fsync(f.fileno())
print(json.dumps(out))
