"""One-shot bounded CPU captures for ten already observed expert spans."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R604';w=root/'.runs/vf-r604-work';raw=(p/'CAPTURE_PLAN.json').read_bytes();plan=json.loads(raw);h=lambda path:hashlib.sha256((root/path).read_bytes()).hexdigest()
assert h('research/R604/capture.py')==plan['runner_sha256'];assert h(plan['binary'])==plan['binary_sha256'];assert h(plan['source'])==plan['source_sha256'];assert h(plan['activation'])==plan['activation_sha256']
for x in plan['experts']:assert h(x['path'])==x['sha256']
assert not any((w/'captures').iterdir())
fd=os.open(w/'CAPTURE_SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(hashlib.sha256(raw).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd);fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for x in plan['experts']:
 subprocess.run([str(root/plan['binary']),str(root/x['path']),str(root/plan['activation']),str(w/'captures'/f"expert{x['expert']}.bin")],check=True,timeout=30)
print('PASS scalar captures',len(plan['experts']))
