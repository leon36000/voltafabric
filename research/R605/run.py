"""One-shot offline layer reassembly; no workers or ledger operations."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R605';w=root/'.runs/vf-r605-work';raw=(p/'PLAN.json').read_bytes();plan=json.loads(raw)
for path,digest in plan['inputs'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(hashlib.sha256(raw).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
subprocess.run([str(root/plan['binary']),str(w/'r428.json'),str(w)],cwd=root,check=True,timeout=30)
