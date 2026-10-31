"""Read the hash-pinned historical local result without displaying its contents."""
import hashlib,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'.runs/vf-r605-work/r428.json';expected='31800dc58ccac050049ce417dd0bdb61143e3d022c8ab9cae210f02ccfec8cd4'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','pc1-lan','cat /home/pc1/NAOS/vf-r428-work/result.json'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20,check=True)
assert len(r.stdout)<1_000_000 and hashlib.sha256(r.stdout).hexdigest()==expected
fd=os.open(out,os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(r.stdout);f.flush();os.fsync(f.fileno())
print('PASS historical result SHA-256')
