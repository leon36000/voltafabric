"""Capture bounded per-expert scalar and vector outputs from retained verified inputs."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r612-pairs'); prior=Path('/home/pc1/NAOS/vf-r612-witness')
raw=(w/'PLAN.json').read_bytes();p=json.loads(raw);ph=hashlib.sha256(raw).hexdigest()
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
assert sha(Path(__file__))==p['runner_sha256'] and sha(w/'expert_pair')==p['binary_sha256']
assert sha(prior/'activation.f32')==p['activation_sha256']
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(ph+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
rows=[]
for e in p['experts']:
 blob=prior/('expert-'+str(e['id'])+'.bin');assert sha(blob)==e['sha256']
 out=w/('expert-'+str(e['id'])+'.bin')
 cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','TasksMax=16','-p','CPUQuota=50%','/usr/bin/nice','-n','10',str(w/'expert_pair'),str(blob),str(e['id']),str(out),str(prior/'activation.f32')]
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if r.returncode or r.stderr:raise RuntimeError('expert '+str(e['id'])+' status '+str(r.returncode))
 assert out.stat().st_size==2560*12
 rows.append({'expert':e['id'],'sha256':sha(out),'output':r.stdout.decode()})
fd=os.open(w/'RESULT.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f:json.dump(rows,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('captured',len(rows))
