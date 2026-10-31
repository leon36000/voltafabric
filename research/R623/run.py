"""Read-only full-vocabulary CPU oracle; no worker or replay of any claim."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r623-work')
p=json.loads((w/'PLAN.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  while chunk:=f.read(1024*1024):h.update(chunk)
 return h.hexdigest()
for path,expected in p['file_sha256'].items():
 if sha(path)!=expected:raise RuntimeError('input or executable changed: '+path)
manifest=json.loads((w/'fixtures/r429/head.json').read_text())
with open(p['shard'],'rb') as f:
 before=os.fstat(f.fileno())
 if before.st_size!=manifest['shard_bytes']:raise RuntimeError('shard size')
 for chunk in manifest['chunks']:
  f.seek(chunk['offset']); left=chunk['bytes'];h=hashlib.sha256()
  while left:
   b=f.read(min(left,1024*1024))
   if not b:raise RuntimeError('truncated chunk')
   left-=len(b);h.update(b)
  if h.hexdigest()!=chunk['sha256']:raise RuntimeError('chunk digest')
 after=os.fstat(f.fileno())
 if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise RuntimeError('shard changed')
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
os.write(fd,(sha(w/'PLAN.json')+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','CPUQuota=50%','-p','TasksMax=16','/usr/bin/nice','-n','10','/usr/bin/ionice','-c','3','/home/pc1/NAOS/vf-r621-work/full-head-check',p['shard'],p['head']]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
current=os.stat(p['shard'])
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns):raise RuntimeError('shard changed during check')
if sha(p['head'])!=p['file_sha256'][p['head']]:raise RuntimeError('head changed during check')
with (w/'EXIT_STATUS').open('x') as f:f.write(str(r.returncode)+'\n');f.flush();os.fsync(f.fileno())
with (w/'STDOUT').open('xb') as f:f.write(r.stdout);f.flush();os.fsync(f.fileno())
with (w/'STDERR').open('xb') as f:f.write(r.stderr);f.flush();os.fsync(f.fileno())
print('exit',r.returncode,'result',r.stdout.decode().strip(),'stderr_bytes',len(r.stderr))
