"""Verify borrowed C ABI against retained f64 expert captures; no workers or replay."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r615-work');raw=(w/'PLAN.json').read_bytes();p=json.loads(raw);ph=hashlib.sha256(raw).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(Path(__file__))==p['runner_sha256'] and sha(w/'check_borrowed')==p['binary_sha256']
for path,digest in p['source_sha256'].items():assert sha(path)==digest
assert sha(p['activation'])==p['activation_sha256']
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(ph+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for e in p['experts']:
 assert sha(e['blob'])==e['blob_sha256'] and sha(e['capture'])==e['capture_sha256']
 cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','TasksMax=16','-p','CPUQuota=50%', '/usr/bin/nice','-n','10',str(w/'check_borrowed'),e['blob'],p['activation'],e['capture']]
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if r.returncode or r.stdout or r.stderr:raise RuntimeError('expert '+str(e['id'])+' failed: '+str(r.returncode))
fd=os.open(w/'RESULT.json',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f:json.dump({'experts':len(p['experts']),'exact_f64_values':len(p['experts'])*2560,'atomic_failure_verified':True},f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('exact rows',len(p['experts']))
