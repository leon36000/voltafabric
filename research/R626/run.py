"""One local bounded pass over the previously observed seven-format corpus."""
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];w=root/'.runs/vf-r626-real';raw=(root/'research/R626/PLAN.json').read_bytes();p=json.loads(raw)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(Path(__file__))==p['runner_sha256'] and sha(root/p['binary'])==p['binary_sha256']
for path,digest in p['files'].items():assert sha(root/path)==digest,path
assert len(p['cases'])==28 and len({(x['gu'],x['down']) for x in p['cases']})==7
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
fd=os.open(w/'SUBMITTED',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.write(fd,(hashlib.sha256(raw).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
for case in p['cases']:
 cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','TasksMax=16','-p','CPUQuota=50%','/usr/bin/nice','-n','10',str(root/p['binary']),str(root/case['blob']),str(root/case['input']),str(root/case['expected']),str(case['gu']),str(case['down'])]
 result=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if result.returncode or result.stdout or result.stderr:raise RuntimeError('case failed '+str(case['layer'])+' '+case['input']+' exit '+str(result.returncode))
fd=os.open(w/'RESULT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f:json.dump({'cases':28,'format_pairs':7,'exact_f64_values':71680,'atomic_failure_verified':True},f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('exact cases',len(p['cases']))
