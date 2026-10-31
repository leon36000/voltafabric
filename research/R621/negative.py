"""One synthetic corruption check of the full-head oracle; no worker."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r621-work');plan=(w/'PLAN.json').read_bytes();p=json.loads(plan)
b=(w/'head-negative.bin').read_bytes()
assert hashlib.sha256(b).hexdigest()=='89b9aab70a949b92eff6354f620dbebdbbaafa3887ea7f6c0e1f03660b9351c8'
marker=w/'NEGATIVE_SUBMITTED';fd=os.open(marker,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,(hashlib.sha256(plan).hexdigest()+' '+hashlib.sha256(b).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd)
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','CPUQuota=50%','-p','TasksMax=16','/usr/bin/nice','-n','10','/usr/bin/ionice','-c','3',str(w/'full-head-check'),p['shard'],str(w/'head-negative.bin')]
r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
with (w/'NEGATIVE_RESULT').open('xb') as f:f.write(r.stdout);f.flush();os.fsync(f.fileno())
x=json.loads(r.stdout);assert r.returncode==9 and not r.stderr and x['violations']>=1
print(json.dumps({'exit':r.returncode,'violations':x['violations'],'observed_argmax':x['observed_argmax'],'reference_argmax':x['reference_argmax']}))
