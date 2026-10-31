"""Read seven bounded, hash-checked observed expert spans from PC1 once."""
import hashlib,json,os,shlex,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R604';w=root/'.runs/vf-r604-work';raw=(p/'FETCH_PLAN.json').read_bytes();plan=json.loads(raw)
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['fetcher_sha256']
assert len(plan['experts'])==7 and sum(x['bytes'] for x in plan['experts'])<15_000_000
assert not any((w/'experts').iterdir())
remote='''import os,sys,hashlib,json
p=json.loads(sys.argv[1]);fd=os.open(p['path'],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
identity=lambda s:[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
assert identity(before)==p['identity']
chunks=[]
for r in p['experts']:
 d=os.pread(fd,r['bytes'],r['offset']);assert len(d)==r['bytes'] and hashlib.sha256(d).hexdigest()==r['sha256'];chunks.append(d)
assert identity(os.fstat(fd))==identity(before)
for d in chunks:sys.stdout.buffer.write(d)
sys.stdout.buffer.flush()
'''
args={'path':plan['path'],'identity':plan['identity'],'experts':plan['experts']}
cmd=' '.join(['/usr/bin/nice','-n','10','/usr/bin/ionice','-c2','-n7','python3','-c',shlex.quote(remote),shlex.quote(json.dumps(args,separators=(',',':')))])
fd=os.open(w/'FETCH_SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(hashlib.sha256(raw).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd);fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=5','-o','StrictHostKeyChecking=yes','pc1-lan',cmd],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
if r.returncode:raise SystemExit('remote read failed: '+r.stderr.decode(errors='replace')[-500:])
assert len(r.stdout)==sum(x['bytes'] for x in plan['experts']);offset=0
for x in plan['experts']:
 data=r.stdout[offset:offset+x['bytes']];offset+=x['bytes'];assert hashlib.sha256(data).hexdigest()==x['sha256']
 fd=os.open(w/'experts'/f"expert{x['expert']}.bin",os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
print('PASS verified bounded expert spans',len(plan['experts']),'bytes',offset)
