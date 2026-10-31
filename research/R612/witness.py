"""One bounded ten-expert witness run; no sessions, replay or GPU."""
import hashlib,json,os,struct,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r612-witness');raw=(w/'PLAN.json').read_bytes();p=json.loads(raw);ph=hashlib.sha256(raw).hexdigest()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['runner_sha256']
assert hashlib.sha256((w/'witness').read_bytes()).hexdigest()==p['binary_sha256']
receipts=[]
for mode in ('scalar','vector'):
 f=Path('/home/pc1/NAOS')/('vf-r612-'+mode)/'RESULT.json';data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==p['receipt_sha256'][mode];receipts.append(json.loads(data))
assert receipts[0]['ids']==receipts[1]['ids']==[e['id'] for e in p['experts']]
assert receipts[0]['weights']==receipts[1]['weights'] and receipts[0]['activation']==receipts[1]['activation']
a=struct.pack('<2560f',*receipts[0]['activation']);assert hashlib.sha256(a).hexdigest()==p['activation_sha256']
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 if v.strip():mem[k]=int(v.split()[0])
assert mem['MemAvailable']>=4*1024**2
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(ph+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
with (w/'activation.f32').open('xb') as f:f.write(a);f.flush();os.fsync(f.fileno())
pack='/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s/experts.bin';fd=os.open(pack,os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd);assert before.st_size==50_292_326_400
rows=[]
for e in p['experts']:
 blob=os.pread(fd,2_176_000,e['id']*2_176_000);assert len(blob)==2_176_000 and hashlib.sha256(blob).hexdigest()==e['sha256']
 path=w/('expert-'+str(e['id'])+'.bin')
 with path.open('xb') as f:f.write(blob);f.flush();os.fsync(f.fileno())
 cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=512M','-p','MemorySwapMax=0','-p','TasksMax=16','-p','CPUQuota=50%', '/usr/bin/nice','-n','10',str(w/'witness'),str(path),str(e['id']),'/dev/null',str(w/'activation.f32')]
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120)
 if r.returncode not in (0,9) or r.stderr:raise RuntimeError('unexpected witness status '+str(e['id'])+' '+str(r.returncode))
 rows.append({'expert':e['id'],'exit':r.returncode,'output':r.stdout.decode()})
after=os.fstat(fd);os.close(fd)
assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns)
fd=os.open(w/'RESULT.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'w') as f:json.dump(rows,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('witnesses',len(rows))
