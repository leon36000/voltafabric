"""Fresh local layer-0 capture, one backend per immutable plan."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
mode=sys.argv[1];assert mode in ('scalar','vector')
w=Path('/home/pc1/NAOS')/('vf-r612-'+mode)
raw=(w/'PLAN.json').read_bytes();p=json.loads(raw);digest=hashlib.sha256(raw).hexdigest()
assert p['backend']==mode and p['run_id'].startswith('r612-')
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==p['runner_sha256']
for name,expected in p['file_sha256'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
assert not (w/'SUBMITTED').exists() and not any((w/'ledger').iterdir())
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 if v.strip():mem[k]=int(v.split()[0])
assert mem['MemAvailable']>=4*1024**2 and mem['SwapFree']>=1024**2
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
assert 'vf-rank-session' not in subprocess.check_output(['ps','-eo','comm'],text=True).splitlines()
fd=os.open(w/'SUBMITTED',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(digest+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
pack=Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s')
shard=Path('/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf')
env={'HOME':'/home/pc1','PATH':'/usr/bin:/bin','TMPDIR':str(w/'tmp'),'OMP_NUM_THREADS':'1',
 'VF_R612_PLAN':digest,'VF_R612_RUN':p['run_id'],'VF_R612_WORKER':p['worker'],
 'VF_R612_WORKER_SHA256':p['worker_sha256'],'VF_R612_BACKEND':mode}
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryMax=2G','-p','MemorySwapMax=0','-p','TasksMax=32','-p','CPUQuota=100%',
 '/usr/bin/env','-i',*[f'{k}={v}' for k,v in env.items()],'/usr/bin/ionice','-c2','-n7','/usr/bin/nice','-n','10',p['binary'],
 str(pack/'dense.bin'),str(shard),str(pack/'experts.bin'),str(pack/'native_experts.txt'),str(w/'ledger')]
with (w/'RESULT.json').open('xb') as out,(w/'error.log').open('xb') as err:
 r=subprocess.run(cmd,stdout=out,stderr=err,env=os.environ.copy());out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
fd=os.open(w/'EXIT_STATUS',os.O_CREAT|os.O_EXCL|os.O_WRONLY|os.O_NOFOLLOW,0o600);os.write(fd,(str(r.returncode)+'\n').encode());os.fsync(fd);os.close(fd)
print(mode,'exit',r.returncode);raise SystemExit(r.returncode)
