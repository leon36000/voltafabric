"""One admitted local scalar run with a private, plan-bound head trace."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r622-work'); raw=(w/'PLAN.json').read_bytes();p=json.loads(raw)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
for path,digest in p['file_sha256'].items():assert sha(path)==digest,path
assert sha(Path(__file__))==p['submit_sha256']
assert json.loads((w/'PREPARED.json').read_bytes())['plan_id']==p['engine_plan_id']
assert not any((w/'ledger').iterdir()) and not (w/'SUBMITTED').exists()
assert os.statvfs(w).f_bavail*os.statvfs(w).f_frsize>=10*1024**3
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
 k,v=line.split(':',1)
 if v.strip():mem[k]=int(v.split()[0])
assert mem['MemAvailable']>=16*1024**2 and mem['SwapFree']>=1024**2
assert 'vf-rank-session' not in subprocess.check_output(['ps','-eo','comm'],text=True).splitlines()
fd=os.open(w/'SUBMITTED',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.write(fd,(hashlib.sha256(raw).hexdigest()+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(w,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
pk=Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s');md=Path('/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S');base='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S';worker=w/'target/release/vf-rank-session';binary=w/'target/release/vf-sc117-prefill-greedy';tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
env={'HOME':'/home/pc1','PATH':'/usr/bin:/bin','TMPDIR':str(w/'tmp'),'OMP_NUM_THREADS':'1','VF_SC117_NATIVE_BACKEND':'scalar','VF_LOCAL_WORKER_SHA256':p['file_sha256'][str(worker)],'VF_PREFILL_GROUP_SIZE':'2','VF_EXPERT_RANK_AFFINITY':'sticky','VF_SC117_HEAD_TRACE':'1','VF_EXPECTED_PLAN_ID':p['engine_plan_id']}
args=[str(binary),str(pk/'dense.bin'),str(md/(base+'-00001-of-00002.gguf')),str(pk/'experts.bin'),str(pk/'native_experts.txt'),str(worker),str(w/'ledger'),str(md/(base+'-00002-of-00002.gguf')),'/home/pc1/NAOS/vf-r443-work/manifest.json','48',','.join(map(str,tokens)),'1','248046','chat']
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryHigh=11G','-p','MemoryMax=13G','-p','MemorySwapMax=0','-p','TasksMax=320','-p','CPUQuota=200%','/usr/bin/env','-i',*[f'{k}={v}' for k,v in env.items()],'/usr/bin/ionice','-c2','-n7','/usr/bin/nice','-n','10',*args]
with (w/'RESULT.json').open('xb') as out,(w/'error.log').open('xb') as err:
 r=subprocess.run(cmd,stdout=out,stderr=err,cwd=w);out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
fd=os.open(w/'EXIT_STATUS',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.write(fd,(str(r.returncode)+'\n').encode());os.fsync(fd);os.close(fd)
print('exit',r.returncode,flush=True)
raise SystemExit(r.returncode)
