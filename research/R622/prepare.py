"""Fresh plan-only local CPU diagnostic; creates no worker or claim."""
import hashlib,json,os,subprocess
from pathlib import Path
w=Path('/home/pc1/NAOS/vf-r622-work');pk=Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s');md=Path('/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S');base='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S';worker=w/'target/release/vf-rank-session';binary=w/'target/release/vf-sc117-prefill-greedy'
tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
hashfile=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (w/'PREPARED.json').exists() and not (w/'SUBMITTED').exists() and not any((w/'ledger').iterdir())
env={'HOME':'/home/pc1','PATH':'/usr/bin:/bin','TMPDIR':str(w/'tmp'),'OMP_NUM_THREADS':'1','VF_SC117_NATIVE_BACKEND':'scalar','VF_LOCAL_WORKER_SHA256':hashfile(worker),'VF_PREFILL_GROUP_SIZE':'2','VF_EXPERT_RANK_AFFINITY':'sticky','VF_SC117_HEAD_TRACE':'1'}
args=[str(binary),'--plan-only',str(pk/'dense.bin'),str(md/(base+'-00001-of-00002.gguf')),str(pk/'experts.bin'),str(pk/'native_experts.txt'),str(worker),str(w/'ledger'),str(md/(base+'-00002-of-00002.gguf')),'/home/pc1/NAOS/vf-r443-work/manifest.json','48',','.join(map(str,tokens)),'1','248046','chat']
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,timeout=120)
assert not r.stderr
prepared=json.loads(r.stdout);assert set(prepared)=={'plan_id','native_math_backend','prompt_tokens_sha256'} and prepared['native_math_backend']=='scalar-reference-ggml-v1'
with (w/'PREPARED.json').open('xb') as f:f.write(r.stdout);f.flush();os.fsync(f.fileno())
print(json.dumps(prepared))
