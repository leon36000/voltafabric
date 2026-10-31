"""One-shot local CPU scalar generation, externally planned; UNKNOWN is never replayed."""
import hashlib,json,os,subprocess
from pathlib import Path
work=Path('/home/pc1/NAOS/vf-r616-work'); plan_bytes=(work/'PLAN.json').read_bytes()
plan=json.loads(plan_bytes); digest=hashlib.sha256(plan_bytes).hexdigest()
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==plan['submit_sha256']
for path,expected in plan['file_sha256'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected, path
assert json.loads((work/'PREPARED.json').read_bytes())['plan_id']==plan['engine_plan_id']
assert not (work/'SUBMITTED').exists() and not (work/'RESULT.json').exists()
assert not any((work/'ledger').iterdir())
assert os.statvfs(work).f_bavail * os.statvfs(work).f_frsize >= 10*1024**3
mem={}
for line in Path('/proc/meminfo').read_text().splitlines():
    k,v=line.split(':',1)
    if v.strip(): mem[k]=int(v.split()[0])
assert mem['MemAvailable'] >= 16*1024**2 and mem['SwapFree'] >= 1024**2
processes=subprocess.check_output(['ps','-eo','comm'],text=True).splitlines()
assert 'vf-rank-session' not in processes
fd=os.open(work/'SUBMITTED',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,(digest+'\n').encode());os.fsync(fd);os.close(fd)
fd=os.open(work,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
pack=Path('/home/pc1/Strata-data/packs/sc117_abliterated_iq3_s')
model=Path('/home/pc1/Strata-data/models/SC117-Qwen3.8-Flash-Next-GSQ-RCO-abliterated/IQ3_S')
base='Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S'
worker=work/'target/release/vf-rank-session'; binary=work/'target/release/vf-sc117-prefill-greedy'
args=['--messages-json','--max-new-tokens','2','--tokenizer-dir',str(pack/'tokenizer'),
      '--tokenizer-manifest',str(work/'adapters/strata_tokenizer/observed-sc117.json'),
      '--binary',str(binary),'--binary-sha256',plan['file_sha256'][str(binary)],
      '--dense',str(pack/'dense.bin'),'--shard',str(model/(base+'-00001-of-00002.gguf')),
      '--pack',str(pack/'experts.bin'),'--index',str(pack/'native_experts.txt'),
      '--session',str(worker),'--ledger',str(work/'ledger'),
      '--ple-shard',str(model/(base+'-00002-of-00002.gguf')),
      '--ple-manifest','/home/pc1/NAOS/vf-r443-work/manifest.json']
env={'HOME':'/home/pc1','PATH':'/usr/bin:/bin','TMPDIR':str(work/'tmp'),
     'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','VF_SC117_NATIVE_BACKEND':'scalar',
     'VF_LOCAL_WORKER_SHA256':plan['file_sha256'][str(worker)], 'VF_PREFILL_GROUP_SIZE':'2',
     'VF_EXPERT_RANK_AFFINITY':'sticky','VF_EXPECTED_PLAN_ID':plan['engine_plan_id']}
cmd=['systemd-run','--user','--scope','--quiet','-p','MemoryHigh=11G','-p','MemoryMax=13G',
     '-p','MemorySwapMax=0','-p','TasksMax=320','-p','CPUQuota=200%',
     '/usr/bin/env','-i',*[f'{k}={v}' for k,v in env.items()],
     '/usr/bin/ionice','-c2','-n7','/usr/bin/nice','-n','10',
     '/home/pc1/Strata/.venv/bin/python','-m','adapters.strata_tokenizer.prompt_runner',*args]
with (work/'PROMPT.json').open('rb') as inp,(work/'RESULT.json').open('xb') as out,(work/'error.log').open('xb') as err:
    process=subprocess.run(cmd,stdin=inp,stdout=out,stderr=err,cwd=work)
    out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno())
fd=os.open(work/'EXIT_STATUS',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
os.write(fd,(str(process.returncode)+'\n').encode());os.fsync(fd);os.close(fd)
print('exit',process.returncode,flush=True)
raise SystemExit(process.returncode)
