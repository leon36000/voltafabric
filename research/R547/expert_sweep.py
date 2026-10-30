"""Characterize two preserved activations against 12 independently observed expert slices.

CPU numerical diagnostic only: no sessions, network, ledger claims, hardware admission,
or changed correctness verdict. Output files must be in an empty dedicated directory.
"""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
if len(sys.argv)!=7:raise SystemExit('usage: expert_sweep PROBE PACK ACT0 ACT1 OBSERVATION OUTDIR')
probe,pack,a0,a1,obs,out=map(Path,sys.argv[1:]);out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected=['c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a','9cdd44775e6d5afd50f61662745765e1b1a5701c3f31a85c766f82efed8a3403']
if sha(probe)!='ce5de908e0493e4e4daf298a368e2a6079a7b03e7a294f5514ba79455ae248b7' or [sha(a0),sha(a1)]!=expected:raise ValueError('probe or activation identity')
record=json.loads(obs.read_text()); experts=record['experts'];
if len(experts)!=12 or len({x['expert'] for x in experts})!=12:raise ValueError('expert set')
fd=os.open(pack,os.O_RDONLY|os.O_NOFOLLOW); before=os.fstat(fd)
if not os.path.isfile(pack):raise ValueError('not regular pack')
for e in experts:
 if e['layer']!=2 or not e['matches_source_shard_slices'] or e['bytes']!=1971200 or e['offset']!=1887436800+e['expert']*1971200:raise ValueError('observation geometry')
 b=os.pread(fd,e['bytes'],e['offset'])
 if len(b)!=e['bytes'] or hashlib.sha256(b).hexdigest()!=e['sha256']:raise ValueError('expert slice mismatch')
rows=[]
for token,act in enumerate((a0,a1)):
 for e in sorted(experts,key=lambda z:z['expert']):
  n=e['expert']; ref=out/f'ref-{token}-{n}.f32'
  result=subprocess.run([str(probe),str(pack),str(n),str(ref),str(act),'2'],capture_output=True,text=True,timeout=60)
  text=result.stdout
  if result.stderr or result.returncode not in (0,19):raise RuntimeError(f'{token}/{n} unexpected exit {result.returncode}: {result.stderr}')
  def take(k):
   m=re.search(r'(?:^|\s)'+re.escape(k)+r'=([^\s]+)',text)
   if not m:raise ValueError(f'missing {k}')
   return m.group(1)
  classification=take('classification')
  if (result.returncode==0)!=(classification=='STRICT_PASS') or (result.returncode==19)!=(classification=='STRICT_FAIL_BOUND_EXPLAINED'):raise ValueError('classification/exit mismatch')
  r={'activation':token,'expert':n,'status':classification,'exit':result.returncode,'gate_violations':int(take('gate_violations')),'up_violations':int(take('up_violations')),'mid_violations':int(take('mid_violations')),'strict_violations':int(take('original_scalar_violations')),'envelope_violations':int(take('explainability_envelope_violations')),'quant_bytes_differ':int(take('quant_bytes_differ'))}
  if r['gate_violations'] or r['up_violations'] or r['mid_violations'] or r['envelope_violations'] or ((r['strict_violations']==0)!=(result.returncode==0)):raise ValueError(f'inconsistent report {r}')
  if result.returncode==0 and (not ref.is_file() or ref.stat().st_size!=30720):raise ValueError('missing success reference')
  if result.returncode!=0 and ref.exists():raise ValueError('failed case emitted reference')
  rows.append(r)
after=os.fstat(fd);os.close(fd)
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('pack changed')
print(json.dumps({'status':'DIAGNOSTIC_COMPLETE','binary_sha256':sha(probe),'activation_sha256':expected,'samples':rows},sort_keys=True,separators=(',',':')))
