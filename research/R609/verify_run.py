"""Verify one local generation and all of its fresh non-replayable claims."""
import hashlib,json,struct,sys
from collections import Counter
from pathlib import Path
p=Path(__file__).resolve().parent;root=p.parents[1];mode=sys.argv[1]
assert mode in ('scalar','vector')
w=root/'.runs'/('vf-r609-'+mode)
plan_bytes=(p/('PLAN-'+mode+'.json')).read_bytes();plan=json.loads(plan_bytes)
assert (w/'SUBMITTED').read_text().strip()==hashlib.sha256(plan_bytes).hexdigest()
assert (w/'EXIT_STATUS').read_text().strip()=='0' and not (w/'error.log').read_bytes()
x=json.loads((w/'RESULT.json').read_bytes());r=x['receipt'];claims=json.loads((w/'CLAIMS.json').read_bytes())
prepared=json.loads((p/('PREPARED-'+mode+'.json')).read_bytes())
assert r['plan_id']==plan['engine_plan_id']==prepared['plan_id']
assert r['prompt_tokens']==prepared['prompt_token_ids']==x['prompt_token_ids']
assert r['prompt_tokens_sha256']==prepared['prompt_tokens_sha256']
assert r['native_math_backend']==('scalar-reference-ggml-v1' if mode=='scalar' else 'vector-ggml-v1')
assert r['layer_count']==48 and r['workers_closed']==96 and set(r['worker_transports'].values())=={'local'}
assert len(r['worker_exec_counts'])==96 and all(a in (2,3) and b==c for a,b,c in r['worker_exec_counts'])
assert x['output']['token_ids']==r['generated_token_ids'] and x['output']['run_id']==r['run_id']
kinds=Counter();names=set()
for rec in claims:
 raw=rec['raw'].encode();assert hashlib.sha256(raw).hexdigest()==rec['sha256'];c=json.loads(raw);s=c['stamp']
 assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and s['run_id']==r['run_id'] and s['plan_id']==r['plan_id']
 assert s['model_revision']=='sc117-local-observation' and s['epoch']==1
 ev='binding' in c;key=b'VF_NATIVE_EVICTION_V1\0' if ev else b'VF_NATIVE_BATCH_V1\0'
 for field in ('run_id','model_revision','plan_id'):
  b=s[field].encode();key+=struct.pack('<Q',len(b))+b
 key+=struct.pack('<I',s['layer'])
 if ev:key+=struct.pack('<I',c['binding']['expert'])
 key+=struct.pack('<QQ',s['batch'],s['epoch']);digest=hashlib.sha256(key).hexdigest()
 name=('native-evict-' if ev else 'native-')+digest+'.claim.json'
 assert name==rec['file'] and name not in names and (not ev or c['operation_id']==digest)
 names.add(name);kinds['evictions' if ev else 'batches']+=1
assert kinds['evictions']==len(r['evictions']) and kinds['batches']==48*len(r['input_tokens'])
print(json.dumps({'backend':mode,'claims':dict(kinds),'generated_token_ids':r['generated_token_ids'],
 'workers_closed':r['workers_closed'],'run_id':r['run_id'],'plan_id':r['plan_id'],
 'receipt_sha256':hashlib.sha256((w/'RESULT.json').read_bytes()).hexdigest()},indent=2,sort_keys=True))
