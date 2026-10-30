#!/usr/bin/env python3
"""Independently verify prompt-bound plan and real local generation."""
import base64,gzip,hashlib,json,runpy,struct
from pathlib import Path
p=Path(__file__).parent
prior=runpy.run_path(str(p.parent/'R493/verify.py')); baseline=prior['r']
def unpack(name,expected,count):
    files=sorted((p/(name+'-parts')).glob('part-*.txt')); assert len(files)==count
    raw=gzip.decompress(base64.b64decode(''.join(f.read_text().strip() for f in files),validate=True))
    assert hashlib.sha256(raw).hexdigest()==expected
    return json.loads(raw)
x=unpack('result','a8636767234ef3b61c23c285db93ebf4d9c9ecee37911ed0141f30881774b0f0',3)
claims_raw=unpack('claims','7be1ce0c1c61fe7d7c6946a470d0776a59696ffeaeaa0fa03e0c035e0387ce88',1)
r=x['receipt']; prompt=[9419,0]; ph=hashlib.sha256(b''.join(struct.pack('<I',t) for t in prompt)).hexdigest();assert ph=='20a3f091c3113f084c97b2119a6f07ffde9f7068aaf310e53de2879be8250ea2'
worker='/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v6','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','epoch':1,'layers':48,'prompt_length':2,'prompt_tokens_sha256':ph,'max_new_tokens':2,'eos_token_id':248046,'prompt_mode':'plain','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':[worker]}}}
pid='r506-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert r['plan_id']==pid and r['prompt_tokens_sha256']==ph and r['prompt_tokens']==x['prompt_token_ids']==prompt
assert r['prompt_mode']=='plain' and r['eos_token_id']==248046 and r['worker_transports']=={'2':'local','3':'local'}
assert r['input_tokens']==prompt+r['generated_token_ids'][:-1] and r['generated_token_ids']==[1165,3849] and x['output']['text']==' We’ve' and x['output']['stop_reason']=='token_limit'
for k in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):assert r[k]==baseline[k],k
counts=r['worker_exec_counts'];rc=r['worker_raw_exec_counts'];assert r['workers_closed']==len(counts)==len(rc)==96
assert all(rank in (2,3) and n==bye and 0<=n<=3 for rank,n,bye in counts)
assert all(rank==c[0] and n==c[1] for (rank,n),c in zip(rc,counts,strict=True)) and sum(n for _,n in rc)==283
claims={k:json.loads(v) for k,v in claims_raw.items()};assert len(claims)==144 and not r['evictions']
assert {(c['stamp']['layer'],c['stamp']['batch']) for c in claims.values()}=={(l,b) for l in range(48) for b in range(3)}
assert all(c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['stamp']['run_id']==r['run_id'] and c['stamp']['plan_id']==pid and c['stamp']['model_revision']=='sc117-local-observation' and c['stamp']['epoch']==1 and c['attempt']==r['attempt_base']+c['stamp']['batch'] for c in claims.values())
known=prior['known'];admitted={}
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    key=(layer,expert)
    if key in admitted:assert admitted[key]==digest
    admitted[key]=digest
    assert known[key]==digest
for layer,hs in enumerate(r['seed_hashes']):
    for expert,digest in zip(r['seed_experts'],hs,strict=True): assert known[(layer,expert)]==digest
sizes=[int(s.split()[4]) for s in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if s and not s.startswith('#')]; assert len(sizes)==48
rank_used={'2':sum(sizes),'3':sum(sizes)}
for layer,expert,rank,local,digest in r['dynamic_admissions']:rank_used[str(rank)]+=sizes[layer]
assert rank_used==r['rank_used_bytes'] and sum(rank_used.values())==r['resident_payload_bytes']
assert all(rank_used[k]<=plan['rank_limits'][k] for k in rank_used)
print(json.dumps({'status':'PASS','plan':pid,'claims':len(claims),'admissions':len(r['dynamic_admissions']),'unique_admitted':len(admitted)}))
