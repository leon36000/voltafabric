#!/usr/bin/env python3
"""Verify a bounded-capture local run against R486 and its known expert digests."""
import hashlib
import json
from pathlib import Path
p = Path(__file__).parent
x = json.loads((p/'RESULT.json').read_text()); r=x['receipt']
b = json.loads((p.parent/'R486/RESULT.json').read_text())['receipt']
worker='/home/pc1/NAOS/vf-r486-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v2','model_revision':'sc117-local-observation',
      'index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
      'epoch':1,'layers':48,'prompt_length':2,'max_new_tokens':2,'payload_budget':3221225472,
      'rank_limits':{'2':1073741824,'3':2147483648},
      'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':[worker]}}}
plan_id='r486-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert r['plan_id']==plan_id and r['worker_transports']=={'2':'local','3':'local'}
assert x['prompt_token_ids']==[9419,0] and r['requested_max_new_tokens']==2 and len(r['input_tokens'])==3
assert len(r['generated_token_ids'])==2 and r['workers_closed']==96 and not r['evictions']
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    assert r[key]==b[key][:len(r[key])], key
assert len(r['worker_exec_counts'])==96
assert all(rank in (2,3) and expected==bye and 0<=expected<=3 for rank,expected,bye in r['worker_exec_counts'])
claims={}
for part in sorted((p/'claims').glob('part-*.json')):
    for name,c in json.loads(part.read_text()).items():
        assert name not in claims
        claims[name]=c
assert len(claims)==144
assert {(c['stamp']['layer'],c['stamp']['batch']) for c in claims.values()}=={(l,batch) for l in range(48) for batch in range(3)}
assert all(c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['stamp']['run_id']==r['run_id'] and c['stamp']['plan_id']==plan_id and c['attempt']==r['attempt_base']+c['stamp']['batch'] for c in claims.values())
known={(layer,expert):digest for layer,hashes in enumerate(b['seed_hashes']) for expert,digest in zip(b['seed_experts'],hashes,strict=True)}
for part in sorted((p.parent/'R486/dynamic_admissions').glob('part-*.json')):
    for layer,expert,rank,local,digest in json.loads(part.read_text()):
        k=(layer,expert)
        if k in known: assert known[k]==digest
        known[k]=digest
admissions=[v for part in sorted((p/'admissions').glob('part-*.json')) for v in json.loads(part.read_text())]
for layer,expert,rank,local,digest in admissions: assert known[(layer,expert)]==digest
assert all(known[(layer,expert)]==digest for layer,hs in enumerate(r['seed_hashes']) for expert,digest in zip(r['seed_experts'],hs,strict=True))
sizes=[int(line.split()[4]) for line in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if line and not line.startswith('#')]
rank_used={'2':sum(sizes),'3':sum(sizes)}
for layer,expert,rank,local,digest in admissions: rank_used[str(rank)]+=sizes[layer]
assert rank_used==r['rank_used_bytes'] and sum(rank_used.values())==r['resident_payload_bytes']
assert all(rank_used[k]<=plan['rank_limits'][k] for k in rank_used)
print(json.dumps({'result':'PASS','text':x['output']['text'],'positions':3,'claims':len(claims),'admissions':len(admissions)},ensure_ascii=False))
