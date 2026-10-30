#!/usr/bin/env python3
"""Verify the fresh local generation, plan, claims and known expert digests."""
import base64, hashlib, json
from pathlib import Path
p=Path(__file__).parent
raw=b''.join(base64.b64decode(q.read_bytes(),validate=False) for q in sorted((p/'result-parts').glob('part-*.txt')))
assert hashlib.sha256(raw).hexdigest()=='52d076dacd2f621cbf5e09226bd1e62157fbd78eb4375623a53f11df19f36c45'
x=json.loads(raw); r=x['receipt']; b=json.loads((p.parent/'R487/RESULT.json').read_text())['receipt']
worker='/home/pc1/NAOS/vf-r493-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v4','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','epoch':1,'layers':48,'prompt_length':2,'max_new_tokens':2,'eos_token_id':248046,'payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':[worker]}}}
plan_id='r492-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert r['plan_id']==plan_id and r['worker_transports']=={'2':'local','3':'local'}
assert x['prompt_token_ids']==[9419,0] and r['eos_token_id']==248046 and r['requested_max_new_tokens']==2
assert r['input_tokens']==x['prompt_token_ids']+r['generated_token_ids'][:-1] and len(r['input_tokens'])==3
assert x['output']['text']==' We’ve' and x['output']['stop_reason']=='token_limit' and r['workers_closed']==96 and not r['evictions']
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    assert r[key]==b[key], key
counts=r['worker_exec_counts']; raw_counts=r['worker_raw_exec_counts']
assert len(counts)==len(raw_counts)==96
assert all(rank in (2,3) and n==bye and 0<=n<=3 for rank,n,bye in counts)
assert all(rank==c[0] and n==c[1] for (rank,n),c in zip(raw_counts,counts,strict=True))
assert sum(n for _,n in raw_counts)==283
claims={}
for part in sorted((p/'claim-parts').glob('part-*.json')):
    for item in json.loads(part.read_text()):
        assert item['name'] not in claims
        bts=item['raw'].encode(); assert hashlib.sha256(bts).hexdigest()==item['sha256']
        claims[item['name']]=json.loads(bts)
assert len(claims)==144
assert {(c['stamp']['layer'],c['stamp']['batch']) for c in claims.values()}=={(l,t) for l in range(48) for t in range(3)}
assert all(c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['stamp']['run_id']==r['run_id'] and c['stamp']['plan_id']==plan_id and c['stamp']['model_revision']=='sc117-local-observation' and c['stamp']['epoch']==1 and c['attempt']==r['attempt_base']+c['stamp']['batch'] for c in claims.values())
baseline=json.loads((p.parent/'R486/RESULT.json').read_text())['receipt']
known={(layer,expert):digest for layer,hashes in enumerate(baseline['seed_hashes']) for expert,digest in zip(baseline['seed_experts'],hashes,strict=True)}
for part in sorted((p.parent/'R486/dynamic_admissions').glob('part-*.json')):
    for layer,expert,rank,local,digest in json.loads(part.read_text()):
        k=(layer,expert)
        if k in known: assert known[k]==digest
        known[k]=digest
for layer,expert,rank,local,digest in r['dynamic_admissions']: assert known[(layer,expert)]==digest
assert all(known[(layer,expert)]==digest for layer,hs in enumerate(r['seed_hashes']) for expert,digest in zip(r['seed_experts'],hs,strict=True))
sizes=[int(s.split()[4]) for s in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if s and not s.startswith('#')]
assert len(sizes)==48
rank_used={'2':sum(sizes),'3':sum(sizes)}
for layer,expert,rank,local,digest in r['dynamic_admissions']: rank_used[str(rank)]+=sizes[layer]
assert rank_used==r['rank_used_bytes'] and sum(rank_used.values())==r['resident_payload_bytes']
assert all(rank_used[k]<=plan['rank_limits'][k] for k in rank_used)
print(json.dumps({'status':'PASS','claims':len(claims),'admissions':len(r['dynamic_admissions']),'raw_executions':sum(n for _,n in raw_counts),'plan':plan_id}))
