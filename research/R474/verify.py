#!/usr/bin/env python3
"""Check the bounded live generation, all claims and local expert observations."""
import json
from pathlib import Path
p=Path(__file__).parent
x=json.loads((p/'RESULT.json').read_text()); r=x['receipt']
for field in ('evictions','dynamic_admissions'):
    r[field]=[item for part in sorted((p/field).glob('part-*.json')) for item in json.loads(part.read_text())]
b=json.loads((p.parent/'R470/RESULT.json').read_text())['receipt']
if x['prompt_token_ids'] != [9419,0] or r['requested_max_new_tokens'] != 8 or len(r['generated_token_ids']) != 8:
    raise ValueError('prompt or generation bound')
if r['input_tokens'] != x['prompt_token_ids'] + r['generated_token_ids'][:-1] or len(r['input_tokens']) != 9 or r['workers_closed'] != 96:
    raise ValueError('autoregression or shutdown')
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    if r[key][:len(b[key])] != b[key]: raise ValueError('reference prefix '+key)
if r['payload_budget_bytes'] != 3221225472 or r['expert_headroom_bytes'] != 982272000 or r['resident_payload_bytes'] > r['payload_budget_bytes']:
    raise ValueError('budget or computed headroom')
claims={}
for path in sorted((p/'claims').glob('part-*.json')):
    for name,value in json.loads(path.read_text()).items():
        if name in claims: raise ValueError('duplicate claim file')
        claims[name]=value
if len(claims)!=1178 or any(v['status']!='RESERVED_UNKNOWN_NO_REPLAY' or v['stamp']['run_id']!=r['run_id'] or v['attempt']!=r['attempt_base']+v['stamp']['batch'] for v in claims.values()):
    raise ValueError('claim identity or status')
evclaims={v['operation_id']:v for k,v in claims.items() if k.startswith('native-evict-')}
batches=[v for k,v in claims.items() if not k.startswith('native-evict-')]
if len(evclaims)!=746 or len(r['evictions'])!=746 or {(v['stamp']['layer'],v['stamp']['batch']) for v in batches}!={(l,batch) for l in range(48) for batch in range(9)}:
    raise ValueError('claim coverage')
for e in r['evictions']:
    c=evclaims.pop(e['operation_id'])
    if any(c[k]!=e[k] for k in ('stamp','attempt','binding')): raise ValueError('eviction identity')
if evclaims: raise ValueError('unmatched eviction')
lines=(p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines()
sizes=[int(s.split()[4]) for s in lines if s and not s.startswith('#')]
if len(sizes)!=48 or sum(sizes)*10!=r['expert_headroom_bytes']: raise ValueError('index headroom')
charge=2*sum(sizes)+sum(sizes[a[0]] for a in r['dynamic_admissions'])-sum(e['released_payload_bytes'] for e in r['evictions'])
if charge!=r['resident_payload_bytes'] or any(e['released_payload_bytes']!=sizes[e['binding']['layer']] for e in r['evictions']): raise ValueError('payload accounting')
expected={(layer,expert):digest for layer,hashes in enumerate(r['seed_hashes']) for expert,digest in zip(r['seed_experts'],hashes,strict=True)}
repeat=0
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    key=(layer,expert)
    if key in expected:
        if expected[key]!=digest: raise ValueError('changed re-admission')
        repeat+=1
    expected[key]=digest
prior=json.loads((p.parent/'R462/WEIGHTS.json').read_text())['experts']+json.loads((p.parent/'R470/WEIGHTS.json').read_text())['experts']
w=json.loads((p/'WEIGHTS.json').read_text()); observed={}
for v in prior+w['experts']:
    key=(v['layer'],v['expert'])
    if not v['matches_source_shard_slices'] or (key in observed and observed[key]!=v['sha256']): raise ValueError('source observation')
    observed[key]=v['sha256']
if any(observed.get(key)!=digest for key,digest in expected.items()) or w['unique_total']!=len(expected) or w['repeat_admissions']!=repeat:
    raise ValueError('unverified admitted expert')
print(json.dumps({'result':'PASS','text':x['output']['text'],'tokens':r['generated_token_ids'],'positions':9,'evictions':len(r['evictions']),'re_admissions':repeat,'experts':len(expected)},ensure_ascii=False))
