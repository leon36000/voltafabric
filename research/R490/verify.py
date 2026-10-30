#!/usr/bin/env python3
"""Check the bounded live generation, all claims and local expert observations."""
import json
from pathlib import Path
p=Path(__file__).parent
x=json.loads((p/'RESULT.json').read_text()); r=x['receipt']
for field in ('evictions','dynamic_admissions'):
    r[field]=[item for part in sorted((p/field).glob('part-*.json')) for item in json.loads(part.read_text())]
b=json.loads((p.parent/'R488/RESULT.json').read_text())['receipt']
expected_plan='r486-4d1110fc2d9cddd088a4d611c40ad5fff585675249c32aee78892edd23a99b69'
if r.get('plan_id')!=expected_plan or r.get('worker_transports')!={'2':'local','3':'local'}: raise ValueError('plan or transport')
if x['prompt_token_ids'] != [9419,11,4338,10033,1092,411,2313,1503,303,4145,3665,13] or r['requested_max_new_tokens'] != 2 or not 1 <= len(r['generated_token_ids']) <= 2:
    raise ValueError('prompt or generation bound')
if r['input_tokens'] != x['prompt_token_ids'] + r['generated_token_ids'][:-1] or len(r['input_tokens']) != len(x['prompt_token_ids'])+len(r['generated_token_ids'])-1 or r['workers_closed'] != 96:
    raise ValueError('autoregression or shutdown')
counts=r['worker_exec_counts']
if len(counts)!=96 or any(rank not in (2,3) or expected!=bye or not 0<=expected<=len(r['input_tokens']) for rank,expected,bye in counts): raise ValueError('worker close counts')
if not any(expected<len(r['input_tokens']) for _,expected,_ in counts): raise ValueError('idle ranks not exercised')
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    if r[key] != b[key]: raise ValueError('reference mismatch '+key)
raw=r['worker_raw_exec_counts']
if len(raw)!=96 or sum(n for _,n in raw)==0 or any(rank!=c[0] or n!=c[1] for (rank,n),c in zip(raw,counts,strict=True)): raise ValueError('raw mode not used for each execution')
if r['payload_budget_bytes'] != 3221225472 or r['expert_headroom_bytes'] != 982272000 or r['resident_payload_bytes'] > r['payload_budget_bytes']:
    raise ValueError('budget or computed headroom')
claims={}
for path in sorted((p/'claims').glob('part-*.json')):
    for name,value in json.loads(path.read_text()).items():
        if name in claims: raise ValueError('duplicate claim file')
        claims[name]=value
if any(v['stamp']['plan_id']!=expected_plan for v in claims.values()): raise ValueError('claim plan mismatch')
if len(claims)!=48*len(r['input_tokens'])+len(r['evictions']) or any(v['status']!='RESERVED_UNKNOWN_NO_REPLAY' or v['stamp']['run_id']!=r['run_id'] or v['attempt']!=r['attempt_base']+v['stamp']['batch'] for v in claims.values()):
    raise ValueError('claim identity or status')
evclaims={v['operation_id']:v for k,v in claims.items() if k.startswith('native-evict-')}
batches=[v for k,v in claims.items() if not k.startswith('native-evict-')]
if len(evclaims)!=len(r['evictions']) or {(v['stamp']['layer'],v['stamp']['batch']) for v in batches}!={(l,batch) for l in range(48) for batch in range(len(r['input_tokens']))}:
    raise ValueError('claim coverage')
for e in r['evictions']:
    c=evclaims.pop(e['operation_id'])
    if any(c[k]!=e[k] for k in ('stamp','attempt','binding')): raise ValueError('eviction identity')
if evclaims: raise ValueError('unmatched eviction')
per_batch={}
for e in r['evictions']:
    age=e['last_used_position']
    if age is not None and (type(age) is not int or age >= e['stamp']['batch']): raise ValueError('last route not protected')
    key=(0 if age is None else age+1,e['binding']['layer'],e['binding']['expert'],e['binding']['rank'])
    per_batch[e['stamp']['batch']]=key
lines=(p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines()
sizes=[int(s.split()[4]) for s in lines if s and not s.startswith('#')]
if len(sizes)!=48 or sum(sizes)*10!=r['expert_headroom_bytes']: raise ValueError('index headroom')
charge=2*sum(sizes)+sum(sizes[a[0]] for a in r['dynamic_admissions'])-sum(e['released_payload_bytes'] for e in r['evictions'])
if charge!=r['resident_payload_bytes'] or any(e['released_payload_bytes']!=sizes[e['binding']['layer']] for e in r['evictions']): raise ValueError('payload accounting')
limits={'2':1073741824,'3':2147483648}
if r['rank_limits']!=limits or len(r['rank_history'])!=len(r['input_tokens']): raise ValueError('rank limits or history')
rank_charge={'2':sum(sizes),'3':sum(sizes)}
for layer,expert,rank,local,digest in r['dynamic_admissions']: rank_charge[str(rank)]+=sizes[layer]
for e in r['evictions']: rank_charge[str(e['binding']['rank'])]-=e['released_payload_bytes']
if rank_charge!=r['rank_used_bytes'] or sum(rank_charge.values())!=charge: raise ValueError('per-rank charge')
for h in r['rank_history']:
    if set(h)!=set(limits) or any(h[k]>limits[k] for k in limits): raise ValueError('per-rank bound')
if r['rank_history'][-1]!=rank_charge: raise ValueError('rank final history')
expected={(layer,expert):digest for layer,hashes in enumerate(r['seed_hashes']) for expert,digest in zip(r['seed_experts'],hashes,strict=True)}
repeat=0
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    key=(layer,expert)
    if key in expected:
        if expected[key]!=digest: raise ValueError('changed re-admission')
        repeat+=1
    expected[key]=digest
baseline={(layer,expert):digest for layer,hashes in enumerate(b['seed_hashes']) for expert,digest in zip(b['seed_experts'],hashes,strict=True)}
for part in sorted((p.parent/'R488/dynamic_admissions').glob('part-*.json')):
    for layer,expert,rank,local,digest in json.loads(part.read_text()):
        key=(layer,expert)
        if key in baseline and baseline[key]!=digest: raise ValueError('baseline digest conflict')
        baseline[key]=digest
if expected!=baseline: raise ValueError('admitted experts differ from source-verified baseline')
print(json.dumps({'result':'PASS','text':x['output']['text'],'tokens':r['generated_token_ids'],'positions':len(r['input_tokens']),'evictions':len(r['evictions']),'re_admissions':repeat,'experts':len(expected)},ensure_ascii=False))
