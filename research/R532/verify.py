"""Validate the fresh 73-position CPU receipt and every preserved claim."""
import base64,gzip,hashlib,json,runpy,struct
from pathlib import Path
p=Path(__file__).parent
meta=json.loads((p/'METADATA.json').read_text())
def unpack(name):
    files=sorted((p/(name+'-parts')).glob('part-*.txt'))
    assert len(files)==meta[name+'_parts']
    raw=gzip.decompress(base64.b64decode(''.join(f.read_text().strip() for f in files),validate=True))
    assert hashlib.sha256(raw).hexdigest()==meta[name+'_sha256']
    return json.loads(raw)
plan=runpy.run_path(str(p/'verify_plan.py')); previous=runpy.run_path(str(p.parent/'R519/verify.py'))
baseline=previous['r']; observed=previous['observed'] | previous['found']
x=unpack('result'); rawclaims=unpack('claims'); r=x['receipt']; ids=plan['tokens']; pid=plan['pid']
assert len(ids)==73 and r['plan_id']==pid and r['prompt_tokens_sha256']==plan['ph']
assert x['prompt_token_ids']==ids and r['prompt_tokens']==r['input_tokens']==ids
assert r['worker_transports']=={'2':'local','3':'local'} and r['workers_closed']==96
assert r['prompt_mode']=='chat' and r['requested_max_new_tokens']==1 and r['eos_token_id']==248046
assert len(r['generated_token_ids'])==1 and x['output']['token_ids']==r['generated_token_ids']
assert x['output']['stop_reason'] in ('eos','token_limit')
for k,n in (('embedding_sha256',73),('output_sha256',73),('logits_sha256',1)):
    assert len(r[k])==n and all(len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in r[k])
for k in ('input_tokens','embedding_sha256','output_sha256'):
    assert r[k][:20]==baseline[k][:20], k
assert len(r['worker_exec_counts'])==len(r['worker_raw_exec_counts'])==96
assert all(rank in (2,3) and 0<=n<=73 and n==bye for rank,n,bye in r['worker_exec_counts'])
assert all(rank==c[0] and 0<=n<=c[1] for (rank,n),c in zip(r['worker_raw_exec_counts'],r['worker_exec_counts'],strict=True))
claims={name:json.loads(raw) for name,raw in rawclaims.items()}
assert len(claims)==48*73+len(r['evictions'])
for filename,c in claims.items():
    s=c['stamp']; eviction='binding' in c
    assert c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and s['run_id']==r['run_id'] and s['plan_id']==pid and s['model_revision']=='sc117-local-observation' and s['epoch']==1 and c['attempt']==r['attempt_base']+s['batch']
    key=b'VF_NATIVE_EVICTION_V1\0' if eviction else b'VF_NATIVE_BATCH_V1\0'
    for val in (s['run_id'],s['model_revision'],s['plan_id']):
        val=val.encode(); key+=struct.pack('<Q',len(val))+val
    key+=struct.pack('<I',s['layer'])
    if eviction: key+=struct.pack('<I',c['binding']['expert'])
    key+=struct.pack('<QQ',s['batch'],s['epoch']); digest=hashlib.sha256(key).hexdigest()
    assert filename==('native-evict-' if eviction else 'native-')+digest+'.claim.json'
    if eviction: assert c['operation_id']==digest
batch=[c for c in claims.values() if 'binding' not in c]
assert {(c['stamp']['layer'],c['stamp']['batch']) for c in batch}=={(l,b) for l in range(48) for b in range(73)}
ev={c['operation_id']:c for c in claims.values() if 'binding' in c}
for event in r['evictions']:
    c=ev.pop(event['operation_id']); assert all(c[k]==event[k] for k in ('stamp','attempt','binding'))
    assert event['last_used_position'] is None or event['last_used_position']<event['stamp']['batch']
assert not ev
sizes=[int(s.split()[4]) for s in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if s and not s.startswith('#')]
used={'2':sum(sizes),'3':sum(sizes)}; admitted={}
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    used[str(rank)]+=sizes[layer]; k=(layer,expert)
    if k in admitted: assert admitted[k]==digest
    admitted[k]=digest
for event in r['evictions']:
    assert event['released_payload_bytes']==sizes[event['binding']['layer']]
    used[str(event['binding']['rank'])]-=event['released_payload_bytes']
for layer,hashes in enumerate(r['seed_hashes']):
    for expert,digest in zip(r['seed_experts'],hashes,strict=True):
        k=(layer,expert)
        if k in admitted: assert admitted[k]==digest
        admitted[k]=digest
assert used==r['rank_used_bytes'] and sum(used.values())==r['resident_payload_bytes']
assert len(r['rank_history'])==73 and r['rank_history'][-1]==used
assert all(0<=h['2']<=1073741824 and 0<=h['3']<=2147483648 for h in r['rank_history'])
assert all(observed[k]==v for k,v in admitted.items() if k in observed)
selection=[{'layer':k[0], 'expert':k[1], 'sha256':v} for k,v in sorted(admitted.items()) if k not in observed]
(p/'selection.json').write_text(json.dumps(selection,sort_keys=True,separators=(',',':')))
print(json.dumps({'status':'PASS_PARTIAL','positions':73,'matched_prefix':20,'claims':len(claims),'evictions':len(r['evictions']),'unique_experts':len(admitted),'previously_checked':sum(k in observed for k in admitted),'new_to_check':len(selection),'text':x['output']['text'],'stop':x['output']['stop_reason']}))
