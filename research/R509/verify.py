#!/usr/bin/env python3
"""Check the rank-headroom local chat result, run identities and expert accounting."""
import base64,gzip,hashlib,json,runpy,struct
from pathlib import Path
p=Path(__file__).parent
def unpack(name,digest,parts):
    fs=sorted((p/(name+'-parts')).glob('part-*.txt')); assert len(fs)==parts
    raw=gzip.decompress(base64.b64decode(''.join(q.read_text().strip() for q in fs),validate=True))
    assert hashlib.sha256(raw).hexdigest()==digest
    return json.loads(raw)
previous=runpy.run_path(str(p.parent/'R500/verify.py'))
baseline=previous['s']['r']
x=unpack('result','c85490cf198495d2d841ac5bd5fb6e1b3753378fe16c592a72ef01e8902642ec',18)
claims_raw=unpack('claims','1ce5f79022c9ba061a6d761741bd2d612eb3f455c0c0c1adef453a1257adb942',13)
r=x['receipt']; worker='/home/pc1/NAOS/vf-r495-work/target/release/vf-rank-session'
plan={'version':'vf-sc117-worker-plan-v7','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-first-feasible-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','epoch':1,'layers':48,'prompt_length':13,'prompt_tokens_sha256':hashlib.sha256(b''.join(struct.pack('<I',t) for t in [248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271])).hexdigest(),'max_new_tokens':1,'eos_token_id':248046,'prompt_mode':'chat','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker]},'3':{'rank':3,'argv':[worker]}}}
pid='r508-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
expected=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
assert r['plan_id']==pid and r['prompt_tokens_sha256']==plan['prompt_tokens_sha256'] and r['prompt_mode']=='chat' and r['eos_token_id']==248046
assert r['worker_transports']=={'2':'local','3':'local'} and x['prompt_token_ids']==expected and r['prompt_tokens']==expected and r['input_tokens']==expected
assert r['requested_max_new_tokens']==1 and r['generated_token_ids']==[9419] and x['output']['text']=='Hello' and x['output']['stop_reason']=='token_limit' and r['workers_closed']==96
assert len(r['output_sha256'])==len(r['embedding_sha256'])==13 and len(r['logits_sha256'])==1
for k in ('output_sha256','embedding_sha256','logits_sha256'):
    assert all(len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in r[k])
for key in ('input_tokens','embedding_sha256','output_sha256','generated_token_ids','logits_sha256'):
    assert r[key]==baseline[key][:len(r[key])], key
counts=r['worker_exec_counts']; rc=r['worker_raw_exec_counts']; assert len(counts)==len(rc)==96
assert all(rank in (2,3) and n==bye and 0<=n<=13 for rank,n,bye in counts)
assert all(rank==c[0] and n==c[1] for (rank,n),c in zip(rc,counts,strict=True)) and sum(n for _,n in rc)==1153
claims={name:json.loads(raw) for name,raw in claims_raw.items()}; assert len(claims)==2570==48*13+len(r['evictions'])
assert all(c['status']=='RESERVED_UNKNOWN_NO_REPLAY' and c['stamp']['run_id']==r['run_id'] and c['stamp']['plan_id']==pid and c['stamp']['model_revision']=='sc117-local-observation' and c['stamp']['epoch']==1 and c['attempt']==r['attempt_base']+c['stamp']['batch'] for c in claims.values())
ev={v['operation_id']:v for k,v in claims.items() if k.startswith('native-evict-')}; batches=[v for k,v in claims.items() if not k.startswith('native-evict-')]
assert len(ev)==len(r['evictions']) and {(v['stamp']['layer'],v['stamp']['batch']) for v in batches}=={(l,b) for l in range(48) for b in range(13)}
for e in r['evictions']:
    c=ev.pop(e['operation_id']); assert all(c[k]==e[k] for k in ('stamp','attempt','binding'))
    assert e['last_used_position'] is None or e['last_used_position']<e['stamp']['batch']
assert not ev
sizes=[int(s.split()[4]) for s in (p.parent.parent/'fixtures/strata_sc117_native_experts_v3.txt').read_text().splitlines() if s and not s.startswith('#')]; assert len(sizes)==48
rank_used={'2':sum(sizes),'3':sum(sizes)}
for layer,expert,rank,local,digest in r['dynamic_admissions']: rank_used[str(rank)]+=sizes[layer]
for e in r['evictions']:
    assert e['released_payload_bytes']==sizes[e['binding']['layer']]
    rank_used[str(e['binding']['rank'])]-=e['released_payload_bytes']
assert rank_used==r['rank_used_bytes'] and sum(rank_used.values())==r['resident_payload_bytes']
assert all(rank_used[k]<=plan['rank_limits'][k] for k in rank_used)
assert len(r['rank_history'])==13 and r['rank_history'][-1]==rank_used
assert all(all(h[k]<=plan['rank_limits'][k] for k in rank_used) for h in r['rank_history'])
observed=previous['s']['admitted']
admitted={}
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    key=(layer,expert)
    if key in admitted: assert admitted[key]==digest
    admitted[key]=digest
for layer,hs in enumerate(r['seed_hashes']):
    for expert,digest in zip(r['seed_experts'],hs,strict=True):
        if (layer,expert) in admitted: assert admitted[(layer,expert)]==digest
        admitted[(layer,expert)]=digest
assert all(observed[k]==v for k,v in admitted.items() if k in observed)
print(json.dumps({'status':'PASS','plan':pid,'claims':len(claims),'evictions':len(r['evictions']),'unique_experts':len(admitted),'previously_source_checked':sum(k in observed for k in admitted),'not_previously_source_checked':sum(k not in observed for k in admitted)}))
