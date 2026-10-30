#!/usr/bin/env python3
"""Check the extended CPU receipt, no-replay claims and weight observations."""
import json
from pathlib import Path
p = Path(__file__).parent
x = json.loads((p/'RESULT.json').read_text()); r=x['receipt']
b = json.loads((p.parent/'R462/RESULT.json').read_text())['receipt']
c = json.loads((p/'CLAIMS.json').read_text())
w = json.loads((p/'WEIGHTS.json').read_text())
prior = json.loads((p.parent/'R462/WEIGHTS.json').read_text())['experts']
if x['prompt_token_ids'] != [9419,0] or r['requested_max_new_tokens'] != 4 or len(r['generated_token_ids']) != 4:
    raise ValueError('prompt or output limit')
if r['input_tokens'] != x['prompt_token_ids'] + r['generated_token_ids'][:-1] or len(r['input_tokens']) != 5 or r['workers_closed'] != 96:
    raise ValueError('autoregressive sequence or shutdown')
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    if r[key][:len(b[key])] != b[key]: raise ValueError('reference prefix: '+key)
if len(c)!=240 or any(v['status']!='RESERVED_UNKNOWN_NO_REPLAY' or v['stamp']['run_id']!=r['run_id'] or v['attempt']!=r['attempt_base']+v['stamp']['batch'] for v in c.values()):
    raise ValueError('claim identity or status')
if {(v['stamp']['layer'],v['stamp']['batch']) for v in c.values()} != {(layer,batch) for layer in range(48) for batch in range(5)}:
    raise ValueError('claim coverage')
expected = {(layer,expert):digest for layer,hashes in enumerate(r['seed_hashes']) for expert,digest in zip(r['seed_experts'],hashes,strict=True)}
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    if (layer,expert) in expected: raise ValueError('duplicate admission')
    expected[layer,expert] = digest
all_records = prior + w['experts']
observed = {(v['layer'],v['expert']):v['sha256'] for v in all_records if v['matches_source_shard_slices']}
if len(observed)!=len(all_records) or observed != expected or w['prior_count']!=len(prior) or w['new_count']!=len(w['experts']) or w['total_count']!=len(expected):
    raise ValueError('weight observations mismatch')
print(json.dumps({'result':'PASS','output':x['output']['text'],'tokens':r['generated_token_ids'],'positions':5,'claims':len(c),'observed_experts':len(expected),'new_experts':len(w['experts'])},ensure_ascii=False))
