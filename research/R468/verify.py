#!/usr/bin/env python3
"""Check real policy receipts, protected routes and prior result equality."""
import ast
import json
from pathlib import Path
p = Path(__file__).parent
r = json.loads((p / 'RESULT.json').read_text())
b = json.loads((p.parent / 'R464/RESULT.json').read_text())['receipt']
claims = json.loads((p / 'CLAIMS.json').read_text())
route_line = (p.parent / 'R466/trace/ROUTES.txt').read_text().splitlines()[0]
prefix, pos, choices = route_line.split(' ', 2)
if prefix != 'VF_R466_ROUTE' or pos != '0':
    raise ValueError('invalid route')
protected = set(ast.literal_eval(choices))
if len(claims) != 146 or any(c['status'] != 'RESERVED_UNKNOWN_NO_REPLAY' for c in claims.values()):
    raise ValueError('claim count or state')
ev = [v for k,v in claims.items() if k.startswith('native-evict-')]
batches = [v for k,v in claims.items() if not k.startswith('native-evict-')]
if len(ev) != 2 or {(v['stamp']['layer'],v['stamp']['batch']) for v in batches} != {(l,pos) for l in range(48) for pos in range(3)}:
    raise ValueError('claim coverage')
if len(r['eviction']) != 2 or {e['operation_id'] for e in ev} != {e['operation_id'] for e in r['eviction']}:
    raise ValueError('eviction identities')
for receipt in r['eviction']:
    claim = next(e for e in ev if e['operation_id'] == receipt['operation_id'])
    if any(claim[k] != receipt[k] for k in ('attempt','stamp','binding')) or claim['stamp']['run_id'] != r['run_id']:
        raise ValueError('eviction receipt and claim differ')
    if receipt['stamp']['batch'] != 0 or receipt['attempt'] != r['attempt_base']:
        raise ValueError('eviction did not follow position zero')
    if receipt['binding']['layer'] != 0 or receipt['binding']['expert'] in protected or receipt['released_payload_bytes'] != 2176000:
        raise ValueError('protected expert selected')
for key in ('input_tokens','generated_token_ids','embedding_sha256','output_sha256','logits_sha256'):
    if r[key] != b[key]:
        raise ValueError('output mismatch: '+key)
if r['workers_closed'] != 96 or r['resident_payload_bytes'] + 2176000 != b['resident_payload_bytes']:
    raise ValueError('shutdown or accounting mismatch')
print(json.dumps({'result':'PASS','protected':sorted(protected),'evicted':[(v['binding']['layer'],v['binding']['expert'],v['binding']['rank']) for v in r['eviction']],'released_bytes':sum(v['released_payload_bytes'] for v in r['eviction']),'outputs_match':True}))
