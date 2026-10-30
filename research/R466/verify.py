#!/usr/bin/env python3
"""Validate recorded routes, admission identities, outputs and consumed claims."""
import ast
import json
from pathlib import Path

root = Path(__file__).parent
reference = json.loads((root.parent / 'R464/RESULT.json').read_text())['receipt']
runs = [json.loads((root / name / 'RESULT.json').read_text()) for name in ('trace', 'readmission')]
routes = []
for name in ('trace', 'readmission'):
    lines = (root / name / 'ROUTES.txt').read_text().splitlines()
    parsed = {}
    for line in lines:
        prefix, position, values = line.split(' ', 2)
        if prefix != 'VF_R466_ROUTE' or int(position) in parsed:
            raise ValueError('duplicate or unexpected route')
        parsed[int(position)] = ast.literal_eval(values)
    if sorted(parsed) != [0, 1, 2]:
        raise ValueError('missing route')
    routes.append(parsed)
    claims = json.loads((root / name / 'CLAIMS.json').read_text())
    if len(claims) != 145 or sum(k.startswith('native-evict-') for k in claims) != 1 or any(c['status'] != 'RESERVED_UNKNOWN_NO_REPLAY' for c in claims.values()):
        raise ValueError('incomplete durable reservations')
    run = runs[0 if name == 'trace' else 1]
    evictions = [v for k, v in claims.items() if k.startswith('native-evict-')]
    eviction = run['eviction']
    if any(evictions[0][k] != eviction[k] for k in ('operation_id', 'attempt', 'stamp', 'binding')):
        raise ValueError('eviction receipt and claim differ')
    if any(c['stamp']['run_id'] != run['run_id'] for c in claims.values()):
        raise ValueError('claim run identity differs')
    batch_claims = [v for k, v in claims.items() if not k.startswith('native-evict-')]
    if {(v['stamp']['layer'], v['stamp']['batch']) for v in batch_claims} != {(layer, position) for layer in range(48) for position in range(3)}:
        raise ValueError('batch claim coverage differs')
if routes[0] != routes[1] or 309 not in routes[0][0] or 309 not in routes[0][2]:
    raise ValueError('route not reproducible')
for run in runs:
    for key in ('input_tokens', 'generated_token_ids', 'embedding_sha256', 'output_sha256', 'logits_sha256'):
        if run[key] != reference[key]:
            raise ValueError('result mismatch: ' + key)
    if run['workers_closed'] != 96:
        raise ValueError('unclosed workers')
if runs[1]['eviction']['binding']['expert'] != 309 or runs[1]['eviction']['released_payload_bytes'] != 2176000:
    raise ValueError('wrong eviction')
load = [a for a in runs[1]['dynamic_admissions'] if a[0] == 0 and a[1] == 309]
if len(load) != 2 or load[0][4] != load[1][4] or load[0][4] != runs[1]['eviction']['binding']['blob_sha256']:
    raise ValueError('expert was not re-admitted with identical digest')
if runs[0]['run_id'] == runs[1]['run_id']:
    raise ValueError('run identity reused')
print(json.dumps({'result': 'PASS', 'expert': 309, 'original': load[0], 'readmission': load[1], 'routes': routes[1], 'outputs_match_reference': True}, sort_keys=True))
