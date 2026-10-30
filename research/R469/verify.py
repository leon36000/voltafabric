#!/usr/bin/env python3
"""Validate the two claim identities and asymmetric lost-ack accounting."""
import json
from pathlib import Path
p = Path(__file__).parent
r = json.loads((p / 'RESULT.json').read_text())
m = json.loads((p / 'SUPPRESSED.json').read_text())
c = json.loads((p / 'CLAIMS.json').read_text())
b = json.loads((p.parent / 'R464/RESULT.json').read_text())['receipt']
if len(c) != 50 or any(v['status'] != 'RESERVED_UNKNOWN_NO_REPLAY' for v in c.values()):
    raise ValueError('claim count or state')
ev = [v for k,v in c.items() if k.startswith('native-evict-')]
batches = [v for k,v in c.items() if not k.startswith('native-evict-')]
if len(ev) != 2 or {(v['stamp']['layer'],v['stamp']['batch']) for v in batches} != {(layer,0) for layer in range(48)}:
    raise ValueError('claim coverage')
if {x['binding']['expert'] for x in ev} != {0,1} or len({x['operation_id'] for x in ev}) != 2:
    raise ValueError('eviction identities')
lost = next(x for x in ev if x['binding']['expert'] == 1)
if any(lost[k] != m[k] for k in ('operation_id','attempt','stamp','binding')) or m['bytes'] != 2176000 or m['rank'] != 3:
    raise ValueError('suppressed acknowledgement differs')
if any(v['stamp']['run_id'] != r['run_id'] or v['attempt'] != r['attempt'] for v in c.values()):
    raise ValueError('operation identity mismatch')
if r['accounted_bytes_after'] + 2176000 != r['accounted_bytes_before'] or not all(r[k] for k in ('first_expert_removed','second_binding_still_accounted','next_batch_refused','poisoned')):
    raise ValueError('controller accounting or poisoned state')
if r['position'] != 1 or r['claims_after_refusal'] != 50 or r['workers_closed'] != 96 or r['workers_clean'] != 95 or r['first_output_sha256'] != b['output_sha256'][0]:
    raise ValueError('continuation, shutdown or reference mismatch')
print(json.dumps({'result':'PASS','acknowledged_expert':0,'uncertain_expert':1,'charged_bytes_removed':2176000,'next_position_refused':True}))
