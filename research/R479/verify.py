#!/usr/bin/env python3
"""Verify preserved claims from the refused R479 run, without replaying them."""
import json
from collections import Counter
from pathlib import Path
p = Path(__file__).parent
claims = {}
for path in sorted((p / 'claims').glob('part-*.json')):
    for name, value in json.loads(path.read_text()).items():
        if name in claims:
            raise ValueError('duplicate claim')
        claims[name] = value
if len(claims) != 1302 or any(v['status'] != 'RESERVED_UNKNOWN_NO_REPLAY' for v in claims.values()):
    raise ValueError('status or count')
run_ids = {v['stamp']['run_id'] for v in claims.values()}
if len(run_ids) != 1 or not next(iter(run_ids)).startswith('r479-prefill-greedy-'):
    raise ValueError('run identity')
layers = Counter(v['stamp']['batch'] for v in claims.values() if 'operation_id' not in v)
evictions = Counter(v['stamp']['batch'] for v in claims.values() if 'operation_id' in v)
if dict(layers) != {i:48 for i in range(9)} or dict(evictions) != {2:108,3:144,4:170,5:134,6:131,7:183}:
    raise ValueError('claim distribution')
print('PASS: 1302 unknown claims preserved; no inference of completion or failure cause')
