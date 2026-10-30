#!/usr/bin/env python3
"""Verify the recorded refused run; claims remain unknown and are never replayed."""
import json
from collections import Counter
from pathlib import Path
p = Path(__file__).parent
claims = {}
for path in sorted((p / 'claims').glob('part-*.json')):
    for key, value in json.loads(path.read_text()).items():
        if key in claims:
            raise ValueError('duplicate claim')
        claims[key] = value
if len(claims) != 335 or any(x['status'] != 'RESERVED_UNKNOWN_NO_REPLAY' for x in claims.values()):
    raise ValueError('claim count or status')
run_ids = {x['stamp']['run_id'] for x in claims.values()}
if len(run_ids) != 1 or not next(iter(run_ids)).startswith('r476-prefill-greedy-'):
    raise ValueError('run identity')
layers = Counter(x['stamp']['batch'] for x in claims.values() if 'operation_id' not in x)
evictions = Counter(x['stamp']['batch'] for x in claims.values() if 'operation_id' in x)
if dict(layers) != {0:48,1:48,2:48,3:48,4:48} or dict(evictions) != {3:80,4:15}:
    raise ValueError('claim distribution')
print('PASS: 335 unknown claims preserved; failure cause and committed results remain unproven')
