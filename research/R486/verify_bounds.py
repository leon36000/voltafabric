#!/usr/bin/env python3
"""Conservative serialized-receipt budget for the fixed SC117 CLI contract."""
import json
from pathlib import Path
r = json.loads((Path(__file__).parent.parent / 'R485/evictions/part-00.json').read_text())[0]
# Widen every runtime-varying integer and the generated run identifier.
for key in ('attempt', 'last_used_position', 'released_payload_bytes'):
    r[key] = 2**64 - 1
for key in ('batch', 'epoch', 'layer'):
    r['stamp'][key] = 2**64 - 1
for key in ('epoch', 'expert', 'layer', 'local', 'rank'):
    r['binding'][key] = 2**64 - 1
r['stamp']['run_id'] = 'r486-prefill-greedy-' + '9'*10 + '-' + '9'*20
r['stamp']['plan_id'] = 'r486-' + 'f'*64
event_size = len(json.dumps(r, separators=(',', ':')).encode()) + 1
admission_size = len(json.dumps([2**32-1]*4 + ['f'*64], separators=(',', ':')).encode()) + 1
assert event_size <= 1024 and admission_size <= 128
# 8 prompt tokens + 32 generated gives <=39 positions; no eviction before
# the first completed position; each later position allows at most 768.
# At most 48 layers x 10 routed experts can be admitted per position.
maximum = 38*768*1024 + 39*48*10*128 + 1024*1024
assert maximum < 32*1024*1024
print(f'PASS: conservative={maximum}, limit={32*1024*1024}, event={event_size}, admission={admission_size}')
