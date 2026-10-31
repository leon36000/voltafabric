#!/usr/bin/env python3
"""Check the persisted fault observation against the independent ledger."""
import hashlib, json
from pathlib import Path
p = Path(__file__).resolve().parent
result = json.loads((p/'result.json').read_bytes())
marker = json.loads((p/'suppressed.json').read_bytes())
claims = [json.loads(x.read_bytes()) for x in p.glob('*.claim.json')]
assert result['status'] == 'pass' and len(claims) == 3
assert all(c['status'] == 'RESERVED_UNKNOWN_NO_REPLAY' for c in claims)
evictions = [c for c in claims if 'operation_id' in c]
assert len(evictions) == 1
claim = evictions[0]
for key in ('operation_id', 'stamp', 'attempt', 'binding'):
    assert marker[key] == claim[key]
assert marker['rank'] == 2 and marker['binding']['expert'] == 0 and marker['bytes'] == 1_971_200
assert marker['stamp']['batch'] == 0 and marker['stamp']['run_id'] == result['lost']['run_id']
assert result['idle']['claims_before'] == result['idle']['claims_after'] == 1
assert result['idle']['error_message'] == 'eviction rank did not participate in last batch'
assert result['lost']['claims_before'] == 1 and result['lost']['claims_after'] == 2
assert result['lost']['error_code'] == 'SESSION_CLOSED'
assert result['idle']['residual_sha256'] == result['lost']['residual_sha256']
print(json.dumps({'status':'pass','claims':len(claims),'suppressed_operation':claim['operation_id'],
                  'result_sha256':hashlib.sha256((p/'result.json').read_bytes()).hexdigest()}))
