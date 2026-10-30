#!/usr/bin/env python3
"""Check preservation of the fresh synthetic two-worker ledger."""
import json
from pathlib import Path
claims = json.loads((Path(__file__).parent / 'CLAIMS.json').read_text())
assert len(claims) == 18
assert all(c['status'] == 'RESERVED_UNKNOWN_NO_REPLAY' for c in claims.values())
run_ids = {c['stamp']['run_id'] for c in claims.values()}
assert len(run_ids) == 2 and all(x.startswith('r482-two-processes-') for x in run_ids)
print('PASS: eighteen consumed claims preserved; they are not replayable')
