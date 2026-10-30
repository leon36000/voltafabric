#!/usr/bin/env python3
import json
from pathlib import Path
claims=json.loads((Path(__file__).parent/'CLAIMS_LOCAL_TEST.json').read_text())
assert len(claims)==27
assert all(v['status']=='RESERVED_UNKNOWN_NO_REPLAY' for v in claims.values())
ids={v['stamp']['run_id'] for v in claims.values()}
assert len(ids)==3 and all(v.startswith('r482-two-processes-') for v in ids)
print('PASS: 27 local synthetic test claims preserved, never replay')
