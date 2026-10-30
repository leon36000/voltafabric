"""Extract a complete, digest-checked QSA trace for the independent oracle."""
import base64
import hashlib
import json
import os
import sys
from pathlib import Path

if len(sys.argv) != 4:
    raise SystemExit('usage: extract_trace RESULT EXPECTED_PLAN OUTPUT')
x = json.loads(Path(sys.argv[1]).read_text())
r = x['receipt']; t = r['qsa_trace']; plan = sys.argv[2]
if (r['plan_id'] != plan or not plan.startswith('r537-') or r['workers_closed'] != 96
        or t['version'] != 'vf-qsa-trace-v1-layer3-f32le' or t['layer'] != 3
        or t['words_per_position'] != 37380 or t['positions'] != len(r['input_tokens'])
        or not 1 <= t['positions'] <= 96):
    raise ValueError('trace identity or bounds')
encoded = t['base64'].encode('ascii')
expected = t['positions'] * 37380 * 4
if len(encoded) != ((expected + 2) // 3) * 4:
    raise ValueError('trace encoded length')
raw = base64.b64decode(encoded, validate=True)
if len(raw) != expected or base64.b64encode(raw) != encoded or hashlib.sha256(raw).hexdigest() != t['sha256']:
    raise ValueError('trace checksum or encoding')
fd = os.open(sys.argv[3], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'wb') as f:
    f.write(raw)
    f.flush()
    os.fsync(f.fileno())
print(json.dumps({'positions':t['positions'], 'bytes':len(raw), 'sha256':t['sha256'], 'plan_id':plan}, sort_keys=True))
