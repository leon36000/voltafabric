"""Retain bounded exact local claim bytes; creates no claim and cannot authorize replay."""
import hashlib
import json
import os
import stat
from pathlib import Path

root = Path('/home/pc1/NAOS/vf-r609-scalar')
assert (root / 'EXIT_STATUS').read_text().strip() == '0'
assert (root / 'SUBMITTED').read_text().strip() == '8765e088eaef8ba59bfde3091f8b164b5ad01a28e37bf46d37ef0951545c6b02'
files = sorted((root / 'ledger').iterdir())
assert 1 <= len(files) <= 10000
records = []
for path in files:
    assert path.name.startswith('native-') and path.name.endswith('.claim.json')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        m = os.fstat(fd)
        assert stat.S_ISREG(m.st_mode) and 0 < m.st_size <= 8192
        raw = b''
        while len(raw) < m.st_size:
            piece = os.read(fd, m.st_size - len(raw))
            if not piece:
                raise ValueError('short claim')
            raw += piece
        assert os.fstat(fd).st_mtime_ns == m.st_mtime_ns
    finally:
        os.close(fd)
    records.append({'file': path.name, 'raw': raw.decode('utf-8'), 'sha256': hashlib.sha256(raw).hexdigest()})
fd = os.open(root / 'CLAIMS.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
with os.fdopen(fd, 'w') as out:
    json.dump(records, out, sort_keys=True, separators=(',', ':'))
    out.write('\n'); out.flush(); os.fsync(out.fileno())
print('retained_claims', len(records))
