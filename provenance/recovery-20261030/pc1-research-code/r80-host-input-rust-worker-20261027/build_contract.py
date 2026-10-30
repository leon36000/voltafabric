"""Read the externally pinned CPU build plan once; no compiler or GPU access."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def unique(pairs):
    out = {}
    for name, value in pairs:
        if name in out:
            raise ValueError('duplicate plan field')
        out[name] = value
    return out


def load_plan(path, expected):
    if not isinstance(expected, str) or re.fullmatch('[0-9a-f]{64}', expected) is None:
        raise ValueError('external SHA required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > 1024 * 1024:
            raise ValueError('bounded regular build plan required')
        raw = stream.read(1024 * 1024 + 1)
        after = os.fstat(stream.fileno())
    signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if len(raw) != before.st_size or signature(before) != signature(after) or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError('consumed build plan differs')
    plan = json.loads(raw, object_pairs_hook=unique)
    if plan.get('no_gpu') is not True or type(plan.get('cargo_jobs')) is not int or plan['cargo_jobs'] != 1:
        raise ValueError('CPU one-job profile required')
    if not isinstance(plan.get('files'), dict) or not plan['files']:
        raise ValueError('file pins required')
    for name, digest in plan['files'].items():
        p = Path(name)
        if p.is_absolute() or p.as_posix() != name or '..' in p.parts or not p.parts:
            raise ValueError('exact relative file name required')
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('file SHA required')
    return plan
