#!/usr/bin/env python3
"""Read-only audit of recovered PC1 session records; never a live lease or R53 ledger.

A missing identity is not evidence that it was never consumed. Done markers
only report recorded hashes; these archives do not contain the FP16 payloads.
"""
import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import PurePosixPath

MAX_PART = 2 * 1024 * 1024
MAX_PARTS = 32
MAX_RECORDS = 10000
ID_FIELDS = ('run_id', 'model_revision', 'plan_id', 'layer', 'epoch')
BATCH = re.compile(r'batch-(\d{20})\.(claim|done)\.json\Z')
HEX = re.compile(r'[0-9a-f]{64}\Z')


def unique(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError('duplicate JSON key')
        obj[key] = value
    return obj


def parse(raw):
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON')))


def read_part(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        a = os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or not 0 < a.st_size <= MAX_PART:
            raise ValueError('invalid archive part')
        chunks, remaining = [], a.st_size
        while remaining:
            part = os.read(fd, min(65536, remaining))
            if not part:
                raise ValueError('short archive part')
            chunks.append(part); remaining -= len(part)
        b = os.fstat(fd)
        identity = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if identity(a) != identity(b):
            raise ValueError('archive part changed')
        raw = b''.join(chunks)
        return parse(raw), hashlib.sha256(raw).hexdigest()
    finally:
        os.close(fd)


def identity(value):
    if not isinstance(value, dict) or set(value) != set(ID_FIELDS):
        raise ValueError('invalid session identity fields')
    for key in ID_FIELDS[:3]:
        if not isinstance(value[key], str) or not value[key].strip() or len(value[key]) > 256:
            raise ValueError('invalid session string identity')
    for key in ID_FIELDS[3:]:
        if type(value[key]) is not int or not 0 <= value[key] < 2**64:
            raise ValueError('invalid session numeric identity')
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def audit(paths):
    if not 1 <= len(paths) <= MAX_PARTS:
        raise ValueError('one to 32 parts required')
    entries = {}; roots = set(); part_hashes = []
    for path in paths:
        part, digest = read_part(path); part_hashes.append(digest)
        if not isinstance(part, dict) or set(part) != {'original_root', 'files'} or not isinstance(part['original_root'], str) or not part['original_root'].startswith('/') or not isinstance(part['files'], dict):
            raise ValueError('invalid archive envelope')
        roots.add(part['original_root'])
        for name, raw in part['files'].items():
            if not isinstance(name, str) or not isinstance(raw, str) or len(name) > 4096 or len(raw.encode()) > 8192:
                raise ValueError('invalid archived record')
            path_obj = PurePosixPath(name)
            if path_obj.is_absolute() or any(v in ('', '.', '..') for v in name.split('/')) or len(path_obj.parts) < 3 or name in entries:
                raise ValueError('duplicate or unsafe record path')
            entries[name] = parse(raw)
            if len(entries) > MAX_RECORDS:
                raise ValueError('too many records')
    if len(roots) != 1:
        raise ValueError('mixed archive roots')
    sessions = {}
    for name, data in entries.items():
        if name.endswith('/identity.json'):
            parent = name.rsplit('/', 1)[0]
            digest = identity(data)
            if parent.rsplit('/', 1)[-1] != digest or digest in sessions:
                raise ValueError('identity path mismatch or duplicate session')
            sessions[digest] = {'path': parent, 'identity': data, 'claims': {}, 'done': {}, 'closed': False, 'close_code': None, 'process': False, 'close_claim': False}
    if not sessions:
        raise ValueError('no sessions')
    for name, data in entries.items():
        parent, filename = name.rsplit('/', 1)
        digest = parent.rsplit('/', 1)[-1]
        s = sessions.get(digest)
        if s is None or s['path'] != parent or not isinstance(data, dict):
            raise ValueError('record without matching identity')
        if filename == 'identity.json':
            continue
        match = BATCH.fullmatch(filename)
        if match:
            index = int(match.group(1)); kind = match.group(2)
            expected = set(ID_FIELDS) | {'batch', 'attempt'} | ({'sha256'} if kind == 'done' else set())
            if set(data) != expected or identity({k: data.get(k) for k in ID_FIELDS}) != digest or type(data.get('batch')) is not int or data['batch'] != index:
                raise ValueError('batch identity mismatch')
            if kind == 'claim' and data['attempt'] != 'assigned-by-rust':
                raise ValueError('invalid claim attempt')
            if kind == 'done' and (type(data['attempt']) is not int or data['attempt'] <= 0 or not isinstance(data['sha256'], str) or not HEX.fullmatch(data['sha256'])):
                raise ValueError('invalid done marker')
            s['claims' if kind == 'claim' else 'done'][index] = name
        elif filename == 'closed.json':
            if set(data) != {'exit_code'} or type(data['exit_code']) is not int:
                raise ValueError('invalid close marker')
            s['closed'] = True
            s['close_code'] = data['exit_code']
        elif filename == 'process.json':
            if set(data) != {'pid'} or type(data['pid']) is not int or data['pid'] <= 0:
                raise ValueError('invalid process marker')
            s['process'] = True
        elif filename == 'close.claim.json':
            if set(data) != {'identity', 'next_batch'} or identity(data['identity']) != digest or type(data['next_batch']) is not int or data['next_batch'] < 0:
                raise ValueError('invalid close claim')
            s['close_claim'] = True
        else:
            raise ValueError('unrecognized record type')
    unknown = sorted(path for s in sessions.values() for i, path in s['claims'].items() if i not in s['done'])
    orphan = sorted(path for s in sessions.values() for i, path in s['done'].items() if i not in s['claims'])
    summary = {'original_root': next(iter(roots)), 'part_sha256': part_hashes, 'records': len(entries),
               'sessions': len(sessions), 'batch_claims': sum(len(s['claims']) for s in sessions.values()),
               'done_markers': sum(len(s['done']) for s in sessions.values()),
               'sessions_without_closed_marker': sum(not s['closed'] for s in sessions.values()),
               'sessions_without_process_marker': sum(not s['process'] for s in sessions.values()),
               'nonzero_close_markers': sorted(s['path'] for s in sessions.values() if s['closed'] and s['close_code'] != 0),
               'close_claims_without_closed_marker': sorted(s['path'] for s in sessions.values() if s['close_claim'] and not s['closed']),
               'claims_without_done': unknown, 'done_without_claim': orphan,
               'done_payloads_verified': False, 'ready_for_live_use': False,
               'limitation': 'Frozen, possibly incomplete PC1 session records; not a live R53 namespace or shared lease. Never replay observed identities; absence is not permission.'}
    return summary, sessions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--identity-json', help='optional JSON file containing exactly the five session identity fields')
    p.add_argument('parts', nargs='+')
    args = p.parse_args()
    result, sessions = audit(args.parts)
    if args.identity_json:
        candidate, _ = read_part(args.identity_json)
        digest = identity(candidate)
        result['candidate'] = {'sha256': digest, 'status': 'CONSUMED_NO_REPLAY' if digest in sessions else 'NOT_OBSERVED_NOT_AUTHORIZED'}
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return 1 if (result['claims_without_done'] or result['done_without_claim'] or result['nonzero_close_markers']
                 or result['sessions_without_closed_marker'] or result['sessions_without_process_marker']) else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as e:
        print('PC1 snapshot inspection refused: ' + str(e), file=sys.stderr)
        raise SystemExit(2)
