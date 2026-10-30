#!/usr/bin/env python3
"""Inspect R53 namespace snapshots without restoring, reserving or replaying them.

Inputs are explicit directories. Exact record bytes are hashed, not merged or
modified. Matching snapshots are not independent proof of ledger completeness.
"""
import argparse
import hashlib
import json
import os
import re
import stat
import sys

MAX_FILES = 10000
MAX_RECORD = 1024 * 1024
BASE = re.compile(r'r53-[A-Za-z0-9_-]{8,60}\Z')
STATUSES = {'RESERVED_UNKNOWN_NO_REPLAY', 'CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY'}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON key')
        result[key] = value
    return result


def snapshot(path):
    root = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        directory = os.fstat(root)
        names = sorted(os.listdir(root))
        if len(names) > MAX_FILES:
            raise ValueError('too many directory entries')
        records = {}
        for name in names:
            if not name.endswith('.json'):
                raise ValueError('unexpected snapshot entry')
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC, dir_fd=root)
            try:
                before = os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_RECORD:
                    raise ValueError('invalid record file')
                chunks, remaining = [], MAX_RECORD + 1
                while remaining:
                    chunk = os.read(fd, min(65536, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    remaining -= len(chunk)
                raw = b''.join(chunks)
                after = os.fstat(fd)
                identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
                if identity(before) != identity(after) or len(raw) != before.st_size:
                    raise ValueError('record changed while reading')
            finally:
                os.close(fd)
            record = json.loads(raw, object_pairs_hook=unique_object, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('non-finite JSON number')))
            if not isinstance(record, dict):
                raise ValueError('record must be an object')
            base, effective, status = (record.get(k) for k in ('base_id', 'effective_run_id', 'status'))
            if (not isinstance(base, str) or not BASE.fullmatch(base)
                    or effective not in (base, base + '-duplicate-slot')
                    or name != effective + '.json' or not isinstance(status, str) or status not in STATUSES):
                raise ValueError('invalid record identity or status')
            if status == 'RESERVED_UNKNOWN_NO_REPLAY' and (
                    record.get('scope') != 'Namespace consumption only; see linked execution receipt for actual outcome'
                    or type(record.get('reserved_ns')) is not int or record['reserved_ns'] <= 0
                    or not isinstance(record.get('context'), dict)):
                raise ValueError('invalid reservation record')
            if status == 'CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY' and (
                    record.get('recorded_after_execution') is not True
                    or record.get('reserved_before_execution') is not False
                    or record.get('automatic_retry') is not False
                    or type(record.get('recorded_ns')) is not int or record['recorded_ns'] <= 0):
                raise ValueError('invalid retrospective status')
            if effective in records:
                raise ValueError('duplicate record identity')
            records[effective] = {'base': base, 'status': status, 'sha256': hashlib.sha256(raw).hexdigest()}
        end = os.fstat(root)
        if (directory.st_dev, directory.st_ino, directory.st_mtime_ns, directory.st_ctime_ns) != (end.st_dev, end.st_ino, end.st_mtime_ns, end.st_ctime_ns):
            raise ValueError('snapshot directory changed')
        return records
    finally:
        os.close(root)


def reconcile(paths):
    if not 1 <= len(paths) <= 32:
        raise ValueError('one to 32 snapshots required')
    snapshots = [snapshot(path) for path in paths]
    identities = sorted(set().union(*(s.keys() for s in snapshots)))
    conflicts, missing, unpaired, retrospective = [], [], [], []
    for identity in identities:
        copies = [s[identity] for s in snapshots if identity in s]
        if len({(v['sha256'], v['base'], v['status']) for v in copies}) != 1:
            conflicts.append(identity)
        if len(copies) != len(snapshots):
            missing.append(identity)
        if copies[0]['status'] == 'CONSUMED_RETROSPECTIVE_CPU_NO_REPLAY':
            retrospective.append(identity)
    for base in sorted({entry['base'] for s in snapshots for entry in s.values()}):
        if base not in identities or base + '-duplicate-slot' not in identities:
            unpaired.append(base)
    return {'sources': len(paths), 'records_per_source': [len(s) for s in snapshots],
            'unique_identities': len(identities), 'conflicts': conflicts, 'missing_from_some': missing,
            'unpaired_bases': unpaired, 'retrospective_count': len(retrospective),
            'matching_content': len(snapshots) > 1 and all(s == snapshots[0] for s in snapshots),
            'ready_for_live_use': False,
            'limitation': 'Snapshots alone cannot prove completeness, reconcile external receipts, or establish a shared ledger.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshots', nargs='+')
    args = parser.parse_args()
    result = reconcile(args.snapshots)
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return 1 if result['conflicts'] or result['missing_from_some'] or result['unpaired_bases'] else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        print('R53 snapshot inspection refused: ' + str(error), file=sys.stderr)
        raise SystemExit(2)
