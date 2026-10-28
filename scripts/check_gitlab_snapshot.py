#!/usr/bin/env python3
"""Verify the imported R59 snapshot; not a CPU, GPU or full-project qualification.

Requires POSIX dir_fd and O_NOFOLLOW. The checkout may be relocated; the manifest
and every input are opened under the supplied root without following symlinks.
"""
import argparse
import hashlib
import json
import os
from pathlib import PurePosixPath
import re
import stat
import sys

MANIFEST = 'provenance/source_final_r59.json'
MANIFEST_SHA256 = '41f179ce7d7076748e907e566182aa1d3298b4deb776cb52ecd0f694edba526d'
MANIFEST_COUNT = 370
MANIFEST_LIMIT = 1024 * 1024
FILE_LIMIT = 128 * 1024 * 1024
CHUNK = 1024 * 1024
HEX = re.compile(r'[0-9a-f]{64}\Z')


class InvalidSnapshot(ValueError):
    pass


def identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def components(name):
    if type(name) is not str or not name or len(name) > 4096 or '\x00' in name or '\\' in name:
        raise InvalidSnapshot('invalid path')
    parts = name.split('/')
    if name.startswith('/') or any(p in ('', '.', '..') for p in parts) or str(PurePosixPath(name)) != name:
        raise InvalidSnapshot('invalid path')
    return parts


def open_root(root):
    """Open all root components without following symlinks, including ancestors."""
    if not os.path.isabs(root):
        raise InvalidSnapshot('root must be absolute')
    parts = root.split('/')[1:]
    if not parts or any(p in ('', '.', '..') for p in parts):
        raise InvalidSnapshot('invalid root')
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    fds = [os.open('/', flags)]
    try:
        for name in parts:
            fds.append(os.open(name, flags, dir_fd=fds[-1]))
        return fds
    except BaseException:
        for fd in reversed(fds):
            os.close(fd)
        raise


def read_relative(rootfd, name, limit, collect=False):
    """Hash one regular file, checked by inode before/after and through parent names."""
    parts = components(name)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened = []
    parent = rootfd
    filefd = None
    try:
        for part in parts[:-1]:
            child = os.open(part, flags, dir_fd=parent)
            before = os.fstat(child)
            opened.append((parent, part, child, identity(before)))
            parent = child
        filefd = os.open(parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=parent)
        before = os.fstat(filefd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise InvalidSnapshot('not a bounded regular file: ' + name)
        hasher = hashlib.sha256()
        total = 0
        chunks = []
        while True:
            chunk = os.read(filefd, min(CHUNK, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise InvalidSnapshot('file exceeds limit: ' + name)
            hasher.update(chunk)
            if collect:
                chunks.append(chunk)
        after = os.fstat(filefd)
        named = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        if identity(before) != identity(after) or identity(before) != identity(named) or total != before.st_size:
            raise InvalidSnapshot('file changed during read: ' + name)
        for pfd, part, dfd, original in reversed(opened):
            if identity(os.fstat(dfd)) != original or identity(os.stat(part, dir_fd=pfd, follow_symlinks=False)) != original:
                raise InvalidSnapshot('directory changed during read: ' + name)
        return hasher.hexdigest(), b''.join(chunks) if collect else None
    finally:
        if filefd is not None:
            os.close(filefd)
        for _, _, fd, _ in reversed(opened):
            os.close(fd)


def unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise InvalidSnapshot('duplicate JSON key')
        out[key] = value
    return out


def check_snapshot(root, expected_sha=MANIFEST_SHA256, expected_count=MANIFEST_COUNT):
    """Private test seam permits synthetic manifests; CLI uses only pinned values."""
    fds = open_root(root)
    try:
        rootfd = fds[-1]
        root_before = identity(os.fstat(rootfd))
        actual, raw = read_relative(rootfd, MANIFEST, MANIFEST_LIMIT, collect=True)
        if actual != expected_sha:
            raise InvalidSnapshot('manifest hash mismatch')
        try:
            manifest = json.loads(raw.decode('utf-8'), object_pairs_hook=unique_pairs,
                                  parse_constant=lambda _: (_ for _ in ()).throw(InvalidSnapshot('non-finite JSON')))
        except (UnicodeError, json.JSONDecodeError, RecursionError) as e:
            raise InvalidSnapshot('invalid manifest JSON') from e
        if type(manifest) is not dict or len(manifest) != expected_count:
            raise InvalidSnapshot('manifest count mismatch')
        for path, expected in manifest.items():
            components(path)
            if type(expected) is not str or HEX.fullmatch(expected) is None:
                raise InvalidSnapshot('invalid SHA256')
            observed, _ = read_relative(rootfd, path, FILE_LIMIT)
            if observed != expected:
                raise InvalidSnapshot('digest mismatch: ' + path)
        if identity(os.fstat(rootfd)) != root_before:
            raise InvalidSnapshot('root changed during verification')
        # Recheck every named root ancestor against the pinned directory handles.
        names = root.split('/')[1:]
        for i, name in enumerate(names):
            if identity(os.stat(name, dir_fd=fds[i], follow_symlinks=False)) != identity(os.fstat(fds[i+1])):
                raise InvalidSnapshot('root ancestor changed during verification')
        return {'status': 'R59_SNAPSHOT_INTEGRITY_PASS', 'manifest_sha256': actual,
                'files_verified': len(manifest), 'scope': 'listed files only; not CPU/GPU or model qualification'}
    finally:
        for fd in reversed(fds):
            os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help='absolute checkout root (default: parent of scripts directory)')
    args = parser.parse_args()
    try:
        print(json.dumps(check_snapshot(args.root), sort_keys=True))
        return 0
    except (OSError, ValueError, RecursionError) as e:
        print(json.dumps({'status': 'R59_SNAPSHOT_INTEGRITY_FAIL', 'error': str(e)}, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
