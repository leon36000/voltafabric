"""Reserve two R53 identities in one pinned directory; no authority is established.

Caller must supply the actual shared directory and cooperative lease. Historical
snapshots and successful local tests are not authorization for live execution.
"""
import json
import os
import re
import stat
import time


def _identity(st):
    return st.st_dev, st.st_ino, st.st_uid


def _lease(spec, fd):
    if set(spec) != {'path', 'dev', 'ino', 'uid'} or not os.path.isabs(spec['path']):
        raise ValueError('invalid lease identity')
    if spec['uid'] != os.geteuid() or fd < 3:
        raise ValueError('invalid lease owner or descriptor')
    expected = spec['dev'], spec['ino'], spec['uid']
    path_st = os.stat(spec['path'], follow_symlinks=False)
    fd_st = os.fstat(fd)
    if not stat.S_ISREG(path_st.st_mode) or not stat.S_ISREG(fd_st.st_mode):
        raise ValueError('lease must be a regular file')
    if _identity(path_st) != expected or _identity(fd_st) != expected:
        raise ValueError('lease identity changed')
    with open(f'/proc/self/fdinfo/{fd}', encoding='ascii') as f:
        locks = [line.split() for line in f if line.startswith('lock:')]
    if len(locks) != 1 or len(locks[0]) != 9 or locks[0][2:5] != ['FLOCK', 'ADVISORY', 'WRITE'] or locks[0][7:9] != ['0', 'EOF']:
        raise ValueError('exclusive flock required')
    parts = locks[0][6].split(':')
    if len(parts) != 3 or (int(parts[0], 16), int(parts[1], 16), int(parts[2])) != (os.major(spec['dev']), os.minor(spec['dev']), spec['ino']):
        raise ValueError('lease lock identity mismatch')


def _root(spec, fd):
    expected = spec['dev'], spec['ino'], spec['uid']
    descriptor = os.fstat(fd)
    current = os.stat(spec['root'], follow_symlinks=False)
    for item in (descriptor, current):
        if not stat.S_ISDIR(item.st_mode) or _identity(item) != expected or item.st_mode & 0o077:
            raise ValueError('ledger directory identity or permissions changed')


def reserve(namespace, plan_id, lease, lease_fd):
    """Consume base and suffix; any created file remains consumed on failure."""
    if set(namespace) != {'root', 'dev', 'ino', 'uid', 'base_id'}:
        raise ValueError('invalid namespace fields')
    root, base = namespace['root'], namespace['base_id']
    if (not isinstance(root, str) or not os.path.isabs(root) or len(root) > 4096
        or namespace['uid'] != os.geteuid() or not isinstance(base, str)
        or re.fullmatch(r'r53-[A-Za-z0-9_-]{8,60}', base) is None
        or not isinstance(plan_id, str) or not plan_id or len(plan_id) > 4096):
        raise ValueError('invalid namespace or plan')
    _lease(lease, lease_fd)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    root_fd = os.open(root, flags)
    try:
        _root(namespace, root_fd)
        ids = [base, base + '-duplicate-slot']
        for effective in ids:
            _lease(lease, lease_fd)
            _root(namespace, root_fd)
            entry = {'status': 'RESERVED_UNKNOWN_NO_REPLAY',
                     'scope': 'Namespace consumption only; see linked execution receipt for actual outcome',
                     'base_id': base, 'effective_run_id': effective,
                     'reserved_ns': time.monotonic_ns(),
                     'context': {'plan_id': plan_id, 'shared_lease': lease}}
            data = (json.dumps(entry, indent=2) + '\n').encode()
            fd = os.open(effective + '.json', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=root_fd)
            try:
                with os.fdopen(fd, 'wb') as output:
                    output.write(data)
                    output.flush()
                    os.fsync(output.fileno())
            finally:
                os.fsync(root_fd)
        _lease(lease, lease_fd)
        _root(namespace, root_fd)
        return ids
    finally:
        os.close(root_fd)
