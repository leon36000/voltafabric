"""Bounded private CPU-side assembly of a multipart resident artifact.

This is a receiver primitive, not yet a Session transport or GPU loader.  The
caller must authenticate the peer, negotiate the protocol, and validate the
artifact before publishing it to an owner. Each operation identity is consumed
on construction, including on failure and uncertain outcome.
"""
import hashlib
import json
import os
import pathlib
import threading

MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_CHUNK_BYTES = 4 * 1024 * 1024

class MultipartError(ValueError):
    pass

def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

class ResidentMultipart:
    """An exclusive, ordered, content-addressed receiver with no replay."""
    def __init__(self, root, identity, total_bytes, sha256):
        fields = {'run_id', 'model_revision', 'plan_id', 'layer', 'epoch', 'rank', 'attempt'}
        if (type(identity) is not dict or set(identity) != fields
                or any(type(identity[k]) is not str or not identity[k] or len(identity[k]) > 512
                       for k in ('run_id', 'model_revision', 'plan_id'))
                or any(type(identity[k]) is not int or not 0 <= identity[k] < 2**64
                       for k in ('layer', 'epoch', 'rank', 'attempt'))
                or type(total_bytes) is not int or not 0 < total_bytes <= MAX_ARTIFACT_BYTES
                or type(sha256) is not str or len(sha256) != 64
                or any(c not in '0123456789abcdef' for c in sha256)):
            raise MultipartError('invalid transfer identity or bounds')
        root = pathlib.Path(root)
        if root.is_symlink() or not root.is_dir():
            raise MultipartError('existing non-symlink root required')
        raw = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()
        self.directory = root / hashlib.sha256(raw).hexdigest()
        self.directory.mkdir(mode=0o700)  # exclusive identity claim
        _fsync_dir(root)
        self._identity = dict(identity)
        self._size = total_bytes
        self._sha = sha256
        self._offset = 0
        self._hash = hashlib.sha256()
        self._state = 'OPEN'
        self._lock = threading.Lock()
        try:
            with (self.directory / 'intent.json').open('xb') as f:
                f.write(json.dumps({'identity': identity, 'bytes': total_bytes, 'sha256': sha256}, sort_keys=True).encode())
                f.flush(); os.fsync(f.fileno())
            self._file = (self.directory / 'staged.bin').open('xb')
            _fsync_dir(self.directory)
        except BaseException:
            self._state = 'UNKNOWN'
            raise

    def _fail(self, message):
        self._state = 'FAILED'
        self._file.close()
        raise MultipartError(message)

    def append(self, offset, data):
        with self._lock:
            if self._state != 'OPEN':
                raise MultipartError('operation is terminal')
            if (type(offset) is not int or offset != self._offset or type(data) is not bytes
                    or not 0 < len(data) <= MAX_CHUNK_BYTES or self._offset + len(data) > self._size):
                self._fail('invalid chunk or offset; identity consumed')
            try:
                written = self._file.write(data)
                if written != len(data):
                    self._fail('short write; identity consumed')
                self._hash.update(data)
                self._offset += len(data)
            except BaseException:
                if self._state == 'OPEN':
                    self._state = 'UNKNOWN'; self._file.close()
                raise

    def finish(self, validate):
        """Validate staged content then link it atomically as resident.bin.

        validate(path, identity) must raise on any format or binding mismatch. It
        must not mutate the staged file; bytes are rehashed afterwards regardless.
        Link/fsync failures are UNKNOWN and must be reconciled, never replayed.
        """
        with self._lock:
            if self._state != 'OPEN':
                raise MultipartError('operation is terminal')
            if not callable(validate) or self._offset != self._size or self._hash.hexdigest() != self._sha:
                self._fail('incomplete, hash mismatch or validator missing')
            try:
                self._file.flush(); os.fsync(self._file.fileno()); self._file.close()
                staged = self.directory / 'staged.bin'
                validate(staged, dict(self._identity))
                h = hashlib.sha256(); size = 0
                with staged.open('rb') as f:
                    while chunk := f.read(MAX_CHUNK_BYTES):
                        size += len(chunk); h.update(chunk)
                if size != self._size or h.hexdigest() != self._sha:
                    self._state = 'FAILED'
                    raise MultipartError('validator changed artifact')
            except BaseException:
                if self._state == 'OPEN': self._state = 'FAILED'
                raise
            target = self.directory / 'resident.bin'
            try:
                os.link(staged, target, follow_symlinks=False)
                _fsync_dir(self.directory)
            except BaseException:
                self._state = 'UNKNOWN'
                raise
            self._state = 'COMMITTED'
            return target
