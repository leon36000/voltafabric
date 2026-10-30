"""Private CPU-only local-reuse admission; no network or GPU contract.

The first header read is untrusted. The owner performs the full digest, tensor,
scale and identity validation before the receiver durably commits its receipt.
"""
import os
import pathlib
import stat
import torch
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from scripts.resident_artifact import _open_header, _stable_stat

def _read_untrusted_info(path):
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0) | getattr(os, 'O_CLOEXEC', 0)
    fd = os.open(pathlib.Path(path), flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('resident source is not a regular file')
        _, info, _ = _open_header(fd, before.st_size)
        if _stable_stat(before) != _stable_stat(os.fstat(fd)):
            raise ValueError('resident source changed during header read')
        return info
    finally:
        os.close(fd)

def admit_owner(receiver, expected_sha256, provider):
    """Create a fully verified owner inside `finish`, then publish after commit.

    Receiver must bind the requested bytes and run its file identity/quiescence
    checks around the callback. An exception after owner creation closes it; it
    is never returned to the caller or sent a work request before commit.
    """
    pending = []
    def validate(path, bound):
        info = _read_untrusted_info(path)
        if any(info[k] != bound[k] for k in ('model_revision', 'layer', 'epoch', 'rank')):
            raise ValueError('resident binding mismatch')
        mapping = torch.full((256,), -1, dtype=torch.int64)
        for physical, expert in enumerate(info['physical_experts']):
            if expert['id'] >= len(mapping):
                raise ValueError('expert outside model domain')
            mapping[expert['id']] = physical
        owner = PersistentQPN2Owner.load(artifact_path=path, expected_info=info,
            expected_sha256=expected_sha256, provider=provider, expert_map=mapping, device='cpu')
        pending.append(owner)
        if owner.state != 'READY':
            raise RuntimeError('owner did not become ready')
    try:
        receiver.finish(validate)
        if len(pending) != 1:
            raise RuntimeError('receiver did not validate exactly one owner')
        return pending.pop()
    except BaseException as error:
        for owner in pending:
            try:
                owner.close()
            except BaseException as cleanup:
                error.add_note('unpublished owner close failed: ' + type(cleanup).__name__)
        raise
