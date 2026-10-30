"""Local consumed namespaces; execution outcomes belong to the linked operation receipts."""

import json, os, re, time
from pathlib import Path


def reserve(root, base, context):
    if (
        not isinstance(base, str)
        or re.fullmatch(r"r53-[A-Za-z0-9_-]{8,60}", base) is None
    ):
        raise ValueError("namespace outside declared ASCII syntax")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("ledger root must be a private directory")
    root.mkdir(mode=0o700, exist_ok=True)
    effective_ids = [base, base + "-duplicate-slot"]
    for effective in effective_ids:
        entry = {
            "status": "RESERVED_UNKNOWN_NO_REPLAY",
            "scope": "Namespace consumption only; see linked execution receipt for actual outcome",
            "base_id": base,
            "effective_run_id": effective,
            "reserved_ns": time.monotonic_ns(),
            "context": context,
        }
        encoded = (json.dumps(entry, indent=2) + "\n").encode()
        # Partial allocation stays consumed if a later alias collides or persistence fails.
        fd = os.open(
            root / (effective + ".json"),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "wb") as f:
            f.write(encoded)
            f.flush()
            os.fsync(f.fileno())
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return effective_ids
