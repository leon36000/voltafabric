"""Read selected CPU metadata only; no model, library, CUDA, or service imports."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import sys


def file_snapshot(name, limit):
    path = Path(name)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
            raise ValueError("nonregular or oversized approved metadata asset")
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, limit + 1 - count))
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise ValueError("asset exceeded read cap")
            digest.update(chunk)
        after = os.fstat(fd)
        pin = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if pin(before) != pin(after) or count != before.st_size:
            raise ValueError("asset changed while read")
        return {"path": name, "bytes": count, "sha256": digest.hexdigest(),
                "device": before.st_dev, "inode": before.st_ino, "stable_read": True}
    finally:
        os.close(fd)


result = {"host": os.uname().nodename, "arch": os.uname().machine,
          "python_executable": sys.executable, "python_version": sys.version.split()[0],
          "distribution_scope": "default python interpreter metadata only",
          "distributions": {}, "approved_assets": [],
          "model_payload_bytes_read": 0, "GPU": False, "NVML": False,
          "engine_started": False}
for name in ("torch", "vllm", "1cat-vllm", "sglang"):
    try:
        result["distributions"][name] = {"version": importlib.metadata.version(name)}
    except importlib.metadata.PackageNotFoundError:
        result["distributions"][name] = {"status": "NOT_IN_THIS_INTERPRETER"}
for name, limit in (
    ("/home/pc1/Strata-data/optimizations/2026-10-14/build-zen5/strata", 128 * 1024 * 1024),
    ("/home/pc1/Strata-data/optimizations/2026-10-14/build-zen5/BUILD.json", 65536),
):
    try:
        result["approved_assets"].append(file_snapshot(name, limit))
    except (OSError, ValueError) as exc:
        result["approved_assets"].append({"path": name, "error": type(exc).__name__})
print(json.dumps(result, sort_keys=True, allow_nan=False))
