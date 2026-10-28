"""Bounded metadata discovery; never import engines, read weights, or query GPU."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat

MAX_ENTRIES = 128
home = Path.home()
roots = [home / ".venvs", home / "venvs", home / ".local/share/uv/python"]
direct = [home / "Strata/.venv",
          Path("/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.venv")]
out = {"host": os.uname().nodename, "architecture": os.uname().machine,
       "scope": "selected roots and PATH only; no exhaustive environment search",
       "path_python312": shutil.which("python3.12"), "roots": [],
       "candidates": [], "model_names": [], "GPU": False, "NVML": False,
       "engine_started": False, "model_payload_bytes_read": 0}


def bounded_dirs(root):
    try:
        entries = []
        with os.scandir(root) as it:
            for entry in it:
                if len(entries) == MAX_ENTRIES:
                    return entries, True, None
                if entry.is_dir(follow_symlinks=False):
                    entries.append(Path(entry.path))
        return entries, False, None
    except OSError as exc:
        return [], False, type(exc).__name__


def inspect_env(path):
    row = {"path": str(path), "pyvenv_cfg": False, "python312_candidate": False,
           "target_distributions": []}
    cfg = path / "pyvenv.cfg"
    try:
        fd = os.open(cfg, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            s = os.fstat(f.fileno())
            if not stat.S_ISREG(s.st_mode) or s.st_size > 16384:
                raise ValueError("invalid configuration")
            raw = f.read(16385)
        if len(raw) > 16384:
            raise ValueError("configuration too large")
        row["pyvenv_cfg"] = True
        row["cfg_sha256"] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8", "strict").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip().lower() in ("version", "version_info"):
                row["declared_version"] = value.strip()
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError, ValueError) as exc:
        row["cfg_error"] = type(exc).__name__
    python = path / "bin/python3.12"
    row["python312_candidate"] = python.exists()
    site = path / "lib/python3.12/site-packages"
    if not site.is_dir():
        return row
    dirs, truncated, error = bounded_dirs(site)
    row["site_entries_truncated"] = truncated
    if error:
        row["site_error"] = error
    for dist in dirs:
        if not re.match(r"(?i)^(?:1cat[_-]?vllm|vllm|torch)[-_].*\.dist-info$", dist.name):
            continue
        meta = dist / "METADATA"
        try:
            fd = os.open(meta, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
            with os.fdopen(fd, "rb") as f:
                s = os.fstat(f.fileno())
                if not stat.S_ISREG(s.st_mode) or s.st_size > 262144:
                    raise ValueError("metadata not bounded regular")
                raw = f.read(262145)
            if len(raw) > 262144:
                raise ValueError("metadata too large")
            fields = {}
            for line in raw.decode("utf-8", "strict").splitlines():
                if line.startswith(("Name: ", "Version: ")):
                    key, value = line.split(": ", 1)
                    fields[key.lower()] = value
                if not line:
                    break
            row["target_distributions"].append({"directory": str(dist), "fields": fields,
                                                 "metadata_sha256": hashlib.sha256(raw).hexdigest()})
        except (OSError, UnicodeError, ValueError) as exc:
            row["target_distributions"].append({"directory": str(dist), "error": type(exc).__name__})
    return row


candidates = []
for root in roots:
    dirs, truncated, error = bounded_dirs(root)
    out["roots"].append({"path": str(root), "entry_count_bounded": len(dirs),
                         "truncated": truncated, "error": error})
    candidates.extend(dirs)
for path in direct:
    if path.is_dir():
        candidates.append(path)
for path in candidates:
    if path.parent.name == "python" and "3.12" not in path.name:
        continue
    row = inspect_env(path)
    if row["python312_candidate"] or row["target_distributions"]:
        out["candidates"].append(row)
models = home / "Strata-data/models"
dirs, truncated, error = bounded_dirs(models)
out["model_root_scan"] = {"path": str(models), "entries_bounded": len(dirs),
                          "truncated": truncated, "error": error}
out["model_names"] = [p.name for p in dirs if "nvfp4" in p.name.casefold()]
print(json.dumps(out, sort_keys=True, allow_nan=False))
