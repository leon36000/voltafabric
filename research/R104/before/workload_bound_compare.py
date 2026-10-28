"""Privately bind R100 workload bytes to both R97 records before comparison.

The result admits declared producer counts only. It does not establish tokenization,
runtime execution, hardware identity, measurement validity, or performance.
"""
import hashlib
import json
import os
import stat
import tempfile
import types

LIMIT = 256 * 1024
R97_SNAPSHOT_SHA256 = "e398a0cc0d36da8b25fb971488b73870f045507326c42f19fccf33d812eeb4d2"
R103_SNAPSHOT_SHA256 = "04ca282e61ac18d16aa63e3ec9626937e613b1c50ff5da1c719754f7d75fa3c9"
_HERE = os.path.dirname(os.path.abspath(__file__))


class Invalid(ValueError):
    """An input could not be safely bound and compared."""


def _load_hashed_snapshot(name, filename, digest):
    path = os.path.join(_HERE, filename)
    try:
        with open(path, "rb") as stream:
            source = stream.read(1024 * 1024 + 1)
    except OSError as exc:
        raise RuntimeError("required source snapshot unavailable") from exc
    if len(source) > 1024 * 1024 or hashlib.sha256(source).hexdigest() != digest:
        raise RuntimeError("required source snapshot hash mismatch")
    module = types.ModuleType(name)
    module.__file__ = path
    exec(compile(source, path, "exec"), module.__dict__)
    return module


_r97 = _load_hashed_snapshot("r104_r97_compare", "r97_compare_snapshot.py", R97_SNAPSHOT_SHA256)
_r103 = _load_hashed_snapshot("r104_workload_admission", "workload_admission.py", R103_SNAPSHOT_SHA256)


def _read_record(path):
    if (type(path) is not str or not path or len(path) > 4096 or
            "\x00" in path or not os.path.isabs(path)):
        raise Invalid("run path must be a nonempty absolute string")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise Invalid("run path is not a regular file")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                raw = stream.read(LIMIT + 1)
        finally:
            if fd >= 0:
                os.close(fd)
    except Invalid:
        raise
    except (OSError, ValueError) as exc:
        raise Invalid("run record cannot be read") from exc
    if len(raw) > LIMIT:
        raise Invalid("run record exceeds byte limit")
    try:
        record = json.loads(raw.decode("utf-8", errors="strict"),
                            object_pairs_hook=_r97.pairs_unique,
                            parse_constant=_r97.invalid_constant)
    except (UnicodeError, json.JSONDecodeError, _r97.Invalid, RecursionError, ValueError) as exc:
        raise Invalid("run record is not strict JSON") from exc
    if type(record) is not dict:
        raise Invalid("run record must be a JSON object")
    return raw, hashlib.sha256(raw).hexdigest(), record


def _workload_expectation(record):
    try:
        work = record["workload"]
        metrics = record["metrics"]
        return {
            "prompt_corpus_sha256": work["prompt_corpus_sha256"],
            "token_counts_sha256": work["token_counts_sha256"],
            "output_token_counts_sha256": metrics["output_token_counts_sha256"],
            "requests": metrics["requests"],
            "output_tokens": metrics["output_tokens"],
        }
    except (KeyError, TypeError) as exc:
        raise Invalid("run record lacks workload binding fields") from exc


def compare_bound_files(left_run_path, right_run_path, corpus_path,
                        input_counts_path, output_counts_path):
    """Bind one admitted set of artifact bytes to both run snapshots, then apply R97."""
    try:
        left_raw, left_hash, left = _read_record(left_run_path)
        right_raw, right_hash, right = _read_record(right_run_path)

        # Admission is performed against left. R97 then requires the two records to
        # agree on every workload hash and on request/output-token counts, binding the
        # exact admitted bytes to right as well without a second workload-file read.
        admitted = _r103.admit_workload(
            corpus_path, input_counts_path, output_counts_path,
            _workload_expectation(left))

        tmp_root = os.path.join(_HERE, ".private-tmp")
        os.makedirs(tmp_root, mode=0o700, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="bound-runs-", dir=tmp_root) as tmp:
            left_snapshot = os.path.join(tmp, "left.json")
            right_snapshot = os.path.join(tmp, "right.json")
            with open(left_snapshot, "wb") as stream:
                stream.write(left_raw)
            with open(right_snapshot, "wb") as stream:
                stream.write(right_raw)
            result = _r97.compare_files(left_snapshot, right_snapshot)

        result["workload_admission"] = admitted
        result["run_snapshot_sha256"] = {"left": left_hash, "right": right_hash}
        result.setdefault("limitations", []).append(
            "Token counts are producer declarations; admission does not prove tokenizer correctness or runtime execution."
        )
        result.setdefault("limitations", []).append(
            "Synthetic records demonstrate schema and byte binding only, not a real benchmark."
        )
        return result
    except (_r97.Invalid, _r103.Invalid, Invalid, OSError, UnicodeError,
            json.JSONDecodeError, RecursionError, ValueError) as exc:
        return {"verdict": "NON_COMPARABLE", "reasons": [str(exc)]}
