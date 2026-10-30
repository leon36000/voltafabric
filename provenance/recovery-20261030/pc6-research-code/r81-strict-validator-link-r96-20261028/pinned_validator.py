"""CPU-only R81 row validator: compile the pinned bytes actually consumed.

This is a private adapter, not the R81 submission or hardware qualification.
Assumes trusted Python runtime and stdlib, not a sandbox against the same UID.
"""
import ast
import functools
import hashlib
import json
import os
from pathlib import Path
import stat
import types

if not __debug__:
    raise RuntimeError("R88 uses assertions; Python optimization is forbidden")

RUNS = Path(__file__).resolve().parent.parent
SOURCE = RUNS.parent / "source"
R88 = RUNS / "r81-evidence-counter-types-r88-20261028/nominal_contract.py"
R55 = RUNS / "gpu-worker-ssh-boundary-r55/verify_wire.py"
R55_MANIFEST = RUNS / "gpu-worker-ssh-boundary-r55/snapshot/MANIFEST.json"
R24 = RUNS / "rdma-worker-20261025-r24/verify_r24.py"
GOLDEN = SOURCE / "fixtures/preweighted_mxfp4_tiny_golden.json"
R88_SHA = "c810319dea715523906ac76a1c2bc2aef71b217a313ec28df4b0e22678d31b3c"
MANIFEST_SHA = "014b00a09ad2d8b2d48fb7d6de0db62babc97de6285656f785cbb22ae4c64cd7"
R24_SHA = "be42d304f987c8615f3676110294a1b4dc473c32f85323d4e1dc9708752bf051"
GOLDEN_SHA = "51e4c60545c4f29dcbe03d57f017e522095ec2d37e19f67d2fe890670227d1ed"


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs)


def read_pinned(path, expected, limit=2 * 1024 * 1024):
    """Read one opened regular file, reject symlinks, bound, hash and return bytes."""
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("invalid expected digest")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("not a regular file")
        chunks, size = [], 0
        while True:
            block = os.read(fd, min(65536, limit + 1 - size))
            if not block:
                break
            size += len(block)
            if size > limit:
                raise ValueError("input exceeds cap")
            chunks.append(block)
        raw = b"".join(chunks)
    finally:
        os.close(fd)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("digest mismatch")
    return raw


def _only_function(tree, name):
    matches = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == name]
    if len(matches) != 1:
        raise ValueError("missing or duplicate function " + name)
    return matches[0]


def load_pinned():
    """Load R88, R55 and R24 from authenticated byte snapshots.

    R88's historical expected_collector dynamically reopens R55, which itself
    dynamically reopens R24. Replace only that path-loading preamble with a
    local reference to the exact pinned oracle functions; retain the original
    R88 collector body and original R55 add_bits AST unchanged.
    """
    validator_raw = read_pinned(R88, R88_SHA)
    manifest_raw = read_pinned(R55_MANIFEST, MANIFEST_SHA)
    oracle_sha = strict_json(manifest_raw)["files"]["verify_wire.py"]
    oracle_raw = read_pinned(R55, oracle_sha)
    rational_raw = read_pinned(R24, R24_SHA)

    rational = types.ModuleType("r96_rational")
    rational.__file__ = str(R24)
    exec(compile(rational_raw, str(R24), "exec"), rational.__dict__)
    oracle = types.ModuleType("r96_pinned_oracle")
    oracle.rational = rational
    oracle.value = functools.lru_cache(None)(rational.value)
    oracle_tree = ast.parse(oracle_raw, filename=str(R55))
    add_bits = _only_function(oracle_tree, "add_bits")
    exec(compile(ast.Module(body=[add_bits], type_ignores=[]), str(R55), "exec"), oracle.__dict__)

    validator = types.ModuleType("r96_pinned_r88")
    validator.__file__ = str(R88)
    # Compile these exact authenticated bytes, not a later read or pyc.
    exec(compile(validator_raw, str(R88), "exec"), validator.__dict__)
    tree = ast.parse(validator_raw, filename=str(R88))
    collector = _only_function(tree, "expected_collector")
    row_indices = [i for i, n in enumerate(collector.body)
                   if isinstance(n, ast.Assign) and len(n.targets) == 1
                   and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "rows"]
    if len(row_indices) != 1:
        raise ValueError("R88 collector layout changed")
    collector.body = [ast.Assign(targets=[ast.Name(id="module", ctx=ast.Store())],
                                 value=ast.Name(id="_r96_oracle", ctx=ast.Load()))] + collector.body[row_indices[0]:]
    transformed = ast.fix_missing_locations(ast.Module(body=[collector], type_ignores=[]))
    validator.__dict__["_r96_oracle"] = oracle
    exec(compile(transformed, str(R88), "exec"), validator.__dict__)
    return validator, dict(validator_sha256=R88_SHA, manifest_sha256=MANIFEST_SHA,
                           oracle_sha256=oracle_sha, rational_sha256=R24_SHA,
                           golden_sha256=GOLDEN_SHA)


def validate_rows_log(log_raw, golden_path=GOLDEN):
    validator, proof = load_pinned()
    golden_raw = read_pinned(golden_path, GOLDEN_SHA, 1024 * 1024)
    result = validator.admit_rows(validator.rows_from_log(log_raw), golden_raw)
    # Row validation is only one gate; R81 needs process exits and postflight.
    if result.get("status") != "R81_ROWS_ADMITTED_NOT_COMPLETE_HARDWARE_QUALIFICATION":
        raise ValueError("unexpected R88 status")
    if result.get("guard_exit_and_postflight_still_required") is not True:
        raise ValueError("missing hardware qualification boundary")
    return result, proof
