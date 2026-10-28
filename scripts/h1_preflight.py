#!/usr/bin/env python3
"""Static, CPU-only validation of a relocatable VoltaFabric H1 bundle.

This never imports CUDA/PyTorch, opens a CUDA library, queries a device, or
runs a GPU canary. A pass means only that the package is structurally usable
on this host for the later, explicitly separate H1 GPU gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import select
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

MAX_MANIFEST = 4 * 1024 * 1024
MAX_ENTRIES = 4096
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_TOOL_OUTPUT = 1024 * 1024
TOOL_TIMEOUT_SECONDS = 30
EXPECTED_ARTIFACTS = {
    "cpu_binary": "bin/vf",
    "sm70_library": "lib/libvf_qpn2_moe_sm70.so",
    "cuda_runtime": "lib/libcudart.so.12",
}
REQUIRED_SOURCE_FILES = [
    "scripts/deepseek_v41_qpn2_layer_canary.py",
    "scripts/deepseek_v41_qpn2_topk6_canary.py",
]
STRICT_PROVIDER_SOURCE_FILES = [
    "adapters/ffi_provider.py",
    "adapters/skinny_moe.py",
    "scripts/hardware_canary.py",
]
HASH_RE = re.compile(r"^[0-9a-f]{64}$")
GLIBC_RE = re.compile(r"\bGLIBC_(\d+(?:\.\d+)+)\b")
NEEDED_RE = re.compile(r"\(NEEDED\).*?\[([^\]]+)\]")
PATHS_RE = re.compile(r"\((?:RPATH|RUNPATH)\).*?\[([^\]]*)\]")
_VERSION_NAME_RE = re.compile(r"\bName:\s+(GLIBC_[A-Za-z0-9_.]+)")


class PreflightError(Exception):
    def __init__(self, message: str, status: str = "FAILED") -> None:
        super().__init__(message)
        self.status = status


def _safe_rel(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise PreflightError("manifest path must be a non-empty POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise PreflightError(f"unsafe manifest path: {value!r}")
    if path.as_posix() != value:
        raise PreflightError(f"manifest path is not canonical POSIX form: {value!r}")
    return value


def _bundle_file(root: Path, rel: str) -> Path:
    rel = _safe_rel(rel)
    path = root.joinpath(*PurePosixPath(rel).parts)
    try:
        path.resolve(strict=True).relative_to(root)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"missing or escaping bundle path: {rel}") from exc
    cur = root
    for part in PurePosixPath(rel).parts:
        cur = cur / part
        if cur.is_symlink():
            raise PreflightError(f"symlink is forbidden in bundle path: {rel}")
    if not path.is_file():
        raise PreflightError(f"bundle path is not a regular file: {rel}")
    return path


def _check_no_symlinks_and_bounds(root: Path) -> None:
    count = 0
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        base = Path(directory)
        count += len(dirs)
        if count > MAX_ENTRIES + 16:
            raise PreflightError("bundle contains too many filesystem entries")
        for name in list(dirs):
            p = base / name
            if p.is_symlink():
                raise PreflightError(
                    f"symlink directory is forbidden: {p.relative_to(root)}"
                )
        for name in files:
            p = base / name
            count += 1
            if count > MAX_ENTRIES + 16:
                raise PreflightError("bundle contains too many filesystem entries")
            if p.is_symlink():
                raise PreflightError(
                    f"symlink file is forbidden: {p.relative_to(root)}"
                )
            try:
                mode = p.stat().st_mode
                size = p.stat().st_size
            except OSError as exc:
                raise PreflightError(
                    f"cannot stat bundle file {p.relative_to(root)}: {exc}"
                ) from exc
            if not stat.S_ISREG(mode):
                raise PreflightError(f"non-regular bundle file: {p.relative_to(root)}")
            if size > MAX_FILE_BYTES:
                raise PreflightError(
                    f"bundle file exceeds {MAX_FILE_BYTES} byte cap: {p.relative_to(root)}"
                )
            total += size
            if total > MAX_TOTAL_BYTES:
                raise PreflightError("bundle exceeds total 1 GiB payload cap")


def _parse_sha256sums(
    path: Path, root: Path, *, max_entries: int = MAX_ENTRIES
) -> dict[str, str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"cannot read {path.name}: {exc}") from exc
    if len(raw) > MAX_MANIFEST:
        raise PreflightError(f"{path.name} exceeds 4 MiB")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreflightError(f"{path.name} is not UTF-8") from exc
    result: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            raise PreflightError(f"malformed {path.name} line {line_no}")
        digest, raw_rel = match.groups()
        rel = _safe_rel(raw_rel)
        if rel in result:
            raise PreflightError(f"duplicate {path.name} path: {rel}")
        result[rel] = digest
        if len(result) > max_entries:
            raise PreflightError(
                f"{path.name} contains more than {max_entries} entries"
            )
    if not result:
        raise PreflightError(f"{path.name} is empty")
    return result


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PreflightError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _verify_hash_list(root: Path, entries: dict[str, str]) -> None:
    for rel, expected in entries.items():
        path = _bundle_file(root, rel)
        actual = _hash_file(path)
        if actual != expected:
            raise PreflightError(f"SHA-256 mismatch for {rel}")


def _read_json(path: Path, label: str, cap: int = MAX_MANIFEST) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise PreflightError(f"cannot read {label}: {exc}") from exc
    if len(raw) > cap:
        raise PreflightError(f"{label} exceeds {cap} byte cap")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PreflightError(f"invalid {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object")
    return value


def _run_bounded(
    argv: list[str],
    *,
    timeout: int = TOOL_TIMEOUT_SECONDS,
    env: dict[str, str] | None = None,
) -> bytes:
    """Run a small CPU metadata command, enforcing a hard combined-output cap."""
    if env is None:
        env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            close_fds=True,
            env=env,
        )
    except OSError as exc:
        raise PreflightError(f"cannot execute {argv[0]}: {exc}", "BLOCKED") from exc
    assert proc.stdout is not None
    output = bytearray()
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                proc.kill()
                proc.wait()
                raise PreflightError(
                    f"command timed out: {Path(argv[0]).name}", "BLOCKED"
                )
            ready, _, _ = select.select([proc.stdout], [], [], min(remaining, 0.25))
            if ready:
                chunk = os.read(
                    proc.stdout.fileno(), min(65536, MAX_TOOL_OUTPUT + 1 - len(output))
                )
                if chunk:
                    output.extend(chunk)
                    if len(output) > MAX_TOOL_OUTPUT:
                        proc.kill()
                        proc.wait()
                        raise PreflightError(
                            f"command output exceeded {MAX_TOOL_OUTPUT} bytes"
                        )
                    continue
                try:
                    proc.wait(timeout=max(0.01, remaining))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    raise PreflightError(
                        f"command timed out: {Path(argv[0]).name}", "BLOCKED"
                    )
                break
            if proc.poll() is not None:
                break
        if proc.returncode != 0:
            detail = bytes(output[:4096]).decode("utf-8", "replace").strip()
            raise PreflightError(f"command failed ({proc.returncode}): {detail}")
        return bytes(output)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()


def _readelf(readelf: str, *args: str, path: Path) -> str:
    raw = _run_bounded([readelf, *args, str(path)])
    return raw.decode("utf-8", "replace")


def _check_elf(
    readelf: str, path: Path, label: str
) -> tuple[set[str], list[str], list[str]]:
    header = _readelf(readelf, "-h", path=path)
    if not re.search(r"^\s*Class:\s+ELF64\s*$", header, re.MULTILINE):
        raise PreflightError(f"{label} is not ELF64")
    if not re.search(
        r"^\s*Data:\s+2's complement, little endian\s*$", header, re.MULTILINE
    ):
        raise PreflightError(f"{label} is not little-endian ELF")
    machine = re.search(r"^\s*Machine:\s+(.+?)\s*$", header, re.MULTILINE)
    if not machine or "X86-64" not in machine.group(1).upper():
        found = machine.group(1) if machine else "unknown"
        raise PreflightError(f"{label} ELF machine is not x86-64: {found}")
    dynamic = _readelf(readelf, "-d", path=path)
    needed = set(NEEDED_RE.findall(dynamic))
    rpaths = PATHS_RE.findall(dynamic)
    versions = _readelf(readelf, "--version-info", "--wide", path=path)
    glibc = sorted(
        set(GLIBC_RE.findall(versions)), key=lambda v: tuple(map(int, v.split(".")))
    )
    if "GLIBC_PRIVATE" in versions:
        raise PreflightError(f"{label} requires non-portable GLIBC_PRIVATE symbols")
    return needed, rpaths, glibc


def _glibc_version() -> tuple[int, ...]:
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, OSError, ValueError):
        value = None
    match = re.fullmatch(r"glibc (\d+(?:\.\d+)+)", value or "")
    if not match:
        raise PreflightError(
            "host glibc version unavailable; compatibility is inconclusive", "BLOCKED"
        )
    return tuple(int(part) for part in match.group(1).split("."))


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _system_libraries() -> set[str]:
    names: set[str] = set()
    ldconfig = shutil.which("ldconfig")
    if ldconfig:
        try:
            text = _run_bounded([ldconfig, "-p"]).decode("utf-8", "replace")
            names.update(
                re.findall(
                    r"\s([A-Za-z0-9_.+-]+\.so(?:\.[A-Za-z0-9_.+-]+)*)\s+\(", text
                )
            )
        except PreflightError:
            pass
    for directory in (
        "/lib",
        "/lib64",
        "/usr/lib",
        "/usr/lib64",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu",
    ):
        p = Path(directory)
        try:
            names.update(x.name for x in p.glob("*.so*"))
        except OSError:
            continue
    return names


def _check_rpaths(rpaths: list[str], label: str) -> None:
    for group in rpaths:
        for item in filter(None, group.split(":")):
            if (
                item.startswith("/")
                and item not in ("/lib", "/lib64", "/usr/lib", "/usr/lib64")
                and not item.startswith(
                    ("/lib/", "/lib64/", "/usr/lib/", "/usr/lib64/")
                )
            ):
                raise PreflightError(
                    f"{label} has non-system absolute RPATH/RUNPATH: {item}"
                )
            if "/run/media/pc1" in item or "/home/pc6" in item:
                raise PreflightError(
                    f"{label} embeds a PC1 absolute library path: {item}"
                )


def _parse_package(root: Path, manifest_hashes: dict[str, str]) -> dict[str, Any]:
    package_path = _bundle_file(root, "PACKAGE.json")
    package = _read_json(package_path, "PACKAGE.json")
    expected = {
        "status": "H1_PACKAGE_PREPARED_NO_GPU",
        "cuda_executed": False,
        "h1_pending": True,
        "entrypoint": "run_h1.sh",
        "required_parameters": ["VF_GPU_PYTHON", "VF_GPU_UUID"],
    }
    schema_version = package.get("schema_version")
    if type(schema_version) is not int or schema_version not in (1, 2, 3):
        raise PreflightError("PACKAGE.json has invalid schema_version")
    expected["schema_version"] = schema_version
    if schema_version == 2:
        expected["required_source_files"] = REQUIRED_SOURCE_FILES
    if schema_version == 3:
        expected["required_source_files"] = (
            REQUIRED_SOURCE_FILES + STRICT_PROVIDER_SOURCE_FILES
        )
        expected["provider_contract"] = "strict_qpn2_torch_v1"
    elif "provider_contract" in package:
        raise PreflightError("a strict provider claim requires schema 3")
    for key, value in expected.items():
        if package.get(key) != value:
            raise PreflightError(f"PACKAGE.json has invalid {key}")
    req = package.get("requirements")
    if not isinstance(req, dict):
        raise PreflightError("PACKAGE.json requirements must be an object")
    requirements = {
        "host_arch": "x86_64",
        "elf_machine": 62,
        "gpu_sm": 70,
        "torch_cuda_major": 12,
        "torch_arch_required": ["sm_70", "compute_70"],
        "cuda_runtime_soname": "libcudart.so.12",
    }
    for key, value in requirements.items():
        if req.get(key) != value:
            raise PreflightError(
                f"PACKAGE.json requirements.{key} must equal {value!r}"
            )
    artifacts = package.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != len(EXPECTED_ARTIFACTS):
        raise PreflightError(
            "PACKAGE.json must declare exactly the three required artifacts"
        )
    by_kind: dict[str, dict[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            raise PreflightError("PACKAGE.json artifact entry must be an object")
        kind = artifact.get("kind")
        if kind not in EXPECTED_ARTIFACTS or kind in by_kind:
            raise PreflightError(f"unknown or duplicate artifact kind: {kind!r}")
        rel = _safe_rel(artifact.get("path"))
        if rel != EXPECTED_ARTIFACTS[kind]:
            raise PreflightError(f"{kind} path must be {EXPECTED_ARTIFACTS[kind]}")
        if artifact.get("elf_machine") != 62:
            raise PreflightError(f"{kind} manifest ELF machine must be 62")
        digest = artifact.get("sha256")
        if not isinstance(digest, str) or not HASH_RE.fullmatch(digest):
            raise PreflightError(f"{kind} has invalid SHA-256")
        if manifest_hashes.get(rel) != digest:
            raise PreflightError(f"{kind} PACKAGE.json hash disagrees with SHA256SUMS")
        path = _bundle_file(root, rel)
        if artifact.get("bytes") != path.stat().st_size:
            raise PreflightError(f"{kind} manifest byte count mismatch")
        by_kind[kind] = artifact
    if set(by_kind) != set(EXPECTED_ARTIFACTS):
        raise PreflightError("PACKAGE.json is missing a required artifact kind")
    cuda_artifact = by_kind["sm70_library"]
    if cuda_artifact.get("device_arch") != "sm_70":
        raise PreflightError(
            "SM70 library manifest does not identify device_arch sm_70"
        )
    if (
        not isinstance(cuda_artifact.get("cuda_toolkit_version"), str)
        or not cuda_artifact["cuda_toolkit_version"]
    ):
        raise PreflightError("SM70 library manifest lacks CUDA toolkit provenance")
    source_digest = package.get("source_manifest_sha256")
    if not isinstance(source_digest, str) or not HASH_RE.fullmatch(source_digest):
        raise PreflightError("PACKAGE.json source_manifest_sha256 is invalid")
    source_manifest = _bundle_file(root, "SOURCE_SHA256SUMS")
    if _hash_file(source_manifest) != source_digest:
        raise PreflightError("SOURCE_SHA256SUMS digest disagrees with PACKAGE.json")
    if manifest_hashes.get("SOURCE_SHA256SUMS") != source_digest:
        raise PreflightError(
            "SOURCE_SHA256SUMS is missing or has inconsistent SHA256SUMS entry"
        )
    if schema_version in (2, 3):
        source_hashes = _parse_sha256sums(source_manifest, root / "source")
        for rel in expected["required_source_files"]:
            path = _bundle_file(root, "source/" + rel)
            if rel not in source_hashes:
                raise PreflightError(
                    f"required canary missing from source manifest: {rel}"
                )
            if manifest_hashes.get("source/" + rel) != source_hashes[rel]:
                raise PreflightError(f"required canary hash mismatch: {rel}")
            if _hash_file(path) != source_hashes[rel]:
                raise PreflightError(f"required canary content mismatch: {rel}")
    return package


def _validate_source_manifest(root: Path, global_hashes: dict[str, str]) -> None:
    source_hashes = _parse_sha256sums(_bundle_file(root, "SOURCE_SHA256SUMS"), root)
    if not source_hashes:
        raise PreflightError("SOURCE_SHA256SUMS is empty")
    for raw_rel, digest in source_hashes.items():
        rel = raw_rel if raw_rel.startswith("source/") else f"source/{raw_rel}"
        if rel not in global_hashes or global_hashes[rel] != digest:
            raise PreflightError(
                f"source manifest entry is missing or inconsistent: {raw_rel}"
            )
        _bundle_file(root, rel)
    if not any(path.startswith("source/") for path in global_hashes):
        raise PreflightError("SHA256SUMS does not cover the source project")


def _step_passed(step: Any) -> bool:
    """verify.py writes each exit code as a JSON integer; only the integer 0 passes.
    `!= 0` alone also passed false, 0.0 and -0.0, and crashed on non-object steps."""
    return (
        isinstance(step, dict)
        and type(step.get("exit_code")) is int
        and step["exit_code"] == 0
    )


def _validate_provenance(
    root: Path, package: dict[str, Any], hashes: dict[str, str]
) -> None:
    cpu_path = package.get("cpu_proof_path")
    if cpu_path != "provenance/CPU_FULL_PASS.json":
        raise PreflightError("CPU qualification provenance is required")

    def receipt(name: str) -> dict[str, Any]:
        p = _bundle_file(root, name)
        if p.stat().st_size > MAX_MANIFEST:
            raise PreflightError("provenance receipt exceeds manifest bound")
        value = json.loads(p.read_text())
        if not isinstance(value, dict):
            raise PreflightError("provenance receipt must be an object")
        return value

    cpu = receipt(cpu_path)
    if (
        cpu.get("status") != "CPU_FULL_PASS"
        or cpu.get("sources_unchanged") is not True
        or cpu.get("cuda_executed") is not False
        or not cpu.get("steps")
        or not all(_step_passed(step) for step in cpu["steps"])
    ):
        raise PreflightError(
            "CPU qualification receipt is not a complete unchanged-source pass"
        )
    if (
        cpu.get("binary_sha256") != hashes["bin/vf"]
        or cpu.get("binary_bytes") != _bundle_file(root, "bin/vf").stat().st_size
    ):
        raise PreflightError("CPU binary differs from qualified revision")
    source_hashes = cpu.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise PreflightError("CPU qualification lacks source revision")
    for name, digest in source_hashes.items():
        rel = _safe_rel(name)
        if hashes.get("source/" + rel) != digest:
            raise PreflightError(f"source differs from CPU-qualified revision: {rel}")
    if package.get("schema_version") == 3:
        required = set(REQUIRED_SOURCE_FILES + STRICT_PROVIDER_SOURCE_FILES)
        missing = required - set(source_hashes)
        if missing:
            raise PreflightError(
                "CPU qualification omits required H1 source: "
                + ", ".join(sorted(missing))
            )
    cuda = receipt("provenance/CUDA_COMPILED.json")
    if (
        cuda.get("status") != "CUDA_COMPILED"
        or cuda.get("device_executed") is not False
        or cuda.get("architecture") != "sm_70"
    ):
        raise PreflightError(
            "expected CUDA_COMPILED provenance without device execution"
        )
    for name in (
        "cuda/qpn2_moe_ffi.h",
        "cuda/qpn2_moe_core.cuh",
        "cuda/qpn2_moe_sm70.cu",
        "cuda/qpn2_moe_validate.c",
    ):
        if cuda.get("hashes", {}).get(name) != hashes.get("source/" + name):
            raise PreflightError(
                f"CUDA source differs from compilation receipt: {name}"
            )
    if (
        cuda.get("hashes", {}).get("build/libvf_qpn2_moe_sm70.so")
        != hashes["lib/libvf_qpn2_moe_sm70.so"]
    ):
        raise PreflightError("SM70 library differs from compilation receipt")


def preflight(
    bundle: Path, *, routing_binary: str | None = None
) -> tuple[str, list[str], dict[str, Any]]:
    checks: list[str] = []
    try:
        root = bundle.resolve(strict=True)
    except OSError as exc:
        raise PreflightError(f"bundle directory unavailable: {exc}") from exc
    if not root.is_dir():
        raise PreflightError("bundle path is not a directory")
    if platform.system() != "Linux":
        raise PreflightError(f"unsupported host OS: {platform.system()}", "BLOCKED")
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        raise PreflightError(
            f"host architecture {machine} rejected; this bundle ships x86_64 binaries",
            "BLOCKED",
        )
    checks.append("host_linux_x86_64")

    _check_no_symlinks_and_bounds(root)
    if len(list(root.rglob("*"))) > MAX_ENTRIES + 16:
        raise PreflightError("bundle contains too many filesystem entries")

    sums_path = _bundle_file(root, "SHA256SUMS")
    hashes = _parse_sha256sums(sums_path, root)
    for required in (
        "PACKAGE.json",
        "SOURCE_SHA256SUMS",
        "run_h1.sh",
        *EXPECTED_ARTIFACTS.values(),
    ):
        if required not in hashes:
            raise PreflightError(f"SHA256SUMS does not cover required file {required}")
    actual_files = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p != root / "SHA256SUMS"
    }
    if actual_files != set(hashes):
        missing = sorted(actual_files - set(hashes))
        absent = sorted(set(hashes) - actual_files)
        raise PreflightError(
            f"SHA256SUMS payload coverage mismatch; unlisted={missing[:3]}, missing={absent[:3]}"
        )
    _verify_hash_list(root, hashes)
    checks.append("sha256_payload_integrity")

    package = _parse_package(root, hashes)
    _validate_source_manifest(root, hashes)
    checks.append("package_manifest_and_source_manifest")
    _validate_provenance(root, package, hashes)
    checks.append("cpu_revision_and_cuda_compilation_provenance")
    _bundle_file(root, "source/Cargo.toml")
    if routing_binary is not None:
        routing_binary = _safe_rel(routing_binary)
        if routing_binary != EXPECTED_ARTIFACTS["cpu_binary"]:
            raise PreflightError(
                "--routing-binary must name the declared verified CPU artifact bin/vf"
            )
    binary_rel = EXPECTED_ARTIFACTS["cpu_binary"]
    if binary_rel not in hashes:
        raise PreflightError("verified CPU routing binary is missing from SHA256SUMS")

    readelf = shutil.which("readelf")
    if not readelf:
        raise PreflightError("readelf is required for static ELF inspection", "BLOCKED")
    host_glibc = _glibc_version()
    system_libs = _system_libraries()
    elf_info: dict[str, dict[str, Any]] = {}
    for kind, rel in EXPECTED_ARTIFACTS.items():
        needed, rpaths, versions = _check_elf(readelf, _bundle_file(root, rel), rel)
        _check_rpaths(rpaths, rel)
        for version in versions:
            if host_glibc < _version_tuple(version):
                raise PreflightError(
                    f"{rel} requires GLIBC_{version}, host provides "
                    + ".".join(map(str, host_glibc)),
                    "BLOCKED",
                )
        elf_info[kind] = {"needed": sorted(needed), "glibc_required": versions}
    checks.append("elf64_little_endian_x86_64_and_glibc_compatibility")

    # Resolve every DT_NEEDED edge by bundle SONAME or system library name; no
    # ldd/loader invocation is used for the CUDA objects.
    bundle_sonames = {Path(rel).name for rel in hashes}
    for kind, info in elf_info.items():
        for soname in info["needed"]:
            if soname not in bundle_sonames and soname not in system_libs:
                raise PreflightError(
                    f"{kind} has unresolved DT_NEEDED dependency {soname}", "BLOCKED"
                )
    if "libcudart.so.12" not in elf_info["sm70_library"]["needed"]:
        raise PreflightError("SM70 library does not declare DT_NEEDED libcudart.so.12")
    if "libcudart.so.12" not in bundle_sonames:
        raise PreflightError("CUDA runtime SONAME is not present in bundle")
    checks.append("static_needed_soname_resolution")

    # This metadata-only CPU command executes the routing binary only after
    # hash and ELF checks. It does not load the CUDA library or initialize CUDA.
    binary = _bundle_file(root, binary_rel)
    if not os.access(binary, os.X_OK):
        raise PreflightError("verified CPU routing binary is not executable")
    clean_env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("LD_PRELOAD", "LD_LIBRARY_PATH", "LD_AUDIT")
    }
    capabilities = _run_bounded([str(binary), "capabilities"], env=clean_env).decode(
        "utf-8", "replace"
    )
    try:
        data = json.loads(capabilities)
        result = data["result"]
        if data.get("ok") is not True or not isinstance(result, dict):
            raise ValueError("bad success envelope")
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PreflightError(
            f"vf capabilities returned invalid JSON/envelope: {exc}"
        ) from exc
    for flag in ("cuda_compiled", "cuda_executed", "pcie_tested", "rdma_tested"):
        if result.get(flag) is not False:
            raise PreflightError(f"vf capabilities must report {flag}=false")
    if result.get("performance_claim") != "NONE":
        raise PreflightError("vf capabilities must report performance_claim=NONE")
    if result.get("cpu_reference") is not True:
        raise PreflightError("vf capabilities does not affirm CPU reference capability")
    checks.append("cpu_capabilities_flags_and_dynamic_loader")

    return (
        "STATIC_PREFLIGHT_PASS_NO_GPU",
        checks,
        {
            "host": {
                "os": platform.system(),
                "architecture": machine,
                "glibc": ".".join(map(str, host_glibc)),
            },
            "elf": elf_info,
            "cuda_artifact_target": {
                "manifest_device_arch": package["requirements"]["gpu_sm"],
                "artifact_device_arch": next(
                    artifact["device_arch"]
                    for artifact in package["artifacts"]
                    if artifact.get("kind") == "sm70_library"
                ),
                "independent_cubin_inspection": "NOT_PERFORMED_NO_CUDA_TOOL_EXECUTED",
            },
            "capabilities": result,
            "interpretation": "Static package/loader admission only; GPU H1 remains pending.",
        },
    )


def _write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise PreflightError(f"refusing to overwrite output receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if len(payload) > MAX_TOOL_OUTPUT:
        raise PreflightError("receipt exceeds 1 MiB")
    try:
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as exc:
        raise PreflightError(f"refusing to overwrite output receipt: {path}") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--routing-binary")
    args = parser.parse_args(argv)
    status = "FAILED"
    checks: list[str] = []
    details: dict[str, Any] = {}
    error = None
    try:
        status, checks, details = preflight(
            args.bundle, routing_binary=args.routing_binary
        )
    except PreflightError as exc:
        status = exc.status
        error = str(exc)
    except Exception as exc:  # produce a receipt for unexpected validation errors
        status = "FAILED"
        error = f"unexpected preflight error: {type(exc).__name__}: {exc}"
    receipt = {
        "schema_version": 1,
        "status": status,
        "cuda_executed": False,
        "h1_pending": True,
        "checks": checks,
        "details": details,
        "error": error,
    }
    try:
        _write_receipt(args.out, receipt)
    except PreflightError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0 if status == "STATIC_PREFLIGHT_PASS_NO_GPU" else 1


if __name__ == "__main__":
    raise SystemExit(main())
