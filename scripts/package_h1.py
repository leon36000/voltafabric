#!/usr/bin/env python3
"""Create one explicit H1 bundle from verified local artifacts; no GPU calls."""

import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil

COPY_DIRS = (
    "src",
    "crates",
    "adapters",
    "scripts",
    "tests",
    "fixtures",
    "cuda",
    "docs",
    "third_party",
    "handoff",
)
SKIP_PARTS = {
    "target",
    ".tools",
    ".runs",
    ".worktrees",
    "__pycache__",
    ".ruff_cache",
    ".git",
    "build",
    "evidence",
}
COPY_FILES = (
    "Cargo.toml",
    "Cargo.lock",
    "rust-toolchain.toml",
    "requirements-cpu.txt",
    "README.md",
    "AGENTS.md",
    "agents/routes.example.json",
    "agents/queue.example.json",
    "evidence/continuation_a105_all_backbone_headers/fetch_headers.py",
    "fixtures/h1_reproduction_files.json",
)
CUDA_FILES = (
    "cuda/qpn2_moe_ffi.h",
    "cuda/qpn2_moe_core.cuh",
    "cuda/qpn2_moe_sm70.cu",
    "cuda/qpn2_moe_validate.c",
)
H1_REQUIRED_SOURCE_FILES = (
    "scripts/deepseek_v41_qpn2_layer_canary.py",
    "scripts/deepseek_v41_qpn2_topk6_canary.py",
    "adapters/ffi_provider.py",
    "adapters/skinny_moe.py",
    "scripts/hardware_canary.py",
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(65536):
            h.update(block)
    return h.hexdigest()


def copy(source, dest):
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def source_snapshot(source):
    paths = set()
    for directory in COPY_DIRS:
        for p in (source / directory).rglob("*"):
            rel = p.relative_to(source)
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            if p.is_symlink():
                raise ValueError(f"source symlink not packageable: {rel}")
            if p.is_file():
                paths.add(rel.as_posix())
    paths.update(COPY_FILES)
    paths.update(reproduction_files(source))
    paths.update(p.name for p in source.glob("LICENSE*"))
    return {name: digest(source / name) for name in sorted(paths)}


def reproduction_files(source):
    """Explicit bounded CPU-oracle inputs; never copy the evidence tree wholesale."""
    manifest = source / "fixtures/h1_reproduction_files.json"
    if manifest.stat().st_size > 32768:
        raise ValueError("reproduction input manifest exceeds size limit")
    names = json.loads(manifest.read_text())
    if not isinstance(names, list) or len(names) > 128:
        raise ValueError("invalid reproduction input manifest")
    seen = set()
    total = 0
    for name in names:
        if not isinstance(name, str):
            raise ValueError("invalid reproduction input path")
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not name.startswith("evidence/")
            or path.as_posix() != name
            or name in seen
        ):
            raise ValueError("invalid reproduction input path")
        p = source / name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(source):
            raise ValueError("invalid reproduction input file")
        total += p.stat().st_size
        if total > 32 * 1024 * 1024:
            raise ValueError("reproduction inputs exceed size limit")
        seen.add(name)
    return names


def _step_passed(step) -> bool:
    """verify.py writes each exit code as a JSON integer; only the integer 0 passes.
    `!= 0` alone also passed false, 0.0 and -0.0, and crashed on non-object steps."""
    return (
        isinstance(step, dict)
        and type(step.get("exit_code")) is int
        and step["exit_code"] == 0
    )


def _check_elf_header(path, label):
    """Bounded static identity check; loader/symbol checks remain preflight's job."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"source must be a regular non-symlink file: {path}")
    with path.open("rb") as stream:
        header = stream.read(64)
    if (
        len(header) != 64
        or header[:7] != b"\x7fELF\x02\x01\x01"
        or int.from_bytes(header[16:18], "little") not in (2, 3)
        or int.from_bytes(header[18:20], "little") != 62
        or int.from_bytes(header[20:24], "little") != 1
        or int.from_bytes(header[52:54], "little") != 64
    ):
        raise ValueError(f"artifact must be little-endian ELF64 x86-64: {label}")


def package(args):
    source = args.source.resolve()
    out = args.out.resolve()
    if out == source or source.is_relative_to(out) or out.is_relative_to(source):
        raise ValueError("bundle must be separate from canonical source")
    if out.exists():
        raise FileExistsError("inspect existing bundle before any new packaging")
    cpu_raw = args.cpu_proof.read_bytes()
    cpu_proof = json.loads(cpu_raw.decode("utf-8"))
    if (
        cpu_proof.get("status") != "CPU_FULL_PASS"
        or cpu_proof.get("sources_unchanged") is not True
        or cpu_proof.get("cuda_executed") is not False
        or not cpu_proof.get("steps")
        or not all(_step_passed(step) for step in cpu_proof["steps"])
    ):
        raise ValueError("a fresh unchanged-source CPU_FULL_PASS receipt is required")
    if digest(args.binary) != cpu_proof.get(
        "binary_sha256"
    ) or args.binary.stat().st_size != cpu_proof.get("binary_bytes"):
        raise ValueError("CPU binary differs from qualification receipt")
    source_hashes = cpu_proof.get("source_sha256")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("CPU qualification must identify its sources")
    snapshot = source_snapshot(source)
    for name, expected in source_hashes.items():
        rel = Path(name)
        candidate = source / rel
        if (
            rel.is_absolute()
            or ".." in rel.parts
            or not candidate.resolve().is_relative_to(source)
        ):
            raise ValueError("unsafe CPU proof source path")
        if name not in snapshot:
            raise ValueError(f"CPU qualification source is not packaged: {name}")
        if digest(candidate) != expected:
            raise ValueError(f"source differs from CPU qualification receipt: {name}")
    missing_canaries = set(H1_REQUIRED_SOURCE_FILES) - set(snapshot)
    if missing_canaries:
        raise ValueError(
            "H1 source is missing required canary: "
            + ", ".join(sorted(missing_canaries))
        )
    missing_qualification = set(H1_REQUIRED_SOURCE_FILES) - set(source_hashes)
    if missing_qualification:
        raise ValueError(
            "CPU qualification omits required H1 source: "
            + ", ".join(sorted(missing_qualification))
        )
    cuda_raw = args.cuda_proof.read_bytes()
    proof = json.loads(cuda_raw.decode("utf-8"))
    if (
        proof.get("status") != "CUDA_COMPILED"
        or proof.get("device_executed") is not False
        or proof.get("architecture") != "sm_70"
    ):
        raise ValueError("expected the existing unexecuted SM70 compilation receipt")
    for name in CUDA_FILES:
        if digest(source / name) != proof["hashes"][name]:
            raise ValueError(f"CUDA source differs from compiled receipt: {name}")
    if digest(args.library) != proof["hashes"]["build/libvf_qpn2_moe_sm70.so"]:
        raise ValueError("SM70 library differs from compilation receipt")
    if args.cuda_license.name != "LICENSE" or not args.cuda_license.is_file():
        raise ValueError("provide the unchanged CUDA runtime component license")
    for label, path in (
        ("cpu_binary", args.binary),
        ("sm70_library", args.library),
        ("cuda_runtime", args.cudart.resolve(strict=True)),
    ):
        _check_elf_header(path, label)
    frozen_inputs = {
        "lib/libvf_qpn2_moe_sm70.so": proof["hashes"]["build/libvf_qpn2_moe_sm70.so"],
        "lib/libcudart.so.12": digest(args.cudart.resolve(strict=True)),
        "provenance/CPU_FULL_PASS.json": hashlib.sha256(cpu_raw).hexdigest(),
        "provenance/CUDA_COMPILED.json": hashlib.sha256(cuda_raw).hexdigest(),
    }
    # Claim the name with an empty directory, build beside it and publish only a
    # complete bundle: a refusal or error leaves neither a partial bundle nor a stage.
    out.mkdir(parents=True)
    stage = out.parent / f".{out.name}.partial"
    try:
        stage.mkdir()
    except BaseException:
        out.rmdir()
        raise
    try:
        build(stage, source, args, cpu_proof, source_hashes, snapshot, frozen_inputs)
        # Replaces only the empty claim; a directory filled meanwhile is refused.
        os.rename(stage, out)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        with contextlib.suppress(OSError):
            out.rmdir()
        raise
    return dict(
        status="H1_PACKAGE_PREPARED_NO_GPU",
        files=len((out / "SHA256SUMS").read_text().splitlines()),
        bytes=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        cuda_executed=False,
        h1_pending=True,
    )


def build(out, source, args, cpu_proof, source_hashes, snapshot, frozen_inputs):
    """Write every bundle file under out, which is the private staging directory."""
    copied = []
    for directory in COPY_DIRS:
        for p in sorted((source / directory).rglob("*")):
            rel = p.relative_to(source)
            if any(part in SKIP_PARTS for part in rel.parts):
                continue
            if p.is_symlink():
                raise ValueError(f"source symlink not packageable: {rel}")
            if p.is_file():
                copy(p, out / "source" / rel)
                copied.append(rel.as_posix())
    for name in (*COPY_FILES, *reproduction_files(source)):
        copy(source / name, out / "source" / name)
        copied.append(name)
    for p in source.glob("LICENSE*"):
        copy(p, out / "source" / p.name)
        copied.append(p.name)
    copy(source / "handoff/h1_bundle_run.sh", out / "run_h1.sh")
    (out / "run_h1.sh").chmod(0o755)
    copy(args.binary, out / "bin/vf")
    (out / "bin/vf").chmod(0o755)
    copy(args.library, out / "lib/libvf_qpn2_moe_sm70.so")
    # Explicit runtime path may be a SONAME symlink; copy its resolved immutable bytes.
    copy(args.cudart.resolve(strict=True), out / "lib/libcudart.so.12")
    copy(args.cuda_license, out / "licenses/CUDA-CUDART-LICENSE")
    copy(args.cuda_proof, out / "provenance/CUDA_COMPILED.json")
    copy(args.cpu_proof, out / "provenance/CPU_FULL_PASS.json")
    if digest(out / "bin/vf") != cpu_proof["binary_sha256"]:
        raise ValueError("CPU binary changed while packaging")
    for name, expected in frozen_inputs.items():
        if digest(out / name) != expected:
            raise ValueError(f"input changed while packaging: {name}")
    for name, expected in source_hashes.items():
        if digest(out / "source" / name) != expected:
            raise ValueError(f"copied source differs from CPU qualification: {name}")
    patch = source / "evidence/continuation_a123/cuda/math_functions.noexcept.patch"
    if patch.is_file():
        copy(patch, out / "provenance/isolated-gcc14-noexcept.patch")
    copied = sorted(set(copied))
    if set(copied) != set(snapshot):
        raise ValueError("source packageable inventory changed while copying")
    for name, expected in snapshot.items():
        if digest(out / "source" / name) != expected:
            raise ValueError(f"source snapshot changed while copying: {name}")
    if source_snapshot(source) != snapshot:
        raise ValueError("canonical packageable snapshot changed during packaging")
    source_sums = "".join(
        f"{digest(out / 'source' / name)}  {name}\n" for name in copied
    )
    (out / "SOURCE_SHA256SUMS").write_text(source_sums)
    # The source tree can also independently reproduce its integrity check.
    (out / "source/SHA256SUMS").write_text(source_sums)
    artifacts = []
    for kind, path in [
        ("cpu_binary", "bin/vf"),
        ("sm70_library", "lib/libvf_qpn2_moe_sm70.so"),
        ("cuda_runtime", "lib/libcudart.so.12"),
    ]:
        p = out / path
        _check_elf_header(p, kind)
        entry = dict(
            path=path,
            kind=kind,
            sha256=digest(p),
            bytes=p.stat().st_size,
            elf_machine=62,
        )
        if kind == "sm70_library":
            entry.update(device_arch="sm_70", cuda_toolkit_version="12.8.93")
        artifacts.append(entry)
    manifest = dict(
        schema_version=3,
        provider_contract="strict_qpn2_torch_v1",
        status="H1_PACKAGE_PREPARED_NO_GPU",
        source_manifest_sha256=digest(out / "SOURCE_SHA256SUMS"),
        cuda_executed=False,
        h1_pending=True,
        cpu_proof_path="provenance/CPU_FULL_PASS.json",
        entrypoint="run_h1.sh",
        required_parameters=["VF_GPU_PYTHON", "VF_GPU_UUID"],
        required_source_files=list(H1_REQUIRED_SOURCE_FILES),
        requirements=dict(
            host_arch="x86_64",
            elf_machine=62,
            gpu_sm=70,
            torch_cuda_major=12,
            torch_arch_required=["sm_70", "compute_70"],
            cuda_runtime_soname="libcudart.so.12",
        ),
        artifacts=artifacts,
        abi=dict(version=1, sizeof=208, alignment=8, fields=31),
        redistribution=dict(
            cuda_runtime="Unchanged bytes; component EULA copied in licenses/CUDA-CUDART-LICENSE",
            project="Directory-scoped MIT/Apache2 notices retained under source/third_party",
            crates="Cargo.lock pins dependencies; crates are not vendored",
        ),
        interpretation="Host binary is x86_64; ARM or older libc requires a native rebuild and another preflight. No GPU correctness, model-wide coverage or performance is claimed.",
    )
    (out / "PACKAGE.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "BUILD_AND_RUN.md").write_text(
        "H1 package: no GPU executed.\n\n"
        "Static check (stdlib Python + readelf, receipt outside bundle):\n"
        "  python3 source/scripts/h1_preflight.py --bundle . --out ../h1-static.json\n\n"
        "GPU run only after docs/HARDWARE_GATE.md and actual V100 reservation:\n"
        "  VF_GPU_PYTHON=/existing/verified/python VF_GPU_UUID=GPU-reserved bash run_h1.sh\n\n"
        "Optional pinned full expert projection gate (after the same reservation):\n"
        "  VF_H1_RESIDENT_IO=safetensors VF_H1_FULL_EXPERT=1 bash run_h1.sh\n"
        "This checks three projections on selected columns, not full layer inference or all H1. "
        "See source/docs/H1_FULL_EXPERT.md for fixed tolerance and unverified GPU arithmetic.\n\n"
        "VF_PREFLIGHT_PYTHON, VF_H1_RUN_ID and VF_H1_RECEIPT_DIR are optional. Receipts go outside the immutable bundle. "
        "A canary pass leaves full H1, H2 and H3 pending.\n\n"
        "The supplied routing binary and CUDA library are Linux x86_64. Preflight rejects incompatible architecture or libc. "
        "For another host rebuild vf from source with Rust 1.90.0 and Cargo.lock; initial crate resolution may require network access. "
        "Keep RUSTUP_HOME/CARGO_HOME/CARGO_TARGET_DIR under the target project. "
        "CUDA source/build_sm70.sh accepts an existing isolated tool root and explicit compiler/include overrides; never changes a driver. "
        "The compiled receipt records CUDA12.8.93/GCC14.3 with six declaration-only isolated compatibility edits; patch is preserved. "
        "The supplied CUDA runtime EULA and donor licenses must remain with redistributed artifacts.\n"
    )
    sums = "".join(
        f"{digest(p)}  {p.relative_to(out).as_posix()}\n"
        for p in sorted(out.rglob("*"))
        if p.is_file() and p != out / "SHA256SUMS"
    )
    (out / "SHA256SUMS").write_text(sums)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[1])
    for name in (
        "binary",
        "library",
        "cudart",
        "cuda-license",
        "cuda-proof",
        "cpu-proof",
        "out",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(package(args)))


if __name__ == "__main__":
    main()
