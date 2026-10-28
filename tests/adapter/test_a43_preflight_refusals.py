"""Each refusal of scripts/h1_preflight.py, driven by an input that violates it alone.

The A43 raise-guard sweep disabled every `or` operand of the preflight's
`if ..: raise` guards; most survived because the older tests stop at the first
failing check of a /bin/true fixture. Here readelf, ldconfig and the
`vf capabilities` call are replaced by canned outputs, so a synthetic bundle
passes completely and each test changes exactly one fact and asserts that
guard's own message. The real-tool path stays covered by test_h1_preflight.py.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from scripts import h1_preflight  # noqa: E402

PreflightError = h1_preflight.PreflightError
CUDA_NAMES = (
    "qpn2_moe_ffi.h",
    "qpn2_moe_core.cuh",
    "qpn2_moe_sm70.cu",
    "qpn2_moe_validate.c",
)
HEADER = (
    "ELF Header:\n"
    "  Class:                             ELF64\n"
    "  Data:                              2's complement, little endian\n"
    "  Machine:                           Advanced Micro Devices X86-64\n"
)
SYSTEM_LIBS = {"libc.so.6", "libm.so.6"}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dynamic(*needed: str, runpath: str | None = None) -> str:
    lines = [
        f" 0x0000000000000001 (NEEDED)             Shared library: [{n}]"
        for n in needed
    ]
    if runpath is not None:
        lines.append(
            f" 0x000000000000001d (RUNPATH)            Library runpath: [{runpath}]"
        )
    return "\n".join(lines) + "\n"


def _versions(*names: str) -> str:
    return "".join(f"  0x0010: Name: {n}  Flags: none  Version: 2\n" for n in names)


class FakeTools:
    """Canned readelf and `vf capabilities` outputs, keyed by file name."""

    def __init__(self) -> None:
        self.header = {
            n: HEADER for n in ("vf", "libvf_qpn2_moe_sm70.so", "libcudart.so.12")
        }
        self.dynamic = {
            "vf": _dynamic("libc.so.6", "libm.so.6"),
            "libvf_qpn2_moe_sm70.so": _dynamic(
                "libcudart.so.12", "libc.so.6", runpath="$ORIGIN"
            ),
            "libcudart.so.12": _dynamic("libc.so.6"),
        }
        self.versions = {
            "vf": _versions("GLIBC_2.17", "GLIBC_2.2.5"),
            "libvf_qpn2_moe_sm70.so": _versions("GLIBC_2.2.5"),
            "libcudart.so.12": _versions("GLIBC_2.3"),
        }
        self.capabilities: object = {
            "ok": True,
            "result": {
                "cuda_compiled": False,
                "cuda_executed": False,
                "pcie_tested": False,
                "rdma_tested": False,
                "performance_claim": "NONE",
                "cpu_reference": True,
            },
        }
        self.readelf: str | None = "/fake/bin/readelf"

    def run(self, argv: list[str], **_kwargs: object) -> bytes:
        tool = Path(argv[0]).name  # a None tool path fails here, as exec would
        if tool == "readelf":
            flag, target = argv[1], Path(argv[-1]).name
            table = {"-h": self.header, "-d": self.dynamic}.get(flag, self.versions)
            return table[target].encode()
        if argv[1:] == ["capabilities"]:
            return json.dumps(self.capabilities).encode()
        raise AssertionError(f"unexpected command: {argv}")

    def which(self, name: str) -> str | None:
        return self.readelf if name == "readelf" else shutil.which(name)

    @contextlib.contextmanager
    def installed(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(h1_preflight, "_run_bounded", self.run)
            )
            stack.enter_context(
                mock.patch.object(h1_preflight.shutil, "which", self.which)
            )
            stack.enter_context(
                mock.patch.object(
                    h1_preflight, "_system_libraries", return_value=set(SYSTEM_LIBS)
                )
            )
            yield


def _apply(change, value):
    if change is None:
        return value
    replaced = change(value)
    return value if replaced is None else replaced


def build(
    root: Path, *, source_sums=None, cpu=None, cuda=None, package=None, sums=None
):
    """Write a bundle that passes; each hook may edit or replace one document."""
    for rel in ("bin", "lib", "source/cuda", "provenance"):
        (root / rel).mkdir(parents=True)
    (root / "run_h1.sh").write_text("#!/bin/sh\nexit 1\n")
    (root / "source/Cargo.toml").write_text("[package]\nname='fixture'\n")
    for name in CUDA_NAMES:
        (root / "source/cuda" / name).write_text(f"synthetic {name}\n")
    # Distinct payloads, so a path/kind mix-up cannot pass on equal hashes.
    for rel in ("bin/vf", "lib/libvf_qpn2_moe_sm70.so", "lib/libcudart.so.12"):
        (root / rel).write_bytes(f"synthetic ELF stand-in for {rel}\n".encode())
    (root / "bin/vf").chmod(0o755)
    text = "".join(
        f"{_sha(p)}  {p.relative_to(root / 'source').as_posix()}\n"
        for p in sorted((root / "source").rglob("*"))
        if p.is_file()
    )
    (root / "SOURCE_SHA256SUMS").write_text(_apply(source_sums, text))
    proof = {
        "status": "CPU_FULL_PASS",
        "sources_unchanged": True,
        "cuda_executed": False,
        "steps": [{"exit_code": 0}],
        "binary_sha256": _sha(root / "bin/vf"),
        "binary_bytes": (root / "bin/vf").stat().st_size,
        "source_sha256": {"Cargo.toml": _sha(root / "source/Cargo.toml")},
    }
    (root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(_apply(cpu, proof)))
    compiled = {
        "status": "CUDA_COMPILED",
        "device_executed": False,
        "architecture": "sm_70",
        "hashes": {f"cuda/{n}": _sha(root / "source/cuda" / n) for n in CUDA_NAMES},
    }
    compiled["hashes"]["build/libvf_qpn2_moe_sm70.so"] = _sha(
        root / "lib/libvf_qpn2_moe_sm70.so"
    )
    (root / "provenance/CUDA_COMPILED.json").write_text(
        json.dumps(_apply(cuda, compiled))
    )

    def artifact(kind: str, rel: str, **extra: object) -> dict:
        return {
            "path": rel,
            "kind": kind,
            "sha256": _sha(root / rel),
            "bytes": (root / rel).stat().st_size,
            "elf_machine": 62,
            **extra,
        }

    manifest = {
        "schema_version": 1,
        "cpu_proof_path": "provenance/CPU_FULL_PASS.json",
        "status": "H1_PACKAGE_PREPARED_NO_GPU",
        "source_manifest_sha256": _sha(root / "SOURCE_SHA256SUMS"),
        "cuda_executed": False,
        "h1_pending": True,
        "entrypoint": "run_h1.sh",
        "required_parameters": ["VF_GPU_PYTHON", "VF_GPU_UUID"],
        "requirements": {
            "host_arch": "x86_64",
            "elf_machine": 62,
            "gpu_sm": 70,
            "torch_cuda_major": 12,
            "torch_arch_required": ["sm_70", "compute_70"],
            "cuda_runtime_soname": "libcudart.so.12",
        },
        "artifacts": [
            artifact("cpu_binary", "bin/vf"),
            artifact(
                "sm70_library",
                "lib/libvf_qpn2_moe_sm70.so",
                device_arch="sm_70",
                cuda_toolkit_version="12.8.93",
            ),
            artifact("cuda_runtime", "lib/libcudart.so.12"),
        ],
    }
    (root / "PACKAGE.json").write_text(json.dumps(_apply(package, manifest)))
    lines = [
        f"{_sha(p)}  {p.relative_to(root).as_posix()}"
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    (root / "SHA256SUMS").write_text("".join(f"{x}\n" for x in _apply(sums, lines)))


class Bundle(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.tools = FakeTools()
        # These fixtures declare x86_64 ELF metadata and exercise the rest of
        # the x86 bundle contract. Simulate that target host on ARM test runners;
        # HostTests separately verifies that a real non-x86 host is blocked.
        self._host_arch = mock.patch.object(
            h1_preflight.platform, "machine", return_value="x86_64"
        )
        self._host_arch.start()
        self.addCleanup(self._host_arch.stop)

    def refused(self, pattern: str, status: str = "FAILED", **hooks) -> None:
        build(self.root, **hooks)
        self.refused_built(pattern, status)

    def refused_built(self, pattern: str, status: str = "FAILED", **kwargs) -> None:
        with self.tools.installed():
            with self.assertRaisesRegex(PreflightError, pattern) as caught:
                h1_preflight.preflight(self.root, **kwargs)
        self.assertEqual(caught.exception.status, status)


class FixtureTests(Bundle):
    def test_the_synthetic_bundle_passes_with_canned_tool_outputs(self) -> None:
        build(self.root)
        with self.tools.installed():
            status, checks, details = h1_preflight.preflight(self.root)
        self.assertEqual(status, "STATIC_PREFLIGHT_PASS_NO_GPU")
        self.assertEqual(len(checks), 7)
        self.assertEqual(
            details["elf"]["sm70_library"]["needed"], ["libc.so.6", "libcudart.so.12"]
        )


class SafeRelTests(unittest.TestCase):
    def test_each_unsafe_spelling_is_refused_by_its_own_check(self) -> None:
        cases = [
            # 5 rather than None: `not None` would refuse through the next operand.
            (5, "non-empty POSIX relative path"),
            ("", "non-empty POSIX relative path"),
            ("a\x00b", "non-empty POSIX relative path"),
            ("a\\b", "non-empty POSIX relative path"),
            ("/etc/passwd", "unsafe manifest path"),
        ]
        for value, pattern in cases:
            with self.subTest(value=value):
                with self.assertRaisesRegex(PreflightError, pattern):
                    h1_preflight._safe_rel(value)


class BundleFileTests(Bundle):
    def test_a_directory_is_not_a_bundle_file(self) -> None:
        build(self.root)
        with self.assertRaisesRegex(PreflightError, "not a regular file: bin"):
            h1_preflight._bundle_file(self.root, "bin")

    def test_a_symlinked_directory_inside_the_bundle_is_refused(self) -> None:
        # The link resolves inside the bundle, so only the per-part walk sees it.
        build(self.root)
        (self.root / "alias").symlink_to(self.root / "bin")
        with self.assertRaisesRegex(PreflightError, "symlink is forbidden.*alias/vf"):
            h1_preflight._bundle_file(self.root, "alias/vf")


class TreeBoundTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)

    def walk(self, pattern: str, **limits: int) -> None:
        with contextlib.ExitStack() as stack:
            for name, value in limits.items():
                stack.enter_context(mock.patch.object(h1_preflight, name, value))
            with self.assertRaisesRegex(PreflightError, pattern):
                h1_preflight._check_no_symlinks_and_bounds(self.root)

    def test_too_many_directories_are_refused_before_their_contents_are_read(self):
        # A symlinked directory follows: without the count check the walk would
        # refuse it as a symlink instead.
        for i in range(20):
            (self.root / f"d{i:02}").mkdir()
        (self.root / "zz").symlink_to(self.root / "d00")
        self.walk("too many filesystem entries", MAX_ENTRIES=0)

    def test_too_many_files_are_refused(self) -> None:
        for i in range(20):
            (self.root / f"f{i:02}").write_text("x")
        self.walk("too many filesystem entries", MAX_ENTRIES=0)

    def test_a_symlinked_directory_is_refused(self) -> None:
        (self.root / "real").mkdir()
        (self.root / "link").symlink_to(self.root / "real")
        self.walk("symlink directory is forbidden: link")

    def test_a_file_over_the_per_file_cap_is_refused(self) -> None:
        (self.root / "big").write_bytes(b"x" * 11)
        self.walk("bundle file exceeds 10 byte cap: big", MAX_FILE_BYTES=10)

    def test_files_over_the_total_cap_are_refused(self) -> None:
        (self.root / "a").write_bytes(b"x" * 6)
        (self.root / "b").write_bytes(b"x" * 6)
        self.walk("exceeds total 1 GiB payload cap", MAX_TOTAL_BYTES=10)


class ManifestParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.path = self.root / "SUMS"

    def parse(self, text: str, pattern: str, **kwargs) -> None:
        self.path.write_text(text)
        with self.assertRaisesRegex(PreflightError, pattern):
            h1_preflight._parse_sha256sums(self.path, self.root, **kwargs)

    def test_an_oversized_but_well_formed_manifest_is_refused(self) -> None:
        text = f"{'a' * 64}  one\n{'b' * 64}  two\n"
        with mock.patch.object(h1_preflight, "MAX_MANIFEST", len(text) - 1):
            self.parse(text, "SUMS exceeds 4 MiB")

    def test_an_empty_manifest_is_refused(self) -> None:
        self.parse("", "SUMS is empty")

    def test_a_malformed_line_is_refused(self) -> None:
        self.parse(f"{'a' * 64} one-space\n", "malformed SUMS line 1")

    def test_more_entries_than_allowed_are_refused(self) -> None:
        self.parse(
            f"{'a' * 64}  one\n{'b' * 64}  two\n",
            "contains more than 1 entries",
            max_entries=1,
        )

    def test_json_documents_over_their_cap_or_not_objects_are_refused(self) -> None:
        self.path.write_text('{"a": 1}')
        with self.assertRaisesRegex(PreflightError, "doc exceeds 4 byte cap"):
            h1_preflight._read_json(self.path, "doc", cap=4)
        self.path.write_text("[]")
        with self.assertRaisesRegex(PreflightError, "doc must be a JSON object"):
            h1_preflight._read_json(self.path, "doc")


class HostTests(Bundle):
    def test_a_bundle_path_that_is_a_file_is_refused(self) -> None:
        build(self.root)
        with self.assertRaisesRegex(PreflightError, "bundle path is not a directory"):
            h1_preflight.preflight(self.root / "run_h1.sh")

    def test_a_non_linux_host_is_blocked(self) -> None:
        build(self.root)
        with mock.patch("platform.system", return_value="Darwin"):
            self.refused_built("unsupported host OS: Darwin", "BLOCKED")

    def test_a_non_x86_64_host_is_blocked(self) -> None:
        build(self.root)
        with mock.patch("platform.machine", return_value="aarch64"):
            self.refused_built("host architecture aarch64 rejected", "BLOCKED")

    def test_an_unreadable_host_glibc_version_is_blocked(self) -> None:
        with mock.patch("os.confstr", return_value="musl libc 1.2.5"):
            with self.assertRaisesRegex(PreflightError, "host glibc version") as ctx:
                h1_preflight._glibc_version()
        self.assertEqual(ctx.exception.status, "BLOCKED")

    def test_a_missing_readelf_is_blocked(self) -> None:
        build(self.root)
        self.tools.readelf = None
        self.refused_built("readelf is required", "BLOCKED")

    def test_a_routing_binary_other_than_the_verified_one_is_refused(self) -> None:
        build(self.root)
        self.refused_built(
            "--routing-binary must name", routing_binary="lib/libcudart.so.12"
        )

    def test_a_routing_binary_without_execute_permission_is_refused(self) -> None:
        build(self.root)
        (self.root / "bin/vf").chmod(0o644)
        self.refused_built("routing binary is not executable")


class CoverageTests(Bundle):
    def test_a_required_file_left_out_of_sha256sums_is_named(self) -> None:
        # The coverage comparison would refuse it too, but without naming it.
        self.refused(
            "does not cover required file run_h1.sh",
            sums=lambda lines: [x for x in lines if not x.endswith("  run_h1.sh")],
        )


class PackageTests(Bundle):
    def package(self, pattern: str, change) -> None:
        self.refused(pattern, package=change)

    @staticmethod
    def artifact(which: str, **fields):
        def change(p):
            entry = next(a for a in p["artifacts"] if a["kind"] == which)
            entry.update(fields)

        return change

    def test_a_top_level_field_with_the_wrong_value_is_refused(self) -> None:
        self.package(
            "PACKAGE.json has invalid status",
            lambda p: p.update(status="H1_PASSED"),
        )

    def test_requirements_must_be_an_object(self) -> None:
        self.package(
            "requirements must be an object", lambda p: p.update(requirements=[])
        )

    def test_a_requirement_with_the_wrong_value_is_refused(self) -> None:
        self.package(
            r"requirements\.gpu_sm must equal 70",
            lambda p: p["requirements"].update(gpu_sm=75),
        )

    def test_artifacts_must_be_a_list(self) -> None:
        # Three keys: only the type is wrong, not the count.
        self.package(
            "exactly the three required artifacts",
            lambda p: p.update(artifacts={a["kind"]: a for a in p["artifacts"]}),
        )

    def test_artifacts_must_number_exactly_three(self) -> None:
        self.package(
            "exactly the three required artifacts",
            lambda p: p.update(artifacts=p["artifacts"][:2]),
        )

    def test_an_artifact_entry_must_be_an_object(self) -> None:
        self.package(
            "artifact entry must be an object",
            lambda p: p.update(artifacts=[1, *p["artifacts"][1:]]),
        )

    def test_an_unknown_artifact_kind_is_refused(self) -> None:
        self.package(
            "unknown or duplicate artifact kind: 'gpu'",
            self.artifact("cuda_runtime", kind="gpu"),
        )

    def test_a_duplicate_artifact_kind_is_refused(self) -> None:
        self.package(
            "unknown or duplicate artifact kind: 'cpu_binary'",
            lambda p: p.update(artifacts=[p["artifacts"][0]] * 2 + p["artifacts"][1:2]),
        )

    def test_an_artifact_at_another_kinds_path_is_refused(self) -> None:
        self.package(
            "cpu_binary path must be bin/vf",
            self.artifact("cpu_binary", path="lib/libcudart.so.12"),
        )

    def test_an_artifact_with_another_elf_machine_is_refused(self) -> None:
        self.package(
            "cuda_runtime manifest ELF machine must be 62",
            self.artifact("cuda_runtime", elf_machine=183),
        )

    def test_artifact_digests_must_be_well_formed_and_agree(self) -> None:
        for digest, pattern in [
            (5, "cpu_binary has invalid SHA-256"),
            ("XYZ", "cpu_binary has invalid SHA-256"),
            ("0" * 64, "cpu_binary PACKAGE.json hash disagrees"),
        ]:
            with self.subTest(digest=digest):
                self.tearDown_bundle()
                self.package(pattern, self.artifact("cpu_binary", sha256=digest))

    def test_an_artifact_byte_count_must_match_the_file(self) -> None:
        self.package(
            "cpu_binary manifest byte count mismatch",
            lambda p: p["artifacts"][0].update(bytes=p["artifacts"][0]["bytes"] + 1),
        )

    def test_the_sm70_library_must_name_its_architecture_and_toolkit(self) -> None:
        for fields, pattern in [
            ({"device_arch": "sm_75"}, "does not identify device_arch sm_70"),
            ({"cuda_toolkit_version": 12}, "lacks CUDA toolkit provenance"),
            ({"cuda_toolkit_version": ""}, "lacks CUDA toolkit provenance"),
        ]:
            with self.subTest(fields=fields):
                self.tearDown_bundle()
                self.package(pattern, self.artifact("sm70_library", **fields))

    def test_the_source_manifest_digest_must_be_well_formed_and_agree(self) -> None:
        for digest, pattern in [
            (5, "source_manifest_sha256 is invalid"),
            ("abc", "source_manifest_sha256 is invalid"),
            # A well-formed but wrong digest is caught against the file itself.
            ("0" * 64, "SOURCE_SHA256SUMS digest disagrees with PACKAGE.json"),
        ]:
            with self.subTest(digest=digest):
                self.tearDown_bundle()
                self.package(
                    pattern, lambda p, d=digest: p.update(source_manifest_sha256=d)
                )

    def tearDown_bundle(self) -> None:
        shutil.rmtree(self.root)
        self.root.mkdir()


class SourceManifestTests(Bundle):
    def test_a_source_entry_missing_from_sha256sums_is_refused(self) -> None:
        self.refused(
            "missing or inconsistent: extra.txt",
            source_sums=lambda text: text + f"{'1' * 64}  extra.txt\n",
        )

    def test_a_source_entry_with_another_digest_is_refused(self) -> None:
        def change(text: str) -> str:
            digest = hashlib.sha256(b"[package]\nname='fixture'\n").hexdigest()
            return text.replace(f"{digest}  Cargo.toml", f"{'1' * 64}  Cargo.toml")

        self.refused("missing or inconsistent: Cargo.toml", source_sums=change)


class ProvenanceTests(Bundle):
    def test_the_cpu_proof_must_be_at_its_fixed_path(self) -> None:
        self.refused(
            "CPU qualification provenance is required",
            package=lambda p: p.update(cpu_proof_path=None),
        )

    def test_an_oversized_receipt_is_refused(self) -> None:
        # Only the receipt crosses the patched bound; the manifests stay below.
        with mock.patch.object(h1_preflight, "MAX_MANIFEST", 10000):
            self.refused(
                "provenance receipt exceeds manifest bound",
                cpu=lambda c: c.update(padding="x" * 20000),
            )

    def test_a_receipt_must_be_an_object(self) -> None:
        self.refused("provenance receipt must be an object", cpu=lambda c: [])

    def test_each_cpu_pass_condition_is_required_on_its_own(self) -> None:
        for field, value in [
            ("status", "CPU_PARTIAL"),
            ("sources_unchanged", False),
            ("cuda_executed", True),
            ("steps", []),
        ]:
            with self.subTest(field=field):
                PackageTests.tearDown_bundle(self)
                self.refused(
                    "not a complete unchanged-source pass",
                    cpu=lambda c, f=field, v=value: c.update({f: v}),
                )

    def test_the_qualified_binary_size_must_match(self) -> None:
        self.refused(
            "CPU binary differs from qualified revision",
            cpu=lambda c: c.update(binary_bytes=c["binary_bytes"] + 1),
        )

    def test_the_cpu_proof_must_name_its_source_revision(self) -> None:
        for value in (["Cargo.toml"], {}):
            with self.subTest(value=value):
                PackageTests.tearDown_bundle(self)
                self.refused(
                    "CPU qualification lacks source revision",
                    cpu=lambda c, v=value: c.update(source_sha256=v),
                )

    def test_each_cuda_compilation_condition_is_required_on_its_own(self) -> None:
        for field, value in [
            ("status", "CUDA_FAILED"),
            ("device_executed", True),
            ("architecture", "sm_75"),
        ]:
            with self.subTest(field=field):
                PackageTests.tearDown_bundle(self)
                self.refused(
                    "expected CUDA_COMPILED provenance without device execution",
                    cuda=lambda c, f=field, v=value: c.update({f: v}),
                )

    def test_the_compiled_library_must_match_the_receipt(self) -> None:
        self.refused(
            "SM70 library differs from compilation receipt",
            cuda=lambda c: c["hashes"].update(
                {"build/libvf_qpn2_moe_sm70.so": "0" * 64}
            ),
        )


class ElfTests(Bundle):
    def elf(self, pattern: str, status: str = "FAILED") -> None:
        build(self.root)
        self.refused_built(pattern, status)

    def test_a_32_bit_elf_is_refused(self) -> None:
        self.tools.header["vf"] = HEADER.replace("ELF64", "ELF32")
        self.elf("bin/vf is not ELF64")

    def test_a_big_endian_elf_is_refused(self) -> None:
        self.tools.header["vf"] = HEADER.replace("little endian", "big endian")
        self.elf("bin/vf is not little-endian ELF")

    def test_glibc_private_symbols_are_refused(self) -> None:
        self.tools.versions["libcudart.so.12"] = _versions("GLIBC_2.3", "GLIBC_PRIVATE")
        self.elf("requires non-portable GLIBC_PRIVATE")

    def test_a_glibc_newer_than_the_host_is_blocked(self) -> None:
        self.tools.versions["vf"] = _versions("GLIBC_2.17", "GLIBC_99.0")
        self.elf("bin/vf requires GLIBC_99.0", "BLOCKED")

    def test_an_unresolvable_needed_library_is_blocked(self) -> None:
        self.tools.dynamic["vf"] = _dynamic("libc.so.6", "libnowhere.so.7")
        self.elf("unresolved DT_NEEDED dependency libnowhere.so.7", "BLOCKED")

    def test_the_sm70_library_must_need_the_cuda_runtime(self) -> None:
        self.tools.dynamic["libvf_qpn2_moe_sm70.so"] = _dynamic("libc.so.6")
        self.elf("does not declare DT_NEEDED libcudart.so.12")

    def test_runpath_entries_outside_system_dirs_or_naming_build_hosts_are_refused(
        self,
    ) -> None:
        for item, pattern in [
            ("/opt/cuda/lib64", "non-system absolute RPATH/RUNPATH: /opt/cuda"),
            # Under a system prefix, so only the build-host check refuses them.
            ("/usr/lib/run/media/pc1/x", "embeds a PC1 absolute library path"),
            ("/usr/lib/home/pc6/x", "embeds a PC1 absolute library path"),
        ]:
            with self.subTest(item=item):
                with self.assertRaisesRegex(PreflightError, pattern):
                    h1_preflight._check_rpaths([f"$ORIGIN:{item}"], "lib/x.so")


class CapabilityTests(Bundle):
    def capabilities(self, pattern: str, change) -> None:
        change(self.tools.capabilities)
        build(self.root)
        self.refused_built(pattern)

    def test_an_unsuccessful_envelope_is_refused(self) -> None:
        self.capabilities("bad success envelope", lambda c: c.update(ok=False))

    def test_a_non_object_result_is_refused(self) -> None:
        self.capabilities("bad success envelope", lambda c: c.update(result=[]))

    def test_each_capability_flag_must_be_false(self) -> None:
        for flag in ("cuda_compiled", "cuda_executed", "pcie_tested", "rdma_tested"):
            with self.subTest(flag=flag):
                self.tools = FakeTools()
                PackageTests.tearDown_bundle(self)
                self.capabilities(
                    f"must report {flag}=false",
                    lambda c, f=flag: c["result"].update({f: True}),
                )

    def test_a_performance_claim_is_refused(self) -> None:
        self.capabilities(
            "performance_claim=NONE",
            lambda c: c["result"].update(performance_claim="FAST"),
        )

    def test_the_cpu_reference_must_be_affirmed(self) -> None:
        self.capabilities(
            "does not affirm CPU reference",
            lambda c: c["result"].update(cpu_reference=False),
        )


class ReceiptTests(unittest.TestCase):
    def test_a_receipt_over_the_output_cap_is_not_written(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "receipt.json"
            with mock.patch.object(h1_preflight, "MAX_TOOL_OUTPUT", 10):
                with self.assertRaisesRegex(PreflightError, "receipt exceeds 1 MiB"):
                    h1_preflight._write_receipt(out, {"status": "FAILED"})
            self.assertFalse(os.path.lexists(out))


if __name__ == "__main__":
    unittest.main()
