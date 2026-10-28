"""Fast, CPU-only rejection tests for scripts/h1_preflight.py."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from scripts import h1_preflight  # noqa: E402


class H1PreflightTests(unittest.TestCase):
    def _bundle(self, root: Path, *, traversal: bool = False) -> None:
        (root / "bin").mkdir(parents=True)
        (root / "lib").mkdir()
        (root / "source").mkdir()
        (root / "run_h1.sh").write_text("#!/bin/sh\nexit 1\n")
        (root / "source" / "Cargo.toml").write_text("[package]\nname='fixture'\n")
        source_sum = hashlib.sha256(
            (root / "source" / "Cargo.toml").read_bytes()
        ).hexdigest()
        (root / "SOURCE_SHA256SUMS").write_text(f"{source_sum}  Cargo.toml\n")
        # Small synthetic provenance only exercises package contracts; it is
        # not represented as an executed qualification of /bin/true.
        cuda_names = (
            "qpn2_moe_ffi.h",
            "qpn2_moe_core.cuh",
            "qpn2_moe_sm70.cu",
            "qpn2_moe_validate.c",
        )
        (root / "source/cuda").mkdir()
        for name in cuda_names:
            (root / "source/cuda" / name).write_text("synthetic fixture")
        (root / "source/scripts").mkdir()
        (root / "source/scripts/deepseek_v41_qpn2_layer_canary.py").write_text(
            "# synthetic packaged canary contract\n"
        )
        (root / "source/scripts/deepseek_v41_qpn2_topk6_canary.py").write_text(
            "# synthetic packaged top-k-6 canary contract\n"
        )
        (root / "SOURCE_SHA256SUMS").write_text(
            "".join(
                f"{self._sha(p)}  {p.relative_to(root / 'source').as_posix()}\n"
                for p in sorted((root / "source").rglob("*"))
                if p.is_file()
            )
        )
        elf = bytearray(Path("/bin/true").read_bytes())
        # Keep a real, inspectable ELF payload while making its machine field
        # match this test's declared x86_64 bundle on ARM qualification hosts.
        if elf[:4] == b"\x7fELF":
            endian = "<" if elf[5] == 1 else ">"
            elf[18:20] = struct.pack(endian + "H", 62)  # EM_X86_64
        for rel in ("bin/vf", "lib/libvf_qpn2_moe_sm70.so", "lib/libcudart.so.12"):
            (root / rel).write_bytes(elf)
        artifacts = [
            {
                "path": "bin/vf",
                "kind": "cpu_binary",
                "sha256": self._sha(root / "bin/vf"),
                "bytes": (root / "bin/vf").stat().st_size,
                "elf_machine": 62,
            },
            {
                "path": "lib/libvf_qpn2_moe_sm70.so",
                "kind": "sm70_library",
                "sha256": self._sha(root / "lib/libvf_qpn2_moe_sm70.so"),
                "bytes": (root / "lib/libvf_qpn2_moe_sm70.so").stat().st_size,
                "elf_machine": 62,
                "device_arch": "sm_70",
                "cuda_toolkit_version": "12.8.93",
            },
            {
                "path": "lib/libcudart.so.12",
                "kind": "cuda_runtime",
                "sha256": self._sha(root / "lib/libcudart.so.12"),
                "bytes": (root / "lib/libcudart.so.12").stat().st_size,
                "elf_machine": 62,
            },
        ]
        (root / "provenance").mkdir()
        cpu_proof = {
            "status": "CPU_FULL_PASS",
            "sources_unchanged": True,
            "cuda_executed": False,
            "steps": [{"exit_code": 0}],
            "binary_sha256": self._sha(root / "bin/vf"),
            "binary_bytes": (root / "bin/vf").stat().st_size,
            "source_sha256": {"Cargo.toml": source_sum},
        }
        (root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(cpu_proof))
        cuda_proof = {
            "status": "CUDA_COMPILED",
            "device_executed": False,
            "architecture": "sm_70",
            "hashes": {
                f"cuda/{name}": self._sha(root / "source/cuda" / name)
                for name in cuda_names
            },
        }
        cuda_proof["hashes"]["build/libvf_qpn2_moe_sm70.so"] = self._sha(
            root / "lib/libvf_qpn2_moe_sm70.so"
        )
        (root / "provenance/CUDA_COMPILED.json").write_text(json.dumps(cuda_proof))
        package = {
            "schema_version": 2,
            "cpu_proof_path": "provenance/CPU_FULL_PASS.json",
            "status": "H1_PACKAGE_PREPARED_NO_GPU",
            "source_manifest_sha256": self._sha(root / "SOURCE_SHA256SUMS"),
            "cuda_executed": False,
            "h1_pending": True,
            "entrypoint": "run_h1.sh",
            "required_parameters": ["VF_GPU_PYTHON", "VF_GPU_UUID"],
            "required_source_files": h1_preflight.REQUIRED_SOURCE_FILES,
            "requirements": {
                "host_arch": "x86_64",
                "elf_machine": 62,
                "gpu_sm": 70,
                "torch_cuda_major": 12,
                "torch_arch_required": ["sm_70", "compute_70"],
                "cuda_runtime_soname": "libcudart.so.12",
            },
            "artifacts": artifacts,
        }
        (root / "PACKAGE.json").write_text(json.dumps(package))
        lines = [
            (p.relative_to(root).as_posix(), self._sha(p))
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
        if traversal:
            lines.append(("../outside", "0" * 64))
        (root / "SHA256SUMS").write_text(
            "".join(f"{digest}  {path}\n" for path, digest in lines)
        )

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _status(self, root: Path) -> str:
        try:
            # Readelf still inspects the real ELF bytes. Never execute the
            # deliberately non-runnable architecture fixture as a CPU worker.
            run_bounded = h1_preflight._run_bounded

            def run(argv, **kwargs):
                if Path(argv[0]) == root / "bin/vf":
                    return b"{}"
                return run_bounded(argv, **kwargs)

            with (
                mock.patch.object(
                    h1_preflight.platform, "machine", return_value="x86_64"
                ),
                mock.patch.object(h1_preflight, "_run_bounded", side_effect=run),
            ):
                h1_preflight.preflight(root)
        except h1_preflight.PreflightError as exc:
            return exc.status + ":" + str(exc)
        self.fail(
            "fixture unexpectedly passed without a validated vf capabilities executable"
        )

    def test_malformed_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root, traversal=True)
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"))
            self.assertIn("unsafe manifest path", outcome)

    def test_package_must_declare_relocatable_deepseek_layer_canary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            package = json.loads((root / "PACKAGE.json").read_text())
            package["required_source_files"] = []
            (root / "PACKAGE.json").write_text(json.dumps(package))
            self._rehash_outer(root, "PACKAGE.json")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            self.assertIn("required_source_files", outcome)

    def test_package_must_declare_topk6_canary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            package = json.loads((root / "PACKAGE.json").read_text())
            package["required_source_files"] = [
                "scripts/deepseek_v41_qpn2_layer_canary.py"
            ]
            (root / "PACKAGE.json").write_text(json.dumps(package))
            self._rehash_outer(root, "PACKAGE.json")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            self.assertIn("required_source_files", outcome)

    def test_path_escape_in_package_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            package = json.loads((root / "PACKAGE.json").read_text())
            package["artifacts"][0]["path"] = "../vf"
            (root / "PACKAGE.json").write_text(json.dumps(package))
            lines = []
            for line in (root / "SHA256SUMS").read_text().splitlines():
                digest, rel = line.split("  ", 1)
                if rel == "PACKAGE.json":
                    digest = self._sha(root / rel)
                lines.append(f"{digest}  {rel}")
            (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"))
            self.assertIn("unsafe manifest path", outcome)

    def test_payload_mutation_after_hashing_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            with (root / "run_h1.sh").open("a") as stream:
                stream.write("# changed\n")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"))
            self.assertIn("SHA-256 mismatch", outcome)

    def test_wrong_elf_architecture_is_rejected_without_running_binary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            cpu = root / "bin/vf"
            contents = bytearray(cpu.read_bytes())
            if contents[:4] != b"\x7fELF":
                self.skipTest("/bin/true is not ELF")
            endian = "<" if contents[5] == 1 else ">"
            contents[18:20] = struct.pack(endian + "H", 183)  # EM_AARCH64
            cpu.write_bytes(contents)
            package = json.loads((root / "PACKAGE.json").read_text())
            package["artifacts"][0]["sha256"] = self._sha(cpu)
            (root / "PACKAGE.json").write_text(json.dumps(package))
            proof_path = root / "provenance/CPU_FULL_PASS.json"
            cpu_proof = json.loads(proof_path.read_text())
            cpu_proof["binary_sha256"] = self._sha(cpu)
            proof_path.write_text(json.dumps(cpu_proof))
            lines = []
            for line in (root / "SHA256SUMS").read_text().splitlines():
                digest, rel = line.split("  ", 1)
                path = root / rel
                if rel in ("bin/vf", "PACKAGE.json", "provenance/CPU_FULL_PASS.json"):
                    digest = self._sha(path)
                lines.append(f"{digest}  {rel}")
            (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"))
            self.assertIn("machine is not x86-64", outcome)

    def _rehash_outer(self, root: Path, *changed: str) -> None:
        lines = []
        for line in (root / "SHA256SUMS").read_text().splitlines():
            digest, rel = line.split("  ", 1)
            if rel in changed:
                digest = self._sha(root / rel)
            lines.append(f"{digest}  {rel}")
        (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")

    def test_symlink_in_a_bundle_path_is_rejected(self) -> None:
        # Without this a bundle entry could be a link, and preflight would hash
        # whatever it points at while the recipient believes it is bundle content.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            target = root / "elsewhere.bin"
            target.write_bytes((root / "bin/vf").read_bytes())
            (root / "bin/vf").unlink()
            (root / "bin/vf").symlink_to(target)
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            # The tree walk reports it first; either guard must refuse it.
            self.assertIn("symlink", outcome)

    def test_duplicate_manifest_path_is_rejected(self) -> None:
        # A repeated path would be keyed once, so one of the two digests would
        # silently never be checked.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            text = (root / "SHA256SUMS").read_text()
            first = text.splitlines()[0]
            digest, rel = first.split("  ", 1)
            (root / "SHA256SUMS").write_text(text + f"{'0' * 64}  {rel}\n")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            self.assertIn("duplicate", outcome)

    def test_non_canonical_manifest_path_is_rejected(self) -> None:
        # ./bin/vf names the same file by another spelling; accepting it would let
        # one entry be verified twice and another not at all.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            lines = []
            for line in (root / "SHA256SUMS").read_text().splitlines():
                digest, rel = line.split("  ", 1)
                lines.append(f"{digest}  ./{rel}" if rel == "run_h1.sh" else line)
            (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            self.assertIn("canonical POSIX form", outcome)

    def test_non_regular_bundle_file_is_rejected(self) -> None:
        # A FIFO where a payload file belongs must not be read as content.
        import os

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self._bundle(root)
            (root / "run_h1.sh").unlink()
            try:
                os.mkfifo(root / "run_h1.sh")
            except (AttributeError, OSError):
                self.skipTest("this platform cannot create a FIFO")
            outcome = self._status(root)
            self.assertTrue(outcome.startswith("FAILED:"), outcome)
            # The tree walk reports it first; either guard must refuse it.
            self.assertIn("regular", outcome)


if __name__ == "__main__":
    unittest.main()
