"""Independent CPU-only tests for H1 source/binary provenance binding."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from tests.adapter.package_fixture_utils import elf_metadata_fixture, provider_sources


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(os.environ.get("VF_H1_SOURCE_SCRIPTS", WORKTREE_ROOT / "scripts"))
sys.path.insert(0, str(MODULE_DIR))

import h1_preflight  # noqa: E402
import package_h1  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


CUDA_NAMES = (
    "cuda/qpn2_moe_ffi.h",
    "cuda/qpn2_moe_core.cuh",
    "cuda/qpn2_moe_sm70.cu",
    "cuda/qpn2_moe_validate.c",
)


class ProvenanceValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "bin").mkdir()
        (self.root / "lib").mkdir()
        (self.root / "source/src").mkdir(parents=True)
        (self.root / "source/cuda").mkdir()
        (self.root / "provenance").mkdir()
        self.binary = b"cpu binary receipt target\n"
        self.library = b"compiled SM70 library receipt target\n"
        (self.root / "bin/vf").write_bytes(self.binary)
        (self.root / "lib/libvf_qpn2_moe_sm70.so").write_bytes(self.library)
        self.source_bytes = b"qualified source revision\n"
        (self.root / "source/src/core.rs").write_bytes(self.source_bytes)
        self.cuda_bytes = {}
        for index, name in enumerate(CUDA_NAMES):
            data = f"cuda source {index}\n".encode()
            self.cuda_bytes[name] = data
            (self.root / "source" / name).write_bytes(data)

        self.cpu = {
            "status": "CPU_FULL_PASS",
            "sources_unchanged": True,
            "cuda_executed": False,
            "steps": [{"step": "cpu gate", "exit_code": 0}],
            "source_sha256": {"src/core.rs": sha(self.source_bytes)},
            "binary_sha256": sha(self.binary),
            "binary_bytes": len(self.binary),
        }
        self.cuda = {
            "status": "CUDA_COMPILED",
            "device_executed": False,
            "architecture": "sm_70",
            "hashes": {name: sha(data) for name, data in self.cuda_bytes.items()},
        }
        self.cuda["hashes"]["build/libvf_qpn2_moe_sm70.so"] = sha(self.library)
        (self.root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(self.cpu))
        (self.root / "provenance/CUDA_COMPILED.json").write_text(json.dumps(self.cuda))
        self.package = {"cpu_proof_path": "provenance/CPU_FULL_PASS.json"}
        self.hashes = {
            "bin/vf": sha(self.binary),
            "lib/libvf_qpn2_moe_sm70.so": sha(self.library),
            "source/src/core.rs": sha(self.source_bytes),
            **{"source/" + name: sha(data) for name, data in self.cuda_bytes.items()},
        }

    def tearDown(self):
        self.temp.cleanup()

    def test_matching_cpu_and_cuda_receipts_pass(self):
        h1_preflight._validate_provenance(self.root, self.package, self.hashes)

    def test_stale_cpu_source_revision_is_rejected(self):
        self.cpu["source_sha256"]["src/core.rs"] = "0" * 64
        (self.root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(self.cpu))
        with self.assertRaises(h1_preflight.PreflightError):
            h1_preflight._validate_provenance(self.root, self.package, self.hashes)

    def test_stale_binary_revision_is_rejected(self):
        self.cpu["binary_sha256"] = "0" * 64
        (self.root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(self.cpu))
        with self.assertRaises(h1_preflight.PreflightError):
            h1_preflight._validate_provenance(self.root, self.package, self.hashes)

    def test_incomplete_cpu_qualification_is_rejected(self):
        self.cpu["steps"] = [{"step": "failed", "exit_code": 2}]
        (self.root / "provenance/CPU_FULL_PASS.json").write_text(json.dumps(self.cpu))
        with self.assertRaises(h1_preflight.PreflightError):
            h1_preflight._validate_provenance(self.root, self.package, self.hashes)

    def test_stale_cuda_source_hash_is_rejected(self):
        self.cuda["hashes"][CUDA_NAMES[0]] = "0" * 64
        (self.root / "provenance/CUDA_COMPILED.json").write_text(json.dumps(self.cuda))
        with self.assertRaises(h1_preflight.PreflightError):
            h1_preflight._validate_provenance(self.root, self.package, self.hashes)


class PackageBindingTests(unittest.TestCase):
    def _fixture(self, root: Path):
        source = root / "source"
        for directory in package_h1.COPY_DIRS:
            (source / directory).mkdir(parents=True, exist_ok=True)
        (source / "src/core.rs").write_text("qualified core source\n")
        for name in package_h1.COPY_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if name.endswith(".json") else "fixture\n")
        (source / "fixtures/h1_reproduction_files.json").write_text("[]\n")
        (source / "handoff/h1_bundle_run.sh").write_text("#!/bin/sh\nexit 0\n")
        canary = source / "scripts/deepseek_v41_qpn2_layer_canary.py"
        canary.parent.mkdir(parents=True, exist_ok=True)
        canary.write_text("# qualified relocatable DeepSeek canary fixture\n")
        topk6_canary = source / "scripts/deepseek_v41_qpn2_topk6_canary.py"
        topk6_canary.write_text("# qualified relocatable top-k-6 canary fixture\n")
        provider_hashes = provider_sources(source)
        for name in package_h1.CUDA_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{name}\n")
        (source / "LICENSE-project").write_text("fixture license\n")

        binary = root / "vf"
        library = root / "libvf.so"
        cudart = root / "libcudart.so.12"
        license_path = root / "LICENSE"
        for path, data in (
            (binary, b"binary fixture\n"),
            (library, b"SM70 library fixture\n"),
            (cudart, b"runtime fixture\n"),
            (license_path, b"runtime license fixture\n"),
        ):
            path.write_bytes(
                elf_metadata_fixture(data) if path != license_path else data
            )
        cpu = {
            "status": "CPU_FULL_PASS",
            "sources_unchanged": True,
            "cuda_executed": False,
            "steps": [{"step": "fixture", "exit_code": 0}],
            "source_sha256": {
                **provider_hashes,
                "src/core.rs": sha((source / "src/core.rs").read_bytes()),
                "scripts/deepseek_v41_qpn2_layer_canary.py": sha(canary.read_bytes()),
                "scripts/deepseek_v41_qpn2_topk6_canary.py": sha(
                    topk6_canary.read_bytes()
                ),
            },
            "binary_sha256": sha(binary.read_bytes()),
            "binary_bytes": binary.stat().st_size,
        }
        cpu_path = root / "CPU_FULL_PASS.json"
        cpu_path.write_text(json.dumps(cpu))
        cuda_hashes = {
            name: package_h1.digest(source / name) for name in package_h1.CUDA_FILES
        }
        cuda_hashes["build/libvf_qpn2_moe_sm70.so"] = package_h1.digest(library)
        cuda = {
            "status": "CUDA_COMPILED",
            "device_executed": False,
            "architecture": "sm_70",
            "hashes": cuda_hashes,
        }
        cuda_path = root / "CUDA_COMPILED.json"
        cuda_path.write_text(json.dumps(cuda))
        return SimpleNamespace(
            source=source,
            out=root / "bundle",
            binary=binary,
            library=library,
            cudart=cudart,
            cuda_license=license_path,
            cuda_proof=cuda_path,
            cpu_proof=cpu_path,
        ), cpu_path

    def test_cpu_proof_mismatch_rejects_before_bundle_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, cpu_path = self._fixture(root)
            cpu = json.loads(cpu_path.read_text())
            cpu["binary_sha256"] = "0" * 64
            cpu_path.write_text(json.dumps(cpu))
            with self.assertRaises(ValueError):
                package_h1.package(args)
            self.assertFalse(args.out.exists())

    def _reject(self, root: Path, mutate, expect=ValueError):
        """Each case must refuse before any bundle directory appears."""
        args, cpu_path = self._fixture(root)
        mutate(args, cpu_path)
        with self.assertRaises(expect):
            package_h1.package(args)
        return args

    def test_receipt_admitting_changed_sources_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:

            def mutate(args, cpu_path):
                cpu = json.loads(cpu_path.read_text())
                cpu["sources_unchanged"] = False
                cpu_path.write_text(json.dumps(cpu))

            args = self._reject(Path(temp), mutate)
            self.assertFalse(args.out.exists())

    def test_receipt_containing_a_failed_step_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:

            def mutate(args, cpu_path):
                cpu = json.loads(cpu_path.read_text())
                cpu["steps"] = list(cpu["steps"]) + [
                    {"step": "rust_debug", "exit_code": 101}
                ]
                cpu_path.write_text(json.dumps(cpu))

            args = self._reject(Path(temp), mutate)
            self.assertFalse(args.out.exists())

    def test_source_that_drifted_from_the_receipt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:

            def mutate(args, cpu_path):
                name = sorted(json.loads(cpu_path.read_text())["source_sha256"])[0]
                (args.source / name).write_text("drifted after qualification\n")

            args = self._reject(Path(temp), mutate)
            self.assertFalse(args.out.exists())

    def test_existing_bundle_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:

            def mutate(args, cpu_path):
                args.out.mkdir(parents=True)
                (args.out / "already-delivered").write_text("previous delivery\n")

            args = self._reject(Path(temp), mutate, expect=FileExistsError)
            self.assertTrue((args.out / "already-delivered").is_file())

    def test_cuda_component_license_must_accompany_the_runtime(self):
        with tempfile.TemporaryDirectory() as temp:

            def mutate(args, cpu_path):
                renamed = args.cuda_license.with_name("NOTICE")
                args.cuda_license.rename(renamed)
                args.cuda_license = renamed

            args = self._reject(Path(temp), mutate)
            self.assertFalse(args.out.exists())

    def test_package_carries_agent_examples_and_cpu_proof(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self._fixture(Path(temp))
            package_h1.package(args)
            self.assertTrue((args.out / "source/agents/routes.example.json").is_file())
            self.assertTrue((args.out / "source/agents/queue.example.json").is_file())
            manifest = json.loads((args.out / "PACKAGE.json").read_text())
            self.assertEqual(
                manifest["cpu_proof_path"], "provenance/CPU_FULL_PASS.json"
            )
            self.assertTrue((args.out / manifest["cpu_proof_path"]).is_file())
            source_hashes = package_h1.source_snapshot(args.source)
            self.assertEqual(
                source_hashes["agents/routes.example.json"],
                package_h1.digest(args.out / "source/agents/routes.example.json"),
            )
            self.assertEqual(
                source_hashes["agents/queue.example.json"],
                package_h1.digest(args.out / "source/agents/queue.example.json"),
            )
            canary = "scripts/deepseek_v41_qpn2_layer_canary.py"
            self.assertEqual(
                source_hashes[canary],
                package_h1.digest(args.out / "source" / canary),
            )
            topk6_canary = "scripts/deepseek_v41_qpn2_topk6_canary.py"
            self.assertEqual(
                source_hashes[topk6_canary],
                package_h1.digest(args.out / "source" / topk6_canary),
            )

    def test_package_refuses_when_required_canary_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            args, cpu_path = self._fixture(Path(temp))
            (args.source / "scripts/deepseek_v41_qpn2_layer_canary.py").unlink()
            cpu = json.loads(cpu_path.read_text())
            cpu["source_sha256"].pop("scripts/deepseek_v41_qpn2_layer_canary.py")
            cpu_path.write_text(json.dumps(cpu))
            with self.assertRaisesRegex(ValueError, "missing required canary"):
                package_h1.package(args)
            self.assertFalse(args.out.exists())

    def test_package_refuses_when_topk6_canary_is_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            args, cpu_path = self._fixture(Path(temp))
            topk6 = "scripts/deepseek_v41_qpn2_topk6_canary.py"
            (args.source / topk6).unlink()
            cpu = json.loads(cpu_path.read_text())
            cpu["source_sha256"].pop(topk6)
            cpu_path.write_text(json.dumps(cpu))
            with self.assertRaisesRegex(ValueError, "missing required canary"):
                package_h1.package(args)
            self.assertFalse(args.out.exists())


if __name__ == "__main__":
    unittest.main()
