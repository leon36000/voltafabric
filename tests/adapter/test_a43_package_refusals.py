"""CPU-only tests pinning each individual refusal of scripts/package_h1.py.

Every case starts from a fixture that packages successfully and changes exactly
one thing, so that exactly one operand of one guard is true. Each assertion
names the exception type and the guard's own message, because several later
checks refuse with a similar type and would otherwise mask a missing guard.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock
from tests.adapter.package_fixture_utils import elf_metadata_fixture, provider_sources


WORKTREE_ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = Path(os.environ.get("VF_H1_SOURCE_SCRIPTS", WORKTREE_ROOT / "scripts"))
sys.path.insert(0, str(MODULE_DIR))

import package_h1  # noqa: E402


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class PackageRefusalTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve()
        self.args = self._fixture(self.root)

    # -- fixture -----------------------------------------------------------

    def _fixture(self, root: Path) -> SimpleNamespace:
        source = root / "source"
        for directory in package_h1.COPY_DIRS:
            (source / directory).mkdir(parents=True, exist_ok=True)
        (source / "src/core.rs").write_text("qualified core source\n")
        for name in package_h1.COPY_FILES:
            path = source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n" if name.endswith(".json") else "fixture\n")
        (source / "fixtures/h1_reproduction_files.json").write_text("[]\n")
        canary = source / "scripts/deepseek_v41_qpn2_layer_canary.py"
        canary.parent.mkdir(parents=True, exist_ok=True)
        canary.write_text("# qualified relocatable DeepSeek canary fixture\n")
        topk6_canary = source / "scripts/deepseek_v41_qpn2_topk6_canary.py"
        topk6_canary.write_text("# qualified relocatable top-k-6 canary fixture\n")
        provider_hashes = provider_sources(source)
        (source / "handoff/h1_bundle_run.sh").write_text("#!/bin/sh\nexit 0\n")
        for name in package_h1.CUDA_FILES:
            (source / name).write_text(f"{name}\n")
        (source / "LICENSE-project").write_text("fixture license\n")
        (root / "outside.txt").write_text("not part of the source tree\n")

        args = SimpleNamespace(
            source=source,
            out=root / "bundle",
            binary=root / "vf",
            library=root / "libvf.so",
            cudart=root / "libcudart.so.12",
            cuda_license=root / "LICENSE",
            cuda_proof=root / "CUDA_COMPILED.json",
            cpu_proof=root / "CPU_FULL_PASS.json",
        )
        args.binary.write_bytes(elf_metadata_fixture(b"binary fixture\n"))
        args.library.write_bytes(elf_metadata_fixture(b"SM70 library fixture\n"))
        args.cudart.write_bytes(elf_metadata_fixture(b"runtime fixture\n"))
        args.cuda_license.write_bytes(b"runtime license fixture\n")
        self.cpu = {
            "status": "CPU_FULL_PASS",
            "sources_unchanged": True,
            "cuda_executed": False,
            "steps": [{"step": "fixture", "exit_code": 0}],
            "source_sha256": {
                **provider_hashes,
                "src/core.rs": package_h1.digest(source / "src/core.rs"),
                "scripts/deepseek_v41_qpn2_layer_canary.py": package_h1.digest(canary),
                "scripts/deepseek_v41_qpn2_topk6_canary.py": package_h1.digest(
                    topk6_canary
                ),
            },
            "binary_sha256": package_h1.digest(args.binary),
            "binary_bytes": args.binary.stat().st_size,
        }
        hashes = {
            name: package_h1.digest(source / name) for name in package_h1.CUDA_FILES
        }
        hashes["build/libvf_qpn2_moe_sm70.so"] = package_h1.digest(args.library)
        self.cuda = {
            "status": "CUDA_COMPILED",
            "device_executed": False,
            "architecture": "sm_70",
            "hashes": hashes,
        }
        self._write_proofs(args)
        return args

    def _write_proofs(self, args):
        args.cpu_proof.write_text(json.dumps(self.cpu))
        args.cuda_proof.write_text(json.dumps(self.cuda))

    # -- helpers -----------------------------------------------------------

    def _refuse(self, message, expect=ValueError, *, bundle_created=False):
        """Package must refuse with exactly this exception type and message."""
        with self.assertRaisesRegex(expect, re.escape(message)):
            package_h1.package(self.args)
        if not bundle_created:
            self.assertFalse(self.args.out.exists(), "refusal must precede the bundle")

    def _refuse_during_copy(self, hook, message):
        """Run package with a hook around each copy() to simulate a concurrent
        change between the up-front checks and the copy (a TOCTOU window).

        Since A46 the bundle is built in a private stage beside the output, so
        a refusal here must leave neither the output nor the stage behind."""
        real_copy = package_h1.copy
        out = self.args.out.resolve()
        stage = out.parent / f".{out.name}.partial"

        def copy(source, dest):
            rel = dest.relative_to(stage).as_posix()
            hook("before", rel, dest)
            real_copy(source, dest)
            hook("after", rel, dest)

        with mock.patch.object(package_h1, "copy", copy):
            self._refuse(message)
        self.assertFalse(stage.exists(), "a refusal must remove the stage")

    def _cpu(self, **changes):
        self.cpu.update(changes)
        self._write_proofs(self.args)

    def _cuda(self, **changes):
        self.cuda.update(changes)
        self._write_proofs(self.args)

    # -- control -----------------------------------------------------------

    def test_fixture_packages_successfully(self):
        # Every refusal below differs from this passing fixture by one change.
        result = package_h1.package(self.args)
        self.assertEqual(result["status"], "H1_PACKAGE_PREPARED_NO_GPU")
        self.assertTrue((self.args.out / "PACKAGE.json").is_file())

    # -- copy() ------------------------------------------------------------

    def test_copy_refuses_symlink_to_regular_file(self):
        # A symlink to a regular file passes is_file(); only is_symlink() refuses.
        link = self.root / "link.txt"
        link.symlink_to(self.root / "outside.txt")
        dest = self.root / "copied/link.txt"
        with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
            package_h1.copy(link, dest)
        self.assertFalse(dest.exists())

    def test_copy_refuses_missing_source(self):
        dest = self.root / "copied/missing.txt"
        with self.assertRaisesRegex(ValueError, "regular non-symlink file"):
            package_h1.copy(self.root / "missing.txt", dest)
        self.assertFalse(dest.exists())

    # -- source_snapshot() -------------------------------------------------

    def test_snapshot_refuses_symlinked_source_file(self):
        (self.args.source / "src/link.rs").symlink_to(self.root / "outside.txt")
        with self.assertRaisesRegex(
            ValueError, re.escape("source symlink not packageable: src/link.rs")
        ):
            package_h1.source_snapshot(self.args.source)

    def test_package_refuses_symlink_before_creating_bundle(self):
        # The copy loop repeats this refusal with the same message, but only
        # after creating the bundle; the snapshot must refuse first.
        (self.args.source / "src/link.rs").symlink_to(self.root / "outside.txt")
        self._refuse("source symlink not packageable: src/link.rs")

    # -- bundle location ---------------------------------------------------

    def test_bundle_equal_to_source_is_refused(self):
        self.args.out = self.args.source
        self._refuse(
            "bundle must be separate from canonical source", bundle_created=True
        )

    def test_bundle_containing_source_is_refused(self):
        # Without this check the existing parent would hit FileExistsError instead.
        self.args.out = self.root
        self._refuse(
            "bundle must be separate from canonical source", bundle_created=True
        )

    def test_bundle_inside_source_is_refused(self):
        self.args.out = self.args.source / "bundle"
        self._refuse("bundle must be separate from canonical source")

    def test_existing_bundle_is_refused_before_reading_receipts(self):
        # out.mkdir(exist_ok=False) later raises FileExistsError too; only the
        # message identifies the explicit up-front refusal.
        self.args.out.mkdir()
        self._refuse(
            "inspect existing bundle before any new packaging",
            FileExistsError,
            bundle_created=True,
        )
        self.assertEqual(list(self.args.out.iterdir()), [])

    # -- CPU qualification receipt ------------------------------------------

    CPU_RECEIPT = "a fresh unchanged-source CPU_FULL_PASS receipt is required"

    def test_cpu_receipt_with_other_status_is_refused(self):
        self._cpu(status="CPU_PARTIAL_PASS")
        self._refuse(self.CPU_RECEIPT)

    def test_cpu_receipt_with_changed_sources_is_refused(self):
        self._cpu(sources_unchanged=False)
        self._refuse(self.CPU_RECEIPT)

    def test_cpu_receipt_that_executed_cuda_is_refused(self):
        self._cpu(cuda_executed=True)
        self._refuse(self.CPU_RECEIPT)

    def test_cpu_receipt_without_steps_is_refused(self):
        self._cpu(steps=[])
        self._refuse(self.CPU_RECEIPT)

    def test_cpu_receipt_with_failed_step_is_refused(self):
        self._cpu(steps=[{"step": "fixture", "exit_code": 0}, {"exit_code": 101}])
        self._refuse(self.CPU_RECEIPT)

    def test_binary_digest_mismatch_is_refused(self):
        self._cpu(binary_sha256="0" * 64)
        self._refuse("CPU binary differs from qualification receipt")

    def test_binary_size_mismatch_is_refused(self):
        # Digest still matches; only the recorded size is wrong.
        self._cpu(binary_bytes=self.cpu["binary_bytes"] + 1)
        self._refuse("CPU binary differs from qualification receipt")

    def test_source_hashes_that_are_not_a_mapping_are_refused(self):
        self._cpu(source_sha256=["src/core.rs"])
        self._refuse("CPU qualification must identify its sources")

    def test_empty_source_hashes_are_refused(self):
        self._cpu(source_sha256={})
        self._refuse("CPU qualification must identify its sources")

    def test_absolute_source_path_is_refused(self):
        # The absolute path names the real file with its real digest, so only
        # is_absolute() refuses it (source / absolute == absolute).
        target = self.args.source / "src/core.rs"
        self._cpu(source_sha256={str(target): package_h1.digest(target)})
        self._refuse("unsafe CPU proof source path")

    def test_dotdot_source_path_is_refused(self):
        # Resolves back inside the source tree; only the ".." part refuses it.
        target = self.args.source / "src/core.rs"
        self._cpu(source_sha256={"src/../src/core.rs": package_h1.digest(target)})
        self._refuse("unsafe CPU proof source path")

    def test_source_path_escaping_through_symlink_is_refused(self):
        # Relative, no "..", but a root-level symlink resolves outside source.
        (self.args.source / "escape.txt").symlink_to(self.root / "outside.txt")
        digest = package_h1.digest(self.root / "outside.txt")
        self._cpu(source_sha256={"escape.txt": digest})
        self._refuse("unsafe CPU proof source path")

    def test_source_drift_from_cpu_receipt_is_refused(self):
        (self.args.source / "src/core.rs").write_text("drifted\n")
        self._refuse("source differs from CPU qualification receipt: src/core.rs")

    # -- CUDA compilation receipt -------------------------------------------

    CUDA_RECEIPT = "expected the existing unexecuted SM70 compilation receipt"

    def test_cuda_receipt_with_other_status_is_refused(self):
        self._cuda(status="CUDA_FAILED")
        self._refuse(self.CUDA_RECEIPT)

    def test_cuda_receipt_that_executed_on_device_is_refused(self):
        self._cuda(device_executed=True)
        self._refuse(self.CUDA_RECEIPT)

    def test_cuda_receipt_for_other_architecture_is_refused(self):
        self._cuda(architecture="sm_80")
        self._refuse(self.CUDA_RECEIPT)

    def test_cuda_source_drift_from_receipt_is_refused(self):
        name = package_h1.CUDA_FILES[0]
        self.cuda["hashes"][name] = "0" * 64
        self._write_proofs(self.args)
        self._refuse(f"CUDA source differs from compiled receipt: {name}")

    def test_library_drift_from_receipt_is_refused(self):
        self.args.library.write_bytes(b"rebuilt SM70 library\n")
        self._refuse("SM70 library differs from compilation receipt")

    def test_cuda_license_with_other_name_is_refused(self):
        renamed = self.args.cuda_license.with_name("NOTICE")
        self.args.cuda_license.rename(renamed)
        self.args.cuda_license = renamed
        self._refuse("provide the unchanged CUDA runtime component license")

    def test_missing_cuda_license_is_refused(self):
        # copy() would later refuse the missing file with another ValueError,
        # after creating the bundle; the message and absent bundle pin this one.
        self.args.cuda_license.unlink()
        self._refuse("provide the unchanged CUDA runtime component license")

    # -- changes during copying (simulated concurrent writers) --------------

    def test_binary_swapped_before_copy_is_refused(self):
        def hook(phase, rel, dest):
            if phase == "before" and rel == "bin/vf":
                self.args.binary.write_bytes(b"swapped binary\n")

        self._refuse_during_copy(hook, "CPU binary changed while packaging")

    def test_qualified_source_corrupted_in_bundle_is_refused(self):
        # The snapshot check after it would also fail, with another message.
        def hook(phase, rel, dest):
            if phase == "after" and rel == "source/src/core.rs":
                dest.write_text("corrupted copy\n")

        self._refuse_during_copy(
            hook, "copied source differs from CPU qualification: src/core.rs"
        )

    def test_unqualified_source_corrupted_in_bundle_is_refused(self):
        # README.md is in the snapshot but not in the CPU receipt.
        def hook(phase, rel, dest):
            if phase == "after" and rel == "source/README.md":
                dest.write_text("corrupted copy\n")

        self._refuse_during_copy(
            hook, "source snapshot changed while copying: README.md"
        )

    def test_file_added_during_copy_is_refused(self):
        # docs/ is walked after src/, so the new file is copied but was not in
        # the snapshot; the final re-snapshot would refuse with another message.
        def hook(phase, rel, dest):
            if phase == "after" and rel == "source/src/core.rs":
                (self.args.source / "docs/late.md").write_text("late\n")

        self._refuse_during_copy(
            hook, "source packageable inventory changed while copying"
        )

    def test_source_changed_after_copy_is_refused(self):
        # Copies match the snapshot; only the final re-snapshot sees the change.
        def hook(phase, rel, dest):
            if phase == "after" and rel == "provenance/CPU_FULL_PASS.json":
                (self.args.source / "README.md").write_text("edited late\n")

        self._refuse_during_copy(
            hook, "canonical packageable snapshot changed during packaging"
        )

    def test_symlink_added_during_copy_is_refused(self):
        # Without the copy loop's own check, copy() refuses with another message.
        def hook(phase, rel, dest):
            if phase == "after" and rel == "source/src/core.rs":
                link = self.args.source / "docs/late-link.md"
                link.symlink_to(self.root / "outside.txt")

        self._refuse_during_copy(
            hook, "source symlink not packageable: docs/late-link.md"
        )


if __name__ == "__main__":
    unittest.main()
