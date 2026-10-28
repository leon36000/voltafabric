"""Successor provider availability/provenance and pre-copy ELF refusals."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.adapter.package_fixture_utils import PROVIDER_SOURCES
from tests.adapter import test_a16_h1_bundle as binding
import h1_preflight
import package_h1

LEGACY_CANARIES = [
    "scripts/deepseek_v41_qpn2_layer_canary.py",
    "scripts/deepseek_v41_qpn2_topk6_canary.py",
]


def refresh_bundle(root):
    """Keep all hashes consistent after an adversarial structural change."""
    source = root / "source"
    rows = [
        (package_h1.digest(p), p.relative_to(source).as_posix())
        for p in sorted(source.rglob("*"))
        if p.is_file()
    ]
    (root / "SOURCE_SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for digest, name in rows)
    )
    path = root / "PACKAGE.json"
    document = json.loads(path.read_text())
    document["source_manifest_sha256"] = package_h1.digest(root / "SOURCE_SHA256SUMS")
    path.write_text(json.dumps(document))
    hashes = {
        p.relative_to(root).as_posix(): package_h1.digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.name != "SHA256SUMS"
    }
    (root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in hashes.items())
    )
    return hashes


class SuccessorTests(unittest.TestCase):
    def fixture(self, root):
        return binding.PackageBindingTests()._fixture(root)

    def test_reproduction_manifest_refuses_unsafe_or_unbounded_inputs(self):
        for names in (
            {},
            ["../outside.txt"],
            ["/etc/hosts"],
            ["src/core.rs"],
            [True],
            ["evidence/./a.json"],
            ["evidence/a.json"] * 129,
        ):
            with (
                self.subTest(names=str(names)[:100]),
                tempfile.TemporaryDirectory() as temp,
            ):
                args, _ = self.fixture(Path(temp))
                (args.source / "fixtures/h1_reproduction_files.json").write_text(
                    json.dumps(names)
                )
                with self.assertRaises(ValueError):
                    package_h1.package(args)
                self.assertFalse(args.out.exists())

    def test_relocation_inputs_are_copied_and_bound_to_cpu_proof(self):
        manifest = "fixtures/h1_reproduction_files.json"
        names = json.loads((binding.WORKTREE_ROOT / manifest).read_text())
        self.assertEqual(len(names), 55)
        self.assertIn(
            "evidence/continuation_a97_model_metadata/upstream_314113f8/LICENSE",
            names,
        )
        with tempfile.TemporaryDirectory() as temp:
            args, proof = self.fixture(Path(temp))
            (args.source / manifest).write_text(json.dumps(names))
            cpu = json.loads(proof.read_text())
            for name in names:
                path = args.source / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("relocation input: " + name)
                cpu["source_sha256"][name] = package_h1.digest(path)
            proof.write_text(json.dumps(cpu))
            try:
                package_h1.package(args)
            except ValueError as error:
                self.fail(f"qualified relocation inputs omitted: {error}")
            for name in names:
                self.assertEqual(
                    (args.out / "source" / name).read_bytes(),
                    (args.source / name).read_bytes(),
                )

    def test_qualified_gate_reproducer_is_packaged_without_other_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            args, proof = self.fixture(Path(temp))
            name = "evidence/continuation_a105_all_backbone_headers/fetch_headers.py"
            helper = args.source / name
            helper.parent.mkdir(parents=True, exist_ok=True)
            helper.write_text("# explicitly CPU-qualified reproduction helper\n")
            unrelated = helper.parent / "not_qualified.json"
            unrelated.write_text("{}\n")
            cpu = json.loads(proof.read_text())
            cpu["source_sha256"][name] = package_h1.digest(helper)
            proof.write_text(json.dumps(cpu))
            try:
                package_h1.package(args)
            except ValueError as error:
                self.fail(
                    f"qualified gate reproduction source must be packageable: {error}"
                )
            self.assertEqual(
                (args.out / "source" / name).read_bytes(), helper.read_bytes()
            )
            self.assertFalse(
                (args.out / "source" / unrelated.relative_to(args.source)).exists()
            )

    def test_successor_declares_and_binds_strict_provider_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            package_h1.package(args)
            document = json.loads((args.out / "PACKAGE.json").read_text())
            self.assertEqual(document["schema_version"], 3)
            self.assertEqual(document["provider_contract"], "strict_qpn2_torch_v1")
            for name in PROVIDER_SOURCES:
                self.assertIn(name, document["required_source_files"])
                self.assertTrue((args.out / "source" / name).is_file())
            hashes = refresh_bundle(args.out)
            parsed = h1_preflight._parse_package(args.out, hashes)
            h1_preflight._validate_provenance(args.out, parsed, hashes)

    def test_missing_provider_rejected_before_output_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            args, proof = self.fixture(Path(temp))
            name = PROVIDER_SOURCES[0]
            (args.source / name).unlink()
            cpu = json.loads(proof.read_text())
            cpu["source_sha256"].pop(name)
            proof.write_text(json.dumps(cpu))
            with self.assertRaisesRegex(ValueError, "missing required"):
                package_h1.package(args)
            self.assertFalse(args.out.exists())

    def test_provider_not_qualified_by_cpu_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            args, proof = self.fixture(Path(temp))
            cpu = json.loads(proof.read_text())
            cpu["source_sha256"].pop(PROVIDER_SOURCES[0])
            proof.write_text(json.dumps(cpu))
            with self.assertRaisesRegex(ValueError, "qualification omits required"):
                package_h1.package(args)
            self.assertFalse(args.out.exists())

    def test_removed_provider_rejected_even_with_consistent_manifests(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            package_h1.package(args)
            name = PROVIDER_SOURCES[0]
            (args.out / "source" / name).unlink()
            proof = args.out / "provenance/CPU_FULL_PASS.json"
            cpu = json.loads(proof.read_text())
            cpu["source_sha256"].pop(name)
            proof.write_text(json.dumps(cpu))
            hashes = refresh_bundle(args.out)
            with self.assertRaisesRegex(h1_preflight.PreflightError, "ffi_provider"):
                h1_preflight._parse_package(args.out, hashes)

    def test_preflight_refuses_omitted_cpu_binding(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            package_h1.package(args)
            proof = args.out / "provenance/CPU_FULL_PASS.json"
            cpu = json.loads(proof.read_text())
            cpu["source_sha256"].pop(PROVIDER_SOURCES[0])
            proof.write_text(json.dumps(cpu))
            hashes = refresh_bundle(args.out)
            package = h1_preflight._parse_package(args.out, hashes)
            with self.assertRaisesRegex(
                h1_preflight.PreflightError, "qualification omits required"
            ):
                h1_preflight._validate_provenance(args.out, package, hashes)

    def test_legacy_schemas_remain_valid_without_strict_provider_claim(self):
        for version in (1, 2):
            with self.subTest(version=version), tempfile.TemporaryDirectory() as temp:
                args, _ = self.fixture(Path(temp))
                package_h1.package(args)
                path = args.out / "PACKAGE.json"
                package = json.loads(path.read_text())
                package["schema_version"] = version
                package.pop("provider_contract", None)
                package["required_source_files"] = LEGACY_CANARIES
                path.write_text(json.dumps(package))
                hashes = refresh_bundle(args.out)
                parsed = h1_preflight._parse_package(args.out, hashes)
                h1_preflight._validate_provenance(args.out, parsed, hashes)

    def test_strict_claim_requires_versioned_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _ = self.fixture(Path(temp))
            package_h1.package(args)
            path = args.out / "PACKAGE.json"
            package = json.loads(path.read_text())
            package.update(
                schema_version=2,
                required_source_files=LEGACY_CANARIES,
                provider_contract="strict_qpn2_torch_v1",
            )
            path.write_text(json.dumps(package))
            hashes = refresh_bundle(args.out)
            with self.assertRaisesRegex(h1_preflight.PreflightError, "schema 3"):
                h1_preflight._parse_package(args.out, hashes)

    def test_wrong_architecture_rejected_for_all_three_artifacts(self):
        for kind in ("binary", "library", "cudart"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                args, proof = self.fixture(Path(temp))
                path = getattr(args, kind)
                data = bytearray(path.read_bytes())
                data[18:20] = (183).to_bytes(2, "little")  # AArch64
                path.write_bytes(data)
                self.update_receipt(args, proof, kind)
                with self.assertRaisesRegex(ValueError, "ELF64.*x86-64"):
                    package_h1.package(args)
                self.assertFalse(args.out.exists())

    def test_malformed_elf_rejected_before_output_claim(self):
        for field, replacement in (
            (0, b"NOPE"),
            (4, b"\x01"),
            (5, b"\x02"),
            (6, b"\x00"),
            (16, b"\x00\x00"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                args, proof = self.fixture(Path(temp))
                data = bytearray(args.binary.read_bytes())
                data[field : field + len(replacement)] = replacement
                args.binary.write_bytes(data)
                self.update_receipt(args, proof, "binary")
                with self.assertRaisesRegex(ValueError, "ELF64.*x86-64"):
                    package_h1.package(args)
                self.assertFalse(args.out.exists())

    def test_validated_artifacts_and_receipts_cannot_change_during_copy(self):
        for target in (
            "lib/libvf_qpn2_moe_sm70.so",
            "lib/libcudart.so.12",
            "provenance/CPU_FULL_PASS.json",
            "provenance/CUDA_COMPILED.json",
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temp:
                args, _ = self.fixture(Path(temp))
                real_copy = package_h1.copy
                stage = args.out.parent / f".{args.out.name}.partial"

                def changed_copy(source, dest):
                    real_copy(source, dest)
                    if dest.relative_to(stage).as_posix() == target:
                        if target.endswith(".json"):
                            document = json.loads(dest.read_text())
                            document["status"] = "ALTERED_AFTER_VALIDATION"
                            dest.write_text(json.dumps(document))
                        else:
                            data = bytearray(dest.read_bytes())
                            data[-1] ^= 1  # Preserve the valid ELF identity.
                            dest.write_bytes(data)

                with patch.object(package_h1, "copy", changed_copy):
                    with self.assertRaisesRegex(ValueError, "changed while packaging"):
                        package_h1.package(args)
                self.assertFalse(args.out.exists())
                self.assertFalse(stage.exists())

    def update_receipt(self, args, proof, kind):
        if kind == "binary":
            cpu = json.loads(proof.read_text())
            cpu.update(
                binary_sha256=package_h1.digest(args.binary),
                binary_bytes=args.binary.stat().st_size,
            )
            proof.write_text(json.dumps(cpu))
        elif kind == "library":
            cuda = json.loads(args.cuda_proof.read_text())
            cuda["hashes"]["build/libvf_qpn2_moe_sm70.so"] = package_h1.digest(
                args.library
            )
            args.cuda_proof.write_text(json.dumps(cuda))


if __name__ == "__main__":
    unittest.main()
