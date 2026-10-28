"""A46: the three behaviours A43/A44 recorded as observations, now fixed.

1. package_h1 built the bundle in place, so a refusal or error after its mkdir
   left a partial bundle at the output path. It now claims the output with an
   empty directory, builds in a private sibling stage and publishes by rename;
   any failure removes only what this run created.
2. A CPU proof naming a source that is absent or outside the packaged set
   (build/, evidence/, ...) failed with a bare FileNotFoundError after mkdir. It
   is now refused up front, before anything is created.
3. resident_artifact let a lone surrogate (JSON "\\ud800") in model_revision or in
   the vf_info string escape as a bare UnicodeEncodeError; both are now refused
   with the reader's own prefixed message.

CPU only; every file lives in a temporary directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
import test_a43_package_refusals as a43  # noqa: E402
import test_a44_artifact_refusals as a44  # noqa: E402

package_h1 = a43.package_h1


class Packaging(unittest.TestCase):
    _fixture = a43.PackageRefusalTests._fixture
    _write_proofs = a43.PackageRefusalTests._write_proofs

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name).resolve()
        self.args = self._fixture(self.root)
        self.out = self.args.out
        self.stage = self.out.parent / f".{self.out.name}.partial"

    def assertNothingCreated(self):
        self.assertFalse(self.out.exists(), "no bundle may remain")
        self.assertFalse(self.stage.exists(), "no stage may remain")

    def test_success_publishes_complete_bundle_without_stage(self):
        result = package_h1.package(self.args)
        self.assertEqual(result["status"], "H1_PACKAGE_PREPARED_NO_GPU")
        self.assertFalse(self.stage.exists())
        sums = (self.out / "SHA256SUMS").read_text().splitlines()
        self.assertEqual(result["files"], len(sums))
        listed = {line.split("  ", 1)[1] for line in sums}
        on_disk = {
            p.relative_to(self.out).as_posix()
            for p in self.out.rglob("*")
            if p.is_file()
        }
        self.assertEqual(listed, on_disk - {"SHA256SUMS"})

    def test_unpackaged_proof_source_is_refused_before_claiming(self):
        built = self.args.source / "build/vf_extra.o"
        built.parent.mkdir()
        built.write_bytes(b"object outside the packaged set\n")
        self.cpu["source_sha256"]["build/vf_extra.o"] = package_h1.digest(built)
        self._write_proofs(self.args)
        with self.assertRaisesRegex(
            ValueError, r"^CPU qualification source is not packaged: build/vf_extra\.o$"
        ):
            package_h1.package(self.args)
        self.assertNothingCreated()

    def test_missing_proof_source_is_refused_before_claiming(self):
        self.cpu["source_sha256"]["src/absent.rs"] = "0" * 64
        self._write_proofs(self.args)
        with self.assertRaisesRegex(
            ValueError, r"^CPU qualification source is not packaged: src/absent\.rs$"
        ):
            package_h1.package(self.args)
        self.assertNothingCreated()

    def test_error_while_building_removes_stage_and_claim(self):
        real_copy = package_h1.copy

        def copy(source, dest):
            if dest.name == "vf" and dest.parent.name == "bin":
                raise OSError(28, "No space left on device")
            real_copy(source, dest)

        with mock.patch.object(package_h1, "copy", copy):
            with self.assertRaisesRegex(OSError, r"No space left on device"):
                package_h1.package(self.args)
        self.assertNothingCreated()

    def test_claim_filled_meanwhile_is_not_replaced(self):
        real_build = package_h1.build

        def build(*args, **kwargs):
            real_build(*args, **kwargs)
            (self.out / "intruder.txt").write_text("written by someone else\n")

        with mock.patch.object(package_h1, "build", build):
            with self.assertRaises(OSError):
                package_h1.package(self.args)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ["intruder.txt"])
        self.assertFalse(self.stage.exists())

    def test_stale_stage_is_kept_and_claim_released(self):
        self.stage.mkdir()
        (self.stage / "marker").write_text("left by an earlier run\n")
        with self.assertRaises(FileExistsError):
            package_h1.package(self.args)
        self.assertFalse(self.out.exists())
        self.assertEqual(
            (self.stage / "marker").read_text(), "left by an earlier run\n"
        )


class ResidentSurrogates(a44.Base):
    def test_revision_with_lone_surrogate(self):
        self.refuse_info(
            "model_revision is invalid", a44.make_info(model_revision="rev\ud800")
        )

    def test_revision_with_lone_surrogate_in_file(self):
        # json.dumps escapes it as \\ud800, which the reader decodes back.
        info = a44.make_info(model_revision="rev\ud800")
        self.assertIn("\\ud800", json.dumps(info))
        self.refuse_file("model_revision is invalid", a44.encode(info))

    def test_vf_info_string_with_lone_surrogate(self):
        def header_mut(header):
            header["__metadata__"]["vf_info"] += "\ud800"

        self.refuse_file(
            "invalid Safetensors header JSON", a44.encode(header_mut=header_mut)
        )


if __name__ == "__main__":
    unittest.main()
