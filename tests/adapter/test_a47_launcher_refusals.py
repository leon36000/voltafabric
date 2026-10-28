"""A47: the refusals of the two H1 bash launchers, run end to end.

handoff/h1_bundle_run.sh (shipped as run_h1.sh) and handoff/hardware_canary.sh
were checked only as text and by `bash -n`. Here each one really runs inside a
synthetic bundle whose path has spaces. Recording stubs replace the preflight and
GPU interpreters, so nothing imports torch, loads CUDA or touches a device.

Every refusal must stop before either stub runs and before any receipt directory
exists. Every accepted case must hand the stubs the exact argv, working directory
and receipt root.

guard_schemata.py --sites sh-launchers switches each refusal off through
VF_MUTANT, which the launchers read themselves, so the environment below keeps it.
The ids in the comments are that catalog's ids.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash") or "/bin/bash"
UUID = "GPU-00000000-0000-0000-0000-cpu-test-only"
CUDART = ".tools/cuda12/components/cuda_cudart-linux-x86_64-12.8.90-archive/lib/libcudart.so.12"
# Records the working directory, the receipt root it inherited and its argv.
STUB = """#!/bin/sh
{{ pwd; printf '%s\\n' "${{VF_H1_RECEIPT_DIR-<unset>}}"; for a in "$@"; do printf '%s\\n' "$a"; done; }} > "$0.call"
exit {code}
"""


class Launchers(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.tmp = Path(temp.name).resolve()
        self.bundle = self.tmp / "H1 bundle with spaces"
        handoff = self.bundle / "source/handoff"
        handoff.mkdir(parents=True)
        shutil.copy(ROOT / "handoff/h1_bundle_run.sh", self.bundle / "run_h1.sh")
        shutil.copy(ROOT / "handoff/hardware_canary.sh", handoff / "hardware_canary.sh")
        self.canary = handoff / "hardware_canary.sh"
        self.stubs = self.tmp / "stub dir"
        self.stubs.mkdir()
        self.preflight = self.stub("preflight")
        self.gpu = self.stub("gpu python")
        self.receipts = self.tmp / "h1-receipts"

    def stub(self, name, code=0, text=None):
        path = self.stubs / name
        path.write_text(text or STUB.format(code=code))
        path.chmod(0o755)
        return path

    def date_stub(self, code):
        """Put a `date` that prints nothing and exits with code first on PATH."""
        self.stub("date", text=f"#!/bin/sh\nexit {code}\n")
        return f"{self.stubs}{os.pathsep}{os.environ.get('PATH', '/usr/bin:/bin')}"

    def run_script(self, script, **values):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("VF_") or k == "VF_MUTANT"
        }
        env.update(
            VF_PREFLIGHT_PYTHON=str(self.preflight),
            VF_GPU_PYTHON=str(self.gpu),
            VF_GPU_UUID=UUID,
        )
        for key, value in values.items():
            if value is None:
                env.pop(key, None)
            else:
                env[key] = value
        return subprocess.run(
            [BASH, str(script)],
            cwd=self.tmp,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    def call(self, stub):
        record = Path(f"{stub}.call")
        return record.read_text().splitlines() if record.exists() else None

    def assertNothingRan(self):
        self.assertIsNone(self.call(self.preflight), "preflight must not run")
        self.assertIsNone(self.call(self.gpu), "the GPU interpreter must not run")

    def canary_argv(self, out, *, library, proof, cudart, binary, extra=()):
        return [
            "scripts/hardware_canary.py",
            "--ack-hardware-gate",
            "--gpu-uuid",
            UUID,
            "--device",
            "0",
            "--library",
            str(library),
            "--cuda-proof",
            str(proof),
            "--cudart",
            str(cudart),
            "--routing-binary",
            str(binary),
            "--out",
            str(out),
            "--resident-io",
            "json",
            *extra,
        ]


class BundleLauncher(Launchers):
    def run_bundle(self, **values):
        return self.run_script(self.bundle / "run_h1.sh", **values)

    def test_accepted_run_hands_exact_argv_to_preflight_then_canary(self):
        done = self.run_bundle(VF_H1_RUN_ID="run-1.2_3")
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(
            self.call(self.preflight),
            [
                str(self.tmp),
                str(self.receipts),
                str(self.bundle / "source/scripts/h1_preflight.py"),
                "--bundle",
                str(self.bundle),
                "--out",
                str(self.receipts / "static-run-1.2_3.json"),
            ],
        )
        self.assertEqual(
            self.call(self.gpu),
            [
                str(self.bundle / "source"),
                str(self.receipts),
                *self.canary_argv(
                    self.receipts / "hardware-run-1.2_3.json",
                    library=self.bundle / "lib/libvf_qpn2_moe_sm70.so",
                    proof=self.bundle / "provenance/CUDA_COMPILED.json",
                    cudart=self.bundle / "lib/libcudart.so.12",
                    binary=self.bundle / "bin/vf",
                ),
            ],
        )

    def test_missing_or_empty_gpu_parameters_are_refused_first(self):
        # ids 4-7: removed check, or `:?` weakened to `?`.
        for name, message in (
            (
                "VF_GPU_PYTHON",
                "Set the existing verified CUDA12 Torch SM70 interpreter",
            ),
            ("VF_GPU_UUID", "Set the available reserved V100 UUID"),
        ):
            for value in (None, ""):
                with self.subTest(name=name, value=value):
                    done = self.run_bundle(**{name: value})
                    self.assertNotEqual(done.returncode, 0)
                    self.assertIn(f"{name}: {message}", done.stderr)
                    self.assertNothingRan()
                    self.assertFalse(self.receipts.exists())

    def test_invalid_run_identifiers_are_refused(self):
        # ids 9 (outside the charset), 10 (.) and 11 (..).
        for run_id in (".", "..", "a/b", "../escape", "a b", "x;y", "é", "a\tb"):
            with self.subTest(run_id=run_id):
                done = self.run_bundle(VF_H1_RUN_ID=run_id)
                self.assertEqual(done.returncode, 2)
                self.assertEqual(done.stderr, "Invalid H1 run identifier\n")
                self.assertNothingRan()
                self.assertFalse(self.receipts.exists())

    def test_empty_generated_identifier_is_refused(self):
        # id 8: an empty VF_H1_RUN_ID is replaced by `date`, so only a `date`
        # that prints nothing and succeeds reaches the "" pattern.
        done = self.run_bundle(VF_H1_RUN_ID=None, PATH=self.date_stub(0))
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stderr, "Invalid H1 run identifier\n")
        self.assertNothingRan()

    def test_failing_identifier_generation_stops_the_launcher(self):
        # id 1: `set -e` stops on the failed $(date ..) before the case is reached.
        done = self.run_bundle(VF_H1_RUN_ID=None, PATH=self.date_stub(1))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr, "")
        self.assertNothingRan()
        self.assertFalse(self.receipts.exists())

    def test_identifiers_near_the_refused_ones_are_accepted(self):
        for run_id in ("a..b", "..x", ".x", "_-.", "Z9"):
            with self.subTest(run_id=run_id):
                for record in self.stubs.glob("*.call"):
                    record.unlink()
                done = self.run_bundle(VF_H1_RUN_ID=run_id)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(
                    self.call(self.gpu)[-3],
                    str(self.receipts / f"hardware-{run_id}.json"),
                )

    def test_failed_static_preflight_stops_before_the_canary(self):
        # id 1: without `set -e` the canary would start after a failed preflight.
        self.preflight = self.stub("preflight", code=3)
        done = self.run_bundle(VF_H1_RUN_ID="r1")
        self.assertEqual(done.returncode, 3)
        self.assertIsNotNone(self.call(self.preflight))
        self.assertIsNone(self.call(self.gpu))

    def test_unusable_receipt_directory_stops_before_preflight(self):
        # id 1 again: mkdir fails inside $(..); without `set -e` the receipt root
        # would become "" and the preflight would write below /.
        blocker = self.tmp / "not a directory"
        blocker.write_text("")
        done = self.run_bundle(VF_H1_RUN_ID="r1", VF_H1_RECEIPT_DIR=str(blocker))
        self.assertNotEqual(done.returncode, 0)
        self.assertNothingRan()

    def test_relative_receipt_directory_is_created_and_made_absolute(self):
        done = self.run_bundle(VF_H1_RUN_ID="r1", VF_H1_RECEIPT_DIR="custom receipts/x")
        self.assertEqual(done.returncode, 0, done.stderr)
        receipts = self.tmp / "custom receipts/x"
        self.assertTrue(receipts.is_dir())
        self.assertEqual(self.call(self.preflight)[1], str(receipts))
        self.assertEqual(self.call(self.gpu)[1], str(receipts))
        self.assertEqual(self.call(self.gpu)[-3], str(receipts / "hardware-r1.json"))

    def test_canary_refusal_after_the_static_preflight(self):
        # The bundle launcher does not check VF_H1_FULL_EXPERT itself: the static
        # receipt is written, then hardware_canary.sh refuses (id 19).
        done = self.run_bundle(VF_H1_RUN_ID="r1", VF_H1_FULL_EXPERT="2")
        self.assertEqual(done.returncode, 2)
        self.assertEqual(done.stderr, "VF_H1_FULL_EXPERT must be 0 or 1\n")
        self.assertIsNotNone(self.call(self.preflight))
        self.assertIsNone(self.call(self.gpu))


class CanaryLauncher(Launchers):
    def run_canary(self, **values):
        return self.run_script(self.canary, **values)

    def test_defaults_point_at_the_canonical_tree(self):
        done = self.run_canary(VF_H1_RUN_ID="r1")
        self.assertEqual(done.returncode, 0, done.stderr)
        source = self.bundle / "source"
        self.assertEqual(
            self.call(self.gpu),
            [
                str(source),
                "<unset>",
                *self.canary_argv(
                    source / "evidence/hardware-H1-r1.json",
                    library=source / "build/libvf_qpn2_moe_sm70.so",
                    proof=source / "evidence/continuation_a23/CUDA_COMPILED.json",
                    cudart=self.bundle / CUDART,
                    binary=self.bundle / "target/release/vf",
                ),
            ],
        )

    def test_overrides_reach_the_canary(self):
        done = self.run_canary(
            VF_GPU_DEVICE="1",
            VF_H1_RESIDENT_IO="safetensors",
            VF_H1_LIBRARY="/l i b.so",
            VF_H1_CUDA_PROOF="/proof.json",
            VF_H1_CUDART="/cudart.so",
            VF_H1_BINARY="/vf",
            VF_H1_OUTPUT="/out.json",
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        argv = self.canary_argv(
            "/out.json",
            library="/l i b.so",
            proof="/proof.json",
            cudart="/cudart.so",
            binary="/vf",
        )
        argv[argv.index("--device") + 1] = "1"
        argv[argv.index("--resident-io") + 1] = "safetensors"
        self.assertEqual(self.call(self.gpu)[2:], argv)

    def test_missing_or_empty_gpu_parameters_are_refused(self):
        # ids 15-18.
        for name, message in (
            (
                "VF_GPU_PYTHON",
                "Set an existing CUDA12 Torch SM70 interpreter; the CPU venv cannot run H1",
            ),
            ("VF_GPU_UUID", "Set the UUID of the available reserved V100"),
        ):
            for value in (None, ""):
                with self.subTest(name=name, value=value):
                    done = self.run_canary(**{name: value})
                    self.assertNotEqual(done.returncode, 0)
                    self.assertIn(f"{name}: {message}", done.stderr)
                    self.assertNothingRan()

    def test_full_expert_switch(self):
        sample = "fixtures/checkpoint_qwen36_expert0_full.safetensors"
        sample = str(self.bundle / "source" / sample)
        for value, override, extra in (
            (None, None, []),
            ("", None, []),
            ("0", None, []),
            ("1", None, ["--full-expert-sample", sample]),
            ("1", "/s.st", ["--full-expert-sample", "/s.st"]),
        ):
            with self.subTest(value=value, sample=override):
                done = self.run_canary(
                    VF_H1_FULL_EXPERT=value, VF_H1_FULL_EXPERT_SAMPLE=override
                )
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertEqual(
                    self.call(self.gpu)[-len(extra) - 2 :],
                    ["--resident-io", "json", *extra],
                )

    def test_other_full_expert_values_are_refused(self):
        # id 19.
        for value in ("2", "01", "yes", " 1", "1 ", "true"):
            with self.subTest(value=value):
                done = self.run_canary(VF_H1_FULL_EXPERT=value)
                self.assertEqual(done.returncode, 2)
                self.assertEqual(done.stderr, "VF_H1_FULL_EXPERT must be 0 or 1\n")
                self.assertNothingRan()

    def test_failing_identifier_generation_stops_the_launcher(self):
        # id 12: without `set -e` the canary would start with the receipt name
        # hardware-H1-.json after `date` failed.
        done = self.run_canary(VF_H1_RUN_ID=None, PATH=self.date_stub(1))
        self.assertEqual(done.returncode, 1)
        self.assertEqual(done.stderr, "")
        self.assertNothingRan()


if __name__ == "__main__":
    unittest.main()
