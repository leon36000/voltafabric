"""CPU-only identity and refusal contracts for the DeepSeek QPN2 layer canary."""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))

from scripts import deepseek_v41_qpn2_layer_canary as canary  # noqa: E402


REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
RUN_ID = "a132-contract-test-r1"
PLAN_ID = "a132-test-plan"
ARTIFACT_IDENTITY = {
    "model_revision": REVISION,
    "layer": 0,
    "rank": 0,
    "epoch": 125,
    "shape": [1, 4608, 5120, 5120, 2304],
    "physical_experts": [{"id": 96, "local": 0}],
    "resident_order": [0],
    "payload_bytes": 19_907_776,
}
DISPATCH_IDENTITY = {
    "run_id": RUN_ID,
    "model_revision": REVISION,
    "plan_id": PLAN_ID,
    "layer": 0,
    "batch": 1,
    "epoch": 126,
}


def rank_dispatch(*, epoch: int = 126) -> dict:
    stamp = {
        "run_id": RUN_ID,
        "model_revision": REVISION,
        "plan_id": PLAN_ID,
        "layer": 0,
        "batch": 1,
        "epoch": epoch,
    }
    slots = [
        {
            "physical_expert": 0,
            "slot": {
                "expert": 96,
                "local": 0,
                "rank": 0,
                "slot": 0,
                "token": token,
                "weight": weight,
            },
        }
        for token, weight in enumerate((0.25, 0.75))
    ]
    return {
        "problem": {
            "stamp": stamp,
            "hidden": 5120,
            "experts": [96],
            "registry": {
                "epoch": epoch,
                "copies": [{"expert": 96, "local": 0, "rank": 0}],
            },
            "routes": [
                [{"expert": 96, "weight": 0.25}],
                [{"expert": 96, "weight": 0.75}],
            ],
        },
        "dispatch": {
            "stamp": stamp,
            "hidden": 5120,
            "rank": 0,
            "resident_order": [0],
            "tokens": 2,
            "slots": slots,
        },
    }


def rust_provenance(dispatch_sha: str) -> dict:
    return {
        "argv": ["/relocated/bin/vf", "routing", "-"],
        "binary_sha256": "a" * 64,
        "stdin_sha256": "b" * 64,
        "stdout_sha256": "c" * 64,
        "exit_code": 0,
        "pinned_rank_dispatch_sha256": dispatch_sha,
        "reproduced_dispatch_matches": True,
    }


def resident_artifact(path: Path) -> str:
    specs = canary.expected_artifact_tensor_specs()
    header = {"__metadata__": {"vf_info": json.dumps(ARTIFACT_IDENTITY)}}
    cursor = 0
    for name, (dtype, shape) in specs.items():
        item_size = 4 if dtype == "F32" else 1
        size = item_size
        for dim in shape:
            size *= dim
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    with path.open("wb") as output:
        output.write(len(encoded).to_bytes(8, "little"))
        output.write(encoded)
        output.truncate(8 + len(encoded) + cursor)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class DeepSeekQPN2LayerCanaryTests(unittest.TestCase):
    def test_only_exact_two_token_layer_shape_is_admitted(self):
        self.assertTrue(
            canary.validate_deepseek_layer_shape(
                rows=2, hidden=5120, intermediate=2304, top_k=1
            )
        )
        for altered in (
            {"rows": 1, "hidden": 5120, "intermediate": 2304, "top_k": 1},
            {"rows": 2, "hidden": 4096, "intermediate": 2304, "top_k": 1},
            {"rows": 2, "hidden": 5120, "intermediate": 2048, "top_k": 1},
            {"rows": 2, "hidden": 5120, "intermediate": 2304, "top_k": 2},
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                canary.validate_deepseek_layer_shape(**altered)

    def test_compressed_artifact_reader_keeps_exactly_six_planes(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_ROOT / ".work") as temporary:
            path = Path(temporary) / "resident.safetensors"
            expected_sha = resident_artifact(path)
            raw, info, planes = canary.read_artifact(
                path, expected_sha, ARTIFACT_IDENTITY
            )
        self.assertEqual(set(planes), set(canary.expected_artifact_tensor_specs()))
        self.assertEqual(info, ARTIFACT_IDENTITY)
        header_bytes = int.from_bytes(raw[:8], "little")
        self.assertEqual(
            len(raw) - 8 - header_bytes, ARTIFACT_IDENTITY["payload_bytes"]
        )
        for name, (dtype, shape) in canary.expected_artifact_tensor_specs().items():
            item_size = 4 if dtype == "F32" else 1
            expected_size = item_size
            for dim in shape:
                expected_size *= dim
            self.assertEqual(len(planes[name]), expected_size)

    def test_artifact_hash_and_epoch_mismatch_refuse_before_gpu(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_ROOT / ".work") as temporary:
            path = Path(temporary) / "resident.safetensors"
            expected_sha = resident_artifact(path)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                canary.read_artifact(path, "0" * 64, ARTIFACT_IDENTITY)
            wrong_epoch = dict(ARTIFACT_IDENTITY, epoch=124)
            with self.assertRaisesRegex(ValueError, "identity"):
                canary.read_artifact(path, expected_sha, wrong_epoch)

    def test_rank_dispatch_preserves_global_local_map_and_coefficients(self):
        document = rank_dispatch()
        # Rust slot order is not semantically significant; token identity is.
        document["dispatch"]["slots"].reverse()
        result = canary.validate_rank_dispatch(
            document, expected_identity=DISPATCH_IDENTITY
        )
        self.assertEqual(result["route_ids"], [[96], [96]])
        self.assertEqual(result["route_weights"], [[0.25], [0.75]])
        self.assertEqual(result["global_to_local"], {96: 0})

    def test_old_epoch_and_incoherent_slot_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "epoch"):
            canary.validate_rank_dispatch(
                rank_dispatch(epoch=125), expected_identity=DISPATCH_IDENTITY
            )
        document = rank_dispatch()
        document["dispatch"]["slots"][0]["slot"]["weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "route|slot|weight"):
            canary.validate_rank_dispatch(document, expected_identity=DISPATCH_IDENTITY)

    def test_global_id_and_fp32_coefficient_domain_are_checked(self):
        out_of_domain = rank_dispatch()
        out_of_domain["problem"]["routes"][0][0]["expert"] = 97
        with self.assertRaisesRegex(ValueError, "unadmitted global expert"):
            canary.validate_rank_dispatch(
                out_of_domain, expected_identity=DISPATCH_IDENTITY
            )
        underflow = rank_dispatch()
        underflow["problem"]["routes"][0][0]["weight"] = 1e-300
        underflow["dispatch"]["slots"][0]["slot"]["weight"] = 1e-300
        with self.assertRaisesRegex(ValueError, "FP32"):
            canary.validate_rank_dispatch(
                underflow, expected_identity=DISPATCH_IDENTITY
            )

    def test_dispatch_provenance_must_match_successful_rust_reproduction(self):
        dispatch_sha = "d" * 64
        self.assertEqual(
            canary.validate_dispatch_provenance(
                rust_provenance(dispatch_sha), dispatch_sha
            )["reproduced_dispatch_matches"],
            True,
        )
        bad = rust_provenance(dispatch_sha)
        bad["exit_code"] = 256
        with self.assertRaisesRegex(ValueError, "Rust routing receipt"):
            canary.validate_dispatch_provenance(bad, dispatch_sha)
        bad = rust_provenance(dispatch_sha)
        bad["pinned_rank_dispatch_sha256"] = "e" * 64
        with self.assertRaisesRegex(ValueError, "Rust routing receipt"):
            canary.validate_dispatch_provenance(bad, dispatch_sha)

    def test_duplicate_token_route_and_incomplete_coverage_are_rejected(self):
        duplicate = rank_dispatch()
        duplicate["problem"]["routes"][0].append({"expert": 96, "weight": 0.01})
        with self.assertRaisesRegex(ValueError, "top-k|route|duplicate"):
            canary.validate_rank_dispatch(
                duplicate, expected_identity=DISPATCH_IDENTITY
            )
        missing = rank_dispatch()
        missing["dispatch"]["slots"].pop()
        with self.assertRaisesRegex(ValueError, "coverage|slot|token"):
            canary.validate_rank_dispatch(missing, expected_identity=DISPATCH_IDENTITY)

    def test_unfinished_receipt_is_unknown_and_never_replayable(self):
        self.assertEqual(
            canary.classify_receipt({"status": "REQUEST_ARMED"}),
            "UNKNOWN_DO_NOT_REPLAY",
        )
        self.assertEqual(
            canary.classify_receipt({"status": "FAILED_BEFORE_GPU"}),
            "FAILED_BEFORE_GPU",
        )
        self.assertEqual(
            canary.classify_receipt({"status": "A132_DEEPSEEK_QPN2_LAYER_PASS"}),
            "A132_DEEPSEEK_QPN2_LAYER_PASS",
        )

    def test_pass_requires_both_projections_combine_and_dispatch(self):
        report = {
            "full_layer_correct": True,
            "gate_up_correct": True,
            "swiglu_correct": True,
            "down_correct": True,
            "weighted_combine_correct": True,
            "rust_rank_dispatch_bound": True,
            "kernel_launch_count": 2,
            "provider_synchronized_launches": 2,
        }
        self.assertTrue(canary.full_layer_pass_eligible(report))
        for field in (
            "full_layer_correct",
            "gate_up_correct",
            "swiglu_correct",
            "down_correct",
            "weighted_combine_correct",
            "rust_rank_dispatch_bound",
        ):
            altered = dict(report, **{field: False})
            with self.subTest(field=field):
                self.assertFalse(canary.full_layer_pass_eligible(altered))
        for field in ("kernel_launch_count", "provider_synchronized_launches"):
            for value in (0, 1, True):
                altered = dict(report, **{field: value})
                with self.subTest(field=field, value=value):
                    self.assertFalse(canary.full_layer_pass_eligible(altered))

    def test_cli_records_pre_gpu_failure_and_refuses_same_run_replay(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_ROOT / ".work") as temporary:
            receipt_dir = Path(temporary)
            argv = [
                "run",
                "--artifact",
                str(receipt_dir / "missing.safetensors"),
                "--artifact-sha256",
                "0" * 64,
                "--dispatch",
                str(receipt_dir / "missing-dispatch.json"),
                "--dispatch-sha256",
                "1" * 64,
                "--dispatch-provenance",
                str(receipt_dir / "missing-provenance.json"),
                "--model-revision",
                REVISION,
                "--plan-id",
                PLAN_ID,
                "--layer",
                "0",
                "--batch",
                "1",
                "--artifact-epoch",
                "125",
                "--dispatch-epoch",
                "126",
                "--run-id",
                RUN_ID,
                "--library",
                str(receipt_dir / "library.so"),
                "--cuda-proof",
                str(receipt_dir / "cuda.json"),
                "--cudart",
                str(receipt_dir / "cudart.so"),
                "--gpu-uuid",
                "GPU-test-explicit",
            ]
            with mock.patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(receipt_dir)}):
                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    self.assertEqual(canary.main(argv), 2)
                first = json.loads(output.getvalue())
                self.assertEqual(first["status"], "FAILED_BEFORE_GPU")
                self.assertFalse(first["cuda_executed"])

                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    self.assertEqual(canary.main(argv), 3)
                second = json.loads(output.getvalue())
                self.assertEqual(second["reconciled_state"], "FAILED_BEFORE_GPU")
                self.assertFalse(second["replay_allowed"])

    def test_cli_reports_unfinished_operation_unknown_and_blocks_replay(self):
        with tempfile.TemporaryDirectory(dir=SCRIPT_ROOT / ".work") as temporary:
            receipt_dir = Path(temporary)
            with mock.patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(receipt_dir)}):
                target = canary.receipt_path(receipt_dir, RUN_ID)
                target.write_text(
                    json.dumps({"run_id": RUN_ID, "status": "REQUEST_ARMED"})
                )
                original = target.read_bytes()
                output = io.StringIO()
                with mock.patch("sys.stdout", output):
                    self.assertEqual(canary.main(["status", "--run-id", RUN_ID]), 3)
                state = json.loads(output.getvalue())
                self.assertEqual(state["reconciled_state"], "UNKNOWN_DO_NOT_REPLAY")
                self.assertFalse(state["replay_allowed"])
                self.assertEqual(target.read_bytes(), original)

    def test_h1_source_inventory_contains_the_relocatable_canary(self):
        from scripts.package_h1 import source_snapshot

        self.assertIn(
            "scripts/deepseek_v41_qpn2_layer_canary.py",
            source_snapshot(SCRIPT_ROOT),
        )

    def test_streamed_activation_bound_matches_independent_dense_product(self):
        import numpy as np

        weights = np.asarray(
            [[-2.0, 0.25, 1.5], [0.0, -4.0, 0.5], [3.0, -0.125, -1.0]],
            dtype=np.float16,
        )
        inputs = np.asarray([[0.5, -2.0, 1.0], [-1.0, 0.0, 4.0]], dtype=np.float16)
        expected = (
            np.abs(inputs.astype(np.float64)) @ np.abs(weights.astype(np.float64)).T
        )
        actual = canary.dense_magnitude_dot_oracle(np, weights, inputs, row_block=2)
        np.testing.assert_array_equal(actual, expected)

    def test_streamed_activation_bound_rejects_invalid_inputs(self):
        import numpy as np

        weights = np.ones((2, 3), dtype=np.float16)
        inputs = np.ones((1, 3), dtype=np.float16)
        for row_block in (0, -1, True, 1.5):
            with self.subTest(row_block=row_block), self.assertRaises(ValueError):
                canary.dense_magnitude_dot_oracle(
                    np, weights, inputs, row_block=row_block
                )
        inputs[0, 1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            canary.dense_magnitude_dot_oracle(np, weights, inputs)


if __name__ == "__main__":
    unittest.main()
