"""CPU-only contracts for the six-expert A133 QPN2 canary."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts import deepseek_v41_qpn2_topk6_canary as canary


RUN_ID = "a133-topk6-contract-r1"
PLAN_ID = "a133-topk6-test-plan"
STAMP = {
    "run_id": RUN_ID,
    "model_revision": canary.REVISION,
    "plan_id": PLAN_ID,
    "layer": 0,
    "batch": 1,
    "epoch": 127,
}
WEIGHTS = (0.31, 0.23, 0.19, 0.12, 0.09, 0.06)


def rank_dispatch():
    ids0 = list(canary.EXPERT_IDS)
    ids1 = list(reversed(canary.EXPERT_IDS))
    weights1 = list(reversed(WEIGHTS))
    routes = [
        [{"expert": expert, "weight": weight} for expert, weight in zip(ids0, WEIGHTS)],
        [
            {"expert": expert, "weight": weight}
            for expert, weight in zip(ids1, weights1)
        ],
    ]
    mapping = dict(zip(canary.EXPERT_IDS, canary.LOCAL_IDS))
    slots = []
    for token, route in enumerate(routes):
        for index, choice in enumerate(route):
            local = mapping[choice["expert"]]
            slots.append(
                {
                    "physical_expert": local,
                    "slot": {
                        "expert": choice["expert"],
                        "local": local,
                        "rank": 0,
                        "slot": index,
                        "token": token,
                        "weight": choice["weight"],
                    },
                }
            )
    copies = [
        {"expert": expert, "local": local, "rank": 0}
        for expert, local in zip(canary.EXPERT_IDS, canary.LOCAL_IDS)
    ]
    return {
        "problem": {
            "stamp": dict(STAMP),
            "hidden": canary.HIDDEN,
            "experts": list(canary.EXPERT_IDS),
            "preferred_owners": {},
            "registry": {"epoch": STAMP["epoch"], "copies": copies},
            "resident_order": {"0": list(canary.LOCAL_IDS)},
            "routes": routes,
        },
        "dispatch": {
            "stamp": dict(STAMP),
            "hidden": canary.HIDDEN,
            "rank": 0,
            "resident_order": list(canary.LOCAL_IDS),
            "tokens": canary.TOKENS,
            "slots": slots,
        },
    }


def expected_identity():
    return dict(STAMP)


class A133TopK6CanaryTests(unittest.TestCase):
    def test_only_exact_two_token_topk6_geometry_is_admitted(self):
        self.assertTrue(
            canary.validate_shape(rows=2, hidden=5120, intermediate=2304, top_k=6)
        )
        for altered in (
            {"rows": 1, "hidden": 5120, "intermediate": 2304, "top_k": 6},
            {"rows": 2, "hidden": 4096, "intermediate": 2304, "top_k": 6},
            {"rows": 2, "hidden": 5120, "intermediate": 2048, "top_k": 6},
            {"rows": 2, "hidden": 5120, "intermediate": 2304, "top_k": 1},
        ):
            with self.subTest(altered=altered), self.assertRaises(ValueError):
                canary.validate_shape(**altered)

    def test_metadata_pins_all_six_ids_order_epoch_shape_and_payload(self):
        good = canary.expected_artifact_info(layer=0, epoch=125)
        self.assertIs(canary.validate_artifact_metadata(good, good), good)
        for field, altered_value in (
            ("epoch", 124),
            ("payload_bytes", good["payload_bytes"] - 1),
            ("resident_order", list(reversed(good["resident_order"]))),
            ("physical_experts", list(reversed(good["physical_experts"]))),
            ("shape", [5, *good["shape"][1:]]),
        ):
            altered = dict(good, **{field: altered_value})
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "identity|shape|size|experts"),
            ):
                canary.validate_artifact_metadata(altered, good)

    def test_rust_dispatch_accepts_permuted_routes_and_preserves_coefficients(self):
        document = rank_dispatch()
        document["dispatch"]["slots"].reverse()
        result = canary.validate_rank_dispatch(
            document, expected_identity=expected_identity()
        )
        self.assertEqual(result["route_ids"][0], list(canary.EXPERT_IDS))
        self.assertEqual(result["route_ids"][1], list(reversed(canary.EXPERT_IDS)))
        self.assertEqual(
            result["global_to_local"], dict(zip(canary.EXPERT_IDS, canary.LOCAL_IDS))
        )
        self.assertEqual(result["route_weights"][0], list(WEIGHTS))
        self.assertEqual(result["route_weights"][1], list(reversed(WEIGHTS)))

    def test_duplicate_id_unknown_id_stale_stamp_and_bad_epoch_are_refused(self):
        bad = rank_dispatch()
        bad["problem"]["routes"][0][1]["expert"] = bad["problem"]["routes"][0][0][
            "expert"
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

        bad = rank_dispatch()
        bad["problem"]["routes"][0][0]["expert"] = 97
        with self.assertRaisesRegex(ValueError, "outside"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

        bad = rank_dispatch()
        bad["dispatch"]["stamp"]["epoch"] = 126
        with self.assertRaisesRegex(ValueError, "Stamp"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

        bad = rank_dispatch()
        bad["problem"]["registry"]["epoch"] = 126
        with self.assertRaisesRegex(ValueError, "registry"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

    def test_duplicate_incomplete_mismapped_and_wrong_weight_slots_are_refused(self):
        cases = []
        duplicate = rank_dispatch()
        duplicate["dispatch"]["slots"][1] = dict(duplicate["dispatch"]["slots"][0])
        cases.append((duplicate, "duplicate"))
        missing = rank_dispatch()
        missing["dispatch"]["slots"].pop()
        cases.append((missing, "cover"))
        foreign = rank_dispatch()
        foreign["dispatch"]["slots"][0]["slot"]["local"] = 4
        cases.append((foreign, "global/local"))
        wrong_weight = rank_dispatch()
        wrong_weight["dispatch"]["slots"][0]["slot"]["weight"] = 0.5
        cases.append((wrong_weight, "coefficient"))
        for document, message in cases:
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                canary.validate_rank_dispatch(
                    document, expected_identity=expected_identity()
                )

    def test_json_booleans_cannot_masquerade_as_slot_integers(self):
        for field_path in (
            ("expert", False),
            ("local", False),
            ("rank", False),
        ):
            bad = rank_dispatch()
            field, value = field_path
            bad["dispatch"]["slots"][0]["slot"][field] = value
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "global/local"),
            ):
                canary.validate_rank_dispatch(
                    bad, expected_identity=expected_identity()
                )

        bad = rank_dispatch()
        bad["dispatch"]["rank"] = False
        with self.assertRaisesRegex(ValueError, "rank, dimensions"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

    def test_metadata_json_booleans_cannot_alias_zero_expert_ids(self):
        expected = canary.expected_artifact_info(layer=0, epoch=125)
        actual = json.loads(json.dumps(expected))
        actual["physical_experts"][0]["id"] = False
        with self.assertRaisesRegex(ValueError, "identity"):
            canary.validate_artifact_metadata(actual, expected)

    def test_dispatch_reads_require_small_regular_non_symlink_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "dispatch.json"
            regular.write_bytes(b"{}")
            self.assertEqual(canary._read_bounded_regular(regular, 2), b"{}")
            with self.assertRaisesRegex(ValueError, "cap"):
                canary._read_bounded_regular(regular, 1)
            link = root / "dispatch-link.json"
            link.symlink_to(regular)
            with self.assertRaises(OSError):
                canary._read_bounded_regular(link, 2)

    def test_zero_nonfinite_and_nonunit_route_weights_are_refused(self):
        for value in (0.0, float("nan"), float("inf"), 1e-300):
            bad = rank_dispatch()
            bad["problem"]["routes"][0][0]["weight"] = value
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "weight|coefficients"),
            ):
                canary.validate_rank_dispatch(
                    bad, expected_identity=expected_identity()
                )
        bad = rank_dispatch()
        bad["problem"]["routes"][0][0]["weight"] = 0.5
        with self.assertRaisesRegex(ValueError, "unit mass"):
            canary.validate_rank_dispatch(bad, expected_identity=expected_identity())

    def test_full_layer_pass_requires_exact_two_synchronized_projection_launches(self):
        report = {
            "rust_rank_dispatch_bound": True,
            "gate_up_correct": True,
            "swiglu_correct": True,
            "down_correct": True,
            "weighted_combine_correct": True,
            "full_layer_correct": True,
            "kernel_launch_count": 2,
            "provider_synchronized_launches": 2,
        }
        self.assertTrue(canary.full_layer_pass_eligible(report))
        for key in (
            "rust_rank_dispatch_bound",
            "gate_up_correct",
            "swiglu_correct",
            "down_correct",
            "weighted_combine_correct",
            "full_layer_correct",
        ):
            self.assertFalse(canary.full_layer_pass_eligible(report | {key: False}))
        self.assertFalse(
            canary.full_layer_pass_eligible(report | {"kernel_launch_count": 1})
        )
        self.assertFalse(
            canary.full_layer_pass_eligible(
                report | {"provider_synchronized_launches": 1}
            )
        )

    def test_missing_artifact_is_a_terminal_pre_gpu_failure_and_cannot_replay(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            argv = [
                "run",
                "--artifact",
                str(root / "missing.safetensors"),
                "--artifact-sha256",
                "a" * 64,
                "--dispatch",
                str(root / "dispatch.json"),
                "--dispatch-sha256",
                "b" * 64,
                "--dispatch-provenance",
                str(root / "provenance.json"),
                "--model-revision",
                canary.REVISION,
                "--plan-id",
                PLAN_ID,
                "--layer",
                "0",
                "--batch",
                "1",
                "--artifact-epoch",
                "125",
                "--dispatch-epoch",
                "127",
                "--run-id",
                RUN_ID,
                "--library",
                str(root / "lib.so"),
                "--cuda-proof",
                str(root / "cuda.json"),
                "--cudart",
                str(root / "cudart.so"),
                "--gpu-uuid",
                "GPU-test-explicit",
            ]
            with mock.patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": temporary}):
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


if __name__ == "__main__":
    unittest.main()
