"""A92: Rust resident artifacts become one private device-resident adapter owner."""

from __future__ import annotations

import json
import inspect
import struct
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from adapters import skinny_moe as adapter
from scripts import resident_artifact


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = (
    ROOT / "evidence/continuation_a57_interpc_stdio/receipts/expert0.qpn2.safetensors"
)
INFO = {
    "model_revision": "1355db6a052410cfd62085d94b58866fd0f2c3c5",
    "layer": 0,
    "rank": 0,
    "epoch": 1,
    "payload_bytes": 1769856,
    "physical_experts": [{"id": 0, "local": 900}],
    "resident_order": [900],
    "shape": [1, 1024, 2048, 2048, 512],
}
SHAPE = (1, 1024, 2048, 2048, 512)
ARTIFACT_SHA256 = "904f7ab1fdc2ed923c596a24db51944afdadc5dc13669940d7a0d9eee8166710"
HAS_DIGEST_LOADER = callable(getattr(adapter, "load_resident_artifact", None)) and (
    "expected_sha256" in inspect.signature(adapter.load_resident_artifact).parameters
)


def stamp(**changes):
    value = {
        "run_id": "a92-run",
        "model_revision": INFO["model_revision"],
        "plan_id": "a92-plan",
        "layer": 0,
        "batch": 1,
        "epoch": 1,
    }
    value.update(changes)
    return value


def rank_dispatch(current_stamp=None, **changes):
    value = {
        "stamp": stamp() if current_stamp is None else dict(current_stamp),
        "rank": 0,
        "hidden": 2048,
        "tokens": 1,
        "resident_order": [900],
        "slots": [
            {
                "physical_expert": 0,
                "slot": {
                    "token": 0,
                    "slot": 0,
                    "expert": 0,
                    "rank": 0,
                    "local": 900,
                    "weight": 1.0,
                },
            }
        ],
    }
    value.update(changes)
    return value


def align_one(ids, block, experts):
    assert block == 8 and experts == 1 and ids.numel() == 1
    return (
        torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
    )


class ZeroProvider:
    def __init__(self):
        self.calls = 0

    def gemm_qpn2_moe(
        self,
        x,
        codes,
        scales,
        gscales,
        blocks,
        slots,
        ntpp,
        rows,
        columns,
        inner,
        in_div,
        splitk,
        nacc,
    ):
        self.calls += 1
        return torch.zeros((rows, columns), dtype=torch.float16, device=x.device)


class ResidentArtifactInstallTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(ARTIFACT.is_file(), "pinned Rust artifact fixture is required")

    def test_adapter_exposes_rust_resident_import(self):
        self.assertTrue(
            callable(getattr(adapter, "load_resident_artifact", None)),
            "Skinny adapter must load Rust QPN2 artifacts into one resident owner",
        )

    def test_adapter_binds_import_to_the_rust_artifact_digest(self):
        self.assertTrue(
            HAS_DIGEST_LOADER,
            "resident import must verify the exact Rust artifact SHA-256",
        )

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_scale_domain_is_checked_before_copy_without_rescanning_owned_planes(self):
        layer = SimpleNamespace(expert_map=None)
        calls = []
        original = adapter._validate_sm70_scale_domain

        def counted(scales, gscales, k, **kwargs):
            calls.append((scales.device.type, k))
            return original(scales, gscales, k, **kwargs)

        with mock.patch.object(adapter, "_validate_sm70_scale_domain", counted):
            adapter.load_resident_artifact(
                layer,
                ARTIFACT,
                expected_info=INFO,
                expected_sha256=ARTIFACT_SHA256,
                device="cpu",
            )

        self.assertEqual(
            calls,
            [("cpu", 2048), ("cpu", 512)],
            "the strict artifact reader admits each immutable source plane once; copies only need layout checks",
        )

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_loads_exact_rust_planes_once_and_dispatch_reuses_resident_owner(self):
        layer = SimpleNamespace(expert_map=None)
        with mock.patch(
            "scripts.resident_artifact.load_stack",
            wraps=__import__(
                "scripts.resident_artifact", fromlist=["load_stack"]
            ).load_stack,
        ) as load_stack:
            owner = adapter.load_resident_artifact(
                layer,
                ARTIFACT,
                expected_info=INFO,
                expected_sha256=ARTIFACT_SHA256,
                device="cpu",
            )
            self.assertIs(layer._skinny_moe_resident, owner)
            self.assertEqual(owner.shape, SHAPE)
            self.assertEqual(
                owner.artifact_binding, (INFO["model_revision"], 0, 0, 1, (900,))
            )
            self.assertEqual(load_stack.call_count, 1)

            raw = ARTIFACT.read_bytes()
            header_size = struct.unpack("<Q", raw[:8])[0]
            header = json.loads(raw[8 : 8 + header_size])
            data_start = 8 + header_size
            tensor_map = {
                "codes13": owner._qc13,
                "scales13": owner._qs13,
                "gscales13": owner._gs13,
                "codes2": owner._qc2,
                "scales2": owner._qs2,
                "gscales2": owner._gs2,
            }
            for name, tensor in tensor_map.items():
                begin, end = header[name]["data_offsets"]
                self.assertEqual(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes(),
                    raw[data_start + begin : data_start + end],
                    name,
                )

            topk_ids = torch.tensor([[0]], dtype=torch.int64)
            topk_weights = torch.tensor([[1.0]], dtype=torch.float32)
            current_stamp = stamp()
            dispatch = rank_dispatch(current_stamp)
            mask = adapter.bind_rank_dispatch(
                layer, dispatch, current_stamp, topk_ids, topk_weights
            )
            self.assertEqual(mask.tolist(), [[True]])

            provider = ZeroProvider()
            for _ in range(2):
                positions, output = adapter.moe_apply_local(
                    layer,
                    torch.ones((1, 2048), dtype=torch.float16),
                    topk_weights,
                    topk_ids,
                    None,
                    None,
                    kernel=provider,
                    align=align_one,
                    selected_mask=mask,
                )
                self.assertEqual(positions.tolist(), [0])
                self.assertEqual(tuple(output.shape), (1, 2048))
                self.assertTrue(torch.equal(output, torch.zeros_like(output)))
            self.assertEqual(provider.calls, 4)
            self.assertEqual(load_stack.call_count, 1)
            self.assertIs(layer._skinny_moe_resident, owner)

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_metadata_mismatch_does_not_publish_any_resident_state(self):
        layer = SimpleNamespace(expert_map=None)
        wrong = dict(INFO, model_revision="other-revision")
        with self.assertRaisesRegex(ValueError, "identity"):
            adapter.load_resident_artifact(
                layer,
                ARTIFACT,
                expected_info=wrong,
                expected_sha256=ARTIFACT_SHA256,
                device="cpu",
            )
        self.assertFalse(hasattr(layer, "_skinny_moe_resident"))
        self.assertFalse(hasattr(layer, "skinny_moe_shape"))

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_wrong_global_to_physical_map_is_rejected_before_publication(self):
        layer = SimpleNamespace(expert_map=torch.tensor([-1], dtype=torch.int64))
        with self.assertRaisesRegex(ValueError, "global-to-physical"):
            adapter.load_resident_artifact(
                layer,
                ARTIFACT,
                expected_info=INFO,
                expected_sha256=ARTIFACT_SHA256,
                device="cpu",
            )
        self.assertFalse(hasattr(layer, "_skinny_moe_resident"))

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_dispatch_must_match_resident_model_layer_rank_and_order(self):
        layer = SimpleNamespace(expert_map=None)
        adapter.load_resident_artifact(
            layer,
            ARTIFACT,
            expected_info=INFO,
            expected_sha256=ARTIFACT_SHA256,
            device="cpu",
        )
        topk_ids = torch.tensor([[0]], dtype=torch.int64)
        topk_weights = torch.tensor([[1.0]], dtype=torch.float32)
        invalid = [
            (stamp(model_revision="old"), rank_dispatch(stamp(model_revision="old"))),
            (stamp(layer=1), rank_dispatch(stamp(layer=1))),
            (stamp(epoch=0), rank_dispatch(stamp(epoch=0))),
            (stamp(), rank_dispatch(rank=1)),
            (stamp(), rank_dispatch(resident_order=[901])),
        ]
        for current_stamp, dispatch in invalid:
            with self.subTest(current_stamp=current_stamp, dispatch=dispatch):
                with self.assertRaisesRegex(ValueError, "artifact identity"):
                    adapter.bind_rank_dispatch(
                        layer, dispatch, current_stamp, topk_ids, topk_weights
                    )

        rebound = stamp(epoch=7)
        accepted = rank_dispatch(rebound)
        mask = adapter.bind_rank_dispatch(
            layer, accepted, rebound, topk_ids, topk_weights
        )
        self.assertEqual(mask.tolist(), [[True]])

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_wrong_content_digest_does_not_publish_any_resident_state(self):
        layer = SimpleNamespace(expert_map=None)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            adapter.load_resident_artifact(
                layer,
                ARTIFACT,
                expected_info=INFO,
                expected_sha256="0" * 64,
                device="cpu",
            )
        self.assertFalse(hasattr(layer, "_skinny_moe_resident"))

    @unittest.skipUnless(
        HAS_DIGEST_LOADER,
        "requires a digest-bound Rust resident artifact loader",
    )
    def test_failed_plane_copy_does_not_publish_a_partial_owner(self):
        layer = SimpleNamespace(expert_map=None)
        original_to = torch.Tensor.to
        calls = 0

        def fail_on_fourth_copy(tensor, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise RuntimeError("injected device-copy failure")
            return original_to(tensor, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "to", fail_on_fourth_copy):
            with self.assertRaisesRegex(RuntimeError, "device-copy failure"):
                adapter.load_resident_artifact(
                    layer,
                    ARTIFACT,
                    expected_info=INFO,
                    expected_sha256=ARTIFACT_SHA256,
                    device="cpu",
                )
        self.assertEqual(calls, 4)
        self.assertFalse(hasattr(layer, "_skinny_moe_resident"))

    def test_oversized_file_is_rejected_before_digest_scan(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oversized.safetensors"
            path.write_bytes(b"\0" * 8)
            with path.open("r+b") as output:
                output.truncate(
                    resident_artifact.MAX_PAYLOAD_BYTES
                    + resident_artifact.MAX_HEADER_BYTES
                    + 9
                )
            with mock.patch.object(resident_artifact, "_sha256_fd") as digest:
                with self.assertRaisesRegex(ValueError, "file size outside bounded"):
                    resident_artifact.load_stack(path, expected_sha256="0" * 64)
                digest.assert_not_called()


if __name__ == "__main__":
    unittest.main()
