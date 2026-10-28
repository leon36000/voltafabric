"""A14 numerical and routing boundaries, explicit CPU provider seams only."""

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from adapters import skinny_moe as adapter
from scripts import hardware_canary as canary
from tests.adapter.test_skinny_adapter import layer, align_reference


class RecordingProvider:
    def __init__(self, gate_value=0.0):
        self.gate_value = gate_value
        self.calls = []
        self.torch = torch
        self.device = torch.device("cpu")
        self.launches = 0
        self.synchronized_launches = 0

    def gemm_qpn2_moe(self, x, *args):
        self.calls.append(bool(torch.isfinite(x).all()))
        self.launches += 1
        return torch.full(
            (args[6], args[7]),
            self.gate_value if len(self.calls) == 1 else 0.0,
            dtype=torch.float16,
        )

    def synchronize(self):
        self.synchronized_launches = self.launches


def apply(l, p, x=None, ids=None, mask=None):
    x = torch.ones((1, 64)) if x is None else x
    ids = torch.tensor([[0]]) if ids is None else ids
    return adapter.moe_apply_local(
        l,
        x,
        torch.ones(ids.shape),
        ids,
        None,
        None,
        kernel=p,
        align=align_reference,
        selected_mask=mask,
    )


class A14BoundaryTests(unittest.TestCase):
    def test_cast_overflow_stops_before_gate_provider(self):
        for dtype in [torch.float32, torch.float64, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                p = RecordingProvider()
                with self.assertRaisesRegex(ValueError, "FP16"):
                    apply(layer(), p, x=torch.full((1, 64), 100000.0, dtype=dtype))
                self.assertEqual(p.calls, [])
        p = RecordingProvider()
        apply(layer(), p, x=torch.full((1, 64), 65504.0, dtype=torch.float32))
        self.assertEqual(p.calls, [True, True])

    def test_swiglu_overflow_stops_before_down_provider(self):
        p = RecordingProvider(300.0)
        with self.assertRaisesRegex(ValueError, "activation"):
            apply(layer(), p)
        self.assertEqual(p.calls, [True])
        p = RecordingProvider(1.0)
        apply(layer(), p)
        self.assertEqual(p.calls, [True, True])

    def test_duplicate_expert_is_rejected_even_if_not_locally_selected(self):
        for partitioned in [False, True]:
            l = layer()
            if partitioned:
                l.expert_map = torch.tensor([-1, 1])
            p = RecordingProvider()
            with self.assertRaisesRegex(ValueError, "duplicate"):
                apply(
                    l,
                    p,
                    ids=torch.tensor([[0, 0]]),
                    mask=torch.zeros((1, 2), dtype=torch.bool) if partitioned else None,
                )
            self.assertEqual(p.calls, [])
        p = RecordingProvider()
        positions, _ = apply(
            layer(), p, x=torch.ones((2, 64)), ids=torch.tensor([[0], [0]])
        )
        self.assertEqual(positions.tolist(), [0, 1])
        p = RecordingProvider()
        apply(
            layer(),
            p,
            x=torch.empty((0, 64)),
            ids=torch.empty((0, 2), dtype=torch.int64),
        )
        self.assertEqual(p.calls, [])

    def test_runtime_half_scale_domain_is_checked_before_any_provider(self):
        for suffix, factor, byte in [("13", 4.0, 0x38), ("2", 0.5, 0x7E)]:
            l = layer()
            getattr(l, "skinny_moe_gs" + suffix).fill_(factor)
            getattr(l, "skinny_moe_qs" + suffix).fill_(byte)
            p = RecordingProvider()
            with self.assertRaisesRegex(ValueError, "SM70"):
                apply(l, p)
            self.assertEqual(p.calls, [])

    def test_unsigned_scale_bytes_in_supported_half_domain_remain_valid(self):
        l = layer()
        l.skinny_moe_gs13.fill_(2**-10)
        l.skinny_moe_gs2.fill_(2**-10)
        for byte in range(127):
            l.skinny_moe_qs13.fill_(byte)
            l.skinny_moe_qs2.fill_(byte)
            p = RecordingProvider()
            apply(l, p)
            self.assertEqual(p.calls, [True, True])

    def test_half_domain_handles_tile_local_maxima_and_rounding_boundary(self):
        factor = 65520.0 / 16384.0
        below = torch.nextafter(torch.tensor(factor), torch.tensor(0.0)).item()
        for f, accepted in [(65504.0 / 16384.0, True), (below, True), (factor, False)]:
            l = layer()
            l.skinny_moe_gs13.fill_(f)
            p = RecordingProvider()
            if accepted:
                apply(l, p)
                self.assertEqual(p.calls, [True, True])
            else:
                with self.assertRaisesRegex(ValueError, "SM70"):
                    apply(l, p)
                self.assertEqual(p.calls, [])
        l = layer()
        l.skinny_moe_gs13.fill_(2**-10)
        l.skinny_moe_qs13.fill_(0x7E)
        # Only first tile uses factor1; make its scales zero. Other large scales
        # have their own tiny global factors and must not be rejected by a global max.
        l.skinny_moe_gs13[0, 0] = 1.0
        l.skinny_moe_qs13[:128] = 0
        apply(l, RecordingProvider())

    def test_prepack_domain_rejects_before_exporter_and_publication(self):
        for factor, byte in [(4.0, 0x38), (0.5, 0x7E)]:
            l = SimpleNamespace(
                w13_weight=SimpleNamespace(
                    data=torch.zeros((1, 128, 32), dtype=torch.uint8)
                ),
                w13_weight_scale=SimpleNamespace(
                    data=torch.full((1, 128, 4), byte, dtype=torch.uint8)
                ),
                w13_weight_scale_2=SimpleNamespace(
                    data=torch.full((1, 2), factor, dtype=torch.float32)
                ),
                w2_weight=SimpleNamespace(
                    data=torch.zeros((1, 64, 32), dtype=torch.uint8)
                ),
                w2_weight_scale=SimpleNamespace(
                    data=torch.full((1, 64, 4), 0x38, dtype=torch.uint8)
                ),
                w2_weight_scale_2=SimpleNamespace(
                    data=torch.ones(1, dtype=torch.float32)
                ),
            )
            calls = []

            def prepack(codes, scales):
                calls.append(True)
                return codes.flatten(), scales.flatten()

            with self.assertRaisesRegex(ValueError, "SM70"):
                adapter.prepack_experts(l, prepack=prepack)
            self.assertEqual(calls, [])
            self.assertFalse(hasattr(l, "skinny_moe_shape"))

    def test_canary_rejects_nonfinite_gate_or_activation_before_down_and_receipts(self):
        binary = Path(os.environ["VF_BINARY"])
        context = canary.prepare_rust_canary(binary, canary.routing_payload(2))[0]
        for gate_value in [300.0, float("nan")]:
            p = RecordingProvider(gate_value)
            commands = []
            original = canary.rust_call

            def observed(binary, command, value):
                commands.append(command)
                return original(binary, command, value)

            with patch.object(canary, "rust_call", side_effect=observed):
                with self.assertRaises(ValueError):
                    canary.execute_rust_rank(
                        p, binary, context, torch.ones((2, 64), dtype=torch.float16)
                    )
            self.assertEqual(len(p.calls), 1)
            self.assertNotIn("resident-completions", commands)
