"""A16 frontier properties; CPU fake providers and real Rust metadata only."""

import itertools
import os
from pathlib import Path
import random
import unittest
from unittest.mock import patch
import warnings
import numpy as np
import torch
from adapters import skinny_moe as adapter
from scripts import hardware_canary as canary
from tests.adapter.test_a14_boundaries import RecordingProvider, apply
from tests.adapter.test_skinny_adapter import align_reference, layer


class SwiProvider(RecordingProvider):
    def __init__(self, gate=0.0, up=0.0, down=0.0):
        super().__init__()
        self.gate, self.up, self.down = gate, up, down
        self.inputs = []

    def gemm_qpn2_moe(self, x, *args):
        self.calls.append(bool(torch.isfinite(x).all()))
        self.inputs.append(x.clone())
        self.launches += 1
        out = torch.full((args[6], args[7]), self.down, dtype=torch.float16)
        if len(self.calls) == 1:
            out[:, : args[7] // 2] = self.gate
            out[:, args[7] // 2 :] = self.up
        return out


class NumericalProperties(unittest.TestCase):
    def test_cast_signs_neighbors_and_nonfinite_stop_before_provider(self):
        threshold = np.float32(65520.0)
        around = [
            np.nextafter(threshold, np.float32(0.0)).item(),
            threshold.item(),
            np.nextafter(threshold, np.float32(np.inf)).item(),
        ]
        values = [0.0, 1.0, 65504.0, 100000.0, *around]
        values = [s * v for s, v in itertools.product([-1.0, 1.0], values)]
        values += [float("nan"), float("inf"), -float("inf")]
        for dtype, value in itertools.product(
            [torch.float16, torch.bfloat16, torch.float32, torch.float64], values
        ):
            with self.subTest(dtype=dtype, value=value):
                x = torch.full((1, 64), value, dtype=torch.float64).to(dtype)
                # NumPy is independent of Torch's cast implementation; compare the
                # actual source value (including BF16 rounding), not its decimal label.
                actual_source = float(x[0, 0])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    finite_half = bool(np.isfinite(np.float16(actual_source)))
                p = SwiProvider()
                if finite_half:
                    positions, _ = apply(layer(), p, x=x)
                    self.assertEqual(positions.tolist(), [0])
                    self.assertEqual(p.calls, [True, True])
                    self.assertEqual(p.inputs[0].dtype, torch.float16)
                else:
                    with self.assertRaises(ValueError):
                        apply(layer(), p, x=x)
                    self.assertEqual(p.calls, [])

    def test_canary_cast_checks_both_signs_before_provider(self):
        binary = Path(os.environ["VF_BINARY"])
        context = canary.prepare_rust_canary(binary, canary.routing_payload(2))[0]
        for value in [-100000.0, 100000.0]:
            with self.subTest(value=value):
                p = SwiProvider()
                with self.assertRaisesRegex(ValueError, "FP16"):
                    canary.execute_rust_rank(
                        p, binary, context, torch.full((2, 64), value)
                    )
                self.assertEqual(p.calls, [])

    def test_swiglu_signed_overflow_negative_and_cancellation_controls(self):
        for gate, up in [
            (300.0, 300.0),
            (300.0, -300.0),
            (65504.0, 2.0),
            (float("nan"), 1.0),
            (1.0, float("inf")),
        ]:
            with self.subTest(gate=gate, up=up):
                p = SwiProvider(gate, up)
                with self.assertRaises(ValueError):
                    apply(layer(), p)
                self.assertEqual(p.calls, [True])
        for gate, up in [
            (300.0, 0.0),
            (0.0, 300.0),
            (-300.0, 65504.0),
            (-300.0, -65504.0),
            (-1.0, 65504.0),
            (1.0, -65504.0),
            (-10.0, 65504.0),
        ]:
            with self.subTest(gate=gate, up=up):
                p = SwiProvider(gate, up)
                apply(layer(), p)
                self.assertEqual(p.calls, [True, True])
                self.assertTrue(bool(torch.isfinite(p.inputs[1]).all()))
                if gate == 0.0 or up == 0.0 or gate == -300.0:
                    self.assertTrue(bool((p.inputs[1] == 0.0).all()))

    def test_canary_swiglu_has_no_down_call_or_completion_publication(self):
        binary = Path(os.environ["VF_BINARY"])
        context = canary.prepare_rust_canary(binary, canary.routing_payload(2))[0]
        for gate, up in [(300.0, 300.0), (300.0, -300.0), (float("inf"), 0.0)]:
            commands = []
            original = canary.rust_call

            def observed(binary, command, value):
                commands.append(command)
                return original(binary, command, value)

            p = SwiProvider(gate, up)
            with patch.object(canary, "rust_call", side_effect=observed):
                with self.assertRaises(ValueError):
                    canary.execute_rust_rank(
                        p, binary, context, torch.ones((2, 64), dtype=torch.float16)
                    )
            self.assertEqual(p.calls, [True])
            self.assertNotIn("resident-completions", commands)

    def test_duplicates_unknown_ids_and_masks_reject_before_selection(self):
        mapping = torch.full((101,), -1, dtype=torch.int64)
        mapping[100] = 0
        for ids in [torch.tensor([[100, 100]]), torch.tensor([[4, 4]])]:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                adapter._select_local_slots(
                    ids, mapping, 1, torch.zeros_like(ids, dtype=torch.bool)
                )
        for ids in [torch.tensor([[-1]]), torch.tensor([[101]])]:
            with self.assertRaises(ValueError):
                adapter._select_local_slots(
                    ids, mapping, 1, torch.zeros_like(ids, dtype=torch.bool)
                )
        for mask in [
            torch.ones((1, 2), dtype=torch.bool),
            torch.ones((1, 1)),
            torch.tensor([[True]]),
        ]:
            # Last case tries selecting an expert not resident at this rank.
            with self.assertRaises(ValueError):
                adapter._select_local_slots(torch.tensor([[4]]), mapping, 1, mask)

    def test_sparse_identifier_permutations_preserve_exact_slot_coverage(self):
        rng = random.Random(16103)
        logical = [0, 4, 20, 100]
        for _ in range(32):
            physical = rng.sample(range(4), 4)
            mapping = torch.full((101,), -1, dtype=torch.int64)
            for expert, local in zip(logical, physical):
                mapping[expert] = local
            rows = [rng.sample(logical, 2) for _ in range(3)]
            ids = torch.tensor(rows)
            mask = torch.tensor(
                [[rng.choice([False, True]) for _ in range(2)] for _ in range(3)]
            )
            actual = adapter._select_local_slots(ids, mapping, 4, mask)
            wanted = [
                (token * 2 + slot, token, physical[logical.index(expert)])
                for token, row in enumerate(rows)
                for slot, expert in enumerate(row)
                if mask[token, slot]
            ]
            self.assertEqual(list(zip(*(a.tolist() for a in actual))), wanted)
            renamed = rng.sample(logical, 4)
            rename = dict(zip(logical, renamed))
            new_mapping = torch.full((101,), -1, dtype=torch.int64)
            for expert, local in zip(logical, physical):
                new_mapping[rename[expert]] = local
            new_ids = torch.tensor([[rename[e] for e in row] for row in rows])
            other = adapter._select_local_slots(new_ids, new_mapping, 4, mask)
            for old, new in zip(actual, other):
                self.assertTrue(torch.equal(old, new))

    def test_replica_owners_combine_coefficients_once_and_initialize_empty_rows(self):
        x = torch.ones((2, 64))
        ids = torch.tensor([[5], [5]])
        weights = torch.tensor([[0.25], [0.75]])
        partials = []
        for mask, value in [
            (torch.tensor([[True], [False]]), 2.0),
            (torch.tensor([[False], [True]]), 4.0),
        ]:
            l = layer(1)
            l.expert_map = torch.tensor([-1, -1, -1, -1, -1, 0])
            p = SwiProvider(down=value)
            partials.append(
                adapter.moe_apply(
                    l,
                    x,
                    weights,
                    ids,
                    None,
                    None,
                    kernel=p,
                    align=align_reference,
                    selected_mask=mask,
                )
            )
            self.assertEqual(p.calls, [True, True])
        expected = torch.tensor([0.5, 3.0]).reshape(2, 1).expand(2, 64)
        self.assertTrue(torch.equal(partials[0] + partials[1], expected))
