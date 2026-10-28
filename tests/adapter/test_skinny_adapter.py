import unittest
from types import SimpleNamespace
from unittest import mock
import torch
from adapters import skinny_moe as adapter


def layer(experts=2):
    n13, k13, n2, k2 = 128, 64, 64, 64
    return SimpleNamespace(
        skinny_moe_shape=(experts, n13, k13, n2, k2),
        skinny_moe_qc13=torch.full(
            (experts * n13 * k13 // 2,), 0x22, dtype=torch.uint8
        ),
        skinny_moe_qs13=torch.full(
            (experts * n13 * k13 // 16,), 0x38, dtype=torch.uint8
        ),
        skinny_moe_gs13=torch.tensor(
            [[0.25] * 2 + [0.5] * 2, [0.5] * 2 + [1.0] * 2][:experts],
            dtype=torch.float32,
        ),
        skinny_moe_qc2=torch.full((experts * n2 * k2 // 2,), 0x22, dtype=torch.uint8),
        skinny_moe_qs2=torch.full((experts * n2 * k2 // 16,), 0x38, dtype=torch.uint8),
        skinny_moe_gs2=torch.ones((experts, n2 // 32), dtype=torch.float32),
        skinny_moe_splitk13=4,
        skinny_moe_splitk2=4,
        expert_map=None,
        apply_router_weight_on_input=False,
    )


def align_reference(ids, block, e):
    groups = []
    experts = []
    for expert in range(e):
        slots = [i for i, v in enumerate(ids.flatten().tolist()) if v == expert]
        slots += [ids.numel()] * ((-len(slots)) % block)
        groups.extend(slots)
        experts.extend([expert] * (len(slots) // block))
    return (
        torch.tensor(groups, dtype=torch.int32),
        torch.tensor(experts, dtype=torch.int32),
        torch.tensor(len(groups), dtype=torch.int32),
    )


class UniformPackedCPUOracle:
    """CPU oracle for homogeneous packed weights, not the CUDA kernel."""

    def __init__(self):
        self.calls = []

    def gemm_qpn2_moe(
        self,
        x,
        codes,
        scales,
        global_scales,
        expert_blocks,
        sorted_slots,
        ntpp,
        s,
        n,
        k,
        in_div,
        splitk,
        nacc,
    ):
        self.calls.append((s, in_div, expert_blocks.tolist(), sorted_slots.tolist()))
        # Every weight is FP4(2)*E4M3(0x38)*per-tile global scale.
        assert codes.dtype == scales.dtype == torch.uint8
        assert bool((codes == 0x22).all()) and bool((scales == 0x38).all())
        output = torch.full((s, n), float("nan"), dtype=torch.float16)
        for block, expert in enumerate(expert_blocks.tolist()):
            for slot in sorted_slots[block * 8 : block * 8 + 8].tolist():
                if slot >= s:
                    continue
                output[slot] = (
                    x[slot // in_div].float().sum()
                    * global_scales[expert].repeat_interleave(32)
                ).half()
        assert bool(torch.isfinite(output).all()), (
            "Oracle ABI did not write every compact row"
        )
        return output


def dense_reference(l, x, weights, ids):
    out = torch.zeros(x.shape[0], 64, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot in range(ids.shape[1]):
            expert = int(ids[token, slot])
            if l.expert_map is not None:
                expert = int(l.expert_map[expert])
            if expert < 0:
                continue
            h1 = (
                x[token].half().float().sum()
                * l.skinny_moe_gs13[expert].repeat_interleave(32)
            ).half()
            act = torch.nn.functional.silu(h1[:64]) * h1[64:]
            h2 = (
                act.float().sum() * l.skinny_moe_gs2[expert].repeat_interleave(32)
            ).half()
            out[token] += weights[token, slot].float() * h2.float()
    return out.to(x.dtype)


class AdapterTests(unittest.TestCase):
    def test_compacts_sparse_global_ids_and_keeps_physical_expert_order(self):
        ids = torch.tensor([[10, 99], [99, 25]])
        mapping = torch.full((100,), -1, dtype=torch.int64)
        mapping[99] = 0
        mapping[10] = 1
        positions, tokens, locals_ = adapter._select_local_slots(ids, mapping, 2, None)
        self.assertEqual(positions.tolist(), [0, 1, 2])
        self.assertEqual(tokens.tolist(), [0, 0, 1])
        self.assertEqual(locals_.tolist(), [1, 0, 0])

    def test_quantized_abi_is_called_only_for_local_compact_rows(self):
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        l.expert_map[99] = 0
        l.expert_map[10] = 1
        x = torch.full((2, 64), 0.01, dtype=torch.float32)
        ids = torch.tensor([[10, 99], [99, 25]])
        weights = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
        oracle = UniformPackedCPUOracle()
        result = adapter.moe_apply(
            l,
            x,
            weights,
            ids,
            None,
            None,
            kernel=oracle,
            align=align_reference,
            selected_mask=l.expert_map[ids] >= 0,
        )
        torch.testing.assert_close(
            result, dense_reference(l, x, weights, ids), rtol=0, atol=0
        )
        self.assertEqual([c[:2] for c in oracle.calls], [(3, 1), (3, 1)])
        self.assertTrue(bool(torch.isfinite(result).all()))

    def test_empty_local_selection_initializes_zero_without_kernel_call(self):
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)

        class Forbidden:
            def gemm_qpn2_moe(self, *args):
                raise AssertionError("Nonlocal slots reached kernel")

        out = adapter.moe_apply(
            l,
            torch.ones((2, 64)),
            torch.ones((2, 2)),
            torch.tensor([[10, 99], [25, 99]]),
            None,
            None,
            kernel=Forbidden(),
            align=align_reference,
            selected_mask=torch.zeros((2, 2), dtype=torch.bool),
        )
        self.assertTrue(torch.equal(out, torch.zeros((2, 64))))

    def test_replica_mask_selects_exactly_one_copy_without_renormalization(self):
        l = layer()
        x = torch.full((1, 64), 0.01)
        ids = torch.tensor([[0, 1]])
        weights = torch.tensor([[0.2, 0.3]])
        results = []
        for mask in [torch.tensor([[True, False]]), torch.tensor([[False, True]])]:
            results.append(
                adapter.moe_apply(
                    l,
                    x,
                    weights,
                    ids,
                    None,
                    None,
                    kernel=UniformPackedCPUOracle(),
                    align=align_reference,
                    selected_mask=mask,
                )
            )
        torch.testing.assert_close(
            results[0] + results[1], dense_reference(l, x, weights, ids), rtol=0, atol=0
        )

    def test_shapes_and_invalid_ids_are_rejected_before_kernel(self):
        bads = [torch.tensor([[-1, 0]]), torch.tensor([[0, 2]])]
        for ids in bads:
            with self.assertRaises(ValueError):
                adapter._select_local_slots(ids, None, 2, None)
        with self.assertRaises(ValueError):
            adapter._select_local_slots(torch.tensor([[0]]), torch.tensor([2]), 2, None)
        with self.assertRaises(ValueError):
            adapter._select_local_slots(
                torch.tensor([[0]]), torch.tensor([-2]), 2, None
            )

    def test_empty_batch_and_zero_weight(self):
        l = layer()
        out = adapter.moe_apply(
            l,
            torch.empty((0, 64)),
            torch.empty((0, 2)),
            torch.empty((0, 2), dtype=torch.int64),
            None,
            None,
            kernel=UniformPackedCPUOracle(),
            align=align_reference,
        )
        self.assertEqual(tuple(out.shape), (0, 64))
        out = adapter.moe_apply(
            l,
            torch.full((1, 64), 0.01),
            torch.zeros((1, 2)),
            torch.tensor([[0, 1]]),
            None,
            None,
            kernel=UniformPackedCPUOracle(),
            align=align_reference,
        )
        self.assertTrue(torch.equal(out, torch.zeros((1, 64))))

    def test_router_weight_on_input_remains_rejected(self):
        l = layer()
        l.apply_router_weight_on_input = True
        with self.assertRaises(ValueError):
            adapter.moe_apply(
                l,
                torch.ones((1, 64)),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=UniformPackedCPUOracle(),
                align=align_reference,
            )


class CombinedSwiGLUGuardTests(unittest.TestCase):
    def test_gate_and_activation_finiteness_share_one_collection(self):
        provider = UniformPackedCPUOracle()
        grouped_checks = []
        original_collect = adapter._collect_any_flags

        def observe_collect(*predicates):
            shapes = tuple(tuple(predicate.shape) for predicate in predicates)
            if shapes == ((1, 128), (1, 64)):
                grouped_checks.append(shapes)
            return original_collect(*predicates)

        with mock.patch.object(
            adapter, "_collect_any_flags", side_effect=observe_collect
        ):
            adapter.moe_apply(
                layer(),
                torch.full((1, 64), 0.01),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=provider,
                align=align_reference,
            )

        self.assertEqual(grouped_checks, [((1, 128), (1, 64))])
        self.assertEqual(len(provider.calls), 2)

    def test_finite_gate_up_with_overflowing_swiglu_never_calls_down(self):
        class OverflowingSwiGLUProvider:
            def __init__(self):
                self.calls = 0

            def gemm_qpn2_moe(
                self,
                x,
                codes,
                scales,
                global_scales,
                blocks,
                slots,
                ntpp,
                s,
                n,
                k,
                in_div,
                splitk,
                nacc,
            ):
                self.calls += 1
                if self.calls != 1:
                    raise AssertionError(
                        "down projection must not run after SwiGLU overflow"
                    )
                return torch.full((s, n), 300.0, dtype=torch.float16)

        provider = OverflowingSwiGLUProvider()
        with self.assertRaisesRegex(ValueError, "nonfinite SwiGLU activation"):
            adapter.moe_apply(
                layer(),
                torch.full((1, 64), 0.01),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=provider,
                align=align_reference,
            )
        self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()


class RustRoutingBridgeTests(unittest.TestCase):
    def test_metadata_only_rust_dispatch_drives_the_same_compact_tensor_slots(self):
        import json, os, subprocess
        from pathlib import Path

        binary = Path(
            os.environ.get(
                "VF_BINARY",
                str(Path(__file__).resolve().parents[3] / "target/release/vf"),
            )
        )
        stamp = dict(
            run_id="adapter-bridge",
            model_revision="uniform-packed-fixture",
            plan_id="sparse-physical-order",
            layer=0,
            batch=1,
            epoch=7,
        )
        payload = dict(
            hidden=64,
            experts=[10, 25, 99],
            routes=[
                [dict(expert=10, weight=0.2), dict(expert=99, weight=0.8)],
                [dict(expert=99, weight=0.3), dict(expert=25, weight=0.7)],
            ],
            registry=dict(
                epoch=7,
                copies=[
                    dict(expert=10, rank=2, local=3),
                    dict(expert=99, rank=2, local=900),
                    dict(expert=25, rank=9, local=0),
                ],
            ),
            stamp=stamp,
            preferred_owners={},
            resident_order={"2": [900, 3], "9": [0]},
        )
        run = subprocess.run(
            [str(binary), "routing", "-"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=10,
        )
        self.assertEqual(run.returncode, 0, run.stdout)
        dispatch = json.loads(run.stdout)["result"]
        rank = next(r for r in dispatch["ranks"] if r["rank"] == 2)
        self.assertEqual([s["physical_expert"] for s in rank["slots"]], [1, 0, 0])
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        l.expert_map[99] = 0
        l.expert_map[10] = 1
        x = torch.full((2, 64), 0.01)
        weights = torch.tensor([[0.2, 0.8], [0.3, 0.7]])
        ids = torch.tensor([[10, 99], [99, 25]])
        mask = adapter.bind_rank_dispatch(l, rank, stamp, ids, weights)
        output = adapter.moe_apply(
            l,
            x,
            weights,
            ids,
            None,
            None,
            kernel=UniformPackedCPUOracle(),
            align=align_reference,
            selected_mask=mask,
        )
        torch.testing.assert_close(
            output, dense_reference(l, x, weights, ids), rtol=0, atol=0
        )


class AdapterAdverseTests(unittest.TestCase):
    def test_distinct_global_experts_cannot_alias_one_physical_weight_stack(self):
        with self.assertRaises(ValueError):
            adapter._select_local_slots(
                torch.tensor([[0, 1]]), torch.tensor([0, 0]), 2, None
            )

    def test_shared_experts_are_scheduled_even_when_no_routed_pair_is_local(self):
        from unittest.mock import patch
        import sys

        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        calls = []
        shared = SimpleNamespace(apply=lambda x, order: calls.append((x, order)))
        module = SimpleNamespace(
            SharedExpertsOrder=SimpleNamespace(MK_INTERNAL_OVERLAPPED="overlapped")
        )
        with patch.dict(
            sys.modules,
            {"vllm.model_executor.layers.fused_moe.runner.shared_experts": module},
        ):
            result = adapter.moe_apply(
                l,
                torch.ones((1, 64)),
                torch.ones((1, 1)),
                torch.tensor([[99]]),
                shared,
                "shared-input",
                kernel=UniformPackedCPUOracle(),
                align=align_reference,
                selected_mask=torch.zeros((1, 1), dtype=torch.bool),
            )
        self.assertEqual(calls, [("shared-input", "overlapped")])
        self.assertTrue(torch.equal(result, torch.zeros((1, 64))))

    def test_nonfinite_provider_buffer_is_rejected(self):
        class Corrupt(UniformPackedCPUOracle):
            def gemm_qpn2_moe(self, *args):
                result = super().gemm_qpn2_moe(*args)
                result[0, 0] = float("nan")
                return result

        with self.assertRaises(ValueError):
            adapter.moe_apply(
                layer(),
                torch.full((1, 64), 0.01),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=Corrupt(),
                align=align_reference,
            )


class DispatchBindingTests(unittest.TestCase):
    def test_six_route_binding_avoids_per_route_tensor_scalar_extraction(self):
        stamp = dict(
            run_id="bind-six",
            model_revision="real",
            plan_id="p",
            layer=0,
            batch=1,
            epoch=7,
        )
        layer_state = layer(experts=6)
        layer_state.expert_map = torch.full((100,), -1, dtype=torch.int64)
        experts = [10, 20, 30, 40, 50, 60]
        for physical, expert in enumerate(experts):
            layer_state.expert_map[expert] = physical
        route_weights = [0.0625, 0.125, 0.1875, 0.25, 0.125, 0.25]
        ids = torch.tensor([experts], dtype=torch.int64)
        weights = torch.tensor([route_weights], dtype=torch.float32)
        rank = dict(
            stamp=stamp,
            rank=2,
            hidden=64,
            tokens=1,
            resident_order=[900 + physical for physical in range(6)],
            slots=[
                dict(
                    physical_expert=physical,
                    slot=dict(
                        token=0,
                        slot=physical,
                        expert=expert,
                        rank=2,
                        local=900 + physical,
                        weight=route_weights[physical],
                    ),
                )
                for physical, expert in enumerate(experts)
            ],
        )

        def reject_scalar_extraction(_tensor):
            raise AssertionError("route binding extracted a tensor scalar in Python")

        with (
            mock.patch.object(torch.Tensor, "__int__", reject_scalar_extraction),
            mock.patch.object(torch.Tensor, "__bool__", reject_scalar_extraction),
        ):
            mask = adapter.bind_rank_dispatch(layer_state, rank, stamp, ids, weights)

        self.assertEqual(mask.tolist(), [[True] * 6])

    def test_empty_rank_dispatch_returns_an_all_false_mask(self):
        stamp = dict(
            run_id="empty-rank-bind",
            model_revision="real",
            plan_id="p",
            layer=0,
            batch=1,
            epoch=9,
        )
        layer_state = layer(experts=2)
        ids = torch.tensor([[0, 1]], dtype=torch.int64)
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        rank = dict(
            stamp=stamp,
            rank=1,
            hidden=64,
            tokens=1,
            resident_order=[900, 901],
            slots=[],
        )

        mask = adapter.bind_rank_dispatch(layer_state, rank, stamp, ids, weights)

        self.assertEqual(tuple(mask.shape), tuple(ids.shape))
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(mask.tolist(), [[False, False]])

    def test_partitioned_execution_requires_explicit_owner_mask(self):
        l = layer()
        l.expert_map = torch.tensor([0, 1])
        l.voltafabric_selected_slot_mask = torch.tensor([[True]])
        with self.assertRaises(ValueError):
            adapter.moe_apply(
                l,
                torch.full((1, 64), 0.01),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=UniformPackedCPUOracle(),
                align=align_reference,
            )

    def test_rank_binding_checks_epoch_physical_order_route_and_coefficient(self):
        import copy

        stamp = dict(
            run_id="bind", model_revision="real", plan_id="p", layer=0, batch=1, epoch=7
        )
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        l.expert_map[99] = 0
        l.expert_map[10] = 1
        ids = torch.tensor([[10, 99]])
        weights = torch.tensor([[0.2, 0.8]])
        rank = dict(
            stamp=stamp,
            rank=2,
            hidden=64,
            tokens=1,
            resident_order=[900, 3],
            slots=[
                dict(
                    physical_expert=1,
                    slot=dict(token=0, slot=0, expert=10, rank=2, local=3, weight=0.2),
                ),
                dict(
                    physical_expert=0,
                    slot=dict(
                        token=0, slot=1, expert=99, rank=2, local=900, weight=0.8
                    ),
                ),
            ],
        )
        mask = adapter.bind_rank_dispatch(l, rank, stamp, ids, weights)
        self.assertEqual(mask.tolist(), [[True, True]])
        mutants = []
        stale = copy.deepcopy(rank)
        stale["stamp"]["epoch"] = 6
        mutants.append(stale)
        bad = copy.deepcopy(rank)
        bad["slots"][0]["physical_expert"] = 0
        mutants.append(bad)
        bad = copy.deepcopy(rank)
        bad["slots"][0]["slot"]["weight"] = 0.9
        mutants.append(bad)
        bad = copy.deepcopy(rank)
        bad["slots"][0]["slot"]["expert"] = 99
        mutants.append(bad)
        bad = copy.deepcopy(rank)
        bad["slots"].append(copy.deepcopy(bad["slots"][0]))
        mutants.append(bad)
        for bad in mutants:
            with self.assertRaises(ValueError):
                adapter.bind_rank_dispatch(l, bad, stamp, ids, weights)

    def test_resident_order_must_agree_with_the_dispatch_local_index(self):
        """Two cases no other clause covers: a local index that disagrees with the
        resident order at that physical slot, and a resident order whose length
        differs from the stack."""
        import copy

        stamp = dict(
            run_id="r", model_revision="m", plan_id="p", layer=3, batch=1, epoch=5
        )
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        l.expert_map[99] = 0
        l.expert_map[10] = 1
        ids = torch.tensor([[10, 99]])
        weights = torch.tensor([[0.2, 0.8]])
        rank = dict(
            stamp=stamp,
            rank=2,
            hidden=64,
            tokens=1,
            resident_order=[900, 3],
            slots=[
                dict(
                    physical_expert=1,
                    slot=dict(token=0, slot=0, expert=10, rank=2, local=3, weight=0.2),
                ),
                dict(
                    physical_expert=0,
                    slot=dict(
                        token=0, slot=1, expert=99, rank=2, local=900, weight=0.8
                    ),
                ),
            ],
        )
        self.assertEqual(
            adapter.bind_rank_dispatch(l, rank, stamp, ids, weights).tolist(),
            [[True, True]],
        )
        local_mismatch = copy.deepcopy(rank)
        # (token, slot) stays unique and the rank still matches, so only the
        # resident-order comparison can reject this.
        local_mismatch["slots"][0]["slot"]["local"] = 900
        longer_order = copy.deepcopy(rank)
        # Every local index still agrees; only the length differs from the stack.
        longer_order["resident_order"] = [900, 3, 7]
        for label, bad in (
            ("local disagrees with resident order", local_mismatch),
            ("resident order longer than the stack", longer_order),
        ):
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    adapter.bind_rank_dispatch(l, bad, stamp, ids, weights)


class PrepackTests(unittest.TestCase):
    def test_real_compressed_crop_prepack_retains_separate_gate_up_factors(self):
        import json
        from pathlib import Path
        from tests.reference.qpn_prepack import _qpn_prepack

        sample = json.loads(
            (
                Path(__file__).resolve().parents[2]
                / "fixtures/checkpoint_qwen36_expert0_crop.json"
            ).read_text()
        )
        planes = sample["planes"]

        def codes(name):
            return torch.tensor(planes[name]["codes"], dtype=torch.uint8).reshape(
                64, 32
            )

        def scales(name):
            return (
                torch.tensor(planes[name]["scales"], dtype=torch.uint8)
                .reshape(64, 4)
                .view(torch.float8_e4m3fn)
            )

        # Independent synthetic factor change makes a mistaken shared factor observable.
        g, u, d = (
            planes[name]["global_dequant"]
            for name in ["gate_proj", "up_proj", "down_proj"]
        )
        u *= 2
        l = SimpleNamespace(
            w13_weight=torch.cat([codes("gate_proj"), codes("up_proj")]).unsqueeze(0),
            w13_weight_scale=torch.cat(
                [
                    scales("gate_proj").view(torch.uint8),
                    scales("up_proj").view(torch.uint8),
                ]
            )
            .view(torch.float8_e4m3fn)
            .unsqueeze(0),
            w13_weight_scale_2=torch.tensor([[g, u]], dtype=torch.float32),
            w2_weight=codes("down_proj").unsqueeze(0),
            w2_weight_scale=scales("down_proj").unsqueeze(0),
            w2_weight_scale_2=torch.tensor([d], dtype=torch.float32),
        )
        adapter.prepack_experts(l, prepack=_qpn_prepack)
        self.assertEqual(l.skinny_moe_shape, (1, 128, 64, 64, 64))
        resident = l._skinny_moe_resident
        self.assertEqual(resident._qc13.dtype, torch.uint8)
        self.assertEqual(resident._qs13.dtype, torch.uint8)
        self.assertFalse(hasattr(l, "skinny_moe_qc13"))
        torch.testing.assert_close(
            resident._gs13,
            torch.tensor([[g, g, u, u]], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            resident._gs2,
            torch.tensor([[d, d]], dtype=torch.float32),
            rtol=0,
            atol=0,
        )
        qc, qs = _qpn_prepack(
            l.w13_weight.reshape(128, 32),
            l.w13_weight_scale.view(torch.uint8).reshape(128, 4),
        )
        self.assertTrue(
            torch.equal(qc, resident._qc13) and torch.equal(qs, resident._qs13)
        )

    def test_prepack_owns_aliasing_outputs_and_skips_per_apply_scale_scan(self):
        l = PrepackValidationTests()._layer()
        validations = []
        validate = adapter._validate_sm70_scale_domain

        def record_validation(*args, **kwargs):
            validations.append(args)
            return validate(*args, **kwargs)

        class ZeroProvider:
            def gemm_qpn2_moe(
                self,
                x,
                _codes,
                _scales,
                _factors,
                _blocks,
                _slots,
                _ntpp,
                s,
                n,
                _k,
                _in_div,
                _splitk,
                _nacc,
            ):
                return torch.zeros((s, n), dtype=torch.float16, device=x.device)

        with mock.patch.object(
            adapter, "_validate_sm70_scale_domain", side_effect=record_validation
        ):
            adapter.prepack_experts(l, prepack=lambda codes, scales: (codes, scales))
            resident = l._skinny_moe_resident
            self.assertEqual(len(validations), 4)
            before = [
                resident._qc13.clone(),
                resident._qs13.clone(),
                resident._gs13.clone(),
                resident._qc2.clone(),
                resident._qs2.clone(),
                resident._gs2.clone(),
            ]
            # The injected prepacker returns views of the mutable checkpoint storage.
            l.w13_weight.fill_(0xAA)
            l.w13_weight_scale.view(torch.uint8).fill_(0x7F)
            l.w13_weight_scale_2.fill_(float("nan"))
            l.w2_weight.fill_(0xBB)
            l.w2_weight_scale.view(torch.uint8).fill_(0x7F)
            l.w2_weight_scale_2.fill_(float("nan"))
            for actual, expected in zip(
                (
                    resident._qc13,
                    resident._qs13,
                    resident._gs13,
                    resident._qc2,
                    resident._qs2,
                    resident._gs2,
                ),
                before,
            ):
                self.assertTrue(torch.equal(actual, expected))

            validations.clear()
            positions, output = adapter.moe_apply_local(
                l,
                torch.zeros((1, 64)),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=ZeroProvider(),
                align=align_reference,
            )
            self.assertEqual(positions.tolist(), [0])
            self.assertEqual(tuple(output.shape), (1, 64))
            self.assertEqual(validations, [])

    def test_invalid_postpack_scale_is_rejected_before_owner_publication(self):
        l = PrepackValidationTests()._layer()
        calls = []

        def corrupt_scale(codes, scales):
            calls.append(1)
            return codes.clone(), torch.full_like(scales, 0x7F)

        with self.assertRaises(ValueError):
            adapter.prepack_experts(l, prepack=corrupt_scale)
        self.assertEqual(calls, [1])
        self.assertFalse(hasattr(l, "_skinny_moe_resident"))


class CombineFiniteTests(unittest.TestCase):
    def test_combine_and_output_cast_overflow_are_rejected(self):
        for dtype, weight in [(torch.float32, 1e38), (torch.float16, 1e6)]:
            with self.assertRaises(ValueError):
                adapter.moe_apply(
                    layer(),
                    torch.full((1, 64), 0.01, dtype=dtype),
                    torch.tensor([[weight]]),
                    torch.tensor([[1]]),
                    None,
                    None,
                    kernel=UniformPackedCPUOracle(),
                    align=align_reference,
                )


class PackedScaleValidationTests(unittest.TestCase):
    def test_nonfinite_packed_scales_and_global_factors_reject_before_provider(self):
        class Forbidden:
            def gemm_qpn2_moe(self, *args):
                raise AssertionError("malformed packed weights reached provider")

        for name, value in [
            ("skinny_moe_qs13", 0x7F),
            ("skinny_moe_gs13", float("nan")),
            ("skinny_moe_gs13", 0.0),
        ]:
            l = layer()
            getattr(l, name).flatten()[0] = value
            with self.assertRaises(ValueError):
                adapter.moe_apply(
                    l,
                    torch.ones((1, 64)),
                    torch.ones((1, 1)),
                    torch.tensor([[0]]),
                    None,
                    None,
                    kernel=Forbidden(),
                    align=align_reference,
                )

    def test_nonfinite_checkpoint_scale_rejects_before_publishing_residency(self):
        from tests.reference.qpn_prepack import _qpn_prepack

        l = PrepackValidationTests()._layer()
        l.w13_weight_scale = torch.full((1, 128, 4), 0x7F, dtype=torch.uint8)
        with self.assertRaises(ValueError):
            adapter.prepack_experts(l, prepack=_qpn_prepack)
        self.assertFalse(hasattr(l, "skinny_moe_qc13"))


class PrepackValidationTests(unittest.TestCase):
    def _layer(self):
        return SimpleNamespace(
            w13_weight=torch.zeros((1, 128, 32), dtype=torch.uint8),
            w13_weight_scale=torch.full((1, 128, 4), 0x38, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            w13_weight_scale_2=torch.ones((1, 2)),
            w2_weight=torch.zeros((1, 64, 32), dtype=torch.uint8),
            w2_weight_scale=torch.full((1, 64, 4), 0x38, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            w2_weight_scale_2=torch.ones((1,)),
        )

    def test_invalid_checkpoint_planes_are_rejected_before_prepack(self):
        from tests.reference.qpn_prepack import _qpn_prepack

        for attribute, value in [
            ("w13_weight", torch.zeros((1, 96, 32), dtype=torch.uint8)),
            ("w2_weight", torch.zeros((2, 64, 32), dtype=torch.uint8)),
            ("w13_weight_scale", torch.zeros((1, 128, 3), dtype=torch.uint8)),
            ("w13_weight_scale_2", torch.tensor([[1.0, float("nan")]])),
            ("w2_weight_scale_2", torch.tensor([0.0])),
        ]:
            l = self._layer()
            setattr(l, attribute, value)
            with self.assertRaises(ValueError):
                adapter.prepack_experts(l, prepack=_qpn_prepack)
            self.assertFalse(hasattr(l, "skinny_moe_qc13"))


class UnsignedScaleContractTests(unittest.TestCase):
    """NVFP4 scale bytes 0..126 are finite unsigned; sign bits include -0."""

    @staticmethod
    def _prepack_recorder(calls):
        def prepack(codes, scales):
            calls.append(scales.flatten()[0].item())
            return codes.flatten().clone(), scales.flatten().clone()

        return prepack

    class RecordingProvider:
        def __init__(self):
            self.calls = []

        def gemm_qpn2_moe(self, *args):
            self.calls.append(args[2].flatten()[0].item())
            # Explicit CPU contract provider; no numerical kernel is executed.
            return torch.zeros((args[7], args[8]), dtype=torch.float16)

    @staticmethod
    def _local(l, provider):
        return adapter.moe_apply_local(
            l,
            torch.zeros((1, 64)),
            torch.ones((1, 1)),
            torch.tensor([[0]]),
            None,
            None,
            kernel=provider,
            align=align_reference,
        )

    def test_signed_checkpoint_scales_reject_before_prepack(self):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            for dtype in (torch.uint8, torch.float8_e4m3fn):
                for byte in (0x80, 0xB8, 0xFE):
                    with self.subTest(name=name, dtype=dtype, byte=byte):
                        l = PrepackValidationTests()._layer()
                        raw = torch.full(
                            getattr(l, name).shape, byte, dtype=torch.uint8
                        )
                        setattr(l, name, raw.view(dtype))
                        calls = []
                        with self.assertRaises(ValueError):
                            adapter.prepack_experts(
                                l, prepack=self._prepack_recorder(calls)
                            )
                        self.assertEqual(calls, [])
                        self.assertFalse(hasattr(l, "skinny_moe_qc13"))

    def test_signed_resident_scales_reject_before_provider(self):
        for name in ("skinny_moe_qs13", "skinny_moe_qs2"):
            for byte in (0x80, 0xB8, 0xFE):
                with self.subTest(name=name, byte=byte):
                    l = layer(1)
                    # An invalid byte anywhere in either projection is rejected.
                    getattr(l, name)[-1] = byte
                    provider = self.RecordingProvider()
                    with self.assertRaises(ValueError):
                        self._local(l, provider)
                    self.assertEqual(provider.calls, [])

    def test_all_checkpoint_scale_bytes_preserve_only_unsigned_finite_encodings(self):
        for name in ("w13_weight_scale", "w2_weight_scale"):
            for dtype in (torch.uint8, torch.float8_e4m3fn):
                for byte in range(256):
                    with self.subTest(name=name, dtype=dtype, byte=byte):
                        l = PrepackValidationTests()._layer()
                        # Isolate byte-format validity from kernel half-range admission.
                        l.w13_weight_scale_2.fill_(2**-10)
                        l.w2_weight_scale_2.fill_(2**-10)
                        raw = torch.full(
                            getattr(l, name).shape, byte, dtype=torch.uint8
                        )
                        setattr(l, name, raw.view(dtype))
                        calls = []
                        prepack = self._prepack_recorder(calls)
                        # Exact allowed byte set from the unsigned E4M3 contract.
                        if byte in range(127):
                            adapter.prepack_experts(l, prepack=prepack)
                            self.assertEqual(len(calls), 2)
                            resident = l._skinny_moe_resident
                            packed = (
                                resident._qs13
                                if name.startswith("w13")
                                else resident._qs2
                            )
                            self.assertTrue(bool((packed == byte).all()))
                        else:
                            with self.assertRaises(ValueError):
                                adapter.prepack_experts(l, prepack=prepack)
                            self.assertEqual(calls, [])
                            self.assertFalse(hasattr(l, "skinny_moe_qc13"))

    def test_all_resident_scale_bytes_match_the_same_contract(self):
        for name in ("skinny_moe_qs13", "skinny_moe_qs2"):
            for byte in range(256):
                with self.subTest(name=name, byte=byte):
                    l = layer(1)
                    l.skinny_moe_gs13.fill_(2**-10)
                    l.skinny_moe_gs2.fill_(2**-10)
                    getattr(l, name).fill_(byte)
                    provider = self.RecordingProvider()
                    if byte in range(127):
                        positions, values = self._local(l, provider)
                        self.assertEqual(positions.tolist(), [0])
                        self.assertEqual(tuple(values.shape), (1, 64))
                        self.assertEqual(len(provider.calls), 2)
                        self.assertIn(byte, provider.calls)
                    else:
                        with self.assertRaises(ValueError):
                            self._local(l, provider)
                        self.assertEqual(provider.calls, [])

    def test_runtime_scale_validation_avoids_plan_sized_temporaries(self):
        # Actual full-expert dimensions, compressed weights, one compact row.
        # The provider is the explicit zero-weight CPU contract oracle.
        l = layer(1)
        l.skinny_moe_shape = (1, 1024, 2048, 2048, 512)
        l.skinny_moe_qc13 = torch.zeros(1024 * 2048 // 2, dtype=torch.uint8)
        l.skinny_moe_qs13 = torch.full((1024 * 2048 // 16,), 0x38, dtype=torch.uint8)
        l.skinny_moe_gs13 = torch.ones((1, 32))
        l.skinny_moe_qc2 = torch.zeros(2048 * 512 // 2, dtype=torch.uint8)
        l.skinny_moe_qs2 = torch.full((2048 * 512 // 16,), 0x38, dtype=torch.uint8)
        l.skinny_moe_gs2 = torch.ones((1, 64))
        provider = self.RecordingProvider()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
        ) as profile:
            positions, values = adapter.moe_apply_local(
                l,
                torch.zeros((1, 2048)),
                torch.ones((1, 1)),
                torch.tensor([[0]]),
                None,
                None,
                kernel=provider,
                align=align_reference,
            )
        largest_operator_allocation = max(
            event.cpu_memory_usage for event in profile.events()
        )
        # Inclusive per-operator allocation includes child scalar conversions; self
        # memory is a net value and understated the old mask by one byte.
        # Reject a temporary as large as the scale plan. This measures CPU tensor
        # allocation only; CUDA reduction scratch and GPU peak remain unqualified.
        self.assertLess(
            largest_operator_allocation,
            min(l.skinny_moe_qs13.numel(), l.skinny_moe_qs2.numel()),
        )
        self.assertEqual(positions.tolist(), [0])
        self.assertEqual(tuple(values.shape), (1, 2048))
        self.assertEqual(len(provider.calls), 2)

    def test_empty_checkpoint_preserves_empty_compressed_buffers(self):
        l = PrepackValidationTests()._layer()
        for name in (
            "w13_weight",
            "w13_weight_scale",
            "w13_weight_scale_2",
            "w2_weight",
            "w2_weight_scale",
            "w2_weight_scale_2",
        ):
            setattr(l, name, getattr(l, name)[:0])
        adapter.prepack_experts(
            l, prepack=lambda codes, scales: (codes.reshape(-1), scales.reshape(-1))
        )
        self.assertEqual(l.skinny_moe_shape, (0, 128, 64, 64, 64))
        for tensor in (
            l._skinny_moe_resident._qc13,
            l._skinny_moe_resident._qs13,
            l._skinny_moe_resident._gs13,
            l._skinny_moe_resident._qc2,
            l._skinny_moe_resident._qs2,
            l._skinny_moe_resident._gs2,
        ):
            self.assertEqual(tensor.numel(), 0)
