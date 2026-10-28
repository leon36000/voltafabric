"""Compact local RankDispatch adapter contract for persistent GPU owners."""

import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from adapters import skinny_moe as adapter
from scripts import hardware_canary as canary
from tests.adapter.test_skinny_adapter import (
    UniformPackedCPUOracle,
    align_reference,
    layer,
)

BINARY = Path(os.environ["VF_BINARY"])


def stamp(**changes):
    result = {
        "run_id": "a152-run",
        "model_revision": "a152-model",
        "plan_id": "a152-plan",
        "layer": 3,
        "batch": 7,
        "epoch": 11,
    }
    result.update(changes)
    return result


def dispatch(**changes):
    result = {
        "stamp": stamp(),
        "rank": 2,
        "hidden": 64,
        "tokens": 6,
        "resident_order": [700, 300],
        "slots": [
            {
                "physical_expert": 0,
                "slot": {
                    "token": 1,
                    "slot": 1,
                    "expert": 99,
                    "rank": 2,
                    "local": 700,
                    "weight": 0.8,
                },
            },
            {
                "physical_expert": 1,
                "slot": {
                    "token": 1,
                    "slot": 2,
                    "expert": 10,
                    "rank": 2,
                    "local": 300,
                    "weight": 0.2,
                },
            },
            {
                "physical_expert": 1,
                "slot": {
                    "token": 4,
                    "slot": 0,
                    "expert": 10,
                    "rank": 2,
                    "local": 300,
                    "weight": 0.75,
                },
            },
        ],
    }
    result.update(changes)
    return result


def routed_layer(*, artifact_epoch=None):
    result = layer()
    result.expert_map = torch.full((100,), -1, dtype=torch.int64)
    result.expert_map[99] = 0
    result.expert_map[10] = 1
    if artifact_epoch is not None:
        result._skinny_moe_resident = adapter._ResidentWeights(
            result.skinny_moe_qc13,
            result.skinny_moe_qs13,
            result.skinny_moe_gs13,
            result.skinny_moe_qc2,
            result.skinny_moe_qs2,
            result.skinny_moe_gs2,
            result.skinny_moe_shape,
            result.skinny_moe_splitk13,
            result.skinny_moe_splitk2,
            ("a152-model", 3, 2, artifact_epoch, (700, 300)),
        )
    return result


class CompactRankDispatchTests(unittest.TestCase):
    def test_compact_slots_match_independent_unweighted_oracle_and_keep_identity(self):
        current = stamp()
        routes = dispatch()
        x = torch.stack(
            (torch.full((64,), 0.015625), torch.full((64,), -0.0078125))
        ).to(torch.float16)
        ids, values = adapter.moe_apply_dispatch_local(
            routed_layer(),
            x,
            routes,
            current,
            token_ids=[1, 4],
            kernel=(provider := UniformPackedCPUOracle()),
            align=align_reference,
        )

        expected_ids = [
            (1, 1, 99, 2, 700),
            (1, 2, 10, 2, 300),
            (4, 0, 10, 2, 300),
        ]
        self.assertEqual(ids, expected_ids)
        self.assertEqual([call[0] for call in provider.calls], [3, 3])

        # Independent per-slot scalar oracle: execute gate, up, SwiGLU and down
        # for each physical resident without applying the route coefficient.
        expected_rows = []
        row_by_token = {1: 0, 4: 1}
        for token, _, expert_id, _, _ in expected_ids:
            physical = {99: 0, 10: 1}[expert_id]
            h1 = (
                x[row_by_token[token]].float().sum()
                * routed_layer().skinny_moe_gs13[physical].repeat_interleave(32)
            ).half()
            activation = torch.nn.functional.silu(h1[:64]) * h1[64:]
            h2 = (
                activation.float().sum()
                * routed_layer().skinny_moe_gs2[physical].repeat_interleave(32)
            ).half()
            expected_rows.append(h2)
        torch.testing.assert_close(values, torch.stack(expected_rows), rtol=0, atol=0)

    def test_real_rust_dispatch_enters_compact_adapter_without_global_routes(self):
        problem = canary.routing_payload(2)
        contexts = canary.prepare_rust_canary(BINARY, problem)
        x = torch.tensor([[2.0**-6] * 64, [2.0**-7] * 64], dtype=torch.float16)
        completions = []
        provider_calls = 0
        for context in contexts:
            if context["plan"]["skip"]:
                self.assertEqual(context["launch"]["dispatch"]["slots"], [])
                continue
            dispatch_value = context["launch"]["dispatch"]
            stack = context["stack"]
            shape = tuple(stack["shape"])
            layer_value = SimpleNamespace(
                skinny_moe_shape=shape,
                skinny_moe_qc13=torch.tensor(stack["codes13"], dtype=torch.uint8),
                skinny_moe_qs13=torch.tensor(stack["scales13"], dtype=torch.uint8),
                skinny_moe_gs13=torch.tensor(
                    stack["gscales13"], dtype=torch.float32
                ).reshape(shape[0], shape[1] // 32),
                skinny_moe_qc2=torch.tensor(stack["codes2"], dtype=torch.uint8),
                skinny_moe_qs2=torch.tensor(stack["scales2"], dtype=torch.uint8),
                skinny_moe_gs2=torch.tensor(
                    stack["gscales2"], dtype=torch.float32
                ).reshape(shape[0], shape[3] // 32),
                skinny_moe_splitk13=4,
                skinny_moe_splitk2=4,
                apply_router_weight_on_input=False,
            )
            max_expert = max(
                (entry["id"] for entry in stack["physical_experts"]), default=-1
            )
            layer_value.expert_map = torch.full(
                (max_expert + 1,), -1, dtype=torch.int64
            )
            for physical, expert in enumerate(stack["physical_experts"]):
                layer_value.expert_map[expert["id"]] = physical

            token_ids = sorted(
                {item["slot"]["token"] for item in dispatch_value["slots"]}
            )
            token_rows = x.index_select(0, torch.tensor(token_ids, dtype=torch.int64))
            provider = UniformPackedCPUOracle()
            identities, values = adapter.moe_apply_dispatch_local(
                layer_value,
                token_rows,
                dispatch_value,
                problem["stamp"],
                token_ids=token_ids,
                kernel=provider,
                align=align_reference,
            )
            provider_calls += len(provider.calls)
            expected_identities = [
                (
                    entry["slot"]["token"],
                    entry["slot"]["slot"],
                    entry["slot"]["expert"],
                    entry["slot"]["rank"],
                    entry["slot"]["local"],
                )
                for entry in dispatch_value["slots"]
            ]
            self.assertEqual(identities, expected_identities)
            self.assertEqual(values.shape[0], len(expected_identities))
            for identity, value in zip(identities, values):
                token, slot, expert, rank, local = identity
                completions.append(
                    {
                        "stamp": problem["stamp"],
                        "token": token,
                        "slot": slot,
                        "expert": expert,
                        "rank": rank,
                        "local": local,
                        "value": value.float().tolist(),
                    }
                )
        self.assertEqual(provider_calls, 4)
        collected = canary.rust_call(
            BINARY,
            "collect-routing",
            {"problem": problem, "completions": completions},
        )
        self.assertEqual(collected, canary.full_uniform_moe_oracle(problem))

    def test_rejects_invalid_local_identity_and_coefficients_before_provider(self):
        mutations = []
        duplicated = dispatch()
        duplicated["slots"].append(copy.deepcopy(duplicated["slots"][0]))
        mutations.append(duplicated)
        wrong_physical = dispatch()
        wrong_physical["slots"][0]["physical_expert"] = 1
        mutations.append(wrong_physical)
        wrong_global = dispatch()
        wrong_global["slots"][0]["slot"]["expert"] = 98
        mutations.append(wrong_global)
        wrong_rank = dispatch()
        wrong_rank["slots"][0]["slot"]["rank"] = 9
        mutations.append(wrong_rank)
        repeated_expert = dispatch()
        repeated = copy.deepcopy(repeated_expert["slots"][0])
        repeated["slot"]["slot"] = 3
        repeated_expert["slots"].append(repeated)
        mutations.append(repeated_expert)
        out_of_range_slot = dispatch()
        out_of_range_slot["slots"][0]["slot"]["slot"] = 64
        mutations.append(out_of_range_slot)
        boolean_expert = dispatch()
        boolean_expert["slots"][0]["slot"]["expert"] = True
        mutations.append(boolean_expert)
        bad_weight = dispatch()
        bad_weight["slots"][0]["slot"]["weight"] = float("nan")
        mutations.append(bad_weight)

        for bad in mutations:
            with self.subTest(dispatch=bad):
                provider = UniformPackedCPUOracle()
                with self.assertRaises(ValueError):
                    adapter.moe_apply_dispatch_local(
                        routed_layer(),
                        torch.ones((2, 64), dtype=torch.float16),
                        bad,
                        stamp(),
                        token_ids=[1, 4],
                        kernel=provider,
                        align=align_reference,
                    )
                self.assertEqual(provider.calls, [])

    def test_rejects_incomplete_extra_or_reordered_compacted_tokens(self):
        for token_ids in ([1], [1, 4, 5], [4, 1], [1, True]):
            with self.subTest(token_ids=token_ids):
                provider = UniformPackedCPUOracle()
                with self.assertRaises(ValueError):
                    adapter.moe_apply_dispatch_local(
                        routed_layer(),
                        torch.ones((len(token_ids), 64), dtype=torch.float16),
                        dispatch(),
                        stamp(),
                        token_ids=token_ids,
                        kernel=provider,
                        align=align_reference,
                    )
                self.assertEqual(provider.calls, [])

    def test_rejects_old_stamp_before_provider(self):
        provider = UniformPackedCPUOracle()
        with self.assertRaises(ValueError):
            adapter.moe_apply_dispatch_local(
                routed_layer(),
                torch.ones((2, 64), dtype=torch.float16),
                dispatch(),
                stamp(epoch=10),
                token_ids=[1, 4],
                kernel=provider,
                align=align_reference,
            )
        self.assertEqual(provider.calls, [])

    def test_fp16_overflow_is_rejected_before_compact_provider(self):
        provider = UniformPackedCPUOracle()
        with self.assertRaisesRegex(ValueError, "FP16 compact input overflow"):
            adapter.moe_apply_dispatch_local(
                routed_layer(),
                torch.full((2, 64), 100000.0, dtype=torch.float32),
                dispatch(),
                stamp(),
                token_ids=[1, 4],
                kernel=provider,
                align=align_reference,
            )
        self.assertEqual(provider.calls, [])

    def test_empty_dispatch_returns_initialized_empty_rows_without_provider(self):
        empty = dispatch(slots=[])
        provider = UniformPackedCPUOracle()
        identities, values = adapter.moe_apply_dispatch_local(
            routed_layer(),
            torch.empty((0, 64), dtype=torch.float16),
            empty,
            stamp(),
            token_ids=[],
            kernel=provider,
            align=align_reference,
        )
        self.assertEqual(identities, [])
        self.assertEqual(tuple(values.shape), (0, 64))
        self.assertEqual(values.dtype, torch.float16)
        self.assertEqual(provider.calls, [])

    def test_loaded_owner_accepts_new_epoch_binding_and_rejects_rollback(self):
        compact_x = torch.full((2, 64), 0.015625, dtype=torch.float16)
        provider = UniformPackedCPUOracle()
        accepted, _ = adapter.moe_apply_dispatch_local(
            routed_layer(artifact_epoch=10),
            compact_x,
            dispatch(),
            stamp(),
            token_ids=[1, 4],
            kernel=provider,
            align=align_reference,
        )
        self.assertEqual(len(accepted), 3)
        self.assertEqual(len(provider.calls), 2)

        stale = stamp(epoch=9)
        stale_dispatch = dispatch(stamp=stale)
        provider = UniformPackedCPUOracle()
        with self.assertRaisesRegex(ValueError, "installed resident"):
            adapter.moe_apply_dispatch_local(
                routed_layer(artifact_epoch=10),
                compact_x,
                stale_dispatch,
                stale,
                token_ids=[1, 4],
                kernel=provider,
                align=align_reference,
            )
        self.assertEqual(provider.calls, [])


if __name__ == "__main__":
    unittest.main()
