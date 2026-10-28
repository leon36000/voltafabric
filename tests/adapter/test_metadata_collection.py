"""CPU verification of raw Skinny outputs -> existing Rust Completion collector."""

import json
import os
import subprocess
import unittest
from pathlib import Path
import torch
from adapters import skinny_moe as adapter
from test_skinny_adapter import layer, align_reference, UniformPackedCPUOracle

BINARY = Path(os.environ["VF_BINARY"])


def invoke(command, payload):
    run = subprocess.run(
        [str(BINARY), command, "-"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=15,
    )
    return run.returncode, json.loads(run.stdout)


def metadata(preferred=2):
    return dict(
        hidden=64,
        experts=[10, 25, 99],
        routes=[
            [dict(expert=10, weight=0.25), dict(expert=99, weight=0.5)],
            [dict(expert=99, weight=0.75), dict(expert=25, weight=0.0)],
        ],
        registry=dict(
            epoch=7,
            copies=[
                dict(expert=10, rank=2, local=3),
                dict(expert=99, rank=2, local=900),
                dict(expert=99, rank=9, local=0),
                dict(expert=25, rank=9, local=3),
            ],
        ),
        stamp=dict(
            run_id="raw-bridge",
            model_revision="uniform-packed",
            plan_id="raw-unweighted",
            layer=0,
            batch=1,
            epoch=7,
        ),
        preferred_owners={"99": preferred},
        resident_order={"2": [900, 3], "9": [3, 0], "17": []},
    )


def rank_layer(rank):
    l = layer(len(rank["resident_order"]))
    l.expert_map = torch.full((100,), -1, dtype=torch.int64)
    global_by_local = (
        {3: 10, 900: 99}
        if rank["rank"] == 2
        else ({3: 25, 0: 99} if rank["rank"] == 9 else {})
    )
    for physical, local in enumerate(rank["resident_order"]):
        expert = global_by_local[local]
        l.expert_map[expert] = physical
        # Identical compressed replicas use the same per-global-expert factors.
        gate, up = {10: (0.5, 1.0), 99: (0.25, 0.5), 25: (0.125, 0.25)}[expert]
        l.skinny_moe_gs13[physical] = torch.tensor([gate, gate, up, up])
    return l


def raw_reference(expert, x):
    gate, up = {10: (0.5, 1.0), 99: (0.25, 0.5), 25: (0.125, 0.25)}[expert]
    gate_value = (x.half().float().sum() * gate).half()
    up_value = (x.half().float().sum() * up).half()
    act = torch.nn.functional.silu(gate_value) * up_value
    return (act.float() * 64).half().double().repeat(64)


class RawCollectionTests(unittest.TestCase):
    def test_raw_slots_have_original_positions_and_no_coefficient_applied(self):
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        l.expert_map[99] = 0
        l.expert_map[10] = 1
        ids = torch.tensor([[10, 99], [99, 25]])
        weights = torch.tensor([[0.25, 0.5], [0.75, 0.0]])
        x = torch.full((2, 64), 2.0**-6)
        pos, values = adapter.moe_apply_local(
            l,
            x,
            weights,
            ids,
            None,
            None,
            kernel=UniformPackedCPUOracle(),
            align=align_reference,
            selected_mask=l.expert_map[ids] >= 0,
        )
        self.assertEqual(pos.tolist(), [0, 1, 2])
        self.assertEqual(tuple(values.shape), (3, 64))
        self.assertEqual(values.dtype, torch.float16)
        for row, p in enumerate(pos.tolist()):
            expert = int(ids.flatten()[p])
            token = p // 2
            torch.testing.assert_close(
                values[row].double(), raw_reference(expert, x[token]), rtol=0, atol=0
            )
        scaled = weights * 2
        pos2, values2 = adapter.moe_apply_local(
            l,
            x,
            scaled,
            ids,
            None,
            None,
            kernel=UniformPackedCPUOracle(),
            align=align_reference,
            selected_mask=l.expert_map[ids] >= 0,
        )
        self.assertTrue(
            torch.equal(pos, pos2) and torch.equal(values, values2),
            "raw outputs must not multiply coefficients",
        )

    def test_two_rank_raw_completions_collect_without_dense_model_for_both_replica_owners(
        self,
    ):
        for preferred, coefficient in ((2, 0.25), (9, 0.25), (2, 0.250000000001)):
            problem = metadata(preferred)
            problem["routes"][0][0]["weight"] = coefficient
            rc, data = invoke("routing", problem)
            self.assertEqual(rc, 0, data)
            ids = torch.tensor([[10, 99], [99, 25]])
            weights = torch.tensor([[0.25, 0.5], [0.75, 0.0]])
            x = torch.tensor([[2.0**-6] * 64, [2.0**-7] * 64], dtype=torch.float32)
            completions = []
            oracle_calls = {}
            for rank in data["result"]["ranks"]:
                l = rank_layer(rank)
                mask = adapter.bind_rank_dispatch(
                    l, rank, problem["stamp"], ids, weights
                )
                oracle = UniformPackedCPUOracle()
                pos, values = adapter.moe_apply_local(
                    l,
                    x,
                    weights,
                    ids,
                    None,
                    None,
                    kernel=oracle,
                    align=align_reference,
                    selected_mask=mask,
                )
                oracle_calls[rank["rank"]] = len(oracle.calls)
                by_position = {
                    s["slot"]["token"] * 2 + s["slot"]["slot"]: s["slot"]
                    for s in rank["slots"]
                }
                self.assertEqual(set(pos.tolist()), set(by_position))
                for row, p in enumerate(pos.tolist()):
                    s = by_position[p]
                    completions.append(
                        dict(
                            stamp=rank["stamp"],
                            **{
                                key: s[key]
                                for key in ("token", "slot", "expert", "rank", "local")
                            },
                            value=values[row].double().tolist(),
                        )
                    )
            self.assertEqual(oracle_calls[17], 0)
            self.assertEqual(
                len(completions), 4, "zero-weight slots are still required"
            )
            completions.reverse()
            rc, collected = invoke(
                "collect-routing", dict(problem=problem, completions=completions)
            )
            self.assertEqual(rc, 0, collected)
            expected = torch.zeros((2, 64), dtype=torch.float64)
            for token, choices in enumerate(problem["routes"]):
                for choice in choices:
                    expected[token] += choice["weight"] * raw_reference(
                        choice["expert"], x[token]
                    )
            actual = torch.tensor(collected["result"], dtype=torch.float64)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            # Sensitivity control: a second application of coefficients changes values.
            doubled = torch.zeros_like(expected)
            for token, choices in enumerate(problem["routes"]):
                for choice in choices:
                    doubled[token] += choice["weight"] ** 2 * raw_reference(
                        choice["expert"], x[token]
                    )
            self.assertFalse(torch.equal(actual, doubled))
            if coefficient != 0.25:
                rounded = expected.clone()
                rounded[0] -= (coefficient - 0.25) * raw_reference(10, x[0])
                self.assertFalse(
                    torch.equal(actual, rounded),
                    "collector must retain metadata f64 coefficient",
                )
            self.assertNotIn("model", problem)
            self.assertNotIn("inputs", problem)

    def test_empty_rank_returns_initialized_empty_shape_without_provider_call(self):
        l = layer()
        l.expert_map = torch.full((100,), -1, dtype=torch.int64)
        oracle = UniformPackedCPUOracle()
        pos, out = adapter.moe_apply_local(
            l,
            torch.ones((2, 64)),
            torch.ones((2, 2)),
            torch.tensor([[10, 99], [25, 99]]),
            None,
            None,
            kernel=oracle,
            align=align_reference,
            selected_mask=torch.zeros((2, 2), dtype=torch.bool),
        )
        self.assertEqual(pos.numel(), 0)
        self.assertEqual(tuple(out.shape), (0, 64))
        self.assertEqual(oracle.calls, [])

    def test_metadata_collect_cli_rejects_missing_duplicate_stale_and_expired_without_result(
        self,
    ):
        problem = metadata()
        rc, d = invoke("routing", problem)
        self.assertEqual(rc, 0, d)
        slots = d["result"]["slots"]
        completions = [
            dict(
                stamp=problem["stamp"],
                **{key: s[key] for key in ("token", "slot", "expert", "rank", "local")},
                value=[1.0] * 64,
            )
            for s in slots
        ]
        variants = [
            dict(problem=problem, completions=completions[:-1]),
            dict(problem=problem, completions=completions + [completions[0]]),
            dict(problem=problem, completions=completions, expired=True),
        ]
        stale = json.loads(json.dumps(completions))
        stale[0]["stamp"]["epoch"] = 6
        variants.append(dict(problem=problem, completions=stale))
        codes = [
            "BATCH_INCOMPLETE",
            "INVALID_COMPLETION",
            "BATCH_ABORTED",
            "INVALID_COMPLETION",
        ]
        for payload, code in zip(variants, codes):
            rc, data = invoke("collect-routing", payload)
            self.assertEqual(rc, 2)
            self.assertFalse(data["ok"])
            self.assertNotIn("result", data)
            self.assertEqual(data["error"]["code"], code)
