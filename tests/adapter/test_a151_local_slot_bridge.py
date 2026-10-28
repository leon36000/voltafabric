"""Real Rust completion conversion of the explicitly injected local adapter."""

import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from scripts import hardware_canary as canary
from tests.adapter.test_canary_integration import BINARY, CPUProvider, ROOT


class LocalAdapterBridgeTests(unittest.TestCase):
    def execute(self, provider, context, x):
        try:
            return canary.execute_rust_rank(
                provider, BINARY, context, x, use_local_adapter=True
            )
        except TypeError as error:
            if "unexpected keyword argument 'use_local_adapter'" in str(error):
                self.fail("canary does not exercise the local-adapter/Rust bridge")
            raise

    def contexts(self, preferred=2, directory=None):
        return canary.prepare_rust_canary(
            BINARY, canary.routing_payload(preferred), resident_dir=directory
        )

    def x(self):
        return torch.tensor([[2**-6] * 64, [2**-7] * 64], dtype=torch.float16)

    def test_both_compressed_inputs_replicas_and_empty_rank_match_scalar_oracle(self):
        for binary_input in (False, True):
            for preferred in (2, 9):
                with self.subTest(binary=binary_input, preferred=preferred):
                    with tempfile.TemporaryDirectory(dir=ROOT / ".work") as tmp:
                        contexts = self.contexts(
                            preferred, Path(tmp) if binary_input else None
                        )
                        provider = CPUProvider()
                        rows = []
                        for context in reversed(contexts):
                            out = self.execute(provider, context, self.x())
                            self.assertEqual(
                                len(out), len(context["plan"]["completion_slots"])
                            )
                            rows.extend(out)
                        problem = canary.routing_payload(preferred)
                        result = canary.rust_call(
                            BINARY,
                            "collect-routing",
                            dict(problem=problem, completions=rows),
                        )
                        self.assertEqual(
                            result, canary.full_uniform_moe_oracle(problem)
                        )
                        self.assertEqual(len(rows), 4)
                        self.assertEqual(provider.launches, 4)
                        self.assertEqual(provider.synchronized_launches, 4)
                        self.assertEqual(provider.sync_count, 2)
                        self.assertEqual(
                            {r["rank"] for r in rows if r["expert"] == 99}, {preferred}
                        )

    def test_mismatched_route_or_coefficient_refuses_before_launch(self):
        for mutation in ("duplicate", "wrong_expert", "coefficient", "ragged", "nan"):
            with self.subTest(mutation=mutation):
                context = self.contexts()[0]
                # The routing matrix is the independently retained logical input,
                # not reconstructed by trusting the selected rank's output slots.
                context.setdefault(
                    "routes", copy.deepcopy(canary.routing_payload(2)["routes"])
                )
                routes = context["routes"]
                if mutation == "duplicate":
                    routes[0][1]["expert"] = routes[0][0]["expert"]
                elif mutation == "wrong_expert":
                    routes[0][0]["expert"] = 25
                elif mutation == "coefficient":
                    routes[0][0]["weight"] = 0.125
                elif mutation == "ragged":
                    routes[1].pop()
                else:
                    routes[0][0]["weight"] = float("nan")
                provider = CPUProvider()
                with self.assertRaises(ValueError):
                    self.execute(provider, context, self.x())
                self.assertEqual(provider.launches, 0)

    def test_slot_order_is_reconciled_to_rust_completion_order(self):
        contexts = self.contexts()
        for context in contexts:
            context["launch"]["dispatch"]["slots"].reverse()
            context["plan"] = canary.rust_call(
                BINARY, "resident-launch", context["launch"]
            )
        rows = []
        for context in contexts:
            rows.extend(self.execute(CPUProvider(), context, self.x()))
        problem = canary.routing_payload(2)
        self.assertEqual(
            canary.rust_call(
                BINARY, "collect-routing", dict(problem=problem, completions=rows)
            ),
            canary.full_uniform_moe_oracle(problem),
        )

    def test_adapter_row_permutation_is_reconciled_by_slot_identity(self):
        from adapters import skinny_moe as adapter

        original = adapter.moe_apply_dispatch_local

        def shuffled(*args, **kwargs):
            identities, values = original(*args, **kwargs)
            return list(reversed(identities)), values.flip(0)

        rows = []
        with patch.object(adapter, "moe_apply_dispatch_local", side_effect=shuffled):
            for context in self.contexts():
                rows.extend(self.execute(CPUProvider(), context, self.x()))
        problem = canary.routing_payload(2)
        self.assertEqual(
            canary.rust_call(
                BINARY, "collect-routing", dict(problem=problem, completions=rows)
            ),
            canary.full_uniform_moe_oracle(problem),
        )

    def test_invalid_logical_domain_is_refused_even_on_empty_rank(self):
        for mutation in (
            "negative",
            "huge",
            "boolean",
            "negative_weight",
            "infinite_weight",
            "missing",
        ):
            with self.subTest(mutation=mutation):
                context = self.contexts()[-1]
                self.assertTrue(context["plan"]["skip"])
                choice = context["routes"][0][0]
                if mutation == "negative":
                    choice["expert"] = -1
                elif mutation == "huge":
                    choice["expert"] = canary.MAX_CANARY_BYTES // 8
                elif mutation == "boolean":
                    choice["expert"] = True
                elif mutation == "negative_weight":
                    choice["weight"] = -0.25
                elif mutation == "infinite_weight":
                    choice["weight"] = float("inf")
                else:
                    del context["routes"]
                provider = CPUProvider()
                with patch.object(
                    canary, "rust_call", wraps=canary.rust_call
                ) as observed:
                    with self.assertRaises(ValueError):
                        self.execute(provider, context, self.x())
                self.assertEqual(provider.launches, 0)
                self.assertNotIn(
                    "resident-completions", [c.args[1] for c in observed.call_args_list]
                )

    def test_nonfinite_swiglu_never_calls_down_or_rust_converter(self):
        class OverflowProvider(CPUProvider):
            def gemm_qpn2_moe(self, *args):
                self.launches += 1
                return torch.full((args[7], args[8]), 300.0, dtype=torch.float16)

        provider = OverflowProvider()
        with patch.object(canary, "rust_call", wraps=canary.rust_call) as observed:
            with self.assertRaisesRegex(ValueError, "SwiGLU"):
                self.execute(provider, self.contexts()[0], self.x())
        self.assertEqual(provider.launches, 1)
        self.assertNotIn(
            "resident-completions", [c.args[1] for c in observed.call_args_list]
        )

    def test_fp16_cast_overflow_refuses_before_provider(self):
        context = self.contexts()[0]
        for value in (100000.0, -100000.0):
            provider = CPUProvider()
            with self.assertRaisesRegex(ValueError, "FP16 compact input overflow"):
                self.execute(provider, context, torch.full((2, 64), value))
            self.assertEqual(provider.launches, 0)

    def test_sync_failure_never_converts_or_publishes_completions(self):
        context = self.contexts()[0]
        provider = CPUProvider()

        def fail():
            raise RuntimeError("injected sync failure")

        provider.synchronize = fail
        with patch.object(canary, "rust_call", wraps=canary.rust_call) as observed:
            with self.assertRaisesRegex(RuntimeError, "injected sync failure"):
                self.execute(provider, context, self.x())
        self.assertNotIn(
            "resident-completions", [c.args[1] for c in observed.call_args_list]
        )
        self.assertEqual(provider.launches, 2)
        self.assertEqual(provider.synchronized_launches, 0)

    def test_disjoint_or_duplicate_slot_identities_never_reach_rust_converter(self):
        from adapters import skinny_moe as adapter

        original = adapter.moe_apply_dispatch_local
        for malformed in (
            "duplicate",
            "foreign",
            "truncated",
            "identity_type",
            "value_dtype",
        ):

            def altered(*args, **kwargs):
                identities, values = original(*args, **kwargs)
                identities = list(identities)
                if malformed == "duplicate":
                    identities[0] = identities[1]
                elif malformed == "foreign":
                    identities[0] = (*identities[0][:-1], identities[0][-1] + 1)
                elif malformed == "truncated":
                    identities, values = identities[:-1], values[:-1]
                elif malformed == "identity_type":
                    identities[0] = None
                else:
                    values = values.to(torch.float32)
                return identities, values

            with self.subTest(malformed=malformed):
                context = self.contexts()[0]
                provider = CPUProvider()
                with patch.object(
                    adapter, "moe_apply_dispatch_local", side_effect=altered
                ):
                    with patch.object(
                        canary, "rust_call", wraps=canary.rust_call
                    ) as observed:
                        with self.assertRaisesRegex(
                            ValueError, "compact adapter slot identities or values"
                        ):
                            self.execute(provider, context, self.x())
                self.assertNotIn(
                    "resident-completions", [c.args[1] for c in observed.call_args_list]
                )


if __name__ == "__main__":
    unittest.main()
