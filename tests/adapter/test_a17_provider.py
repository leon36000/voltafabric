"""Direct QPN2 provider contracts, with CPU tensors and a recorded fake enqueue.

No CUDA library/runtime is loaded and fake launches are not numerical GPU proof.
"""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from scripts import hardware_canary as canary


class ProviderAdmissionTests(unittest.TestCase):
    def provider(self):
        p = object.__new__(canary.QPN2Provider)
        p.torch, p.device = torch, torch.device("cpu")
        p.pending, p.admissions, p.launches, p.synchronized_launches = [], 0, 0, 0
        p.progress = None
        calls = []
        p.library = SimpleNamespace(
            vf_qpn2_moe_validate=lambda *_: calls.append("validate") or 0,
            vf_qpn2_moe_launch=lambda *_: calls.append("launch") or 0,
        )
        return p, calls

    def data(self, n=32):
        return dict(
            x=torch.ones((1, 64), dtype=torch.float16),
            codes=torch.full((n * 32,), 0x22, dtype=torch.uint8),
            scales=torch.full((n * 4,), 0x38, dtype=torch.uint8),
            global_scales=torch.full((1, n // 32), 0.25, dtype=torch.float32),
            expert_blocks=torch.tensor([0], dtype=torch.int32),
            sorted_slots=torch.tensor([0] + [1] * 7, dtype=torch.int32),
            ntpp=torch.tensor(8, dtype=torch.int32),
            s=1,
            n=n,
            k=64,
            in_div=1,
            splitk=4,
            nacc=1,
        )

    def call(self, provider, data):
        stream = SimpleNamespace(cuda_stream=0)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            return provider.gemm_qpn2_moe(**data)

    def reject(self, data):
        provider, calls = self.provider()
        caught = None
        try:
            self.call(provider, data)
        except Exception as error:
            caught = error
        self.assertEqual(calls, [], "invalid values reached C validation/enqueue")
        self.assertEqual(provider.pending, [])
        self.assertEqual(provider.launches, 0)
        self.assertIsInstance(caught, ValueError)

    def test_direct_nonfinite_activations_never_reach_c(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                data = self.data()
                data["x"][0, 0] = value
                self.reject(data)

    def test_two_rounded_scale_overflows_never_reach_c(self):
        for factor, scale in ((4.0, 0x38), (0.5, 0x7E), (3.9990234375, 0x38)):
            with self.subTest(factor=factor, scale=scale):
                data = self.data()
                data["global_scales"].fill_(factor)
                data["scales"].fill_(scale)
                self.reject(data)

    def test_invalid_global_and_unsigned_scale_contract(self):
        for factor in (0.0, -0.25, float("nan"), float("inf"), -float("inf")):
            with self.subTest(factor=factor):
                data = self.data()
                data["global_scales"].fill_(factor)
                self.reject(data)
        for scale in (0x7F, 0x80, 0xB8, 0xFF):
            with self.subTest(scale=scale):
                data = self.data()
                data["scales"][0] = scale
                self.reject(data)

    def test_valid_finite_boundary_and_underflow_are_admitted(self):
        for factor, scale, activation in (
            (0.25, 0x38, 1.0),
            (3.998046875, 0x38, 65504.0),
            (2.0**-149, 0x7E, -65504.0),
            (0.25, 0x00, -0.0),
            (0.25, 0x01, 2.0**-24),
        ):
            with self.subTest(factor=factor, scale=scale, activation=activation):
                provider, calls = self.provider()
                data = self.data()
                data["global_scales"].fill_(factor)
                data["scales"].fill_(scale)
                data["x"].fill_(activation)
                output = self.call(provider, data)
                self.assertEqual(calls, ["validate", "launch"])
                self.assertEqual(provider.admissions, 1)
                self.assertEqual(provider.launches, 1)
                self.assertEqual(len(provider.pending), 1)
                self.assertEqual(tuple(output.shape), (1, 32))
                # No output or accuracy assertion: the fake enqueue computes nothing.

    def test_every_tile_checked_in_gate_up_and_down_sized_calls(self):
        for n in (32, 64, 128):
            for tile in range(n // 32):
                with self.subTest(n=n, tile=tile):
                    data = self.data(n)
                    data["global_scales"][0, tile] = 4.0
                    self.reject(data)

    def test_same_tensor_pointer_changed_contents_are_revalidated(self):
        for name in ("scales", "global_scales", "x"):
            with self.subTest(name=name):
                provider, calls = self.provider()
                data = self.data()
                self.call(provider, data)
                provider.pending.clear()  # Fake enqueue, no outstanding device work.
                before = data[name].data_ptr()
                data[name].reshape(-1)[0] = (
                    0x80
                    if name == "scales"
                    else 4.0
                    if name == "global_scales"
                    else float("inf")
                )
                self.assertEqual(data[name].data_ptr(), before)
                caught = None
                try:
                    self.call(provider, data)
                except Exception as error:
                    caught = error
                self.assertEqual(calls, ["validate", "launch"])
                self.assertEqual(provider.admissions, 1)
                self.assertEqual(provider.pending, [])
                self.assertEqual(provider.launches, 1)
                self.assertIsInstance(caught, ValueError)

    def test_preflight_refusal_is_not_counted_as_provider_admission(self):
        provider, calls = self.provider()
        data = self.data()
        data["x"][0, 0] = float("inf")

        with self.assertRaisesRegex(ValueError, "nonfinite FP16 provider activations"):
            self.call(provider, data)

        self.assertEqual(provider.admissions, 0)
        self.assertEqual(calls, [])
        self.assertEqual(provider.pending, [])

    def test_opaque_launch_failure_is_counted_before_call_and_retains_pending(self):
        provider, calls = self.provider()

        def fail_after_entry(*_args):
            calls.append("launch-entered")
            raise RuntimeError("opaque provider failure")

        provider.library.vf_qpn2_moe_launch = fail_after_entry
        with self.assertRaisesRegex(RuntimeError, "opaque provider failure"):
            self.call(provider, self.data())

        self.assertEqual(calls, ["validate", "launch-entered"])
        self.assertEqual(provider.admissions, 1)
        self.assertEqual(provider.launches, 0)
        self.assertEqual(len(provider.pending), 1)


if __name__ == "__main__":
    unittest.main()
