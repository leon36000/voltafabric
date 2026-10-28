"""Real CPU C validation plus fake CUDA enqueue: no numerical GPU claim.

The tests catch lost host mirrors/borrowed buffers, accepting invalid coverage,
replay after uncertain enqueue, and releasing buffers after failed completion.
"""

import ctypes as C
from contextlib import nullcontext
import gc
import os
from pathlib import Path
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import weakref

import torch
from adapters.ffi_provider import StrictQPN2Provider
from scripts.hardware_canary import Params

ROOT = Path(__file__).resolve().parents[2]


class Stream:
    cuda_stream = 0

    def __init__(self):
        self.failure = False

    def synchronize(self):
        if self.failure:
            raise RuntimeError("injected async completion error")


class FfiProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = ROOT / ".work/a147-resident-provider"
        directory.mkdir(exist_ok=True, parents=True)
        cls.temp = tempfile.TemporaryDirectory(dir=directory)
        path = Path(cls.temp.name) / "validator.so"
        subprocess.run(
            [
                os.environ.get("CC", "cc"),
                "-shared",
                "-fPIC",
                "-O2",
                "-I",
                str(ROOT / "cuda"),
                str(ROOT / "cuda/qpn2_moe_validate.c"),
                "-o",
                str(path),
            ],
            check=True,
            capture_output=True,
        )
        cls.native = C.CDLL(str(path))
        cls.native.vf_qpn2_moe_validate.argtypes = [C.POINTER(Params)]
        cls.native.vf_qpn2_moe_validate.restype = C.c_int

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def provider(self):
        # Bypass GPU identity constructor only inside this CPU test fixture.
        p = object.__new__(StrictQPN2Provider)
        p.torch, p.device = torch, torch.device("cpu")
        p.pending, p.admissions, p.launches, p.synchronized_launches = [], 0, 0, 0
        p.progress = None
        p.max_resident_bytes, p.max_pending_bytes = 1 << 20, 1 << 16
        p._pending_bytes, p._failed, p._closed = 0, False, False
        p._lock = threading.RLock()
        p.enqueued = []
        p.recorded = []

        def launch(pointer, stream, error):
            params = C.cast(pointer, C.POINTER(Params)).contents
            p.enqueued.append(
                (
                    params.ntpp_host,
                    params.block_expert_host[0],
                    params.sorted_slots_host[0],
                    len(p.pending),
                )
            )
            return 0

        p.library = SimpleNamespace(
            vf_qpn2_moe_validate=self.native.vf_qpn2_moe_validate,
            vf_qpn2_moe_launch=launch,
        )
        return p

    def data(self):
        # Three physical experts is outside the diagnostic canary's tiny E<=2
        # envelope, but is inside the real bounded C ABI contract.
        return dict(
            x=torch.ones((1, 64), dtype=torch.float16),
            codes=torch.full((3 * 64 * 64 // 2,), 0x22, dtype=torch.uint8),
            scales=torch.full((3 * 64 * 64 // 16,), 0x38, dtype=torch.uint8),
            global_scales=torch.full((3, 2), 0.25, dtype=torch.float32),
            expert_blocks=torch.tensor([2], dtype=torch.int32),
            sorted_slots=torch.tensor([0] + [1] * 7, dtype=torch.int32),
            ntpp=torch.tensor(8, dtype=torch.int32),
            s=1,
            n=64,
            k=64,
            in_div=1,
            splitk=4,
            nacc=1,
        )

    def call(self, p, data, stream=None):
        stream = stream or Stream()
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
            patch.object(
                torch.Tensor,
                "record_stream",
                lambda tensor, active: p.recorded.append((id(tensor), active)),
            ),
        ):
            return p.gemm_qpn2_moe(**data)

    def test_constructor_requires_hardware_gate_before_runtime_load(self):
        with self.assertRaises(PermissionError):
            StrictQPN2Provider(
                "absent-library",
                "absent-runtime",
                0,
                "GPU-reserved",
                max_resident_bytes=1024,
                max_pending_bytes=1024,
            )

    def test_external_kernel_registers_all_borrowed_tensor_streams(self):
        p, data, stream = self.provider(), self.data(), Stream()
        self.call(p, data, stream)
        held = p.pending[0][1]
        self.assertEqual(p.recorded, [(id(tensor), stream) for tensor in held])

    def test_real_c_coverage_rejection_never_enqueues(self):
        p, data = self.provider(), self.data()
        data["sorted_slots"][1] = 0
        with self.assertRaisesRegex(ValueError, "C FFI"):
            self.call(p, data)
        self.assertEqual(p.enqueued, [])
        self.assertEqual(p.pending, [])

    def test_real_c_rejects_indices_ntpp_and_misaligned_input(self):
        for fault in (
            "foreign_expert",
            "skipped_active_block",
            "negative_slot",
            "unaligned_ntpp",
            "missing_coverage",
            "misaligned_input",
        ):
            with self.subTest(fault=fault):
                p, data = self.provider(), self.data()
                if fault == "foreign_expert":
                    data["expert_blocks"][0] = 3
                elif fault == "skipped_active_block":
                    data["expert_blocks"][0] = -1
                elif fault == "negative_slot":
                    data["sorted_slots"][0] = -1
                elif fault == "unaligned_ntpp":
                    data["ntpp"].fill_(7)
                elif fault == "missing_coverage":
                    data["ntpp"].fill_(0)
                else:
                    data["x"] = torch.ones((1, 65), dtype=torch.float16)[:, 1:]
                    self.assertTrue(data["x"].is_contiguous())
                    self.assertNotEqual(data["x"].data_ptr() % 16, 0)
                with self.assertRaisesRegex(ValueError, "C FFI"):
                    self.call(p, data)
                self.assertEqual(p.enqueued, [])
                self.assertEqual(p.pending, [])
                self.assertEqual(p.admissions, 0)

    def test_padding_order_permutations_preserve_exact_coverage(self):
        for offset in range(8):
            p, data = self.provider(), self.data()
            data["sorted_slots"].fill_(1)
            data["sorted_slots"][offset] = 0
            self.call(p, data)
            self.assertEqual(
                p.pending[0][1][5].tolist(), [0 if i == offset else 1 for i in range(8)]
            )
            p.close()

    def test_general_projection_uses_exact_owned_metadata_mirrors(self):
        p, data = self.provider(), self.data()
        y = self.call(p, data)
        data["expert_blocks"].fill_(0)
        data["sorted_slots"].fill_(1)
        data["ntpp"].fill_(0)
        self.assertEqual(p.enqueued, [(8, 2, 0, 1)])
        held = p.pending[0][1]
        self.assertEqual(held[4].tolist(), [2])
        self.assertEqual(held[5].tolist(), [0] + [1] * 7)
        self.assertEqual(held[6].item(), 8)
        self.assertEqual(tuple(y.shape), (1, 64))
        self.assertTrue(torch.isnan(y).all())  # Fake kernel computes nothing.
        p.synchronize()
        self.assertEqual(p.synchronized_launches, 1)
        self.assertEqual(p.pending, [])

    def test_buffers_survive_caller_drop_until_completion(self):
        p, data = self.provider(), self.data()
        borrowed = weakref.ref(data["x"])
        self.call(p, data)
        del data
        gc.collect()
        self.assertIsNotNone(borrowed())
        p.synchronize()
        gc.collect()
        self.assertIsNone(borrowed())

    def test_failed_completion_retains_buffers_and_blocks_new_execution(self):
        p, data, stream = self.provider(), self.data(), Stream()
        self.call(p, data, stream)
        stream.failure = True
        with self.assertRaisesRegex(RuntimeError, "completion"):
            p.synchronize()
        self.assertEqual(len(p.pending), 1)
        self.assertEqual(p.synchronized_launches, 0)
        with self.assertRaisesRegex(RuntimeError, "failed|unknown"):
            self.call(p, data)
        self.assertEqual(len(p.enqueued), 1)

    def test_uncertain_enqueue_is_not_replayed_and_can_be_drained(self):
        p, data = self.provider(), self.data()
        p.library.vf_qpn2_moe_launch = lambda *_: -8
        with self.assertRaisesRegex(RuntimeError, "launch"):
            self.call(p, data)
        self.assertEqual(len(p.pending), 1)
        self.assertEqual(p.admissions, 1)
        with self.assertRaisesRegex(RuntimeError, "failed|unknown"):
            self.call(p, data)
        p.synchronize()
        self.assertEqual(p.pending, [])
        with self.assertRaises(RuntimeError):
            self.call(p, data)

    def test_nonfinite_and_mutated_weight_values_are_revalidated(self):
        for field, value in [
            ("x", float("inf")),
            ("scales", 0x80),
            ("global_scales", 4.0),
        ]:
            with self.subTest(field=field):
                p, data = self.provider(), self.data()
                self.call(p, data)
                p.synchronize()
                data[field].reshape(-1)[0] = value
                with self.assertRaises(ValueError):
                    self.call(p, data)
                self.assertEqual(len(p.enqueued), 1)
                self.assertEqual(p.pending, [])

    def test_explicit_memory_budgets_reject_before_output_allocation(self):
        for budget in ["max_resident_bytes", "max_pending_bytes"]:
            with self.subTest(budget=budget):
                p, data = self.provider(), self.data()
                setattr(p, budget, 1)
                with patch.object(
                    torch, "full", side_effect=AssertionError("allocation")
                ):
                    with self.assertRaisesRegex(ValueError, "budget"):
                        self.call(p, data)
                self.assertEqual(p.pending, [])

    def test_pending_budget_is_released_only_by_successful_sync(self):
        p, data = self.provider(), self.data()
        # Exact workspace: FP16 input/output (128+128), metadata (4+32+4).
        p.max_pending_bytes = 296
        self.call(p, data)
        with self.assertRaisesRegex(ValueError, "budget"):
            self.call(p, data)
        p.synchronize()
        self.call(p, data)
        self.assertEqual(len(p.enqueued), 2)

    def test_close_drains_and_permanently_refuses_launches(self):
        p, data = self.provider(), self.data()
        self.call(p, data)
        p.close()
        self.assertEqual(p.pending, [])
        with self.assertRaisesRegex(RuntimeError, "closed"):
            self.call(p, data)


if __name__ == "__main__":
    unittest.main()
