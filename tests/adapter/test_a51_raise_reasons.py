"""A51 raise-reason sweep: each refusal below was reached by a test that checked only
the exception class, or by none, so replacing its message went unnoticed.

Every test drives one refusal of the H1-path Python and pins its class and its whole
message (anchored), plus the PreflightError status where the preflight sets one. The
fixtures are borrowed from the A43 refusal suite and the adapter tests by module
attribute, so none of their test cases is collected twice. CPU only.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from adapters import skinny_moe as adapter  # noqa: E402
from scripts import full_expert_canary as full_canary  # noqa: E402
from scripts import h1_preflight  # noqa: E402
from scripts import hardware_canary as canary  # noqa: E402

import test_a43_preflight_refusals as a43p  # noqa: E402
import test_skinny_adapter as skinny  # noqa: E402


def exact(message: str) -> str:
    return "^" + re.escape(message) + "$"


class PreflightReasons(a43p.Bundle):
    def test_source_file_differing_from_the_cpu_proof_names_it(self) -> None:
        def stale(proof):
            proof["source_sha256"]["Cargo.toml"] = "0" * 64

        self.refused(
            exact("source differs from CPU-qualified revision: Cargo.toml"), cpu=stale
        )

    def test_cuda_source_differing_from_the_compilation_receipt_names_it(self) -> None:
        def stale(compiled):
            compiled["hashes"]["cuda/qpn2_moe_sm70.cu"] = "0" * 64

        self.refused(
            exact(
                "CUDA source differs from compilation receipt: cuda/qpn2_moe_sm70.cu"
            ),
            cuda=stale,
        )

    def test_existing_output_receipt_is_never_overwritten(self) -> None:
        out = self.root / "receipt.json"
        out.write_text("prior\n")
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            code = h1_preflight.main(["--bundle", str(self.root), "--out", str(out)])
        self.assertEqual(code, 2)
        self.assertEqual(
            stderr.getvalue(), f"refusing to overwrite output receipt: {out}\n"
        )
        self.assertEqual(out.read_text(), "prior\n")


class CanaryReceiptReasons(unittest.TestCase):
    def setUp(self) -> None:
        # Receipts must stay below the receipt root; make it this test's directory.
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        root = patch.dict(os.environ, VF_H1_RECEIPT_DIR=str(self.dir))
        root.start()
        self.addCleanup(root.stop)

    def test_prior_receipt_path_is_refused(self) -> None:
        path = self.dir / "receipt.json"
        path.write_text("{}\n")
        with self.assertRaisesRegex(
            FileExistsError,
            exact("prior receipt path exists; inspect it before a new attempt"),
        ):
            canary.Receipt(path, {"status": "STARTING"})
        self.assertEqual(path.read_text(), "{}\n")

    def test_receipt_taken_over_by_another_execution_is_not_replaced(self) -> None:
        path = self.dir / "receipt.json"
        receipt = canary.Receipt(path, {"status": "STARTING"})
        foreign = {"status": "STARTING", "execution_id": "someone-else"}
        path.write_text(json.dumps(foreign))
        with self.assertRaisesRegex(
            RuntimeError, exact("H1 receipt ownership changed; refuse replacement")
        ):
            receipt.update({"status": "DONE"})
        self.assertEqual(json.loads(path.read_text()), foreign)


class CanaryProviderReasons(unittest.TestCase):
    """Direct gemm_qpn2_moe admission with CPU tensors, as in the A43 canary suite."""

    def provider(self):
        p = object.__new__(canary.QPN2Provider)
        p.torch, p.device = torch, torch.device("cpu")
        p.pending, p.launches, p.synchronized_launches = [], 0, 0
        p.progress = None
        p.library = SimpleNamespace(
            vf_qpn2_moe_validate=lambda *_: self.fail("validated"),
            vf_qpn2_moe_launch=lambda *_: self.fail("launched"),
        )
        return p

    def data(self, **changes):
        data = dict(
            x=torch.ones((1, 64), dtype=torch.float16),
            codes=torch.full((32 * 32,), 0x22, dtype=torch.uint8),
            scales=torch.full((32 * 4,), 0x38, dtype=torch.uint8),
            global_scales=torch.full((1, 1), 0.25, dtype=torch.float32),
            expert_blocks=torch.tensor([0], dtype=torch.int32),
            sorted_slots=torch.tensor([0] + [1] * 7, dtype=torch.int32),
            ntpp=torch.tensor(8, dtype=torch.int32),
            s=1,
            n=32,
            k=64,
            in_div=1,
            splitk=4,
            nacc=1,
        )
        data.update(changes)
        return data

    def refuse(self, data, message):
        with (
            patch.object(
                torch.cuda,
                "current_stream",
                return_value=SimpleNamespace(cuda_stream=0),
            ),
            patch.object(torch.cuda, "device", return_value=contextlib.nullcontext()),
        ):
            with self.assertRaisesRegex(ValueError, exact(message)):
                self.provider().gemm_qpn2_moe(**data)

    def test_provider_without_the_hardware_gate_acknowledgement(self) -> None:
        with self.assertRaisesRegex(
            PermissionError, exact("--ack-hardware-gate is required")
        ):
            canary.QPN2Provider("lib.so", "cudart.so", 0, "GPU-uuid")

    def test_shape_outside_the_tiny_crop_and_the_full_expert_mode(self) -> None:
        # Nine rows: beyond the tiny crop, and full-expert mode is not enabled.
        self.refuse(
            self.data(s=9),
            "H1 requires the tiny crop or explicit bounded full-expert mode",
        )

    def test_nonfinite_activation(self) -> None:
        x = torch.ones((1, 64), dtype=torch.float16)
        x[0, 5] = float("nan")
        self.refuse(self.data(x=x), "nonfinite FP16 provider activations")


class CanaryOracleReasons(unittest.TestCase):
    def test_nonfinite_e4m3_scale_byte(self) -> None:
        for byte in (0x7F, 0xFF):
            with self.subTest(byte=byte):
                with self.assertRaisesRegex(ValueError, exact("nonfinite E4M3 scale")):
                    canary.fp8_e4m3(byte)

    def test_comparison_without_stream_synchronization(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            exact("GPU comparison requires explicit successful stream synchronization"),
        ):
            canary.compare([[1.0]], [[1.0]], dict(atol=0.0, rtol=0.0))

    def test_tolerance_failure_reports_both_values(self) -> None:
        with self.assertRaisesRegex(
            AssertionError, exact("fixed predeclared tolerance exceeded: 1.5 vs 2")
        ):
            canary.compare(
                [[1.5]], [[2.0]], dict(atol=0.25, rtol=0.0), synchronized=True
            )

    def test_full_expert_half_overflow(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            exact("nonfinite half intermediate in strict full-expert reference"),
        ):
            full_canary._half(1e6)


class AdapterReasons(unittest.TestCase):
    def select(self, message, topk, expert_map=None, mask=None, local=2):
        with self.assertRaisesRegex(ValueError, exact(message)):
            adapter._select_local_slots(
                torch.tensor(topk),
                None if expert_map is None else torch.tensor(expert_map),
                local,
                None if mask is None else torch.tensor(mask),
            )

    def test_sm70_scale_byte_at_or_above_0x7f(self) -> None:
        scales = torch.full((128,), 0x38, dtype=torch.uint8)
        scales[17] = 0x7F
        with self.assertRaisesRegex(
            ValueError, exact("SM70 group scales must be unsigned finite E4M3")
        ):
            adapter._validate_sm70_scale_domain(
                scales, torch.ones((1, 1), dtype=torch.float32), 64
            )

    def test_local_slot_selection_refusals(self) -> None:
        self.select("negative global expert ID", [[-1]])
        self.select("global expert outside unpartitioned resident stack", [[2]])
        self.select("expert_map physical index outside resident stack", [[0]], [0, 2])
        self.select("unknown global expert ID", [[2]], [0, 1])
        self.select("distinct global experts alias one physical stack", [[0]], [0, 0])
        self.select(
            "selected slot has no local resident replica",
            [[1]],
            [0, -1],
            mask=[[True]],
        )

    def apply(self, layer, weights=None):
        return adapter.moe_apply(
            layer,
            torch.ones((1, 64)),
            torch.ones((1, 1)) if weights is None else weights,
            torch.tensor([[0]]),
            None,
            None,
            kernel=skinny.UniformPackedCPUOracle(),
            align=skinny.align_reference,
        )

    def test_router_weight_on_input_is_refused(self) -> None:
        layer = skinny.layer()
        layer.apply_router_weight_on_input = True
        with self.assertRaisesRegex(
            ValueError, exact("Skinny applies router weights on output")
        ):
            self.apply(layer)

    def test_partitioned_layer_without_an_owner_mask_is_refused(self) -> None:
        layer = skinny.layer()
        layer.expert_map = torch.tensor([0, 1])
        with self.assertRaisesRegex(
            ValueError,
            exact("partitioned execution requires an explicit dispatch owner mask"),
        ):
            self.apply(layer)

    def test_weighted_combine_overflowing_fp32_is_refused(self) -> None:
        # A finite, nonnegative coefficient whose product with the expert output
        # exceeds FP32: admission passes, the combine itself overflows.
        layer = skinny.layer()
        self.assertTrue(bool(self.apply(layer).abs().max() > 1))
        with self.assertRaisesRegex(
            ValueError, exact("nonfinite FP32 combine or output cast")
        ):
            self.apply(layer, torch.full((1, 1), 3e38))


if __name__ == "__main__":
    unittest.main()
