"""CPU contracts for durable H1 admission and Rust-driven canary cases."""

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid
import torch
from scripts import hardware_canary as canary
from tests.adapter.test_skinny_adapter import UniformPackedCPUOracle

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ["VF_BINARY"])


class ReceiptTests(unittest.TestCase):
    def path(self):
        p = ROOT / "evidence/canary-integration" / (uuid.uuid4().hex + ".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def test_exclusive_claim_never_replaces_prior_unknown_receipt(self):
        p = self.path()
        p.write_text('{"status":"ENQUEUE_PENDING","unknown":true}')
        with self.assertRaises(FileExistsError):
            canary.Receipt(p, dict(status="STARTING"))
        self.assertEqual(json.loads(p.read_text())["status"], "ENQUEUE_PENDING")

    def test_atomic_progress_preserves_attempt_identity(self):
        p = self.path()
        r = canary.Receipt(p, dict(status="STARTING"))
        first = json.loads(p.read_text())
        r.update(
            dict(
                status="RUNNING",
                execution_phase="ENQUEUE_PENDING",
                may_have_executed=True,
            )
        )
        second = json.loads(p.read_text())
        self.assertEqual(first["execution_id"], second["execution_id"])
        self.assertTrue(second["may_have_executed"])

    def test_failed_atomic_replace_keeps_prior_readable_receipt(self):
        p = self.path()
        r = canary.Receipt(p, dict(status="STARTING"))
        before = p.read_text()
        with patch.object(canary.os, "replace", side_effect=OSError("injected")):
            with self.assertRaises(OSError):
                r.update(dict(status="RUNNING"))
        self.assertEqual(p.read_text(), before)

    def test_foreign_receipt_cannot_be_updated(self):
        p = self.path()
        r = canary.Receipt(p, dict(status="STARTING"))
        p.write_text('{"execution_id":"foreign","status":"UNKNOWN"}')
        with self.assertRaises(RuntimeError):
            r.update(dict(status="PASS"))
        self.assertEqual(json.loads(p.read_text())["status"], "UNKNOWN")

    def test_two_real_process_claims_have_one_winner(self):
        p = self.path()
        code = "from pathlib import Path; from scripts.hardware_canary import Receipt; import sys;\ntry: Receipt(Path(sys.argv[1]),{'status':'STARTING'})\nexcept FileExistsError: sys.exit(2)\n"
        children = [
            subprocess.Popen(
                [sys.executable, "-c", code, str(p)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            for _ in range(2)
        ]
        results = []
        for child in children:
            stdout, stderr = child.communicate(timeout=20)
            results.append(child.returncode)
            self.assertIn(child.returncode, [0, 2], stderr.decode())
        self.assertEqual(sorted(results), [0, 2])
        self.assertEqual(json.loads(p.read_text())["status"], "STARTING")

    def test_crash_after_persisted_pending_blocks_replay(self):
        p = self.path()
        code = "from pathlib import Path; from scripts.hardware_canary import Receipt; import os,sys; r=Receipt(Path(sys.argv[1]),{'status':'STARTING'}); r.update({'status':'RUNNING','execution_phase':'ENQUEUE_PENDING','may_have_executed':True,'cpu_fault_injection':True}); os._exit(23)"
        result = subprocess.run(
            [sys.executable, "-c", code, str(p)],
            cwd=ROOT,
            capture_output=True,
            timeout=20,
        )
        self.assertEqual(result.returncode, 23)
        self.assertEqual(
            json.loads(p.read_text())["execution_phase"], "ENQUEUE_PENDING"
        )
        with self.assertRaises(FileExistsError):
            canary.Receipt(p, dict(status="STARTING"))

    def test_sync_failure_keeps_pending_lifetimes_and_uncertain_receipt(self):
        p = self.path()
        r = canary.Receipt(p, dict(status="STARTING"))

        class Stream:
            def synchronize(self):
                raise RuntimeError("injected async failure")

        provider = object.__new__(canary.QPN2Provider)
        provider.pending = [(Stream(), object())]
        provider.launches = 1
        provider.synchronized_launches = 0
        provider.progress = lambda event: r.update(dict(status="RUNNING", **event))
        r.update(
            dict(status="RUNNING", execution_phase="ENQUEUED", may_have_executed=True)
        )
        with self.assertRaises(RuntimeError):
            provider.synchronize()
        self.assertEqual(len(provider.pending), 1)
        self.assertEqual(provider.synchronized_launches, 0)
        self.assertTrue(json.loads(p.read_text())["may_have_executed"])

    def test_cleanup_progress_cannot_promote_failed_attempt(self):
        from types import SimpleNamespace

        p = self.path()
        report = dict(
            status="FAIL",
            cuda_executed=False,
            error="injected",
            cpu_fault_injection=True,
        )
        receipt = canary.Receipt(p, report)
        canary.record_progress(
            report,
            receipt,
            SimpleNamespace(launches=0),
            dict(execution_phase="SYNCHRONIZED", may_have_executed=False),
        )
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(json.loads(p.read_text())["status"], "FAIL")

    def test_successful_sync_releases_only_after_stream_completion(self):
        p = self.path()
        receipt = canary.Receipt(p, dict(status="STARTING", cpu_fault_injection=True))
        visits = []

        class Stream:
            def synchronize(self):
                visits.append("completed")

        provider = object.__new__(canary.QPN2Provider)
        provider.pending = [(Stream(), object()), (Stream(), object())]
        provider.launches = 2
        provider.synchronized_launches = 0
        provider.progress = lambda event: receipt.update(
            dict(status="RUNNING", **event)
        )
        provider.synchronize()
        self.assertEqual(visits, ["completed", "completed"])
        self.assertEqual(provider.pending, [])
        self.assertEqual(provider.synchronized_launches, 2)
        self.assertFalse(json.loads(p.read_text())["may_have_executed"])


class CPUProvider(UniformPackedCPUOracle):
    """Explicit existing CPU oracle, never selected automatically."""

    def __init__(self):
        super().__init__()
        self.torch = torch
        self.device = torch.device("cpu")
        self.launches = 0
        self.synchronized_launches = 0
        self.sync_count = 0

    def gemm_qpn2_moe(self, *args):
        out = super().gemm_qpn2_moe(*args)
        self.launches += 1
        return out

    def synchronize(self):
        self.sync_count += 1
        self.synchronized_launches = self.launches


class RustCanaryTests(unittest.TestCase):
    def test_all_ranks_use_owned_compressed_weights_prepared_slots_and_receipts(self):
        for preferred in [2, 9]:
            problem = canary.routing_payload(preferred)
            contexts = canary.prepare_rust_canary(BINARY, problem)
            provider = CPUProvider()
            x = torch.tensor([[2**-6] * 64, [2**-7] * 64], dtype=torch.float16)
            receipts = []
            for context in reversed(contexts):
                rows = canary.execute_rust_rank(provider, BINARY, context, x)
                receipts.extend(rows)
                plan = context["plan"]
                if plan["skip"]:
                    self.assertEqual(rows, [])
                else:
                    self.assertEqual(len(rows), len(plan["completion_slots"]))
            result = canary.rust_call(
                BINARY, "collect-routing", dict(problem=problem, completions=receipts)
            )
            self.assertEqual(result, canary.full_uniform_moe_oracle(problem))
            self.assertEqual(provider.launches, 4)
            self.assertEqual(provider.synchronized_launches, 4)
            self.assertEqual(provider.sync_count, 2)

    def test_bad_full_stamp_is_rejected_before_provider_or_tensor_work(self):
        problem = canary.routing_payload(2)
        contexts = canary.prepare_rust_canary(BINARY, problem)
        context = contexts[0]
        context["launch"]["expected_stamp"]["batch"] += 1
        provider = CPUProvider()
        with self.assertRaises(RuntimeError):
            canary.execute_rust_rank(
                provider, BINARY, context, torch.ones((2, 64), dtype=torch.float16)
            )
        self.assertEqual(provider.launches, 0)

    def test_sync_failure_never_emits_rust_completion_receipts(self):
        problem = canary.routing_payload(2)
        context = canary.prepare_rust_canary(BINARY, problem)[0]
        provider = CPUProvider()
        provider.synchronize = lambda: (_ for _ in ()).throw(
            RuntimeError("injected sync failure")
        )
        commands = []
        original = canary.rust_call

        def observed(binary, command, value):
            commands.append(command)
            return original(binary, command, value)

        with patch.object(canary, "rust_call", side_effect=observed):
            with self.assertRaisesRegex(RuntimeError, "injected sync failure"):
                canary.execute_rust_rank(
                    provider,
                    BINARY,
                    context,
                    torch.full((2, 64), 2**-6, dtype=torch.float16),
                )
        self.assertNotIn("resident-completions", commands)

    def test_overflowing_uniform_input_never_becomes_a_completion(self):
        context = canary.prepare_rust_canary(BINARY, canary.routing_payload(2))[0]
        with self.assertRaisesRegex(AssertionError, "Oracle ABI did not write"):
            canary.execute_rust_rank(
                CPUProvider(), BINARY, context, torch.ones((2, 64), dtype=torch.float16)
            )
