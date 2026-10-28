"""Real Rust binary export -> strict CPU reader -> existing H1 provider contract."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from scripts import hardware_canary as canary
from tests.adapter.test_canary_integration import CPUProvider

BINARY = Path(os.environ["VF_BINARY"])
ROOT = Path(__file__).resolve().parents[2]
RUN_TMP = Path(os.environ.get("TMPDIR", str(ROOT / "evidence/adapter-temp")))


def stub_cuda_proof(directory, library_sha256):
    """Minimal CUDA_COMPILED receipt matching the stubbed library hash."""
    path = Path(directory) / "CUDA_COMPILED.json"
    path.write_text(
        json.dumps(
            {
                "status": "CUDA_COMPILED",
                "architecture": "sm_70",
                "device_executed": False,
                "hashes": {"build/libvf_qpn2_moe_sm70.so": library_sha256},
            }
        )
    )
    return str(path)


class BinaryCanaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUN_TMP.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=RUN_TMP, prefix="a19-canary-")
        self.directory = Path(self.temp.name)
        self.problem = canary.routing_payload(2)
        self.x = torch.tensor([[2**-6] * 64, [2**-7] * 64], dtype=torch.float16)

    def tearDown(self):
        self.temp.cleanup()

    def contexts(self):
        return canary.prepare_rust_canary(
            BINARY, self.problem, resident_dir=self.directory
        )

    def test_temporaries_stay_in_configured_or_project_tmpdir(self):
        self.assertTrue(self.directory.resolve().is_relative_to(RUN_TMP.resolve()))
        configured = os.environ.get("TMPDIR")
        expected = Path(configured) if configured else ROOT
        self.assertTrue(RUN_TMP.resolve().is_relative_to(expected.resolve()))

    def test_broken_receipt_symlink_blocks_claim_without_creating_its_target(self):
        receipt = self.directory / "prior.json"
        missing = self.directory / "missing-prior.json"
        receipt.symlink_to(missing)
        with patch.object(canary, "ROOT", self.directory):
            with self.assertRaises(FileExistsError):
                canary.Receipt(receipt, {"status": "CPU_TEST"})
        self.assertTrue(receipt.is_symlink())
        self.assertFalse(missing.exists())

    def test_bundle_receipts_use_explicit_external_root_and_reject_escape(self):
        bundle = self.directory / "bundle"
        bundle.mkdir()
        receipts = self.directory / "receipts"
        receipts.mkdir()
        with (
            patch.object(canary, "ROOT", bundle),
            patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(receipts)}),
        ):
            target = receipts / "hardware.json"
            self.assertEqual(canary.output_path(target), target)
            canary.Receipt(target, {"status": "CPU_TEST"})
            with self.assertRaises(ValueError):
                canary.output_path(self.directory / "escape.json")

    def test_binary_mode_requires_export_capability_before_provider(self):
        from types import SimpleNamespace

        output = self.directory / "old-binary.json"
        argv = [
            "--ack-hardware-gate",
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            str(self.directory / "unused.so"),
            "--cudart",
            str(self.directory / "unused-runtime.so"),
            "--routing-binary",
            str(BINARY),
            "--out",
            str(output),
            "--resident-io",
            "safetensors",
            "--cuda-proof",
            stub_cuda_proof(self.directory, "cpu-test"),
        ]
        old_capabilities = dict(
            ok=True,
            result=dict(
                owned_quantized_buffers=True,
                host_launch_metadata=True,
                compact_output_receipts=True,
            ),
        )
        response = SimpleNamespace(returncode=0, stdout=json.dumps(old_capabilities))
        with (
            patch.object(canary, "ROOT", self.directory),
            patch.object(canary, "sha", return_value="cpu-test"),
            patch.object(
                canary, "check_sample", return_value={"source": {"cpu_test": True}}
            ),
            patch.object(canary.subprocess, "run", return_value=response),
            patch.object(
                canary, "QPN2Provider", side_effect=RuntimeError("CPU stop")
            ) as provider,
        ):
            self.assertEqual(canary.main(argv), 2)
        provider.assert_not_called()
        self.assertIn(
            "export capability", json.loads(output.read_text())["error"]["message"]
        )

    def test_main_binary_mode_claims_private_attempt_directory_before_provider(self):
        output = self.directory / "h1.json"
        argv = [
            "--ack-hardware-gate",
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            str(self.directory / "unused.so"),
            "--cudart",
            str(self.directory / "unused-runtime.so"),
            "--routing-binary",
            str(BINARY),
            "--out",
            str(output),
            "--resident-io",
            "safetensors",
            "--cuda-proof",
            stub_cuda_proof(self.directory, "cpu-test"),
        ]
        with (
            patch.object(canary, "ROOT", self.directory),
            patch.object(canary, "sha", return_value="cpu-test"),
            patch.object(
                canary, "check_sample", return_value={"source": {"cpu_test": True}}
            ),
            patch.object(
                canary, "QPN2Provider", side_effect=RuntimeError("CPU stop before CUDA")
            ) as provider,
        ):
            self.assertEqual(canary.main(argv), 2)
        self.assertEqual(provider.call_count, 1)
        report = json.loads(output.read_text())
        self.assertEqual(report["resident_io"], "safetensors")
        self.assertFalse(report["cuda_executed"])
        self.assertEqual(report["status"], "BLOCKED")
        artifact_directory = Path(report["resident_artifact_directory"])
        self.assertEqual(artifact_directory.parent, self.directory)
        self.assertEqual(artifact_directory.stat().st_mode & 0o777, 0o700)

    def test_binary_all_ranks_and_replica_selection_equal_independent_oracle(self):
        for preferred in [2, 9]:
            self.problem = canary.routing_payload(preferred)
            contexts = self.contexts()
            provider = CPUProvider()
            completions = []
            for context in reversed(contexts):
                completions.extend(
                    canary.execute_rust_rank(provider, BINARY, context, self.x)
                )
            result = canary.rust_call(
                BINARY,
                "collect-routing",
                dict(problem=self.problem, completions=completions),
            )
            self.assertEqual(result, canary.full_uniform_moe_oracle(self.problem))
            self.assertEqual(provider.launches, 4)
            self.assertEqual(provider.synchronized_launches, 4)
            self.assertEqual(provider.sync_count, 2)
            self.assertEqual(
                len(list(self.directory.glob("*.safetensors"))),
                6 * (1 if preferred == 2 else 2),
            )

    def test_json_reference_and_binary_metadata_and_bytes_match(self):
        reference = canary.prepare_rust_canary(BINARY, self.problem)
        for legacy, binary in zip(reference, self.contexts()):
            artifact = binary["artifact"]
            self.assertEqual(binary["plan"], legacy["plan"])
            for name, tensor in artifact.tensors.items():
                dtype = torch.float32 if name.startswith("gscales") else torch.uint8
                expected = torch.tensor(legacy["stack"][name], dtype=dtype).reshape(
                    tensor.shape
                )
                self.assertTrue(
                    torch.equal(
                        tensor.view(torch.uint8).reshape(-1),
                        expected.view(torch.uint8).reshape(-1),
                    ),
                    name,
                )
            for key, value in artifact.metadata.items():
                self.assertEqual(value, legacy["stack"][key])

    def test_changes_to_prepared_weights_metadata_and_tensors_fail_before_provider(
        self,
    ):
        for change in ["weights", "metadata", "dtype", "tensor", "epoch"]:
            context = self.contexts()[0]
            if change == "weights":
                context["launch"]["weights"]["experts"][0]["gate"]["codes"][0] ^= 1
            elif change == "metadata":
                context["stack"]["rank"] = float(context["stack"]["rank"])
            elif change == "dtype":
                context["artifact"].tensors["codes13"] = (
                    context["artifact"].tensors["codes13"].float()
                )
            elif change == "tensor":
                context["artifact"].tensors["codes13"][0] ^= 1
            else:
                context["launch"]["expected_stamp"]["epoch"] += 1
            provider = CPUProvider()
            with self.assertRaises((RuntimeError, ValueError)):
                canary.execute_rust_rank(provider, BINARY, context, self.x)
            self.assertEqual(provider.launches, 0, change)

    def test_mutation_after_load_is_not_hidden_by_a_tensor_pointer_cache(self):
        context = self.contexts()[0]
        tensor = context["artifact"].tensors["gscales13"]
        pointer = tensor.data_ptr()
        tensor[0, 0] = 4
        self.assertEqual(pointer, tensor.data_ptr())
        provider = CPUProvider()
        with self.assertRaises((RuntimeError, ValueError)):
            canary.execute_rust_rank(provider, BINARY, context, self.x)
        self.assertEqual(provider.launches, 0)

    def test_binary_invalid_cast_and_swiglu_never_publish_completions(self):
        context = self.contexts()[0]
        provider = CPUProvider()
        with self.assertRaisesRegex(ValueError, "FP16 compact input overflow"):
            canary.execute_rust_rank(
                provider, BINARY, context, torch.full((2, 64), 100000.0)
            )
        self.assertEqual(provider.launches, 0)
        provider = CPUProvider()

        def overflow(*args):
            provider.launches += 1
            return torch.full((args[7], args[8]), 300.0, dtype=torch.float16)

        provider.gemm_qpn2_moe = overflow
        commands = []
        original = canary.rust_call

        def observed(binary, command, request):
            commands.append(command)
            return original(binary, command, request)

        with patch.object(canary, "rust_call", side_effect=observed):
            with self.assertRaisesRegex(ValueError, "nonfinite SwiGLU"):
                canary.execute_rust_rank(provider, BINARY, context, self.x)
        self.assertEqual(provider.launches, 1)
        self.assertNotIn("resident-completions", commands)


if __name__ == "__main__":
    unittest.main()
