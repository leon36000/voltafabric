"""CPU-only H1 ABI/provider contracts; no CUDA runtime library is loaded."""

import ctypes as C
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import unittest
import uuid
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ["VF_BINARY"])
spec = importlib.util.spec_from_file_location(
    "hardware_canary", ROOT / "scripts/hardware_canary.py"
)
canary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canary)


class HardwareContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence = ROOT / "evidence/hardware-contract"
        cls.evidence.mkdir(parents=True, exist_ok=True)
        cls.validator = canary.CPUValidator(ROOT, cls.evidence)

    def params(self):
        dimensions = dict(
            experts=2, n=32, k=64, slots=4, input_rows=2, in_div=2, splitk=2, nacc=1
        )
        lengths = dict(
            codes=2048,
            scales=256,
            gscales=2,
            x=128,
            y=128,
            block_expert=2,
            sorted_slots=16,
            ntpp=1,
        )
        buffers = {
            name: (0x100000 + index * 0x100000, count)
            for index, (name, count) in enumerate(lengths.items())
        }
        return canary.make_params(
            dimensions=dimensions,
            buffers=buffers,
            expert_blocks=[0, 1],
            sorted_slots=[0, 2] + [4] * 6 + [1, 3] + [4] * 6,
            ntpp=16,
        )

    def test_bounded_full_expert_shape_admits_only_declared_geometries(self):
        base = dict(
            full_expert_enabled=True,
            deepseek_probe_enabled=True,
            experts=1,
            slots=2,
            n=2304,
            k=5120,
            in_div=1,
            splitk=4,
            nacc=1,
            input_rows=2,
            expert_blocks=1,
        )

        def allowed(**changes):
            return canary.bounded_full_expert_shape(**(base | changes))

        self.assertTrue(allowed())  # gate
        self.assertTrue(allowed(n=5120, k=2304))  # direct down
        self.assertTrue(allowed(n=4608, k=5120, splitk=8))  # combined gate/up
        self.assertFalse(allowed(n=4608, k=5120, splitk=4))
        self.assertFalse(allowed(n=5120, k=2304, splitk=8))
        self.assertTrue(allowed(n=5120, k=2304, splitk=8, adapter_layer_enabled=True))
        self.assertFalse(allowed(n=5120, k=2304, splitk=8, adapter_layer_enabled=1))
        self.assertFalse(allowed(n=5120, k=2368, splitk=8, adapter_layer_enabled=True))
        self.assertFalse(allowed(full_expert_enabled=1))
        self.assertFalse(allowed(deepseek_probe_enabled=1))
        for changes in (
            {"experts": 2},
            {"slots": 3, "input_rows": 3},
            {"input_rows": 1},
            {"in_div": 2},
            {"nacc": 2},
            {"expert_blocks": 2},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(allowed(**changes))

        legacy = dict(
            full_expert_enabled=True,
            deepseek_probe_enabled=False,
            experts=1,
            slots=6,
            n=512,
            k=2048,
            in_div=1,
            splitk=4,
            nacc=1,
            input_rows=6,
            expert_blocks=1,
        )
        self.assertTrue(canary.bounded_full_expert_shape(**legacy))
        self.assertFalse(canary.bounded_full_expert_shape(**(legacy | {"slots": 2})))

    def test_a133_admits_only_exact_six_expert_topk6_projections(self):
        base = dict(
            full_expert_enabled=True,
            deepseek_probe_enabled=True,
            adapter_layer_enabled=True,
            experts=6,
            slots=12,
            n=4608,
            k=5120,
            in_div=1,
            splitk=8,
            nacc=1,
            input_rows=12,
            expert_blocks=6,
        )
        self.assertTrue(canary.bounded_full_expert_shape(**base))
        self.assertTrue(
            canary.bounded_full_expert_shape(**(base | {"n": 5120, "k": 2304}))
        )
        for changes in (
            {"experts": 5},
            {"slots": 11},
            {"input_rows": 6},
            {"expert_blocks": 5},
            {"in_div": 2},
            {"splitk": 4},
            {"n": 5120, "k": 4608},
            {"adapter_layer_enabled": False},
            {"deepseek_probe_enabled": False},
            {"full_expert_enabled": False},
        ):
            with self.subTest(changes=changes):
                self.assertFalse(canary.bounded_full_expert_shape(**(base | changes)))

    def test_ctypes_size_and_every_offset_equal_compiled_c_header(self):
        source = self.evidence / "abi_probe.c"
        executable = self.evidence / "abi_probe"
        fields = [name for name, _ in canary.Params._fields_]
        body = [
            "#include <stdio.h>",
            "#include <stddef.h>",
            '#include "../../cuda/qpn2_moe_ffi.h"',
            "int main(void) {",
            'printf("{\\"sizeof\\":%zu,\\"offsets\\":{", sizeof(vf_qpn2_moe_params));',
        ]
        for index, name in enumerate(fields):
            body.append(
                'printf("%s\\"%s\\":%%zu", offsetof(vf_qpn2_moe_params, %s));'
                % ("," if index else "", name, name)
            )
        body.extend(['puts("}}");', "return 0;", "}"])
        source.write_text("\n".join(body) + "\n")
        run = subprocess.run(
            [
                "gcc",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-pedantic",
                str(source),
                "-o",
                str(executable),
            ],
            text=True,
            capture_output=True,
            timeout=30,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        actual = json.loads(subprocess.check_output([str(executable)], text=True))
        self.assertEqual(actual, canary.abi_layout())
        (self.evidence / "abi-layout.json").write_text(
            json.dumps(actual, indent=2) + "\n"
        )

    def test_provider_requires_gate_before_import_or_loading_cuda(self):
        with patch.object(
            canary.C, "CDLL", side_effect=AssertionError("runtime loaded")
        ):
            with self.assertRaises(PermissionError):
                canary.QPN2Provider("unused.so", "unused-cudart.so", 0, "GPU-unused")

    def test_gpu_provider_rejects_torch_without_sm70_arch_before_runtime_load(self):
        from types import SimpleNamespace
        import torch

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(major=7, minor=0),
            ),
            patch.object(torch.cuda, "get_arch_list", return_value=["sm_80"]),
            patch.object(torch.version, "cuda", "12.8"),
            patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")),
        ):
            with self.assertRaisesRegex(RuntimeError, "SM70"):
                canary.QPN2Provider(
                    "unused.so",
                    "unused-cudart.so",
                    0,
                    "GPU-unused",
                    ack_hardware_gate=True,
                )

    def test_gpu_provider_rejects_cuda13_torch_without_runtime_load(self):
        import torch

        with (
            patch.object(torch.cuda, "is_available", return_value=True),
            patch.object(torch.version, "cuda", "13.0"),
            patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA 12"):
                canary.QPN2Provider(
                    "unused.so",
                    "unused-cudart.so",
                    0,
                    "GPU-unused",
                    ack_hardware_gate=True,
                )

    def test_output_stays_below_resolved_project_root(self):
        self.assertEqual(
            canary.output_path(ROOT / "evidence/canary.json"),
            (ROOT / "evidence/canary.json").resolve(),
        )
        with self.assertRaises(ValueError):
            canary.output_path(ROOT.parent / "outside-canary.json")

    def test_cpu_validator_accepts_opaque_device_addresses_and_owned_mirrors(self):
        p = self.params()
        self.assertEqual(self.validator.validate(p), 0)
        self.assertEqual(list(p._host_mirrors[0]), [0, 1])
        self.assertIn("validate_cpu", str(self.validator.path))
        dynamic = subprocess.check_output(
            ["readelf", "-d", str(self.validator.path)], text=True
        )
        self.assertNotIn("libcudart", dynamic)
        self.assertNotIn("libcuda.", dynamic)

    def test_cpu_validator_rejects_bad_abi_shapes_lengths_and_ids(self):
        mutations = [
            ("abi_version", 2, -1),
            ("struct_size", 0, -1),
            ("n", 31, -2),
            ("k", 32, -2),
            ("in_div", 0, -2),
            ("input_rows", 1, -2),
            ("codes_bytes", 2047, -3),
            ("sorted_slots_count", 15, -3),
            ("ntpp_host", 17, -5),
            ("ntpp_host", 8, -6),
        ]
        for name, value, expected in mutations:
            with self.subTest(field=name):
                p = self.params()
                setattr(p, name, value)
                self.assertEqual(self.validator.validate(p), expected)
        p = self.params()
        p._host_mirrors[0][0] = 2
        self.assertEqual(self.validator.validate(p), -5)
        p = self.params()
        p._host_mirrors[1][0] = -1
        self.assertEqual(self.validator.validate(p), -5)
        p = self.params()
        p._host_mirrors[1][0] = 2
        self.assertEqual(self.validator.validate(p), -6)

    def test_output_alias_rejected_by_real_cpu_validator(self):
        p = self.params()
        p.y = C.cast(p.codes, canary.U16P)
        self.assertEqual(self.validator.validate(p), -4)

    def test_python_builder_rejects_wrapping_fields_before_c_call(self):
        dimensions = dict(
            experts=2**32, n=32, k=64, slots=4, input_rows=2, in_div=2, splitk=2, nacc=1
        )
        with self.assertRaises(ValueError):
            canary.make_params(
                dimensions=dimensions,
                buffers={},
                expert_blocks=[],
                sorted_slots=[],
                ntpp=0,
            )

    def test_independent_cpu_byte_decode_and_scalar_oracle(self):
        self.assertEqual(canary.fp8_e4m3(0x38), 1.0)
        self.assertEqual(canary.fp8_e4m3(0x01), 2**-9)
        self.assertEqual(canary.fp8_e4m3(0xB8), -1.0)
        with self.assertRaises(ValueError):
            canary.fp8_e4m3(0x7F)
        plane = dict(
            rows=32,
            columns=64,
            codes=[0x22] * (32 * 32),
            scales=[0x38] * (32 * 4),
            global_dequant=0.25,
        )
        expected = canary.oracle_plane(plane, [[2**-5] * 64, [0.5] + [0.0] * 63])
        self.assertEqual(expected, [[0.5] * 32, [0.125] * 32])
        plane["codes"][0] = 0x2A  # low nibble -1, next nibble +1.
        self.assertEqual(canary.oracle_plane(plane, [[0.5] + [0.0] * 63])[0][0], -0.125)

    def test_no_comparison_pass_without_sync_or_with_nan_or_changed_tolerance(self):
        policy = canary.POLICIES["uniform_gemm"]
        with self.assertRaises(RuntimeError):
            canary.compare([[1.0]], [[1.0]], policy)
        with self.assertRaises(AssertionError):
            canary.compare([[float("nan")]], [[1.0]], policy, synchronized=True)
        with self.assertRaises(AssertionError):
            canary.compare([[1.001]], [[1.0]], policy, synchronized=True)
        result = canary.compare([[1.0]], [[1.0]], policy, synchronized=True)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["max_abs_error"], 0.0)

    def test_sparse_physical_bind_masks_replica_and_remote_slots_on_cpu(self):
        import torch
        from adapters import skinny_moe as adapter

        layer = canary.uniform_layer(torch, torch.device("cpu"))
        payload = canary.routing_payload(9)
        ids = torch.tensor([[10, 99], [99, 25]])
        weights = torch.tensor([[0.25, 0.5], [0.75, 1.0]])
        rank = dict(
            stamp=payload["stamp"],
            rank=2,
            hidden=64,
            tokens=2,
            resident_order=[900, 3],
            slots=[
                dict(
                    physical_expert=1,
                    slot=dict(token=0, slot=0, expert=10, rank=2, local=3, weight=0.25),
                )
            ],
        )
        mask = adapter.bind_rank_dispatch(layer, rank, payload["stamp"], ids, weights)
        self.assertEqual(mask.tolist(), [[True, False], [False, False]])
        expected = canary.uniform_moe_oracle(9)
        self.assertTrue(expected[0][0] > 0)
        self.assertEqual(expected[1], [0.0] * 64)

    def test_real_crop_oracle_is_finite_nonzero_and_bounded(self):
        sample = canary.check_sample(
            ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json"
        )
        for plane in sample["planes"].values():
            result = canary.oracle_plane(plane, [[0.5] + [0.0] * 63])
            self.assertGreater(max(abs(v) for v in result[0]), 0)
        bad = dict(sample["planes"]["gate_proj"])
        bad["rows"] = 512
        with self.assertRaises(ValueError):
            canary.oracle_plane(bad, [[0.0] * 64])

    def test_existing_hardware_receipt_is_not_overwritten_or_replayed(self):
        output = self.evidence / "prior-hardware-receipt.json"
        original = '{"status":"STARTING","operation":"unknown"}'
        output.write_text(original)
        with patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")):
            rc = canary.main(
                [
                    "--ack-hardware-gate",
                    "--gpu-uuid",
                    "GPU-unreserved",
                    "--library",
                    "unused.so",
                    "--cudart",
                    "unused-cudart.so",
                    "--routing-binary",
                    "unused",
                    "--out",
                    str(output),
                ]
            )
        self.assertEqual(rc, 2)
        self.assertEqual(output.read_text(), original)

    def cuda_proof(self, name, library_sha256, **overrides):
        receipt = {
            "status": "CUDA_COMPILED",
            "architecture": "sm_70",
            "device_executed": False,
            "hashes": {"build/libvf_qpn2_moe_sm70.so": library_sha256},
        }
        receipt.update(overrides)
        path = self.evidence / ("cuda-proof-" + name + "-" + uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(receipt))
        return path

    def canary_argv(
        self, output, library, proof=None, routing_binary="unused", ack=True
    ):
        argv = (["--ack-hardware-gate"] if ack else []) + [
            "--gpu-uuid",
            "GPU-unreserved",
            "--library",
            str(library),
            "--cudart",
            str(library),
            "--routing-binary",
            str(routing_binary),
            "--out",
            str(output),
        ]
        if proof is not None:
            argv += ["--cuda-proof", str(proof)]
        return argv

    def run_canary(self, argv):
        """Any outcome here must stay on the CPU side of the gate."""
        with (
            patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")),
            patch.object(
                canary, "QPN2Provider", side_effect=AssertionError("provider built")
            ) as provider,
        ):
            rc = canary.main(argv)
        provider.assert_not_called()
        return rc

    def test_library_without_a_cuda_receipt_is_refused_before_any_cuda_action(self):
        library = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        library.write_bytes(b"not a real library")
        output = self.evidence / ("no-proof-" + uuid.uuid4().hex + ".json")
        self.assertEqual(self.run_canary(self.canary_argv(output, library)), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["cuda_executed"])
        self.assertIn("--cuda-proof is required", report["error"]["message"])

    def test_library_that_differs_from_its_receipt_is_refused(self):
        library = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        library.write_bytes(b"shipped bytes")
        other = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        other.write_bytes(b"different bytes")
        proof = self.cuda_proof("mismatch", canary.sha(other))
        output = self.evidence / ("mismatch-" + uuid.uuid4().hex + ".json")
        self.assertEqual(self.run_canary(self.canary_argv(output, library, proof)), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn(
            "differs from its compilation receipt", report["error"]["message"]
        )

    def test_receipt_claiming_device_execution_or_another_arch_is_refused(self):
        library = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        library.write_bytes(b"shipped bytes")
        digest = canary.sha(library)
        for name, overrides in (
            ("executed", {"device_executed": True}),
            ("sm80", {"architecture": "sm_80"}),
            ("status", {"status": "V100_EXECUTED"}),
            ("nolib", {"hashes": {}}),
        ):
            with self.subTest(receipt=name):
                proof = self.cuda_proof(name, digest, **overrides)
                output = self.evidence / (name + "-" + uuid.uuid4().hex + ".json")
                self.assertEqual(
                    self.run_canary(self.canary_argv(output, library, proof)), 2
                )
                report = json.loads(output.read_text())
                self.assertEqual(report["status"], "BLOCKED")
                self.assertFalse(report["cuda_executed"])

    def test_matching_receipt_passes_the_guard_and_is_recorded(self):
        library = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        library.write_bytes(b"shipped bytes")
        digest = canary.sha(library)
        proof = self.cuda_proof("match", digest)
        self.assertEqual(
            canary.check_cuda_proof(proof, digest)["architecture"], "sm_70"
        )
        output = self.evidence / ("match-" + uuid.uuid4().hex + ".json")
        argv = self.canary_argv(output, library, proof, routing_binary=BINARY) + [
            "--sample",
            str(library),
        ]
        with (
            patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")),
            patch.object(canary, "check_sample", return_value={"source": {}}),
            patch.object(
                canary, "QPN2Provider", side_effect=RuntimeError("CPU stop")
            ) as provider,
        ):
            self.assertEqual(canary.main(argv), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["library_sha256"], digest)
        self.assertEqual(report["cuda_proof_sha256"], canary.sha(proof))
        self.assertFalse(report["cuda_executed"])
        self.assertEqual(provider.call_count, 1)

    def test_shipped_launchers_pass_the_receipt_and_default_to_the_built_library(self):
        launcher = (ROOT / "handoff/hardware_canary.sh").read_text()
        self.assertIn('--cuda-proof "$VF_H1_CUDA_PROOF"', launcher)
        self.assertIn(
            'VF_H1_LIBRARY:-"$VF_SOURCE_DIR/build/libvf_qpn2_moe_sm70.so"', launcher
        )
        self.assertNotIn("continuation_a123/cuda/artifacts", launcher)
        bundle = (ROOT / "handoff/h1_bundle_run.sh").read_text()
        self.assertIn(
            'export VF_H1_CUDA_PROOF="$VF_BUNDLE_DIR/provenance/CUDA_COMPILED.json"',
            bundle,
        )

    def test_shipped_library_matches_the_receipt_the_launcher_would_use(self):
        """Canonical source checks build/ against its receipt; inside an H1 bundle,
        where build/ and evidence/ are not packaged, the bundle pair is checked."""
        pairs = (
            (
                ROOT / "build/libvf_qpn2_moe_sm70.so",
                ROOT / "evidence/continuation_a23/CUDA_COMPILED.json",
            ),
            (
                ROOT.parent / "lib/libvf_qpn2_moe_sm70.so",
                ROOT.parent / "provenance/CUDA_COMPILED.json",
            ),
        )
        checked = 0
        for library, proof in pairs:
            if library.is_file() and proof.is_file():
                self.assertEqual(
                    canary.check_cuda_proof(proof, canary.sha(library))["status"],
                    "CUDA_COMPILED",
                )
                checked += 1
        self.assertGreater(checked, 0, "no shipped library/receipt pair was found")

    def valid_canary_inputs(self):
        """Everything downstream valid, so only the guard under test can refuse."""
        library = self.evidence / ("lib-" + uuid.uuid4().hex + ".so")
        library.write_bytes(b"not a real library")
        return library, self.cuda_proof("valid", canary.sha(library))

    def test_hardware_gate_acknowledgement_is_required_with_everything_else_valid(self):
        # The older test omitted the flag but also passed unreadable artifacts, so
        # its assertions were satisfied by a later file-not-found. Here every other
        # input is valid and only the missing acknowledgement can stop the run.
        library, proof = self.valid_canary_inputs()
        output = self.evidence / ("noack-valid-" + uuid.uuid4().hex + ".json")
        argv = self.canary_argv(
            output, library, proof, routing_binary=BINARY, ack=False
        )
        self.assertEqual(self.run_canary(argv), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["cuda_executed"])
        self.assertIn("ack-hardware-gate", report["error"]["message"])

    def test_device_ordinal_outside_its_bound_is_refused_before_cuda(self):
        library, proof = self.valid_canary_inputs()
        output = self.evidence / ("device-" + uuid.uuid4().hex + ".json")
        argv = self.canary_argv(output, library, proof, routing_binary=BINARY) + [
            "--device",
            "99",
        ]
        self.assertEqual(self.run_canary(argv), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "BLOCKED")
        self.assertIn("device ordinal", report["error"]["message"])

    def test_checkpoint_sample_revision_is_pinned(self):
        sample = json.loads(
            (ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json").read_text()
        )
        self.assertEqual(
            canary.check_sample(ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json")[
                "source"
            ]["revision"],
            "1355db6a052410cfd62085d94b58866fd0f2c3c5",
        )
        for field, value in (
            ("revision", "0000000000000000000000000000000000000000"),
            ("model", "someone-else/another-model"),
        ):
            altered = json.loads(json.dumps(sample))
            altered["source"][field] = value
            path = self.evidence / (field + "-" + uuid.uuid4().hex + ".json")
            path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError, msg=field):
                canary.check_sample(path)

    def test_a_pending_request_makes_the_receipt_admit_a_possible_execution(self):
        # This flag is the receipt's own admission that a kernel may have run; a
        # receipt that always denies it would hide an uncertain launch.
        provider = canary.QPN2Provider.__new__(canary.QPN2Provider)
        provider.launches, provider.synchronized_launches = 1, 0
        seen = []
        provider.progress = seen.append
        provider.pending = ["one enqueued request"]
        provider._notify("ENQUEUE_PENDING")
        provider.pending = []
        provider._notify("COMPLETED")
        self.assertEqual([x["may_have_executed"] for x in seen], [True, False])
        self.assertEqual([x["pending_requests"] for x in seen], [1, 0])

    def test_main_without_ack_reports_blocked_without_runtime(self):
        output = self.evidence / ("missing-ack-" + uuid.uuid4().hex + ".json")
        with patch.object(canary.C, "CDLL", side_effect=AssertionError("CUDA loaded")):
            rc = canary.main(
                [
                    "--gpu-uuid",
                    "GPU-unreserved",
                    "--library",
                    "unused.so",
                    "--cudart",
                    "unused-cudart.so",
                    "--routing-binary",
                    "unused",
                    "--out",
                    str(output),
                ]
            )
        report = json.loads(output.read_text())
        self.assertEqual(rc, 2)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(report["h1_status"], "OPEN")
        self.assertFalse(report["cuda_executed"])
        self.assertFalse(report["performance_measured"])


if __name__ == "__main__":
    unittest.main()
