"""CPU-only full-real H1 preparation, packed-provider and admission contracts."""

import fcntl

# Linux fcntl seal commands and flags; this CPython build omits their names.
from contextlib import redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from adapters.persistent_gpu_owner import OwnerOutput
from scripts import hardware_canary as canary
from scripts import full_expert_canary as full

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ["VF_BINARY"])
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


class PackedSparseCPUOracle:
    """Explicit test only: decode packed coordinates and half arithmetic via Torch."""

    torch = torch
    device = torch.device("cpu")

    def __init__(self):
        self.launches = 0
        self.synchronized_launches = 0

    def synchronize(self):
        self.synchronized_launches = self.launches

    def gemm_qpn2_moe(
        self, x, codes, scales, gs, blocks, slots, padded, s, n, k, in_div, splitk, nacc
    ):
        self.launches += 1
        assert (s, in_div, splitk, nacc) == (6, 1, 4, 1)
        assert slots.tolist() == [0, 1, 2, 3, 4, 5, 6, 6]
        assert blocks.tolist() == [0] and int(padded) == 8
        out = torch.empty((s, n), dtype=torch.float16)
        order = [0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15]
        levels = [0, 0.5, 1, 1.5, 2, 3, 4, 6]
        for token in range(s):
            nonzero = torch.nonzero(x[token]).flatten()
            assert nonzero.numel() == 1
            col = int(nonzero[0])
            a = float(x[token, col])
            group = col // 16
            position = order.index(col % 16)
            for row in range(n):
                r = row % 32
                lane = (r // 8) * 4 + (r % 4) + (16 if r % 8 >= 4 else 0)
                index = ((row // 32) * (k // 16) + group) * 32 + lane
                byte = int(codes[index * 8 + position // 2])
                nibble = (byte >> (4 * (position % 2))) & 15
                b = int(scales[index])
                e, m = b >> 3, b & 7
                scale = m * 2**-9 if e == 0 else (1 + m / 8) * 2 ** (e - 7)
                gm = (gs[0, row // 32] * 16384).half()
                combined = (torch.tensor(scale, dtype=torch.float16) * gm).half()
                signed = -levels[nibble & 7] if nibble & 8 else levels[nibble & 7]
                weight = (
                    torch.tensor(signed / 16384, dtype=torch.float16) * combined
                ).half()
                out[token, row] = (weight.float() * a).half()
        return out


class FullExpertAdapterShapeOracle:
    """CPU shape/plumbing double; numeric output shares the dense reference."""

    torch = torch
    device = torch.device("cpu")

    def __init__(self, sample):
        self.sample = sample
        self.launches = 0
        self.admissions = 0
        self.synchronized_launches = 0
        self.shapes = []

    def synchronize(self):
        self.synchronized_launches = self.launches

    def gemm_qpn2_moe(
        self, x, codes, scales, gs, blocks, slots, padded, s, n, k, in_div, splitk, nacc
    ):
        self.admissions += 1
        self.launches += 1
        self.shapes.append((s, n, k, in_div, splitk, nacc))
        self.assert_metadata(blocks, slots, padded, s)
        if (n, k) == (1024, 2048):
            gate = full.dense_projection_reference(self.sample, "gate_proj", x[:s])
            up = full.dense_projection_reference(self.sample, "up_proj", x[:s])
            return torch.cat((gate, up), dim=1)
        if (n, k) == (2048, 512):
            return full.dense_projection_reference(self.sample, "down_proj", x[:s])
        raise AssertionError("unexpected real-expert adapter provider shape")

    @staticmethod
    def assert_metadata(blocks, slots, padded, s):
        assert s == 6
        assert blocks.tolist() == [0]
        assert slots.tolist() == [0, 1, 2, 3, 4, 5, 6, 6]
        assert int(padded) == 8


class FullExpertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUN_TMP.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=RUN_TMP, prefix="a20-")
        self.directory = Path(self.temp.name)
        self.sample = ROOT / "fixtures/checkpoint_qwen36_expert0_full.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def test_temporaries_stay_in_configured_or_project_tmpdir(self):
        self.assertTrue(self.directory.resolve().is_relative_to(RUN_TMP.resolve()))
        configured = os.environ.get("TMPDIR")
        expected = Path(configured) if configured else ROOT
        self.assertTrue(RUN_TMP.resolve().is_relative_to(expected.resolve()))

    def prepared(self):
        return full.prepare_full_expert(BINARY, self.sample, self.directory)

    def test_real_three_projections_use_exact_packed_buffers_and_staged_native_reference(
        self,
    ):
        prepared = self.prepared()
        provider = PackedSparseCPUOracle()
        cases = []
        full.execute_full_expert(provider, prepared, cases)
        self.assertEqual(provider.launches, 3)
        self.assertEqual(provider.synchronized_launches, 3)
        self.assertEqual(
            [(x["plane"], x["shape"]) for x in cases],
            [
                ("gate_proj", [512, 2048]),
                ("up_proj", [512, 2048]),
                ("down_proj", [2048, 512]),
            ],
        )
        self.assertTrue(all(x["max_abs_error"] == 0 for x in cases))
        self.assertTrue(all(x["full_layer_inference"] is False for x in cases))
        reference = full.references(self.sample)
        self.assertTrue(
            all(
                any(v != 0 for row in reference[name]["values"] for v in row)
                for name in reference
            )
        )
        self.assertEqual(
            [x["subnormal_weights"] for x in cases],
            [reference[name]["subnormal_weights"] for name in reference],
        )
        self.assertTrue(any(v["subnormal_fp4_inputs"] > 0 for v in reference.values()))
        for case in cases:
            self.assertEqual(
                case["subnormal_fp4_inputs"],
                reference[case["plane"]]["subnormal_fp4_inputs"],
            )
            width = case["shape"][1]
            self.assertEqual(
                {c // (width // 4) for c, _ in case["probes"]}, {0, 1, 2, 3}
            )

    def test_dense_projection_oracle_matches_sparse_staged_reference(self):
        references = full.references(self.sample)
        for plane, expected_shape in (
            ("gate_proj", (512, 2048)),
            ("up_proj", (512, 2048)),
            ("down_proj", (2048, 512)),
        ):
            with self.subTest(plane=plane):
                reference = references[plane]
                inputs = torch.zeros((6, expected_shape[1]), dtype=torch.float16)
                for row, (column, amplitude) in enumerate(reference["probes"]):
                    inputs[row, column] = amplitude
                actual = full.dense_projection_reference(self.sample, plane, inputs)
                expected = torch.tensor(reference["values"], dtype=torch.float16)
                self.assertTrue(torch.equal(actual, expected))

    def test_real_expert_flows_through_persistent_owner_for_two_exec_batches(self):
        prepared = self.prepared()
        provider = FullExpertAdapterShapeOracle(self.sample)
        cases = []
        from adapters import skinny_moe as adapter

        with patch.object(
            adapter,
            "load_resident_artifact",
            wraps=adapter.load_resident_artifact,
        ) as load_resident_artifact:
            with patch.object(
                OwnerOutput,
                "as_raw_output",
                side_effect=AssertionError("full-expert path decoded FP16 scalars"),
            ):
                full.execute_full_expert_adapter(provider, prepared, cases)
        self.assertEqual(load_resident_artifact.call_count, 1)
        self.assertEqual(
            load_resident_artifact.call_args.kwargs["expected_info"], full.META
        )
        self.assertEqual(
            load_resident_artifact.call_args.kwargs["expected_sha256"],
            prepared.artifact_sha256,
        )
        self.assertEqual(provider.launches, 4)
        self.assertEqual(provider.admissions, 4)
        self.assertEqual(provider.synchronized_launches, 4)
        self.assertEqual(
            provider.shapes,
            [
                (6, 1024, 2048, 1, 8, 1),
                (6, 2048, 512, 1, 8, 1),
                (6, 1024, 2048, 1, 8, 1),
                (6, 2048, 512, 1, 8, 1),
            ],
        )
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["name"], "real_full_expert_persistent_owner")
        self.assertEqual(cases[0]["status"], "PASS")
        self.assertEqual(cases[0]["exec_batches"], 2)
        self.assertEqual(cases[0]["artifact_loads"], 1)
        self.assertEqual(cases[0]["attempts"], [1, 2])
        self.assertEqual(cases[0]["batch_ids"], [6, 7])
        self.assertEqual(cases[0]["projection_calls"], 4)
        self.assertEqual(cases[0]["synchronized_projection_calls"], 4)
        self.assertEqual(
            cases[0]["oracle"], "FP64_DENSE_DOT_AFTER_EXACT_HALF_WEIGHT_STAGES"
        )
        self.assertFalse(cases[0]["full_layer_inference"])

        self.assertTrue(cases[0]["compressed_owner_path"])
        self.assertTrue(cases[0]["resident_dispatch_bound"])
        self.assertEqual(cases[0]["resident_artifact_sha256"], prepared.artifact_sha256)
        self.assertEqual(cases[0]["resident_payload_bytes"], full.META["payload_bytes"])

    def test_resident_import_rejects_changed_artifact_before_any_projection(self):
        prepared = self.prepared()
        raw = bytearray(prepared.path.read_bytes())
        raw[-1] ^= 1
        prepared.path.write_bytes(raw)
        provider = FullExpertAdapterShapeOracle(self.sample)

        with self.assertRaisesRegex(ValueError, "SHA-256"):
            full.execute_full_expert_adapter(provider, prepared, [])

        self.assertEqual(provider.launches, 0)
        self.assertEqual(provider.synchronized_launches, 0)

    def test_export_uses_the_same_pinned_sample_snapshot_as_reference(self):
        baseline = self.prepared()
        original = self.sample.read_bytes()
        sample = self.directory / "race.safetensors"
        sample.write_bytes(original)

        header_size = int.from_bytes(original[:8], "little")
        header = json.loads(original[8 : 8 + header_size])
        tensor = header["model.language_model.layers.0.mlp.experts.0.gate_proj.weight"]
        start, _ = tensor["data_offsets"]
        changed = bytearray(original)
        changed[8 + header_size + start] ^= 1
        self.assertNotEqual(bytes(changed), original)

        original_rust_call = full.rust_call
        observed = {}

        def replace_after_reference(binary, operation, request, *, pass_fds=()):
            export_path = Path(request["source"]["request"]["path"])
            self.assertEqual(len(pass_fds), 1)
            source_fd = pass_fds[0]
            observed["path"] = str(export_path)
            observed["fd"] = source_fd
            observed["bytes_before"] = export_path.read_bytes()
            observed["mode"] = export_path.stat().st_mode & 0o777
            required_seals = 0x0008 | 0x0002 | 0x0004 | 0x0001
            self.assertEqual(
                fcntl.fcntl(source_fd, 1034) & required_seals,
                required_seals,
            )
            # A same-UID writer at the Rust handoff boundary cannot mutate the sealed inode.
            with self.assertRaises(OSError):
                os.pwrite(source_fd, changed, 0)
            # This runs after references(sample), exactly at the export boundary.
            sample.write_bytes(changed)
            try:
                return original_rust_call(binary, operation, request, pass_fds=pass_fds)
            finally:
                sample.write_bytes(original)

        with patch.object(full, "rust_call", side_effect=replace_after_reference):
            raced = full.prepare_full_expert(BINARY, sample, self.directory)

        self.assertEqual(sample.read_bytes(), original)
        self.assertEqual(
            full._digest(raced.artifact),
            full._digest(baseline.artifact),
            "exported artifact must come from the exact bytes used by the oracle",
        )
        self.assertEqual(raced.sample_sha256, full.SAMPLE_SHA)
        self.assertEqual(observed["bytes_before"], original)
        self.assertEqual(observed["mode"], 0o400)
        self.assertTrue(observed["path"].startswith("/proc/self/fd/"))
        self.assertNotEqual(observed["path"], str(sample.resolve()))
        with self.assertRaises(OSError):
            os.fstat(observed["fd"])

    def test_failed_export_closes_sealed_source_memfd(self):
        snapshots = []

        def fail_after_observing_snapshot(binary, operation, request, *, pass_fds=()):
            snapshot = Path(request["source"]["request"]["path"])
            self.assertEqual(len(pass_fds), 1)
            source_fd = pass_fds[0]
            snapshots.append((snapshot, source_fd))
            self.assertTrue(snapshot.is_file())
            self.assertEqual(snapshot.read_bytes(), self.sample.read_bytes())
            self.assertTrue(fcntl.fcntl(source_fd, 1034) & 0x0008)
            with self.assertRaises(OSError):
                os.pwrite(source_fd, b"\x00", 0)
            raise RuntimeError("injected export failure")

        with patch.object(full, "rust_call", side_effect=fail_after_observing_snapshot):
            with self.assertRaisesRegex(RuntimeError, "injected export failure"):
                full.prepare_full_expert(BINARY, self.sample, self.directory)

        self.assertEqual(len(snapshots), 1)
        with self.assertRaises(OSError):
            os.fstat(snapshots[0][1])

    def test_changed_tensor_file_or_static_identity_never_calls_provider(self):
        for kind in ["tensor", "dtype", "metadata", "file"]:
            prepared = self.prepared()
            if kind == "tensor":
                prepared.artifact.tensors["codes13"][0] ^= 1
            elif kind == "dtype":
                prepared.artifact.tensors["codes13"] = prepared.artifact.tensors[
                    "codes13"
                ].float()
            elif kind == "metadata":
                prepared.artifact.metadata["epoch"] = 2
            else:
                raw = bytearray(prepared.path.read_bytes())
                raw[-1] ^= 1
                prepared.path.write_bytes(raw)
            provider = PackedSparseCPUOracle()
            cases = []
            with self.assertRaises((ValueError, RuntimeError)):
                full.execute_full_expert(provider, prepared, cases)
            self.assertEqual(provider.launches, 0, kind)
            self.assertEqual(cases, [])

    def test_bad_sample_and_fifo_fail_before_rust_export(self):
        wrong = self.directory / "wrong.safetensors"
        wrong.write_bytes(b"invalid")
        fifo = self.directory / "fifo"
        os.mkfifo(fifo)
        for path in [wrong, fifo]:
            with patch.object(full, "rust_call") as called:
                with self.assertRaises(ValueError):
                    full.prepare_full_expert(BINARY, path, self.directory)
                called.assert_not_called()

    def test_staged_reference_retains_overflow_and_underflow_cases(self):
        for g, b in [(4.0, 0x38), (0.5, 0x7E)]:
            with self.assertRaises(ValueError):
                full.staged_weight(2, b, g)
        self.assertEqual(full.staged_weight(2, 0, 0.5), 0)
        self.assertEqual(full.staged_weight(1, 0x38, 2**-15), 2**-16)
        self.assertEqual(full.staged_weight(2, 126, 2**-149), 0)
        for b in [127, 128, 255]:
            with self.assertRaises(ValueError):
                full.staged_weight(2, b, 0.25)

    def test_sync_or_numerical_failure_never_publishes_case(self):
        prepared = self.prepared()
        provider = PackedSparseCPUOracle()
        provider.synchronize = lambda: (_ for _ in ()).throw(RuntimeError("sync fault"))
        cases = []
        with self.assertRaisesRegex(RuntimeError, "sync fault"):
            full.execute_full_expert(provider, prepared, cases)
        self.assertEqual(cases, [])

    def test_keyboard_interrupt_closes_sealed_source_memfd(self):
        descriptors = []

        def interrupt_export(binary, operation, request, *, pass_fds=()):
            descriptors.extend(pass_fds)
            raise KeyboardInterrupt

        with patch.object(full, "rust_call", side_effect=interrupt_export):
            with self.assertRaises(KeyboardInterrupt):
                full.prepare_full_expert(BINARY, self.sample, self.directory)

        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    def test_full_provider_bounds_require_explicit_opt_in_before_stream(self):
        provider = object.__new__(canary.QPN2Provider)
        provider.torch = torch
        provider.device = torch.device("cpu")
        args = (
            torch.zeros((6, 2048), dtype=torch.float16),
            torch.zeros(524288, dtype=torch.uint8),
            torch.zeros(65536, dtype=torch.uint8),
            torch.full((1, 16), 0.25),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([0, 1, 2, 3, 4, 5, 6, 6], dtype=torch.int32),
            torch.tensor(8, dtype=torch.int32),
            6,
            512,
            2048,
            1,
            4,
            1,
        )
        with patch.object(
            torch.cuda,
            "current_stream",
            side_effect=RuntimeError("CPU stop before CUDA"),
        ) as stream:
            with self.assertRaises(ValueError):
                provider.gemm_qpn2_moe(*args)
            stream.assert_not_called()
            provider.full_expert_enabled = True
            with self.assertRaisesRegex(RuntimeError, "CPU stop before CUDA"):
                provider.gemm_qpn2_moe(*args)
            self.assertEqual(stream.call_count, 1)
            for index, bad in [(7, 5), (8, 513), (9, 1024), (10, 2), (11, 2), (12, 2)]:
                wrong = list(args)
                wrong[index] = bad
                stream.reset_mock()
                with self.assertRaises(ValueError):
                    provider.gemm_qpn2_moe(*wrong)
                stream.assert_not_called()

    def test_adapter_full_provider_admits_only_fused_real_expert_shapes(self):
        provider = object.__new__(canary.QPN2Provider)
        provider.torch = torch
        provider.device = torch.device("cpu")
        provider.adapter_full_expert_enabled = True

        def arguments(n, k, splitk, nacc):
            return (
                torch.zeros((6, k), dtype=torch.float16),
                torch.zeros(n * k // 2, dtype=torch.uint8),
                torch.full((n * k // 16,), 0x38, dtype=torch.uint8),
                torch.full((1, n // 32), 0.25, dtype=torch.float32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([0, 1, 2, 3, 4, 5, 6, 6], dtype=torch.int32),
                torch.tensor(8, dtype=torch.int32),
                6,
                n,
                k,
                1,
                splitk,
                nacc,
            )

        for n, k, splitk, nacc in ((1024, 2048, 8, 1), (2048, 512, 8, 1)):
            with self.subTest(n=n, k=k, splitk=splitk, nacc=nacc):
                with patch.object(
                    torch.cuda,
                    "current_stream",
                    side_effect=RuntimeError("bounded guard reached stream"),
                ) as stream:
                    with self.assertRaisesRegex(
                        RuntimeError, "bounded guard reached stream"
                    ):
                        provider.gemm_qpn2_moe(*arguments(n, k, splitk, nacc))
                    stream.assert_called_once()

        for n, k, splitk, nacc in (
            (1024, 2048, 4, 1),
            (1024, 2048, 8, 2),
            (1024, 1024, 8, 1),
            (512, 2048, 8, 1),
        ):
            with self.subTest(n=n, k=k, splitk=splitk, nacc=nacc):
                with patch.object(torch.cuda, "current_stream") as stream:
                    with self.assertRaises(ValueError):
                        provider.gemm_qpn2_moe(*arguments(n, k, splitk, nacc))
                    stream.assert_not_called()

    def test_wrong_numeric_result_is_atomic(self):
        prepared = self.prepared()

        class Wrong(PackedSparseCPUOracle):
            def gemm_qpn2_moe(self, *args):
                out = super().gemm_qpn2_moe(*args)
                out[0, 0] = float("nan")
                return out

        cases = []
        with self.assertRaisesRegex(AssertionError, "nonfinite/unwritten"):
            full.execute_full_expert(Wrong(), prepared, cases)
        self.assertEqual(cases, [])

    def test_main_prepares_full_sample_before_provider(self):
        output = self.directory / "valid-full.json"
        argv = [
            "--ack-hardware-gate",
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            "unused",
            "--cudart",
            "unused",
            "--routing-binary",
            str(BINARY),
            "--out",
            str(output),
            "--cuda-proof",
            stub_cuda_proof(self.directory, "test"),
            "--full-expert-sample",
            str(self.sample),
            "--full-expert-adapter",
        ]

        providers = []

        class StopAfterAdmission:
            launches = 0
            synchronized_launches = 0

            def __init__(self):
                self.torch = torch
                self.device = torch.device("cpu")
                providers.append(self)

            @property
            def identity(self):
                raise RuntimeError("CPU stop after admission")

            def synchronize(self):
                return None

        with (
            patch.object(canary, "ROOT", ROOT),
            patch.object(canary, "output_path", side_effect=lambda p: Path(p)),
            patch.object(canary, "sha", return_value="test"),
            patch.object(canary, "check_sample", return_value={"source": {}}),
            patch.object(
                canary, "QPN2Provider", side_effect=lambda *a, **k: StopAfterAdmission()
            ) as provider,
        ):
            self.assertEqual(canary.main(argv), 2)
        provider.assert_called_once()
        self.assertEqual(len(providers), 1)
        self.assertTrue(providers[0].full_expert_enabled)
        self.assertTrue(providers[0].adapter_full_expert_enabled)
        report = json.loads(output.read_text())
        self.assertFalse(report["cuda_executed"])
        self.assertEqual(report["full_expert"]["sample_sha256"], full.SAMPLE_SHA)
        self.assertEqual(report["full_expert"]["payload_bytes"], 1769856)
        self.assertEqual(report["full_expert"]["gpu_status"], "PENDING")
        self.assertEqual(report["full_expert"]["adapter_gpu_status"], "PENDING")
        self.assertEqual(report["cases"], [])

    def test_adapter_full_mode_requires_sample_before_receipt_or_provider(self):
        output = self.directory / "missing-adapter-sample.json"
        argv = [
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            "unused",
            "--cudart",
            "unused",
            "--routing-binary",
            str(BINARY),
            "--out",
            str(output),
            "--full-expert-adapter",
        ]
        stderr = io.StringIO()
        with (
            patch.object(canary, "QPN2Provider") as provider,
            redirect_stderr(stderr),
        ):
            with self.assertRaises(SystemExit) as raised:
                canary.main(argv)
        self.assertEqual(raised.exception.code, 2)
        self.assertIn(
            "--full-expert-adapter requires --full-expert-sample", stderr.getvalue()
        )
        provider.assert_not_called()
        self.assertFalse(output.exists())

    def test_main_rejects_bad_full_sample_before_provider(self):
        output = self.directory / "hardware.json"
        argv = [
            "--ack-hardware-gate",
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            "unused",
            "--cudart",
            "unused",
            "--routing-binary",
            str(BINARY),
            "--out",
            str(output),
            "--cuda-proof",
            stub_cuda_proof(self.directory, "test"),
            "--full-expert-sample",
            str(self.directory / "missing"),
        ]
        with (
            patch.object(canary, "ROOT", ROOT),
            patch.object(canary, "output_path", side_effect=lambda p: Path(p)),
            patch.object(canary, "sha", return_value="test"),
            patch.object(canary, "check_sample", return_value={"source": {}}),
            patch.object(canary, "QPN2Provider") as provider,
        ):
            self.assertEqual(canary.main(argv), 2)
        provider.assert_not_called()
        self.assertFalse(json.loads(output.read_text())["cuda_executed"])


if __name__ == "__main__":
    unittest.main()
