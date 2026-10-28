"""A43: one refusal per guard operand of scripts/hardware_canary.py.

Each case makes exactly one operand of one ``if ...: raise`` guard true, with every
earlier check satisfied, and pins both the exception type and its distinctive
message, so a neighbouring or later check cannot satisfy the assertion instead.
Each fixture also has a positive control proving that, without the bad value, the
call runs past the guard. CPU only: no CUDA runtime, library or device is used;
every ctypes/Torch CUDA entry point is replaced by a recorded fake.
"""

from contextlib import nullcontext, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import torch

from scripts import hardware_canary as canary

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json"


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory(prefix="a43-canary-")
        self.addCleanup(self._temp.cleanup)
        self.directory = Path(self._temp.name)


class OutputPathTests(TempDirTest):
    def test_receipt_root_that_is_a_file_is_refused(self):
        # A regular file resolves strictly and a path "below" it is lexically
        # relative to it, so only the is_dir guard can refuse.
        root = self.directory / "receipts"
        root.write_text("not a directory")
        with patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(root)}):
            with self.assertRaisesRegex(
                ValueError, "receipt root must be an existing directory"
            ):
                canary.output_path(root / "hardware.json")
        directory = self.directory / "receipt-dir"
        directory.mkdir()
        with patch.dict(os.environ, {"VF_H1_RECEIPT_DIR": str(directory)}):
            self.assertEqual(
                canary.output_path(directory / "hardware.json"),
                (directory / "hardware.json").resolve(),
            )


class MakeParamsTests(unittest.TestCase):
    DIMENSIONS = dict(
        experts=2, n=32, k=64, slots=4, input_rows=2, in_div=2, splitk=2, nacc=1
    )
    LENGTHS = dict(
        codes=2048,
        scales=256,
        gscales=2,
        x=128,
        y=128,
        block_expert=2,
        sorted_slots=16,
        ntpp=1,
    )

    def build(self, dimensions=None, buffers=None, blocks=None, ntpp=16):
        if buffers is None:
            buffers = {
                name: (0x100000 + index * 0x100000, count)
                for index, (name, count) in enumerate(self.LENGTHS.items())
            }
        return canary.make_params(
            dimensions=dict(self.DIMENSIONS, **(dimensions or {})),
            buffers=buffers,
            expert_blocks=[0, 1] if blocks is None else blocks,
            sorted_slots=[0, 2] + [4] * 6 + [1, 3] + [4] * 6,
            ntpp=ntpp,
        )

    def test_valid_builder_control(self):
        p = self.build()
        self.assertEqual((p.experts, p.codes_bytes, p.ntpp_host), (2, 2048, 16))

    def test_non_integer_dimension_is_refused(self):
        with self.assertRaisesRegex(ValueError, "dimension is not uint32: experts"):
            self.build(dimensions=dict(experts=2.0))

    def buffers_with(self, address, length):
        buffers = {
            name: (0x100000 + index * 0x100000, count)
            for index, (name, count) in enumerate(self.LENGTHS.items())
        }
        buffers["codes"] = (address, length)
        return buffers

    def test_invalid_opaque_address_is_refused(self):
        # Float: only the isinstance operand. -1 and 2**64: only the range operand
        # (ctypes would otherwise wrap them silently into a valid-looking pointer).
        for address in (float(0x100000), -1, 2**64):
            with self.subTest(address=address):
                with self.assertRaisesRegex(ValueError, "invalid opaque address"):
                    self.build(buffers=self.buffers_with(address, 2048))

    def test_invalid_buffer_capacity_is_refused(self):
        for length in (2048.0, -1, 2**64):
            with self.subTest(length=length):
                with self.assertRaisesRegex(ValueError, "invalid buffer capacity"):
                    self.build(buffers=self.buffers_with(0x100000, length))

    def test_metadata_outside_int32_is_refused(self):
        for blocks, ntpp in (([0, 2**31], 16), ([-(2**31) - 1, 1], 16), ([0, 1], 16.0)):
            with self.subTest(blocks=blocks, ntpp=ntpp):
                with self.assertRaisesRegex(ValueError, "metadata is not int32"):
                    self.build(blocks=blocks, ntpp=ntpp)


class CPUValidatorTests(TempDirTest):
    def test_failed_validator_compile_is_refused_before_loading(self):
        loaded = []
        library = SimpleNamespace(vf_qpn2_moe_validate=SimpleNamespace())
        for returncode in (0, 1):
            with self.subTest(returncode=returncode):
                compiled = SimpleNamespace(
                    returncode=returncode, stdout="", stderr="E1"
                )
                with (
                    patch.object(canary.subprocess, "run", return_value=compiled),
                    patch.object(
                        canary.C,
                        "CDLL",
                        side_effect=lambda path: loaded.append(path) or library,
                    ),
                ):
                    if returncode:
                        with self.assertRaisesRegex(
                            RuntimeError, "CPU validator compile failed: E1"
                        ):
                            canary.CPUValidator(ROOT, self.directory)
                    else:
                        canary.CPUValidator(ROOT, self.directory)
        self.assertEqual(len(loaded), 1, "a failed compile was loaded")


GATE_UUID = uuid.UUID("12345678-9abc-def0-1234-56789abcdef0")


class FakeFunction:
    """ctypes-like callable that accepts argtypes/restype assignment."""

    def __init__(self, behaviour):
        self.behaviour = behaviour

    def __call__(self, *args):
        return self.behaviour(*args)


def fake_runtime(driver_rc=0, runtime_rc=0, runtime_version=12080):
    def version(value, rc):
        def call(reference):
            reference._obj.value = value
            return rc

        return FakeFunction(call)

    return SimpleNamespace(
        cudaDriverGetVersion=version(12080, driver_rc),
        cudaRuntimeGetVersion=version(runtime_version, runtime_rc),
    )


def fake_driver(init_rc=0, device_rc=0, uuid_rc=0):
    def get_device(reference, ordinal):
        reference._obj.value = ordinal
        return device_rc

    def get_uuid(reference, device):
        # The UUID is written even on failure, so only the rc guard can refuse.
        for index, byte in enumerate(GATE_UUID.bytes):
            reference._obj[index] = byte
        return uuid_rc

    return SimpleNamespace(
        cuInit=FakeFunction(lambda flags: init_rc),
        cuDeviceGet=FakeFunction(get_device),
        cuDeviceGetUuid_v2=FakeFunction(get_uuid),
    )


class ProviderInitTests(TempDirTest):
    """QPN2Provider admission with every Torch CUDA query and dlopen faked."""

    def setUp(self):
        super().setUp()
        self.gate = self.directory / "HARDWARE_GATE.md"
        self.gate.write_text("gate\n")
        self.expected = "GPU-" + str(GATE_UUID)

    def construct(
        self,
        *,
        gate=None,
        expected=None,
        available=True,
        torch_cuda="12.8",
        props=(7, 0),
        runtime=None,
        driver=None,
    ):
        runtime = runtime or fake_runtime()
        driver = driver or fake_driver()
        library = SimpleNamespace(
            vf_qpn2_moe_validate=FakeFunction(lambda *_: 0),
            vf_qpn2_moe_launch=FakeFunction(lambda *_: 0),
        )

        def cdll(path, mode=None):
            if "cudart" in path:
                return runtime
            if "libcuda" in path:
                return driver
            return library

        with (
            patch.object(torch.cuda, "is_available", return_value=available),
            patch.object(torch.version, "cuda", torch_cuda),
            patch.object(torch.cuda, "get_arch_list", return_value=["sm_70"]),
            patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(
                    major=props[0], minor=props[1], name="fake", total_memory=1
                ),
            ),
            patch.object(canary.C, "CDLL", side_effect=cdll),
        ):
            return canary.QPN2Provider(
                str(self.directory / "fake-library.so"),
                str(self.directory / "fake-cudart.so"),
                0,
                self.expected if expected is None else expected,
                ack_hardware_gate=True,
                gate_path=self.gate if gate is None else gate,
            )

    def test_fully_valid_fake_environment_constructs(self):
        provider = self.construct(expected=self.expected.upper())
        self.assertEqual(provider.identity["uuid"], self.expected)
        self.assertEqual(provider.identity["runtime_api_version"], 12080)
        self.assertEqual((provider.launches, provider.pending), (0, []))
        self.assertEqual(provider.admissions, 0)

    def test_missing_gate_document_or_uuid_is_refused(self):
        message = "hardware gate document and expected GPU UUID required"
        with self.assertRaisesRegex(ValueError, message):
            self.construct(gate=self.directory / "missing.md")
        with self.assertRaisesRegex(ValueError, message):
            self.construct(expected="")

    def test_unavailable_cuda_is_refused(self):
        with self.assertRaisesRegex(
            RuntimeError, "CUDA-enabled Torch and an available canary GPU required"
        ):
            self.construct(available=False)

    def test_torch_without_cuda_build_is_refused(self):
        # torch.version.cuda None alone (not a non-12 version string).
        with self.assertRaisesRegex(
            RuntimeError, "existing CUDA 12-enabled Torch is required"
        ):
            self.construct(torch_cuda=None)

    def test_non_sm70_device_is_refused(self):
        for props in ((8, 0), (7, 5)):
            with self.subTest(props=props):
                with self.assertRaisesRegex(RuntimeError, "canary accepts SM70 only"):
                    self.construct(props=props)

    def test_cuda12_runtime_need_not_export_uuid_query(self):
        # CUDA Runtime 12 exposes no cudaDeviceGetUuid symbol; identity uses Driver API.
        provider = self.construct(runtime=fake_runtime())
        self.assertEqual(provider.identity["uuid"], self.expected)

    def test_failed_driver_uuid_query_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "cuDeviceGetUuid_v2 failed: 2"):
            self.construct(driver=fake_driver(uuid_rc=2))

    def test_failed_driver_init_and_ordinal_queries_are_refused(self):
        for driver, message in (
            (fake_driver(init_rc=3), "cuInit failed: 3"),
            (fake_driver(device_rc=101), "cuDeviceGet failed: 101"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    self.construct(driver=driver)

    def test_uuid_of_another_gpu_is_refused(self):
        with self.assertRaisesRegex(
            RuntimeError, "selected GPU UUID differs from explicitly reserved canary"
        ):
            self.construct(expected="GPU-" + str(uuid.UUID(int=1)))

    def test_failed_version_queries_are_refused(self):
        for keyword, symbol in (
            ("driver_rc", "cudaDriverGetVersion"),
            ("runtime_rc", "cudaRuntimeGetVersion"),
        ):
            with self.subTest(symbol=symbol):
                with self.assertRaisesRegex(RuntimeError, symbol + " failed: 3"):
                    self.construct(runtime=fake_runtime(**{keyword: 3}))

    def test_non_cuda12_runtime_is_refused(self):
        for version in (11080, 13000):
            with self.subTest(version=version):
                with self.assertRaisesRegex(
                    RuntimeError, "the FFI must use the CUDA 12 runtime"
                ):
                    self.construct(runtime=fake_runtime(runtime_version=version))


class ProviderGemmTests(unittest.TestCase):
    """Direct gemm_qpn2_moe admission with CPU tensors and a recorded fake C ABI."""

    def provider(self, validate=0, launch=0, cuda_error=0):
        p = object.__new__(canary.QPN2Provider)
        p.torch, p.device = torch, torch.device("cpu")
        p.pending, p.admissions, p.launches, p.synchronized_launches = [], 0, 0, 0
        p.progress = None
        self.calls = []

        def do_launch(params, stream, error):
            self.calls.append("launch")
            error._obj.value = cuda_error
            return launch

        p.library = SimpleNamespace(
            vf_qpn2_moe_validate=lambda *_: self.calls.append("validate") or validate,
            vf_qpn2_moe_launch=do_launch,
        )
        return p

    def data(self):
        return dict(
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

    def call(self, provider, data):
        stream = SimpleNamespace(cuda_stream=0)
        with (
            patch.object(torch.cuda, "current_stream", return_value=stream),
            patch.object(torch.cuda, "device", return_value=nullcontext()),
        ):
            return provider.gemm_qpn2_moe(**data)

    def refuse(self, data, message, provider=None, error=ValueError, calls=()):
        provider = provider or self.provider()
        with self.assertRaisesRegex(error, message):
            self.call(provider, data)
        self.assertEqual(self.calls, list(calls))
        self.assertEqual(provider.admissions, int("launch" in calls))
        self.assertEqual(provider.launches, 0)

    def test_valid_call_control(self):
        provider = self.provider()
        y = self.call(provider, self.data())
        self.assertEqual(tuple(y.shape), (1, 32))
        self.assertEqual(self.calls, ["validate", "launch"])
        self.assertEqual(provider.admissions, 1)
        self.assertEqual(provider.launches, 1)

    def test_tensor_on_another_device_dtype_or_layout_is_refused(self):
        message = "FFI tensors must be contiguous typed buffers on the canary GPU"
        cases = {
            "device": ("codes", lambda v: v.to("meta")),
            "dtype": ("x", lambda v: v.float()),
            # Same numel, strided view: only is_contiguous is false.
            "layout": ("codes", lambda v: v.repeat(2)[::2]),
        }
        for name, (key, change) in cases.items():
            with self.subTest(case=name):
                data = self.data()
                data[key] = change(data[key])
                self.refuse(data, message)

    def test_malformed_activation_global_or_ntpp_shape_is_refused(self):
        message = "malformed activation/global/ntpp shape"
        # Each change keeps x.shape[1] == k and the global tile count consistent,
        # so later shape guards would admit the call.
        cases = {
            "x_ndim": ("x", lambda v: v.reshape(1, 64, 1)),
            "global_ndim": ("global_scales", lambda v: v.reshape(1, 1, 1)),
            "ntpp_numel": ("ntpp", lambda v: torch.tensor([8, 8], dtype=torch.int32)),
        }
        for name, (key, change) in cases.items():
            with self.subTest(case=name):
                data = self.data()
                data[key] = change(data[key])
                self.refuse(data, message)

    def test_compressed_capacity_or_metadata_count_is_refused(self):
        message = "compressed capacities/metadata exceed the admitted H1 shape"
        cases = {
            "codes": dict(codes=torch.full((32 * 32 + 1,), 0x22, dtype=torch.uint8)),
            "scales": dict(scales=torch.full((32 * 4 * 2,), 0x38, dtype=torch.uint8)),
            "blocks": dict(
                expert_blocks=torch.zeros(3, dtype=torch.int32),
                sorted_slots=torch.tensor(([0] + [1] * 7) * 3, dtype=torch.int32),
            ),
            "slots": dict(sorted_slots=torch.tensor([0] + [1] * 8, dtype=torch.int32)),
        }
        for name, change in cases.items():
            with self.subTest(case=name):
                self.refuse(dict(self.data(), **change), message)

    def test_memory_cap_is_enforced_at_its_exact_boundary(self):
        data = self.data()
        inputs = (
            "x",
            "codes",
            "scales",
            "global_scales",
            "expert_blocks",
            "sorted_slots",
            "ntpp",
        )
        required = sum(data[n].numel() * data[n].element_size() for n in inputs)
        required += data["s"] * data["n"] * 2  # FP16 output.
        with patch.object(canary, "MAX_CANARY_BYTES", required - 1):
            self.refuse(self.data(), "H1 canary memory cap exceeded")
        with patch.object(canary, "MAX_CANARY_BYTES", required):
            self.call(self.provider(), self.data())
        self.assertEqual(self.calls, ["validate", "launch"])

    def test_a133_six_expert_projection_uses_only_its_explicit_128_mib_cap(self):
        n, k, experts, slots = 4608, 5120, 6, 12
        blocks = torch.arange(experts, dtype=torch.int32)
        aligned = []
        for local in range(experts):
            aligned.extend((local, 11 - local, *([slots] * 6)))
        data = dict(
            x=torch.ones((slots, k), dtype=torch.float16),
            codes=torch.full((experts * n * k // 2,), 0x22, dtype=torch.uint8),
            scales=torch.full((experts * n * k // 16,), 0x38, dtype=torch.uint8),
            global_scales=torch.full((experts, n // 32), 0.25, dtype=torch.float32),
            expert_blocks=blocks,
            sorted_slots=torch.tensor(aligned, dtype=torch.int32),
            ntpp=torch.tensor(len(aligned), dtype=torch.int32),
            s=slots,
            n=n,
            k=k,
            in_div=1,
            splitk=8,
            nacc=1,
        )
        names = (
            "x",
            "codes",
            "scales",
            "global_scales",
            "expert_blocks",
            "sorted_slots",
            "ntpp",
        )
        required = sum(data[name].numel() * data[name].element_size() for name in names)
        required += slots * n * 2
        self.assertGreater(required, canary.MAX_CANARY_BYTES)
        self.assertLessEqual(required, canary.MAX_TOPK6_CANARY_BYTES)

        provider = self.provider()
        provider.full_expert_enabled = True
        provider.deepseek_probe_enabled = True
        provider.adapter_layer_enabled = True
        with patch.object(canary, "MAX_TOPK6_CANARY_BYTES", required - 1):
            self.refuse(data, "H1 canary memory cap exceeded", provider=provider)
        self.assertEqual(self.calls, [])

        provider = self.provider()
        provider.full_expert_enabled = True
        provider.deepseek_probe_enabled = True
        provider.adapter_layer_enabled = True
        with (
            patch.object(canary, "MAX_TOPK6_CANARY_BYTES", required),
            patch.object(canary, "MAX_CANARY_BYTES", 1),
        ):
            output = self.call(provider, data)
        self.assertEqual(tuple(output.shape), (slots, n))
        self.assertEqual(self.calls, ["validate", "launch"])
        self.assertEqual(provider.admissions, 1)
        self.assertEqual(provider.launches, 1)

    def test_c_validation_failure_is_refused_before_launch(self):
        self.refuse(
            self.data(),
            "C FFI metadata validation rejected canary: -3",
            provider=self.provider(validate=-3),
            calls=["validate"],
        )

    def test_launch_status_or_cuda_error_is_refused(self):
        for status, error, text in ((7, 0, "FFI=7 CUDA=0"), (0, 5, "FFI=0 CUDA=5")):
            with self.subTest(status=status, error=error):
                provider = self.provider(launch=status, cuda_error=error)
                self.refuse(
                    self.data(),
                    "real QPN2 launch failed: " + text,
                    provider=provider,
                    error=RuntimeError,
                    calls=["validate", "launch"],
                )
                # Borrowed buffers stay pending after an uncertain launch.
                self.assertEqual(len(provider.pending), 1)


def plane(rows=32, columns=64, factor=0.25, code=0x22):
    return dict(
        rows=rows,
        columns=columns,
        codes=[code] * (rows * columns // 2),
        scales=[0x38] * (rows * columns // 16),
        global_dequant=factor,
    )


class OracleTests(unittest.TestCase):
    def test_valid_plane_control(self):
        self.assertEqual(canary.oracle_plane(plane(), [[2**-5] * 64]), [[0.5] * 32])

    def test_fp8_code_that_is_not_a_byte_is_refused(self):
        # 56.0 is in range but not an int; 256 is an int whose low bits decode.
        for value in (56.0, 256):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "FP8 code is not a byte"):
                    canary.fp8_e4m3(value)

    def test_crop_dimensions_outside_h1_are_refused(self):
        for rows, columns in ((48, 64), (32, 96), (160, 64)):
            with self.subTest(rows=rows, columns=columns):
                with self.assertRaisesRegex(ValueError, "crop dimensions outside H1"):
                    canary.oracle_plane(plane(rows, columns), [[0.0] * columns])

    def test_crop_compressed_lengths_are_refused(self):
        for name, delta in (("codes", 1), ("codes", -1), ("scales", 1), ("scales", -1)):
            with self.subTest(name=name, delta=delta):
                bad = plane()
                bad[name] = bad[name] + [0x22] if delta > 0 else bad[name][:-1]
                with self.assertRaisesRegex(
                    ValueError, "crop compressed lengths differ"
                ):
                    canary.oracle_plane(bad, [[0.0] * 64])

    def test_invalid_checkpoint_global_scale_is_refused(self):
        for factor in (float("nan"), float("inf"), -0.25):
            with self.subTest(factor=factor):
                with self.assertRaisesRegex(
                    ValueError, "invalid checkpoint global scale"
                ):
                    canary.oracle_plane(plane(factor=factor), [[0.0] * 64])

    def test_invalid_oracle_input_is_refused(self):
        for x in ([0.0] * 63, [0.0] * 63 + [float("nan")], [float("inf")] + [0.0] * 63):
            with self.subTest(length=len(x), last=x[-1]):
                with self.assertRaisesRegex(ValueError, "invalid oracle input"):
                    canary.oracle_plane(plane(), [x])

    def test_fp4_code_that_is_not_a_byte_is_refused(self):
        for value in (34.0, 256, -1):
            with self.subTest(value=value):
                bad = plane()
                bad["codes"][0] = value
                with self.assertRaisesRegex(ValueError, "FP4 code is not a byte"):
                    canary.oracle_plane(bad, [[0.0] * 64])


class CompareTests(unittest.TestCase):
    POLICY = canary.POLICIES["uniform_gemm"]

    def test_shape_differences_are_refused(self):
        # zip() would otherwise truncate both cases into a PASS.
        for actual, expected in (
            ([[1.0]], [[1.0], [2.0]]),
            ([[1.0], [2.0]], [[1.0]]),
            ([[1.0, 2.0]], [[1.0]]),
            ([[1.0]], [[1.0, 2.0]]),
        ):
            with self.subTest(actual=actual, expected=expected):
                with self.assertRaisesRegex(
                    AssertionError, "output shape differs from oracle"
                ):
                    canary.compare(actual, expected, self.POLICY, synchronized=True)

    def test_nonfinite_oracle_value_is_refused(self):
        for wanted in (float("nan"), float("inf")):
            with self.subTest(wanted=wanted):
                with self.assertRaisesRegex(
                    AssertionError, "nonfinite/unwritten canary output"
                ):
                    canary.compare([[1.0]], [[wanted]], self.POLICY, synchronized=True)


class RustSubprocessTests(TempDirTest):
    def response(self, returncode, ok):
        payload = dict(ok=ok, result=dict(ranks=[dict(rank=2, resident_order=[])]))
        return SimpleNamespace(returncode=returncode, stdout=json.dumps(payload))

    def test_rust_rank_requires_zero_exit_and_ok(self):
        with patch.object(
            canary.subprocess, "run", return_value=self.response(0, True)
        ):
            self.assertEqual(canary.rust_rank("vf", 2)[0]["rank"], 2)
        for returncode, ok in ((1, True), (0, False)):
            with self.subTest(returncode=returncode, ok=ok):
                with patch.object(
                    canary.subprocess, "run", return_value=self.response(returncode, ok)
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "real Rust routing rejected canary plan"
                    ):
                        canary.rust_rank("vf", 2)

    def test_rust_call_requires_zero_exit_and_ok(self):
        with patch.object(
            canary.subprocess, "run", return_value=self.response(0, True)
        ):
            self.assertIn("ranks", canary.rust_call("vf", "routing", {}))
        for returncode, ok in ((1, True), (0, False)):
            with self.subTest(returncode=returncode, ok=ok):
                with patch.object(
                    canary.subprocess, "run", return_value=self.response(returncode, ok)
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, "Rust canary routing rejected"
                    ):
                        canary.rust_call("vf", "routing", {})

    def test_export_outside_cpu_compressed_scope_is_refused(self):
        from scripts import resident_artifact

        loaded = object()
        for scope, executed, admitted in (
            ("CPU_COMPRESSED_EXPORT_ONLY", False, True),
            ("CUDA_EXPORT", False, False),
            ("CPU_COMPRESSED_EXPORT_ONLY", True, False),
            ("CPU_COMPRESSED_EXPORT_ONLY", None, False),
        ):
            with self.subTest(scope=scope, executed=executed):
                result = dict(scope=scope, cuda_executed=executed, meta={})
                with (
                    patch.object(canary, "rust_call", return_value=result),
                    patch.object(resident_artifact, "load_stack", return_value=loaded),
                ):
                    if admitted:
                        self.assertIs(
                            canary.export_canary_stack("vf", {}, self.directory),
                            loaded,
                        )
                    else:
                        with self.assertRaisesRegex(
                            RuntimeError, "unexpected resident export scope"
                        ):
                            canary.export_canary_stack("vf", {}, self.directory)


class ScriptedProvider:
    """Explicitly injected CPU provider returning scripted projection outputs."""

    def __init__(self, outputs, sync_ok=True):
        self.torch, self.device = torch, torch.device("cpu")
        self.launches, self.synchronized_launches = 0, 0
        self.outputs, self.sync_ok = list(outputs), sync_ok

    def gemm_qpn2_moe(self, *args):
        self.launches += 1
        return self.outputs.pop(0)

    def synchronize(self):
        if self.sync_ok:
            self.synchronized_launches = self.launches


class ExecuteRustRankTests(unittest.TestCase):
    """execute_rust_rank with a scripted Rust boundary and scripted provider."""

    def fixture(self, e=1, n13=128, n2=64, tokens=2):
        k = 64
        plan = dict(
            skip=False,
            output_columns=64,
            source_tokens=[index % 2 for index in range(tokens)],
            block_expert=[0],
            sorted_slots=[0, 1] + [2] * 6,
            ntpp=8,
            completion_slots=[0, 1],
            projection13=dict(n=n13, k=k, in_div=1, splitk=4, nacc=1),
            projection2=dict(n=n2, k=k, in_div=1, splitk=4, nacc=1),
            stamp=dict(run_id="cpu-a43"),
            rank=2,
        )
        stack = dict(
            shape=[e, n13, k, n2, k],
            codes13=[0x22] * (e * n13 * k // 2),
            scales13=[0x38] * (e * n13 * k // 16),
            codes2=[0x22] * (e * n2 * k // 2),
            scales2=[0x38] * (e * n2 * k // 16),
            gscales13=[0.25] * (e * n13 // 32),
            gscales2=[0.25] * (e * n2 // 32),
        )
        request = dict(weights=dict(cpu=True), dispatch=dict(tokens=2))
        context = dict(launch=request, plan=copy.deepcopy(plan), stack=stack)
        return plan, context

    def outputs(self, n13=128, n2=64, h1=None, h2=None):
        return [
            torch.full((2, n13), 0.5, dtype=torch.float16) if h1 is None else h1,
            torch.full((2, n2), 0.25, dtype=torch.float16) if h2 is None else h2,
        ]

    def execute(self, plan, context, provider, x=None):
        self.commands = []
        stack = copy.deepcopy(context["stack"])

        def rust_call(binary, command, request, **_):
            self.commands.append(command)
            if command == "resident-launch":
                return copy.deepcopy(plan)
            if command == "resident-pack":
                return copy.deepcopy(stack)
            if command == "resident-completions":
                return dict(completions=["receipt"])
            raise AssertionError("unexpected Rust command " + command)

        if x is None:
            x = torch.full((2, 64), 2.0**-6, dtype=torch.float16)
        with patch.object(canary, "rust_call", side_effect=rust_call):
            return canary.execute_rust_rank(provider, "vf", context, x)

    def refuse(self, error, message, plan, context, provider, x=None):
        with self.assertRaisesRegex(error, message):
            self.execute(plan, context, provider, x)
        self.assertNotIn("resident-completions", self.commands)

    def test_scripted_rank_control_reaches_completions(self):
        plan, context = self.fixture()
        provider = ScriptedProvider(self.outputs())
        self.assertEqual(self.execute(plan, context, provider), ["receipt"])
        self.assertEqual(provider.launches, 2)
        self.assertEqual(self.commands[-1], "resident-completions")

    def test_changed_launch_plan_is_refused(self):
        plan, context = self.fixture()
        context["plan"]["rank"] = 3  # Stack unchanged: only the plan differs.
        self.refuse(
            RuntimeError,
            "canary prepared identity or compressed snapshot changed",
            plan,
            context,
            ScriptedProvider(self.outputs()),
        )

    def test_activation_device_or_shape_mismatch_is_refused(self):
        message = "canary activation shape/device differs from prepared dispatch"
        for name, x in (
            (
                "device",
                torch.full((2, 64), 2.0**-6, dtype=torch.float16, device="meta"),
            ),
            ("rows", torch.full((3, 64), 2.0**-6, dtype=torch.float16)),
            ("columns", torch.full((2, 65), 2.0**-6, dtype=torch.float16)),
        ):
            with self.subTest(case=name):
                plan, context = self.fixture()
                provider = ScriptedProvider(self.outputs())
                self.refuse(ValueError, message, plan, context, provider, x)
                self.assertEqual(provider.launches, 0)

    def test_nonfinite_activation_is_refused_before_compaction(self):
        x = torch.full((2, 64), 2.0**-6, dtype=torch.float16)
        x[1, 3] = float("nan")
        plan, context = self.fixture()
        provider = ScriptedProvider(self.outputs())
        self.refuse(
            ValueError, "nonfinite canary activations", plan, context, provider, x
        )
        self.assertEqual(provider.launches, 0)

    def test_stack_outside_tiny_bound_is_refused(self):
        message = "Rust H1 path is bounded to two experts and tiny uniform planes"
        for name, shape in (
            ("experts", dict(e=3)),
            ("n13", dict(n13=64)),
            ("n2", dict(n2=32)),
            ("tokens", dict(tokens=9)),
        ):
            with self.subTest(case=name):
                plan, context = self.fixture(**shape)
                dims = dict(n13=shape.get("n13", 128), n2=shape.get("n2", 64))
                provider = ScriptedProvider(self.outputs(**dims))
                self.refuse(ValueError, message, plan, context, provider)
                self.assertEqual(provider.launches, 0)

    def test_gate_up_output_contract_is_enforced(self):
        message = "gate/up provider output differs from prepared shape"
        good = torch.full((2, 128), 0.5, dtype=torch.float16)
        for name, h1 in (
            ("shape", torch.full((3, 128), 0.5, dtype=torch.float16)),
            ("dtype", good.float()),
            ("device", good.to("meta")),
        ):
            with self.subTest(case=name):
                plan, context = self.fixture()
                provider = ScriptedProvider(self.outputs(h1=h1))
                self.refuse(ValueError, message, plan, context, provider)
                self.assertEqual(provider.launches, 1)

    def test_nonfinite_gate_up_output_is_refused(self):
        h1 = torch.full((2, 128), 0.5, dtype=torch.float16)
        h1[0, 0] = float("inf")
        plan, context = self.fixture()
        provider = ScriptedProvider(self.outputs(h1=h1))
        self.refuse(
            ValueError, "nonfinite gate/up provider output", plan, context, provider
        )

    def test_down_output_contract_is_enforced(self):
        message = "down provider output differs from prepared shape"
        good = torch.full((2, 64), 0.25, dtype=torch.float16)
        for name, h2 in (
            ("shape", torch.full((2, 63), 0.25, dtype=torch.float16)),
            ("dtype", good.float()),
            ("device", good.to("meta")),
        ):
            with self.subTest(case=name):
                plan, context = self.fixture()
                provider = ScriptedProvider(self.outputs(h2=h2))
                self.refuse(ValueError, message, plan, context, provider)
                self.assertEqual(provider.synchronized_launches, 0)

    def test_unproven_completion_is_refused(self):
        plan, context = self.fixture()
        provider = ScriptedProvider(self.outputs(), sync_ok=False)
        self.refuse(
            RuntimeError,
            "provider has not proved completion of its accepted launches",
            plan,
            context,
            provider,
        )


class CudaProofAndSampleTests(TempDirTest):
    def proof(self, **overrides):
        receipt = {
            "status": "CUDA_COMPILED",
            "architecture": "sm_70",
            "device_executed": False,
            "hashes": {"build/libvf_qpn2_moe_sm70.so": "abc"},
        }
        receipt.update(overrides)
        path = self.directory / ("proof-" + uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(receipt))
        return path

    def test_valid_proof_control(self):
        self.assertEqual(
            canary.check_cuda_proof(self.proof(), "abc")["status"], "CUDA_COMPILED"
        )

    def test_unqualified_receipt_is_refused(self):
        message = "expected a CUDA_COMPILED sm_70 receipt without device execution"
        for overrides in (
            dict(status="V100_EXECUTED"),
            dict(device_executed=True),
            dict(device_executed=None),
            dict(architecture="sm_80"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    canary.check_cuda_proof(self.proof(**overrides), "abc")

    def test_receipt_without_library_hash_is_refused(self):
        for hashes in ({}, {"build/libvf_qpn2_moe_sm70.so": ""}):
            with self.subTest(hashes=hashes):
                with self.assertRaisesRegex(
                    ValueError, "CUDA receipt does not identify the SM70 library"
                ):
                    canary.check_cuda_proof(self.proof(hashes=hashes), "abc")

    def sample(self, change):
        sample = json.loads(SAMPLE.read_text())
        change(sample)
        path = self.directory / ("sample-" + uuid.uuid4().hex + ".json")
        path.write_text(json.dumps(sample))
        return path

    def test_valid_sample_control(self):
        self.assertEqual(
            set(canary.check_sample(self.sample(lambda s: None))["planes"]),
            {"gate_proj", "up_proj", "down_proj"},
        )

    def test_sample_of_another_kind_is_refused(self):
        with self.assertRaisesRegex(
            ValueError, "H1 requires the pinned real checkpoint crop"
        ):
            canary.check_sample(self.sample(lambda s: s.update(kind="full_model")))

    def test_sample_with_other_planes_is_refused(self):
        def drop(sample):
            del sample["planes"]["down_proj"]

        def extra(sample):
            sample["planes"]["extra_proj"] = sample["planes"]["gate_proj"]

        for change in (drop, extra):
            with self.subTest(change=change.__name__):
                with self.assertRaisesRegex(
                    ValueError, "expected exactly gate/up/down crop planes"
                ):
                    canary.check_sample(self.sample(change))

    def test_crop_plane_other_than_64x64_is_refused(self):
        # A self-consistent 32x64 plane is valid for oracle_plane, so only the
        # 64x64 bound can refuse it.
        def shrink(sample):
            sample["planes"]["up_proj"] = plane(32, 64)

        with self.assertRaisesRegex(ValueError, "H1 real crop is bounded to 64x64"):
            canary.check_sample(self.sample(shrink))


class RunCasesTests(unittest.TestCase):
    """run_cases with every earlier stage stubbed, to reach the real-crop check."""

    def run_cases(self, sample):
        provider = ScriptedProvider([])
        provider.gemm_qpn2_moe = lambda *args: torch.zeros((args[7], args[8]))
        rank = dict(resident_order=[900, 3])
        from adapters import skinny_moe as adapter

        with (
            patch.object(canary, "compare", return_value=dict(status="PASS")),
            patch.object(canary, "rust_rank", return_value=(rank, {})),
            patch.object(canary, "prepare_rust_canary", return_value=[]),
            patch.object(canary, "rust_call", return_value=[[0.0]]),
            patch.object(
                adapter, "bind_rank_dispatch", return_value=torch.tensor([[True]])
            ),
            patch.object(adapter, "moe_apply", return_value=torch.zeros((2, 64))),
        ):
            return canary.run_cases(provider, sample, "vf", [])

    def test_nonvacuous_crop_control(self):
        cases = self.run_cases(dict(planes=dict(gate_proj=plane(64, 64))))
        self.assertEqual(cases[-1]["name"], "real_checkpoint_plane")

    def test_vacuous_checkpoint_oracle_is_refused(self):
        with self.assertRaisesRegex(
            AssertionError, "checkpoint oracle trace is vacuous"
        ):
            self.run_cases(dict(planes=dict(gate_proj=plane(64, 64, code=0x00))))


class MainCapabilityTests(TempDirTest):
    """main() capability admission, everything before it valid and CPU-only."""

    CAPABILITIES = dict(
        owned_quantized_buffers=True,
        host_launch_metadata=True,
        compact_output_receipts=True,
        compressed_resident_artifact=True,
    )

    def run_main(self, returncode=0, ok=True, **capabilities):
        proof = self.directory / ("proof-" + uuid.uuid4().hex + ".json")
        proof.write_text(
            json.dumps(
                {
                    "status": "CUDA_COMPILED",
                    "architecture": "sm_70",
                    "device_executed": False,
                    "hashes": {"build/libvf_qpn2_moe_sm70.so": "cpu-test"},
                }
            )
        )
        output = self.directory / ("h1-" + uuid.uuid4().hex + ".json")
        argv = [
            "--ack-hardware-gate",
            "--gpu-uuid",
            "CPU-NO-GPU",
            "--library",
            str(self.directory / "unused.so"),
            "--cudart",
            str(self.directory / "unused-runtime.so"),
            "--routing-binary",
            str(self.directory / "unused-vf"),
            "--out",
            str(output),
            "--cuda-proof",
            str(proof),
        ]
        payload = dict(ok=ok, result=dict(self.CAPABILITIES, **capabilities))
        response = SimpleNamespace(returncode=returncode, stdout=json.dumps(payload))
        environment = {k: v for k, v in os.environ.items() if k != "VF_H1_RECEIPT_DIR"}
        with (
            patch.dict(os.environ, environment, clear=True),
            patch.object(canary, "ROOT", self.directory),
            patch.object(canary, "sha", return_value="cpu-test"),
            patch.object(canary, "check_sample", return_value={"source": {}}),
            patch.object(canary.subprocess, "run", return_value=response),
            patch.object(
                canary, "QPN2Provider", side_effect=RuntimeError("CPU stop")
            ) as provider,
        ):
            with redirect_stdout(io.StringIO()):
                self.assertEqual(canary.main(argv), 2)
        report = json.loads(output.read_text())
        self.assertEqual(report["status"], "BLOCKED")
        return report["error"]["message"], provider.call_count

    def test_complete_capabilities_reach_the_provider(self):
        self.assertEqual(self.run_main(), ("CPU stop", 1))

    def test_missing_canary_capabilities_are_refused_before_provider(self):
        message = "Rust binary lacks canary integration capabilities"
        cases = [dict(returncode=1), dict(ok=False)] + [
            {name: False}
            for name in (
                "owned_quantized_buffers",
                "host_launch_metadata",
                "compact_output_receipts",
            )
        ]
        for case in cases:
            with self.subTest(case=case):
                self.assertEqual(self.run_main(**case), (message, 0))


if __name__ == "__main__":
    unittest.main()
