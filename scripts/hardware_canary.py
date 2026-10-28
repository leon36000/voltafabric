#!/usr/bin/env python3
"""Small SM70 QPN2 canary. Import and CPU contract tests never initialize CUDA.

VF_GPU_PYTHON must point to an existing CUDA 12-enabled Torch environment with
sm_70/compute_70 support. The isolated adapter-cpu Torch 2.10 CPU venv must never
be used to run this hardware command; no installation or rebuild is automated.
"""

from __future__ import annotations

import argparse
import ctypes as C
import ctypes.util
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import copy
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
import uuid

ROOT = Path(__file__).resolve().parents[1]
ABI_VERSION = 1
MAX_CANARY_BYTES = 16 * 1024 * 1024
# A single explicit A133 DeepSeek top-k=6 layer needs one six-expert compressed
# projection buffer at a time. This is a canary-only cap, not a production limit.
MAX_TOPK6_CANARY_BYTES = 128 * 1024 * 1024
# Uniform dyadic GEMMs are exact in FP16/FP32: no tolerance.
# Real crop: one-hot dyadic inputs eliminate summation cancellation; 2^-9 relative
# covers three FP16 scale/weight roundings (~3*2^-11), plus subnormal abs rounding.
# Uniform MoE additionally includes rounded FP16 SiLU/product and output combine.
POLICIES = {
    "uniform_gemm": {"atol": 0.0, "rtol": 0.0},
    "real_crop": {"atol": 2.0**-22, "rtol": 2.0**-9},
    "uniform_moe": {"atol": 2.0**-10, "rtol": 2.0**-9},
}


def bounded_full_expert_shape(
    *,
    full_expert_enabled,
    deepseek_probe_enabled,
    experts,
    slots,
    n,
    k,
    in_div,
    splitk,
    nacc,
    input_rows,
    expert_blocks,
    adapter_layer_enabled=False,
):
    """Admit predeclared full-expert shapes and one adapter-only split-K."""
    topk6 = (
        full_expert_enabled is True
        and deepseek_probe_enabled is True
        and adapter_layer_enabled is True
        and experts == 6
        and slots == 12
        and in_div == 1
        and nacc == 1
        and input_rows == 12
        and expert_blocks == 6
    )
    if topk6:
        return (n, k, splitk) in (
            (4608, 5120, 8),
            (5120, 2304, 8),
        )
    common = (
        full_expert_enabled is True
        and experts == 1
        and in_div == 1
        and nacc == 1
        and input_rows == slots
        and expert_blocks == 1
    )
    if not common:
        return False
    legacy = (n, k) in ((512, 2048), (2048, 512))
    if legacy and slots == 6 and splitk == 4:
        return True
    if deepseek_probe_enabled is not True or slots != 2:
        return False
    if (n, k) == (2304, 5120):
        return splitk == 4
    if (n, k) == (5120, 2304):
        return splitk == 4 or (adapter_layer_enabled is True and splitk == 8)
    # The resident adapter uses split-K=8 for the combined gate/up plane.
    return (n, k) == (4608, 5120) and splitk == 8


sys.path.insert(0, str(ROOT))
U8P, U16P = C.POINTER(C.c_uint8), C.POINTER(C.c_uint16)
F32P, I32P = C.POINTER(C.c_float), C.POINTER(C.c_int32)


class Params(C.Structure):
    """Exact vf_qpn2_moe_params v1. size_t/pointers follow the host C ABI."""

    _fields_ = [
        (name, C.c_uint32)
        for name in (
            "abi_version",
            "struct_size",
            "experts",
            "n",
            "k",
            "slots",
            "input_rows",
            "in_div",
            "splitk",
            "nacc",
        )
    ] + [
        ("codes", U8P),
        ("codes_bytes", C.c_size_t),
        ("scales", U8P),
        ("scales_bytes", C.c_size_t),
        ("gscales", F32P),
        ("gscales_count", C.c_size_t),
        ("x", U16P),
        ("x_count", C.c_size_t),
        ("y", U16P),
        ("y_count", C.c_size_t),
        ("block_expert", I32P),
        ("block_expert_count", C.c_size_t),
        ("sorted_slots", I32P),
        ("sorted_slots_count", C.c_size_t),
        ("ntpp", I32P),
        ("ntpp_count", C.c_size_t),
        ("block_expert_host", I32P),
        ("block_expert_host_count", C.c_size_t),
        ("sorted_slots_host", I32P),
        ("sorted_slots_host_count", C.c_size_t),
        ("ntpp_host", C.c_int32),
    ]


def abi_layout():
    return {
        "sizeof": C.sizeof(Params),
        "offsets": {name: getattr(Params, name).offset for name, _ in Params._fields_},
    }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_path(path):
    """Reject traversal and symlink escape before GPU actions or report writes."""
    candidate = Path(path).resolve()
    explicit = os.environ.get("VF_H1_RECEIPT_DIR")
    project = Path(explicit).resolve(strict=True) if explicit else ROOT.resolve()
    if not project.is_dir():
        raise ValueError("receipt root must be an existing directory")
    try:
        candidate.relative_to(project)
    except ValueError as exc:
        raise ValueError(
            "--out must stay below the resolved project root: " + str(project)
        ) from exc
    return candidate


def make_params(*, dimensions, buffers, expert_blocks, sorted_slots, ntpp):
    """CPU-only builder. Keep immutable host mirrors alive with this Params."""
    p = Params()
    p.abi_version, p.struct_size = ABI_VERSION, C.sizeof(Params)
    for name in (
        "experts",
        "n",
        "k",
        "slots",
        "input_rows",
        "in_div",
        "splitk",
        "nacc",
    ):
        value = dimensions[name]
        if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
            raise ValueError("dimension is not uint32: " + name)
        setattr(p, name, value)
    pairs = {
        "codes": (U8P, "codes_bytes"),
        "scales": (U8P, "scales_bytes"),
        "gscales": (F32P, "gscales_count"),
        "x": (U16P, "x_count"),
        "y": (U16P, "y_count"),
        "block_expert": (I32P, "block_expert_count"),
        "sorted_slots": (I32P, "sorted_slots_count"),
        "ntpp": (I32P, "ntpp_count"),
    }
    for name, (pointer_type, length_name) in pairs.items():
        address, length = buffers[name]
        if not isinstance(address, int) or not 0 <= address <= C.c_size_t(-1).value:
            raise ValueError("invalid opaque address")
        if not isinstance(length, int) or not 0 <= length <= C.c_size_t(-1).value:
            raise ValueError("invalid buffer capacity")
        setattr(p, name, C.cast(C.c_void_p(address), pointer_type))
        setattr(p, length_name, length)
    for values in (expert_blocks, sorted_slots, [ntpp]):
        if any(not isinstance(v, int) or not -(2**31) <= v < 2**31 for v in values):
            raise ValueError("metadata is not int32")
    blocks = (C.c_int32 * len(expert_blocks))(*expert_blocks)
    slots = (C.c_int32 * len(sorted_slots))(*sorted_slots)
    p.block_expert_host = C.cast(blocks, I32P)
    p.block_expert_host_count = len(expert_blocks)
    p.sorted_slots_host = C.cast(slots, I32P)
    p.sorted_slots_host_count = len(sorted_slots)
    p.ntpp_host = ntpp
    p._host_mirrors = (blocks, slots)
    return p


class Receipt:
    """Exclusive H1 attempt and atomically replaced, fsynced progress; no replay."""

    def __init__(self, path, report):
        if os.path.lexists(path):
            raise FileExistsError(
                "prior receipt path exists; inspect it before a new attempt"
            )
        self.path = output_path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_id = uuid.uuid4().hex
        report["execution_id"] = self.execution_id
        self._write_new(self.path, report)
        self._sync_directory()

    def _write_new(self, path, report):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _sync_directory(self):
        fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def update(self, report):
        previous = json.loads(self.path.read_text())
        if previous.get("execution_id") != self.execution_id:
            raise RuntimeError("H1 receipt ownership changed; refuse replacement")
        value = dict(report, execution_id=self.execution_id)
        temporary = self.path.with_name(
            self.path.name + "." + uuid.uuid4().hex + ".tmp"
        )
        try:
            self._write_new(temporary, value)
            os.replace(temporary, self.path)
            self._sync_directory()
        finally:
            temporary.unlink(missing_ok=True)


class CPUValidator:
    """Compile/load the C validator alone, with no CUDA dependency or runtime."""

    def __init__(self, source_root, evidence_dir):
        source_root, evidence_dir = Path(source_root), Path(evidence_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        self.path = evidence_dir / "libvf_qpn2_validate_cpu.so"
        command = [
            "gcc",
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-fPIC",
            "-shared",
            str(source_root / "cuda/qpn2_moe_validate.c"),
            "-o",
            str(self.path),
        ]
        run = subprocess.run(command, text=True, capture_output=True, timeout=30)
        (evidence_dir / "cpu_validator_compile.log").write_text(
            json.dumps(command) + "\n" + run.stdout + run.stderr
        )
        if run.returncode:
            raise RuntimeError("CPU validator compile failed: " + run.stderr)
        self.library = C.CDLL(str(self.path))
        self.library.vf_qpn2_moe_validate.argtypes = [C.POINTER(Params)]
        self.library.vf_qpn2_moe_validate.restype = C.c_int

    def validate(self, params):
        return self.library.vf_qpn2_moe_validate(C.byref(params))


class QPN2Provider:
    """Borrowed compressed CUDA tensors, real C launch, explicit sync required."""

    def __init__(
        self,
        library,
        cudart,
        device,
        expected_uuid,
        *,
        ack_hardware_gate=False,
        gate_path=None,
    ):
        # This guard precedes every Torch CUDA/runtime/library action.
        if not ack_hardware_gate:
            raise PermissionError("--ack-hardware-gate is required")
        gate_path = Path(gate_path or ROOT / "docs/HARDWARE_GATE.md")
        if not gate_path.is_file() or not expected_uuid:
            raise ValueError("hardware gate document and expected GPU UUID required")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-enabled Torch and an available canary GPU required"
            )
        if torch.version.cuda is None or not torch.version.cuda.startswith("12."):
            raise RuntimeError(
                "existing CUDA 12-enabled Torch is required with this FFI runtime"
            )
        arches = torch.cuda.get_arch_list()
        if not ({"sm_70", "compute_70"} & set(arches)):
            raise RuntimeError("Torch GPU operations have no compiled SM70 support")
        self.torch = torch
        self.device = torch.device("cuda", device)
        props = torch.cuda.get_device_properties(self.device)
        if (props.major, props.minor) != (7, 0):
            raise RuntimeError("canary accepts SM70 only")
        self.runtime = C.CDLL(str(Path(cudart).resolve()), mode=C.RTLD_GLOBAL)
        # CUDA Runtime 12's cudart does not export cudaDeviceGetUuid. Query the
        # selected ordinal through the Driver API, which owns CUdevice identity.
        driver_name = ctypes.util.find_library("cuda") or "libcuda.so.1"
        self.driver = C.CDLL(driver_name)
        self.driver.cuInit.argtypes = [C.c_uint]
        self.driver.cuInit.restype = C.c_int
        rc = self.driver.cuInit(0)
        if rc:
            raise RuntimeError("cuInit failed: " + str(rc))
        self.driver.cuDeviceGet.argtypes = [C.POINTER(C.c_int), C.c_int]
        self.driver.cuDeviceGet.restype = C.c_int
        driver_device = C.c_int()
        rc = self.driver.cuDeviceGet(C.byref(driver_device), device)
        if rc:
            raise RuntimeError("cuDeviceGet failed: " + str(rc))
        self.driver.cuDeviceGetUuid_v2.argtypes = [C.c_void_p, C.c_int]
        self.driver.cuDeviceGetUuid_v2.restype = C.c_int
        raw_uuid = (C.c_ubyte * 16)()
        rc = self.driver.cuDeviceGetUuid_v2(C.byref(raw_uuid), driver_device)
        if rc:
            raise RuntimeError("cuDeviceGetUuid_v2 failed: " + str(rc))
        actual_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw_uuid)))
        if actual_uuid.lower() != expected_uuid.lower():
            raise RuntimeError(
                "selected GPU UUID differs from explicitly reserved canary"
            )
        self.identity = {
            "uuid": actual_uuid,
            "ordinal": device,
            "name": props.name,
            "sm": "7.0",
            "total_memory_bytes": props.total_memory,
        }
        for symbol, key in (
            ("cudaDriverGetVersion", "driver_api_version"),
            ("cudaRuntimeGetVersion", "runtime_api_version"),
        ):
            function = getattr(self.runtime, symbol)
            function.argtypes, function.restype = [C.POINTER(C.c_int)], C.c_int
            value = C.c_int()
            rc = function(C.byref(value))
            if rc:
                raise RuntimeError(symbol + " failed: " + str(rc))
            self.identity[key] = value.value
        if not 12000 <= self.identity["runtime_api_version"] < 13000:
            raise RuntimeError("the FFI must use the CUDA 12 runtime")
        self.identity.update(
            torch_version=torch.__version__,
            torch_cuda=torch.version.cuda,
            torch_arch_list=arches,
        )
        self.library_path = Path(library).resolve()
        self.library = C.CDLL(str(self.library_path))
        self.library.vf_qpn2_moe_validate.argtypes = [C.POINTER(Params)]
        self.library.vf_qpn2_moe_validate.restype = C.c_int
        self.library.vf_qpn2_moe_launch.argtypes = [
            C.POINTER(Params),
            C.c_void_p,
            C.POINTER(C.c_int32),
        ]
        self.library.vf_qpn2_moe_launch.restype = C.c_int
        self.pending, self.admissions, self.launches, self.synchronized_launches = (
            [],
            0,
            0,
            0,
        )
        self.progress = None

    def _notify(self, phase, **details):
        callback = getattr(self, "progress", None)
        if callback is not None:
            callback(
                dict(
                    execution_phase=phase,
                    launches=self.launches,
                    synchronized_launches=self.synchronized_launches,
                    pending_requests=len(self.pending),
                    may_have_executed=bool(self.pending),
                    **details,
                )
            )

    def gemm_qpn2_moe(
        self,
        x,
        codes,
        scales,
        global_scales,
        expert_blocks,
        sorted_slots,
        ntpp,
        s,
        n,
        k,
        in_div,
        splitk,
        nacc,
    ):
        t = self.torch
        tensors = (x, codes, scales, global_scales, expert_blocks, sorted_slots, ntpp)
        expected_dtypes = (
            t.float16,
            t.uint8,
            t.uint8,
            t.float32,
            t.int32,
            t.int32,
            t.int32,
        )
        for tensor, dtype in zip(tensors, expected_dtypes):
            if (
                tensor.device != self.device
                or tensor.dtype != dtype
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "FFI tensors must be contiguous typed buffers on the canary GPU"
                )
        if x.ndim != 2 or global_scales.ndim != 2 or ntpp.numel() != 1:
            raise ValueError("malformed activation/global/ntpp shape")
        e = global_scales.shape[0]
        tiny = 1 <= e <= 2 and 1 <= s <= 8 and 0 < n <= 128 and 0 < k <= 256
        full = bounded_full_expert_shape(
            full_expert_enabled=getattr(self, "full_expert_enabled", False),
            deepseek_probe_enabled=getattr(self, "deepseek_probe_enabled", False),
            adapter_layer_enabled=getattr(self, "adapter_layer_enabled", False),
            experts=e,
            slots=s,
            n=n,
            k=k,
            in_div=in_div,
            splitk=splitk,
            nacc=nacc,
            input_rows=x.shape[0],
            expert_blocks=expert_blocks.numel(),
        )
        adapter_full = (
            getattr(self, "adapter_full_expert_enabled", False) is True
            and e == 1
            and s == 6
            and (n, k, splitk, nacc) in ((1024, 2048, 8, 1), (2048, 512, 8, 1))
            and in_div == 1
            and x.shape[0] == 6
            and expert_blocks.numel() == 1
        )
        if not (
            (tiny or full or adapter_full)
            and x.shape[1] == k
            and global_scales.shape[1] == n // 32
        ):
            raise ValueError(
                "H1 requires the tiny crop or explicit bounded full-expert mode"
            )
        topk6 = (
            e == 6
            and s == 12
            and x.shape[0] == 12
            and in_div == 1
            and expert_blocks.numel() == 6
            and getattr(self, "adapter_layer_enabled", False) is True
        )
        if (
            codes.numel() != e * n * k // 2
            or scales.numel() != e * n * k // 16
            or not 1 <= expert_blocks.numel() <= (6 if topk6 else 2)
            or sorted_slots.numel() != expert_blocks.numel() * 8
        ):
            raise ValueError(
                "compressed capacities/metadata exceed the admitted H1 shape"
            )
        # The C validator checks opaque capacities/metadata, not DEVICE values.
        # Direct GEMM canaries also enter here without the MoE adapter's guards.
        # Revalidate content every call; a mutable tensor pointer is not a cache key.
        from adapters.skinny_moe import _validate_sm70_scale_domain

        if not bool(t.isfinite(x).all()):
            raise ValueError("nonfinite FP16 provider activations")
        _validate_sm70_scale_domain(scales, global_scales, k)
        # Buffers remain borrowed: callers must not mutate them while pending.
        # These diagnostic GPU reductions are not performance measurements.
        stream = t.cuda.current_stream(self.device)
        # CPU copies synchronize prior metadata production. Fresh immutable mirrors
        # are uploaded on this same stream; device pointers are never read by CPU.
        host_blocks = expert_blocks.detach().cpu().contiguous().clone()
        host_slots = sorted_slots.detach().cpu().contiguous().clone()
        host_ntpp = ntpp.detach().cpu().contiguous().clone()
        blocks = host_blocks.to(self.device).contiguous()
        slots = host_slots.to(self.device).contiguous()
        padded = host_ntpp.to(self.device).contiguous()
        y = t.full((s, n), float("nan"), dtype=t.float16, device=self.device)
        held = (x, codes, scales, global_scales, blocks, slots, padded, y)
        required = sum(a.numel() * a.element_size() for a in held)
        memory_limit = MAX_TOPK6_CANARY_BYTES if topk6 else MAX_CANARY_BYTES
        if required > memory_limit:
            raise ValueError("H1 canary memory cap exceeded")
        addresses = {
            name: (tensor.data_ptr(), tensor.numel())
            for name, tensor in zip(
                (
                    "x",
                    "codes",
                    "scales",
                    "gscales",
                    "block_expert",
                    "sorted_slots",
                    "ntpp",
                    "y",
                ),
                held,
            )
        }
        p = make_params(
            dimensions=dict(
                experts=e,
                n=n,
                k=k,
                slots=s,
                input_rows=x.shape[0],
                in_div=in_div,
                splitk=splitk,
                nacc=nacc,
            ),
            buffers=addresses,
            expert_blocks=host_blocks.reshape(-1).tolist(),
            sorted_slots=host_slots.reshape(-1).tolist(),
            ntpp=int(host_ntpp.item()),
        )
        status = self.library.vf_qpn2_moe_validate(C.byref(p))
        if status:
            raise ValueError(
                "C FFI metadata validation rejected canary: " + str(status)
            )
        # Hold references BEFORE launching, including possible async-failure paths.
        self.pending.append((stream, held, host_blocks, host_slots, host_ntpp, p))
        cuda_error = C.c_int32()
        self._notify("ENQUEUE_PENDING", projection=dict(experts=e, n=n, k=k, slots=s))
        with t.cuda.device(self.device):
            # Count entry before the opaque ABI call. If that call raises, the
            # owner must treat execution as unknown and must not replay it.
            self.admissions += 1
            status = self.library.vf_qpn2_moe_launch(
                C.byref(p), C.c_void_p(stream.cuda_stream), C.byref(cuda_error)
            )
        if status or cuda_error.value:
            raise RuntimeError(
                "real QPN2 launch failed: FFI=%d CUDA=%d" % (status, cuda_error.value)
            )
        self.launches += 1
        self._notify("ENQUEUED")
        return y

    def synchronize(self):
        # Failure propagates and keeps borrowed buffers alive; no PASS is emitted.
        for stream in {entry[0] for entry in self.pending}:
            stream.synchronize()
        self.synchronized_launches = self.launches
        self.pending.clear()
        self._notify("SYNCHRONIZED")


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def fp8_e4m3(byte):
    if not isinstance(byte, int) or not 0 <= byte <= 255:
        raise ValueError("FP8 code is not a byte")
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        raise ValueError("nonfinite E4M3 scale")
    magnitude = (
        mantissa * 2.0**-9
        if exponent == 0
        else (1 + mantissa / 8) * 2.0 ** (exponent - 7)
    )
    return -magnitude if byte & 128 else magnitude


def oracle_plane(plane, activations):
    """Independent scalar checkpoint-native FP4/E4M3 decode, never QPN fragments."""
    n, k = plane["rows"], plane["columns"]
    if not (0 < n <= 128 and n % 32 == 0 and 0 < k <= 256 and k % 64 == 0):
        raise ValueError("crop dimensions outside H1")
    codes, scales, factor = plane["codes"], plane["scales"], plane["global_dequant"]
    if len(codes) != n * k // 2 or len(scales) != n * k // 16:
        raise ValueError("crop compressed lengths differ")
    if not math.isfinite(factor) or factor < 0:
        raise ValueError("invalid checkpoint global scale")
    lut = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    result = []
    for x in activations:
        if len(x) != k or any(not math.isfinite(v) for v in x):
            raise ValueError("invalid oracle input")
        row = []
        for output in range(n):
            terms = []
            for column, activation in enumerate(x):
                packed = codes[output * (k // 2) + column // 2]
                if not isinstance(packed, int) or not 0 <= packed <= 255:
                    raise ValueError("FP4 code is not a byte")
                nibble = (packed >> (4 * (column & 1))) & 15
                code = (-1 if nibble & 8 else 1) * lut[nibble & 7]
                scale = fp8_e4m3(scales[output * (k // 16) + column // 16])
                terms.append(activation * code * scale * factor)
            row.append(math.fsum(terms))
        result.append(row)
    return result


def compare(actual, expected, policy, *, synchronized=False):
    if not synchronized:
        raise RuntimeError(
            "GPU comparison requires explicit successful stream synchronization"
        )
    if len(actual) != len(expected) or any(
        len(a) != len(e) for a, e in zip(actual, expected)
    ):
        raise AssertionError("output shape differs from oracle")
    maximum, maximum_relative = 0.0, 0.0
    for row, reference in zip(actual, expected):
        for value, wanted in zip(row, reference):
            if not math.isfinite(value) or not math.isfinite(wanted):
                raise AssertionError("nonfinite/unwritten canary output")
            error = abs(value - wanted)
            maximum = max(maximum, error)
            maximum_relative = max(
                maximum_relative, error / abs(wanted) if wanted else 0.0
            )
            if error > policy["atol"] + policy["rtol"] * abs(wanted):
                raise AssertionError(
                    "fixed predeclared tolerance exceeded: %.9g vs %.9g"
                    % (value, wanted)
                )
    return dict(
        status="PASS",
        max_abs_error=maximum,
        max_relative_error=maximum_relative,
        atol=policy["atol"],
        rtol=policy["rtol"],
        synchronized=True,
    )


def align_reference(ids, block, experts):
    """Tiny canary alignment only; no serving scheduler or backend fallback."""
    import torch

    groups, blocks = [], []
    values = ids.detach().cpu().reshape(-1).tolist()
    for expert in range(experts):
        selected = [slot for slot, owner in enumerate(values) if owner == expert]
        selected += [len(values)] * ((-len(selected)) % block)
        groups.extend(selected)
        blocks.extend([expert] * (len(selected) // block))
    return (
        torch.tensor(groups, dtype=torch.int32, device=ids.device),
        torch.tensor(blocks, dtype=torch.int32, device=ids.device),
        torch.tensor(len(groups), dtype=torch.int32, device=ids.device),
    )


def routing_payload(preferred):
    stamp = dict(
        run_id="hardware-canary-H1",
        model_revision="uniform-dyadic-v1",
        plan_id="sparse-replicas-owner-%d" % preferred,
        layer=0,
        batch=1,
        epoch=7,
    )
    return dict(
        hidden=64,
        experts=[10, 25, 99],
        routes=[
            [dict(expert=10, weight=0.25), dict(expert=99, weight=0.5)],
            [dict(expert=99, weight=0.75), dict(expert=25, weight=1.0)],
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
        stamp=stamp,
        preferred_owners={"99": preferred},
        resident_order={"2": [900, 3], "9": [3, 0], "17": []},
    )


def rust_rank(binary, preferred):
    payload = routing_payload(preferred)
    run = subprocess.run(
        [str(binary), "routing", "-"],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=15,
    )
    data = json.loads(run.stdout)
    if run.returncode or not data.get("ok"):
        raise RuntimeError("real Rust routing rejected canary plan: " + run.stdout)
    rank = next(r for r in data["result"]["ranks"] if r["rank"] == 2)
    return rank, payload["stamp"]


def rust_call(binary, command, request, *, pass_fds=(), raw_body=None):
    payload = json.dumps(request).encode("utf-8")
    if raw_body is not None:
        if type(raw_body) is not bytes:
            raise TypeError("raw Rust request body must be immutable bytes")
        payload += b"\n" + raw_body
    run = subprocess.run(
        [str(binary), command, "-"],
        input=payload,
        capture_output=True,
        timeout=30,
        pass_fds=pass_fds,
    )
    response = json.loads(run.stdout)
    if run.returncode or not response.get("ok"):
        stdout = (
            run.stdout.decode("utf-8", errors="replace")
            if isinstance(run.stdout, bytes)
            else str(run.stdout)
        )
        raise RuntimeError("Rust canary " + command + " rejected: " + stdout[:2000])
    return response["result"]


def uniform_expert(global_id, local):
    factors = {10: (0.5, 1.0, 0.5), 99: (0.25, 0.5, 0.25), 25: (0.125, 0.25, 0.125)}

    def plane(factor):
        return dict(
            rows=64,
            columns=64,
            codes=[0x22] * 2048,
            scales=[0x38] * 256,
            global_dequant=factor,
        )

    gate, up, down = factors[global_id]
    return dict(
        id=global_id, local=local, gate=plane(gate), up=plane(up), down=plane(down)
    )


def export_canary_stack(binary, weights, directory):
    """New exclusive files only; an uncertain export is never retried implicitly."""
    from scripts.resident_artifact import load_stack

    path = Path(directory) / ("resident-" + uuid.uuid4().hex + ".safetensors")
    result = rust_call(
        binary,
        "resident-export",
        dict(source=dict(kind="resident", request=weights), output=str(path)),
    )
    if (
        result["scope"] != "CPU_COMPRESSED_EXPORT_ONLY"
        or result["cuda_executed"] is not False
    ):
        raise RuntimeError("unexpected resident export scope")
    artifact = load_stack(path, expected=result["meta"])
    return artifact


def prepare_rust_canary(binary, problem, *, resident_dir=None):
    dispatched = rust_call(binary, "routing", problem)
    contexts = []
    for rank in dispatched["ranks"]:
        experts = [
            uniform_expert(r["expert"], r["local"])
            for r in problem["registry"]["copies"]
            if r["rank"] == rank["rank"]
        ]
        weights = dict(
            model_revision=problem["stamp"]["model_revision"],
            layer=0,
            rank=rank["rank"],
            registry=copy.deepcopy(problem["registry"]),
            resident_order=rank["resident_order"],
            experts=experts,
        )
        request = dict(
            weights=weights,
            dispatch=rank,
            expected_stamp=copy.deepcopy(problem["stamp"]),
            options=dict(splitk13=4, splitk2=4, max_workspace_bytes=MAX_CANARY_BYTES),
        )
        context = dict(
            launch=request,
            plan=rust_call(binary, "resident-launch", request),
            routes=copy.deepcopy(problem["routes"]),
        )
        if resident_dir is None:
            context["stack"] = rust_call(binary, "resident-pack", weights)
        else:
            artifact = export_canary_stack(binary, weights, resident_dir)
            context.update(
                stack=copy.deepcopy(artifact.metadata),
                artifact=artifact,
                resident_dir=Path(resident_dir),
            )
        contexts.append(context)
    return contexts


def _canary_route_tensors(provider, context):
    """Bounded H1 logical input, retained independently of selected dispatch slots."""
    t, device = provider.torch, provider.device
    routes = context.get("routes")
    tokens = context["launch"]["dispatch"]["tokens"]
    if (
        not isinstance(routes, list)
        or len(routes) != tokens
        or not 1 <= tokens <= 8
        or not isinstance(routes[0], list)
        or not 1 <= len(routes[0]) <= 8
    ):
        raise ValueError("H1 logical route dimensions differ")
    width = len(routes[0])
    ids, weights = [], []
    for row in routes:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError("H1 logical routes must be rectangular")
        row_ids, row_weights = [], []
        for choice in row:
            if not isinstance(choice, dict) or set(choice) != {"expert", "weight"}:
                raise ValueError("H1 logical route fields differ")
            expert, weight = choice["expert"], choice["weight"]
            if (
                type(expert) is not int
                or not 0 <= expert < MAX_CANARY_BYTES // 8
                or type(weight) not in (int, float)
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("H1 logical route value outside bounded domain")
            row_ids.append(expert)
            row_weights.append(weight)
        if len(set(row_ids)) != width:
            raise ValueError("duplicate logical expert within token")
        ids.append(row_ids)
        weights.append(row_weights)
    return (
        t.tensor(ids, dtype=t.int64, device=device),
        t.tensor(weights, dtype=t.float64, device=device),
    )


def _local_rank_values(provider, context, x, tensors, ids, weights):
    """Borrow packed planes, execute local slots, restore Rust completion order.

    This canary does not own a persistent GPU allocation. No routing coefficient
    is applied here: the existing Rust Collector applies it exactly once.
    """
    from adapters import skinny_moe as adapter

    t, device = provider.torch, provider.device
    request, plan = context["launch"], context["plan"]
    resident = request["weights"]["resident_order"]
    experts = request["weights"]["experts"]
    domain = max([int(ids.max())] + [v["id"] for v in experts]) + 1
    if domain > MAX_CANARY_BYTES // 8:
        raise ValueError("H1 expert map exceeds byte budget")
    layer = SimpleNamespace(
        skinny_moe_shape=tuple(context["stack"]["shape"]),
        skinny_moe_splitk13=request["options"]["splitk13"],
        skinny_moe_splitk2=request["options"]["splitk2"],
        apply_router_weight_on_input=False,
        expert_map=t.full((domain,), -1, dtype=t.int64, device=device),
    )
    physical = {local: index for index, local in enumerate(resident)}
    for expert in experts:
        layer.expert_map[expert["id"]] = physical[expert["local"]]
    for suffix in ("13", "2"):
        for attr, name in (("qc", "codes"), ("qs", "scales"), ("gs", "gscales")):
            setattr(layer, "skinny_moe_" + attr + suffix, tensors[name + suffix])
    mask = adapter.bind_rank_dispatch(
        layer, request["dispatch"], request["expected_stamp"], ids, weights
    )
    dispatch_slots = request["dispatch"]["slots"]
    token_ids = sorted({entry["slot"]["token"] for entry in dispatch_slots})
    if int(mask.sum().item()) != len(dispatch_slots):
        raise ValueError("logical route binding differs from Rust local dispatch")
    token_rows = t.tensor(token_ids, dtype=t.int64, device=device)
    identities, values = adapter.moe_apply_dispatch_local(
        layer,
        x.index_select(0, token_rows),
        request["dispatch"],
        request["expected_stamp"],
        token_ids=token_ids,
        kernel=provider,
        align=align_reference,
    )
    expected = [
        (v["token"], v["slot"], v["expert"], v["rank"], v["local"])
        for v in plan["completion_slots"]
    ]
    if (
        not isinstance(identities, list)
        or any(
            type(identity) is not tuple
            or len(identity) != 5
            or any(type(value) is not int for value in identity)
            for identity in identities
        )
        or len(identities) != len(expected)
        or len(set(identities)) != len(identities)
        or set(identities) != set(expected)
        or not isinstance(values, t.Tensor)
        or tuple(values.shape) != (len(expected), plan["output_columns"])
        or values.dtype != t.float16
        or values.device != device
    ):
        raise ValueError("compact adapter slot identities or values are malformed")
    # A bijection, not an assumption about dispatch order or compact row order.
    row_for_identity = {identity: row for row, identity in enumerate(identities)}
    permutation = t.tensor(
        [row_for_identity[identity] for identity in expected],
        dtype=t.int64,
        device=device,
    )
    return values.index_select(0, permutation)


def execute_rust_rank(provider, binary, context, x, *, use_local_adapter=False):
    """H1 only. Rust owned buffers/metadata -> real provider -> sync -> Rust receipts.
    CPU providers are only explicitly injected by tests; no automatic fallback.
    """
    request = context["launch"]
    plan = rust_call(binary, "resident-launch", request)
    artifact = context.get("artifact")
    if artifact is None:
        stack = rust_call(binary, "resident-pack", request["weights"])
        unchanged = stack == context["stack"]
    else:
        from scripts.resident_artifact import _validate_info

        # Re-export current source content and compare owned CPU bytes. Frozen fields
        # and tensor addresses do not make caller-owned contents immutable.
        fresh = export_canary_stack(binary, request["weights"], context["resident_dir"])
        _validate_info(context["stack"])
        _validate_info(artifact.metadata)
        stack = fresh.metadata
        unchanged = stack == context["stack"] == artifact.metadata
        unchanged = unchanged and set(fresh.tensors) == set(artifact.tensors)
        if unchanged:
            for name, tensor in fresh.tensors.items():
                prepared = artifact.tensors[name]
                if (
                    prepared.device != tensor.device
                    or prepared.dtype != tensor.dtype
                    or prepared.shape != tensor.shape
                    or not prepared.is_contiguous()
                    or not provider.torch.equal(
                        prepared.view(provider.torch.uint8).reshape(-1),
                        tensor.view(provider.torch.uint8).reshape(-1),
                    )
                ):
                    unchanged = False
                    break
    if plan != context["plan"] or not unchanged:
        raise RuntimeError("canary prepared identity or compressed snapshot changed")
    t, device = provider.torch, provider.device
    if x.device != device or tuple(x.shape) != (
        request["dispatch"]["tokens"],
        plan["output_columns"],
    ):
        raise ValueError(
            "canary activation shape/device differs from prepared dispatch"
        )
    if not bool(t.isfinite(x).all()):
        raise ValueError("nonfinite canary activations")
    route_tensors = (
        _canary_route_tensors(provider, context) if use_local_adapter else None
    )
    fp16_le_bytes = b""
    if not plan["skip"]:
        e, n13, k13, n2, k2 = stack["shape"]
        if not (
            1 <= e <= 2
            and n13 == 128
            and n2 == 64
            and k13 == k2 == 64
            and len(plan["source_tokens"]) <= 8
        ):
            raise ValueError(
                "Rust H1 path is bounded to two experts and tiny uniform planes"
            )
        if artifact is not None:
            tensors = {
                name: tensor.to(device).contiguous()
                for name, tensor in fresh.tensors.items()
            }
        else:
            tensors = {}
            for name in ("codes13", "scales13", "codes2", "scales2"):
                tensors[name] = t.tensor(
                    stack[name], dtype=t.uint8, device=device
                ).contiguous()
            tensors["gscales13"] = t.tensor(
                stack["gscales13"], dtype=t.float32, device=device
            ).reshape(e, n13 // 32)
            tensors["gscales2"] = t.tensor(
                stack["gscales2"], dtype=t.float32, device=device
            ).reshape(e, n2 // 32)
        if use_local_adapter:
            h2 = _local_rank_values(provider, context, x, tensors, *route_tensors)
        else:
            blocks = t.tensor(plan["block_expert"], dtype=t.int32, device=device)
            slots = t.tensor(plan["sorted_slots"], dtype=t.int32, device=device)
            padded = t.tensor(plan["ntpp"], dtype=t.int32, device=device)
            source = t.tensor(plan["source_tokens"], dtype=t.int64, device=device)
            compact = x.index_select(0, source).to(t.float16).contiguous()
            if not bool(t.isfinite(compact).all()):
                raise ValueError("FP16 compact input overflow")
            s = len(plan["completion_slots"])
            a, b = plan["projection13"], plan["projection2"]
            h1 = provider.gemm_qpn2_moe(
                compact,
                tensors["codes13"],
                tensors["scales13"],
                tensors["gscales13"],
                blocks,
                slots,
                padded,
                s,
                a["n"],
                a["k"],
                a["in_div"],
                a["splitk"],
                a["nacc"],
            )
            if (
                tuple(h1.shape) != (s, n13)
                or h1.dtype != t.float16
                or h1.device != device
            ):
                raise ValueError("gate/up provider output differs from prepared shape")
            if not bool(t.isfinite(h1).all()):
                raise ValueError("nonfinite gate/up provider output")
            act = (t.nn.functional.silu(h1[:, :k2]) * h1[:, k2:]).contiguous()
            if not bool(t.isfinite(act).all()):
                raise ValueError("nonfinite SwiGLU activation")
            h2 = provider.gemm_qpn2_moe(
                act,
                tensors["codes2"],
                tensors["scales2"],
                tensors["gscales2"],
                blocks,
                slots,
                padded,
                s,
                b["n"],
                b["k"],
                b["in_div"],
                b["splitk"],
                b["nacc"],
            )
            if (
                tuple(h2.shape) != (s, n2)
                or h2.dtype != t.float16
                or h2.device != device
            ):
                raise ValueError("down provider output differs from prepared shape")
        if not bool(t.isfinite(h2).all()):
            raise ValueError("nonfinite down provider output")
        provider.synchronize()
        if provider.synchronized_launches != provider.launches:
            raise RuntimeError(
                "provider has not proved completion of its accepted launches"
            )
        host_bits = h2.detach().contiguous().view(t.int16).to("cpu").numpy()
        fp16_le_bytes = host_bits.astype("<i2", copy=False).tobytes(order="C")
    output = dict(
        stamp=plan["stamp"],
        rank=plan["rank"],
        rows=len(plan["completion_slots"]),
        columns=plan["output_columns"],
    )
    return rust_call(
        binary,
        "resident-completions",
        dict(launch=request, output=output),
        raw_body=fp16_le_bytes,
    )["completions"]


def full_uniform_moe_oracle(problem):
    """Independent scalar dyadic/FP16 stages and f64 collector sum."""
    factors = {10: (0.5, 1.0, 0.5), 99: (0.25, 0.5, 0.25), 25: (0.125, 0.25, 0.125)}
    output = []
    for token, route in enumerate(problem["routes"]):
        total = 0.0
        for choice in route:
            gate_factor, up_factor, down_factor = factors[choice["expert"]]
            dot = 64 * 2.0 ** (-6 - token)
            gate, up = half(dot * gate_factor), half(dot * up_factor)
            act = half(half(gate / (1 + math.exp(-gate))) * up)
            down = half(64 * act * down_factor)
            total += choice["weight"] * down
        output.append([total] * 64)
    return output


def record_progress(report, receipt, provider, event):
    report.update(event)
    if report["status"] not in ("FAIL", "BLOCKED", "GPU_CANARY_PASS"):
        report["status"] = "RUNNING"
    if provider.launches:
        report["cuda_executed"] = True
    receipt.update(report)


def uniform_layer(torch, device):
    layer = SimpleNamespace(
        skinny_moe_shape=(2, 128, 64, 64, 64),
        skinny_moe_splitk13=4,
        skinny_moe_splitk2=4,
        apply_router_weight_on_input=False,
    )
    for suffix, n in (("13", 128), ("2", 64)):
        setattr(
            layer,
            "skinny_moe_qc" + suffix,
            torch.full((2 * n * 64 // 2,), 0x22, dtype=torch.uint8, device=device),
        )
        setattr(
            layer,
            "skinny_moe_qs" + suffix,
            torch.full((2 * n * 64 // 16,), 0x38, dtype=torch.uint8, device=device),
        )
    layer.skinny_moe_gs13 = torch.tensor(
        [[0.25, 0.25, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
        dtype=torch.float32,
        device=device,
    )
    layer.skinny_moe_gs2 = torch.tensor(
        [[0.25, 0.25], [0.5, 0.5]], dtype=torch.float32, device=device
    )
    layer.expert_map = torch.full((100,), -1, dtype=torch.int64, device=device)
    layer.expert_map[99], layer.expert_map[10] = 0, 1
    return layer


def uniform_moe_oracle(preferred):
    # Explicit scalar FP16 stage roundings; no adapter/FFI kernel in this oracle.
    totals = [0.0, 0.0]
    selections = [(0, 1, 0.25)] + (
        [(0, 0, 0.5), (1, 0, 0.75)] if preferred == 2 else []
    )
    for token, expert, coefficient in selections:
        dot = 64 * (2.0 ** (-6 - token))
        gate = half(dot * (0.25 if expert == 0 else 0.5))
        up = half(dot * (0.5 if expert == 0 else 1.0))
        silu = half(gate / (1 + math.exp(-gate)))
        activation = half(silu * up)
        down = half(64 * activation * (0.25 if expert == 0 else 0.5))
        totals[token] = f32(totals[token] + f32(coefficient * down))
    return [[v] * 64 for v in totals]


def check_cuda_proof(path, library_sha256):
    """Refuse an SM70 library that no CUDA_COMPILED receipt qualifies, before CUDA."""
    if path is None:
        raise PermissionError(
            "--cuda-proof is required: the SM70 library must match a CUDA_COMPILED receipt"
        )
    proof = json.loads(Path(path).read_text())
    if (
        proof.get("status") != "CUDA_COMPILED"
        or proof.get("device_executed") is not False
        or proof.get("architecture") != "sm_70"
    ):
        raise ValueError(
            "expected a CUDA_COMPILED sm_70 receipt without device execution"
        )
    expected = proof.get("hashes", {}).get("build/libvf_qpn2_moe_sm70.so")
    if not expected:
        raise ValueError("CUDA receipt does not identify the SM70 library")
    if expected != library_sha256:
        raise ValueError(
            "SM70 library differs from its compilation receipt: "
            f"{library_sha256} is not {expected}"
        )
    return proof


def check_sample(path):
    sample = json.loads(Path(path).read_text())
    if (
        sample.get("kind") != "real_checkpoint_cropped_expert_not_full_model"
        or sample.get("source", {}).get("revision")
        != "1355db6a052410cfd62085d94b58866fd0f2c3c5"
        or sample["source"].get("model") != "nvidia/Qwen3.6-35B-A3B-NVFP4"
    ):
        raise ValueError("H1 requires the pinned real checkpoint crop")
    if set(sample.get("planes", {})) != {"gate_proj", "up_proj", "down_proj"}:
        raise ValueError("expected exactly gate/up/down crop planes")
    for plane in sample["planes"].values():
        if (plane["rows"], plane["columns"]) != (64, 64):
            raise ValueError("H1 real crop is bounded to 64x64")
        oracle_plane(plane, [[0.0] * 64])  # CPU shape/finite/code validation.
    return sample


def run_cases(provider, sample, binary, cases, *, resident_dir=None):
    t, device = provider.torch, provider.device
    from adapters import skinny_moe as adapter
    from tests.reference.qpn_prepack import _qpn_prepack

    configurations = ((2, 1), (4, 1), (8, 1), (8, 2), (16, 1), (16, 2))
    for split, nacc in configurations:
        k = 64 if split <= 4 else 256
        x = t.tensor([[2.0**-5] * k, [2.0**-4] * k], dtype=t.float16, device=device)
        codes = t.full((2 * 32 * k // 2,), 0x22, dtype=t.uint8, device=device)
        scales = t.full((2 * 32 * k // 16,), 0x38, dtype=t.uint8, device=device)
        globals_ = t.tensor([[0.25], [0.5]], dtype=t.float32, device=device)
        ids = t.tensor([[0, 1], [0, 1]], device=device)
        slots, blocks, padded = align_reference(ids, 8, 2)
        y = provider.gemm_qpn2_moe(
            x, codes, scales, globals_, blocks, slots, padded, 4, 32, k, 2, split, nacc
        )
        provider.synchronize()
        expected = [
            [k * 2.0 ** (-5 - 0) * 0.25] * 32,
            [k * 2.0**-5 * 0.5] * 32,
            [k * 2.0**-4 * 0.25] * 32,
            [k * 2.0**-4 * 0.5] * 32,
        ]
        result = compare(
            y.cpu().tolist(),
            expected,
            POLICIES["uniform_gemm"],
            synchronized=provider.synchronized_launches == provider.launches,
        )
        cases.append(dict(name="uniform_gemm", splitk=split, nacc=nacc, **result))
    for preferred in (2, 9):
        rank, stamp = rust_rank(binary, preferred)
        layer = uniform_layer(t, device)
        x = t.tensor([[2.0**-6] * 64, [2.0**-7] * 64], device=device)
        ids = t.tensor([[10, 99], [99, 25]], device=device)
        weights = t.tensor([[0.25, 0.5], [0.75, 1.0]], device=device)
        mask = adapter.bind_rank_dispatch(layer, rank, stamp, ids, weights)
        y = adapter.moe_apply(
            layer,
            x,
            weights,
            ids,
            None,
            None,
            kernel=provider,
            align=align_reference,
            selected_mask=mask,
        )
        provider.synchronize()
        result = compare(
            y.cpu().tolist(),
            uniform_moe_oracle(preferred),
            POLICIES["uniform_moe"],
            synchronized=provider.synchronized_launches == provider.launches,
        )
        cases.append(
            dict(
                name="uniform_adapter_moe",
                preferred_owner_99=preferred,
                physical_order=rank["resident_order"],
                mask=mask.cpu().tolist(),
                global_ids=ids.cpu().tolist(),
                **result,
            )
        )
    for use_local_adapter in (False, True):
        for preferred in (2, 9):
            problem = routing_payload(preferred)
            contexts = prepare_rust_canary(binary, problem, resident_dir=resident_dir)
            x = t.tensor(
                [[2.0**-6] * 64, [2.0**-7] * 64], dtype=t.float16, device=device
            )
            completions = []
            for context in reversed(contexts):
                completions.extend(
                    execute_rust_rank(
                        provider,
                        binary,
                        context,
                        x,
                        use_local_adapter=use_local_adapter,
                    )
                )
            result = rust_call(
                binary,
                "collect-routing",
                dict(problem=problem, completions=completions),
            )
            compared = compare(
                result,
                full_uniform_moe_oracle(problem),
                POLICIES["uniform_moe"],
                synchronized=provider.synchronized_launches == provider.launches,
            )
            cases.append(
                dict(
                    name=(
                        "rust_local_adapter_collect"
                        if use_local_adapter
                        else "rust_resident_launch_collect"
                    ),
                    resident_io="safetensors" if resident_dir is not None else "json",
                    preferred_owner_99=preferred,
                    ranks=[c["plan"]["rank"] for c in contexts],
                    completions=len(completions),
                    compressed_stack_bytes=[
                        c["stack"]["payload_bytes"] for c in contexts
                    ],
                    **compared,
                )
            )
    activations = [
        [0.5 if c == selected else 0.0 for c in range(64)] for selected in (0, 17)
    ]
    for name, plane in sample["planes"].items():
        cpu_codes = t.tensor(plane["codes"], dtype=t.uint8).reshape(64, 32)
        cpu_scales = t.tensor(plane["scales"], dtype=t.uint8).reshape(64, 4)
        qc, qs = _qpn_prepack(cpu_codes, cpu_scales)
        x = t.tensor(activations, dtype=t.float16, device=device)
        globals_ = t.full(
            (1, 2), plane["global_dequant"], dtype=t.float32, device=device
        )
        ids = t.tensor([[0], [0]], device=device)
        slots, blocks, padded = align_reference(ids, 8, 1)
        y = provider.gemm_qpn2_moe(
            x,
            qc.to(device),
            qs.to(device),
            globals_,
            blocks,
            slots,
            padded,
            2,
            64,
            64,
            1,
            4,
            1,
        )
        provider.synchronize()
        expected = oracle_plane(plane, activations)
        if max(abs(v) for row in expected for v in row) == 0:
            raise AssertionError("checkpoint oracle trace is vacuous")
        result = compare(
            y.cpu().tolist(),
            expected,
            POLICIES["real_crop"],
            synchronized=provider.synchronized_launches == provider.launches,
        )
        cases.append(
            dict(
                name="real_checkpoint_plane",
                plane=name,
                crop=[64, 64],
                one_hot_columns=[0, 17],
                **result,
            )
        )
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ack-hardware-gate",
        action="store_true",
        help="Confirm docs/HARDWARE_GATE.md, canary availability/reservation",
    )
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--cudart", type=Path, required=True)
    parser.add_argument("--routing-binary", type=Path, required=True)
    parser.add_argument(
        "--cuda-proof",
        type=Path,
        help="CUDA_COMPILED receipt qualifying the SM70 library; checked before any CUDA action",
    )
    parser.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--full-expert-sample",
        type=Path,
        help="Optional pinned full expert, three strict one-hot projection checks; full H1 remains OPEN",
    )
    parser.add_argument(
        "--full-expert-adapter",
        action="store_true",
        help="Also run the pinned compressed expert through Skinny's owned adapter path",
    )
    parser.add_argument(
        "--resident-io", choices=("json", "safetensors"), default="json"
    )
    args = parser.parse_args(argv)
    if args.full_expert_adapter and args.full_expert_sample is None:
        parser.error("--full-expert-adapter requires --full-expert-sample")
    # A prior hardware receipt may represent an interrupted enqueue; never replace it.
    if os.path.lexists(args.out):
        print(
            json.dumps(
                {
                    "status": "BLOCKED",
                    "error": "hardware receipt already exists; inspect it and the launcher before choosing a new run",
                }
            )
        )
        return 2
    try:
        args.out = output_path(args.out)
    except ValueError as exc:
        parser.error(str(exc))
    report = dict(
        schema_version=1,
        utc=datetime.now(timezone.utc).isoformat(),
        status="BLOCKED",
        scope="H1 bounded canary; full hardware qualification OPEN",
        h1_status="OPEN",
        cuda_executed=False,
        performance_measured=False,
        tolerance_policy=POLICIES,
        resident_io=args.resident_io,
        abi=abi_layout(),
        cases=[],
    )
    report.update(execution_phase="PRE_GPU", may_have_executed=False)
    try:
        receipt = Receipt(args.out, report)
    except FileExistsError:
        print(
            json.dumps(
                dict(
                    status="BLOCKED",
                    error="H1 receipt concurrently claimed; inspect it before choosing a new run",
                )
            )
        )
        return 2
    provider = None
    try:
        if not args.ack_hardware_gate:
            raise PermissionError(
                "--ack-hardware-gate is mandatory before any GPU action"
            )
        if not 0 <= args.device < 32:
            raise ValueError("invalid device ordinal")
        gate = ROOT / "docs/HARDWARE_GATE.md"
        sample = check_sample(args.sample)
        library_sha256 = sha(args.library)
        check_cuda_proof(args.cuda_proof, library_sha256)
        report.update(
            gate_sha256=sha(gate),
            checkpoint_sha256=sha(args.sample),
            checkpoint_source=sample["source"],
            library_sha256=library_sha256,
            cuda_proof=str(args.cuda_proof),
            cuda_proof_sha256=sha(args.cuda_proof),
            cudart_sha256=sha(args.cudart),
            routing_binary_sha256=sha(args.routing_binary),
            source_sha256={
                name: sha(ROOT / name)
                for name in (
                    "scripts/hardware_canary.py",
                    "scripts/resident_artifact.py",
                    "scripts/full_expert_canary.py",
                    "handoff/hardware_canary.sh",
                    "src/resident_artifact.rs",
                    "Cargo.toml",
                    "Cargo.lock",
                    "cuda/qpn2_moe_ffi.h",
                    "cuda/qpn2_moe_sm70.cu",
                    "cuda/qpn2_moe_core.cuh",
                    "cuda/qpn2_moe_validate.c",
                    "adapters/skinny_moe.py",
                    "src/resident.rs",
                    "src/launch.rs",
                    "src/completion.rs",
                    "src/moe.rs",
                )
            },
        )
        capabilities_run = subprocess.run(
            [str(args.routing_binary), "capabilities"],
            text=True,
            capture_output=True,
            timeout=15,
        )
        capabilities = json.loads(capabilities_run.stdout)
        if (
            capabilities_run.returncode
            or not capabilities.get("ok")
            or not all(
                capabilities["result"].get(k)
                for k in (
                    "owned_quantized_buffers",
                    "host_launch_metadata",
                    "compact_output_receipts",
                )
            )
        ):
            raise RuntimeError("Rust binary lacks canary integration capabilities")
        if (
            args.resident_io == "safetensors" or args.full_expert_sample is not None
        ) and capabilities["result"].get("compressed_resident_artifact") is not True:
            raise RuntimeError(
                "Rust binary lacks compressed resident export capability"
            )
        report["status"] = "STARTING"
        receipt.update(report)
        resident_dir = None
        if args.resident_io == "safetensors" or args.full_expert_sample is not None:
            resident_dir = args.out.parent / ("resident-" + receipt.execution_id)
            resident_dir.mkdir(mode=0o700)
            report["resident_artifact_directory"] = str(resident_dir)
            receipt.update(report)
        full_prepared = None
        if args.full_expert_sample is not None:
            from scripts.full_expert_canary import prepare_full_expert

            full_prepared = prepare_full_expert(
                args.routing_binary, args.full_expert_sample, resident_dir
            )
            report["full_expert"] = dict(
                sample_sha256=full_prepared.sample_sha256,
                payload_bytes=full_prepared.artifact.metadata["payload_bytes"],
                shape=full_prepared.artifact.metadata["shape"],
                full_layer_inference=False,
                gpu_status="PENDING",
                adapter_requested=args.full_expert_adapter,
                adapter_gpu_status=(
                    "PENDING" if args.full_expert_adapter else "NOT_REQUESTED"
                ),
            )
            receipt.update(report)
        provider = QPN2Provider(
            args.library,
            args.cudart,
            args.device,
            args.gpu_uuid,
            ack_hardware_gate=True,
            gate_path=gate,
        )

        provider.full_expert_enabled = full_prepared is not None
        provider.adapter_full_expert_enabled = args.full_expert_adapter

        def progress(event):
            record_progress(report, receipt, provider, event)

        provider.progress = progress
        report["gpu"] = provider.identity
        t = provider.torch
        free_before, total = t.cuda.mem_get_info(provider.device)
        allocated_before = t.cuda.memory_allocated(provider.device)
        t.cuda.reset_peak_memory_stats(provider.device)
        stream = t.cuda.Stream(device=provider.device)
        with t.cuda.stream(stream):
            run_cases(
                provider,
                sample,
                args.routing_binary,
                report["cases"],
                resident_dir=resident_dir,
            )
            if full_prepared is not None:
                from scripts.full_expert_canary import execute_full_expert

                execute_full_expert(provider, full_prepared, report["cases"])
                report["full_expert"]["gpu_status"] = (
                    "PROJECTIONS_MATCHED_STRICT_REFERENCE_ONLY"
                )
                if args.full_expert_adapter:
                    from scripts.full_expert_canary import execute_full_expert_adapter

                    execute_full_expert_adapter(
                        provider, full_prepared, report["cases"]
                    )
                    report["full_expert"]["adapter_gpu_status"] = (
                        "SKINNY_ADAPTER_MATCHED_FP64_DENSE_CPU_ORACLE"
                    )
            provider.synchronize()
        report.update(
            status="GPU_CANARY_PASS",
            launches=provider.launches,
            synchronized_launches=provider.synchronized_launches,
            memory=dict(
                limit_payload_bytes=MAX_CANARY_BYTES,
                allocated_before=allocated_before,
                allocated_after=t.cuda.memory_allocated(provider.device),
                peak_allocated=t.cuda.max_memory_allocated(provider.device),
                device_free_before=free_before,
                device_total=total,
            ),
            numerical_max_abs_error=max(c["max_abs_error"] for c in report["cases"]),
            no_performance_claim=True,
        )
    except Exception as exc:
        report.update(
            status="FAIL"
            if report["cuda_executed"] or report.get("may_have_executed")
            else "BLOCKED",
            error=dict(type=type(exc).__name__, message=str(exc)),
        )
        if provider is not None:
            report.update(
                launches=provider.launches,
                synchronized_launches=provider.synchronized_launches,
            )
            # Best effort completion for borrowed buffers; this never promotes FAIL.
            try:
                provider.synchronize()
            except Exception as cleanup:
                report["cleanup_error"] = str(cleanup)
    try:
        receipt.update(report)
    except Exception as persist_error:
        report.update(status="FAIL", receipt_update_error=str(persist_error))

    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "GPU_CANARY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
