#!/usr/bin/env python3
"""Relocatable, receipt-guarded DeepSeek QPN2 adapter-layer canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FIXTURE_SHA256 = "e6d39f1821c0e3f7f1cc4827d206391aa428a6f3944731069d45dee4135f8c00"
K_ORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)
DEEPSEEK_ARTIFACT_SHAPE = [1, 4608, 5120, 5120, 2304]
DEEPSEEK_PAYLOAD_BYTES = 19_907_776
TERMINAL_STATUSES = {
    "FAILED_BEFORE_GPU",
    "A132_DEEPSEEK_QPN2_LAYER_PASS",
    "A132_DEEPSEEK_QPN2_LAYER_FAILURE_NO_REPLAY",
    "TERMINAL_GPU_FAILURE_AFTER_LAUNCH_NO_REPLAY",
    "TERMINAL_FAILURE_AFTER_CONTEXT_NO_KERNEL_NO_REPLAY",
}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def is_finite_f32(value):
    if not is_finite_number(value):
        return False
    try:
        converted = struct.unpack("<f", struct.pack("<f", float(value)))[0]
    except OverflowError:
        return False
    return math.isfinite(converted) and (value == 0 or converted != 0)


def expected_artifact_tensor_specs():
    """The exact six compressed QPN2 planes consumed by the SM70 provider."""
    return {
        "codes13": ("U8", [4608 * 5120 // 2]),
        "scales13": ("U8", [4608 * 5120 // 16]),
        "gscales13": ("F32", [1, 4608 // 32]),
        "codes2": ("U8", [5120 * 2304 // 2]),
        "scales2": ("U8", [5120 * 2304 // 16]),
        "gscales2": ("F32", [1, 5120 // 32]),
    }


def validate_deepseek_layer_shape(*, rows, hidden, intermediate, top_k):
    """Admit only the measured A130 two-row, top-k-one QPN2 layer geometry."""
    values = (rows, hidden, intermediate, top_k)
    if any(type(value) is not int for value in values):
        raise ValueError("DeepSeek layer dimensions must be integer values")
    if values != (2, 5120, 2304, 1):
        raise ValueError(
            "only the exact two-token DeepSeek QPN2 layer shape is admitted"
        )
    return True


def validate_artifact_identity(info, expected_identity):
    """Compare all resident ownership fields before accepting compressed bytes."""
    required = {
        "model_revision",
        "layer",
        "rank",
        "epoch",
        "shape",
        "physical_experts",
        "resident_order",
        "payload_bytes",
    }
    if not isinstance(info, dict) or not isinstance(expected_identity, dict):
        raise ValueError("resident artifact identity must be an object")
    if set(expected_identity) != required or set(info) < required:
        raise ValueError("resident artifact identity fields are incomplete")
    for name in ("layer", "rank", "epoch", "payload_bytes"):
        if (
            type(info.get(name)) is not int
            or type(expected_identity.get(name)) is not int
        ):
            raise ValueError(f"resident artifact {name} must be an integer")
    if (
        not isinstance(info.get("model_revision"), str)
        or type(info.get("shape")) is not list
        or any(type(value) is not int for value in info["shape"])
        or type(info.get("physical_experts")) is not list
        or type(info.get("resident_order")) is not list
    ):
        raise ValueError("resident artifact identity field types are invalid")
    if any(type(value) is not int for value in info["resident_order"]) or any(
        not isinstance(expert, dict)
        or set(expert) != {"id", "local"}
        or type(expert["id"]) is not int
        or type(expert["local"]) is not int
        for expert in info["physical_experts"]
    ):
        raise ValueError("resident artifact ownership types are invalid")
    if info != expected_identity:
        raise ValueError("resident artifact identity changed")
    if info.get("shape") != DEEPSEEK_ARTIFACT_SHAPE:
        raise ValueError("resident artifact shape is outside the A130 admission")
    if info.get("payload_bytes") != DEEPSEEK_PAYLOAD_BYTES:
        raise ValueError("resident artifact compressed payload size changed")
    return info


def read_artifact(path, expected_sha256, expected_identity):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("resident artifact must be a regular non-symlink file")
    raw = path.read_bytes()
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("resident artifact SHA-256 must be 64 lowercase hex digits")
    if sha256(raw) != expected_sha256:
        raise ValueError("resident artifact SHA-256 mismatch")
    if len(raw) < 8:
        raise ValueError("resident artifact truncated before header")
    header_len = struct.unpack_from("<Q", raw)[0]
    data_start = 8 + header_len
    if data_start > len(raw):
        raise ValueError("resident SafeTensors header is out of bounds")
    header = json.loads(raw[8:data_start].rstrip(b" "))
    payload = memoryview(raw)[data_start:]
    try:
        info = json.loads(header["__metadata__"]["vf_info"])
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("resident artifact is missing vf_info metadata") from exc
    validate_artifact_identity(info, expected_identity)
    expected = expected_artifact_tensor_specs()
    if set(header) != {"__metadata__", *expected}:
        raise ValueError("resident artifact tensor set differs")
    spans = {}
    ordered_spans = []
    for name, (dtype, shape) in expected.items():
        tensor = header[name]
        try:
            start, stop = tensor["data_offsets"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"resident artifact tensor span invalid: {name}") from exc
        item = 4 if dtype == "F32" else 1
        if tensor.get("dtype") != dtype or tensor.get("shape") != shape:
            raise ValueError(f"resident artifact tensor metadata changed: {name}")
        if (
            type(start) is not int
            or type(stop) is not int
            or type(tensor.get("shape")) is not list
            or any(type(value) is not int for value in tensor["shape"])
        ):
            raise ValueError(f"resident artifact tensor field types changed: {name}")
        if (
            start < 0
            or stop < start
            or stop > len(payload)
            or stop - start != math.prod(shape) * item
        ):
            raise ValueError(f"resident artifact tensor span invalid: {name}")
        spans[name] = payload[start:stop]
        ordered_spans.append((start, stop, name))
    cursor = 0
    for start, stop, name in sorted(ordered_spans):
        if start != cursor:
            raise ValueError(
                f"resident artifact payload has a gap or overlap at {name}"
            )
        cursor = stop
    if cursor != len(payload) or cursor != info["payload_bytes"]:
        raise ValueError("resident artifact payload length differs from its six planes")
    return raw, info, spans


def validate_rank_dispatch(document, *, expected_identity):
    """Validate Rust routing and return token-major IDs/weights without renormalizing."""
    required_identity = {
        "run_id",
        "model_revision",
        "plan_id",
        "layer",
        "batch",
        "epoch",
    }
    if not isinstance(document, dict) or set(document) != {"problem", "dispatch"}:
        raise ValueError("RankDispatch document must contain problem and dispatch")
    if (
        not isinstance(expected_identity, dict)
        or set(expected_identity) != required_identity
    ):
        raise ValueError("expected RankDispatch identity is incomplete")
    problem, dispatch = document["problem"], document["dispatch"]
    if not isinstance(problem, dict) or not isinstance(dispatch, dict):
        raise ValueError("RankDispatch problem and dispatch must be objects")
    problem_stamp, dispatch_stamp = problem.get("stamp"), dispatch.get("stamp")
    if not isinstance(problem_stamp, dict) or problem_stamp != dispatch_stamp:
        raise ValueError("RankDispatch stamps disagree")
    if set(problem_stamp) != required_identity:
        raise ValueError("RankDispatch stamp identity is incomplete")
    for key, expected in expected_identity.items():
        if problem_stamp.get(key) != expected or type(
            problem_stamp.get(key)
        ) is not type(expected):
            raise ValueError(f"RankDispatch {key} identity mismatch")
    epoch = expected_identity["epoch"]
    registry = problem.get("registry")
    if (
        not isinstance(registry, dict)
        or type(epoch) is not int
        or type(registry.get("epoch")) is not int
        or registry["epoch"] != epoch
    ):
        raise ValueError("RankDispatch registry epoch mismatch")
    if type(problem.get("hidden")) is not int or problem["hidden"] != 5120:
        raise ValueError("RankDispatch hidden size is outside the A130 admission")
    if problem.get("experts") != [96]:
        raise ValueError("RankDispatch global expert set is outside the A130 admission")
    copies = registry.get("copies")
    if copies != [{"expert": 96, "local": 0, "rank": 0}]:
        raise ValueError("RankDispatch resident global/local mapping changed")
    copy = copies[0]
    if any(type(copy[key]) is not int for key in ("expert", "local", "rank")):
        raise ValueError("RankDispatch resident mapping fields must be integers")
    routes = problem.get("routes")
    if not isinstance(routes, list) or len(routes) != 2:
        raise ValueError("RankDispatch must contain exactly two token routes")
    route_ids, route_weights = [], []
    for token, route in enumerate(routes):
        if not isinstance(route, list) or len(route) != 1:
            raise ValueError("only one unique top-k expert per token is admitted")
        choice = route[0]
        if not isinstance(choice, dict) or set(choice) != {"expert", "weight"}:
            raise ValueError("RankDispatch route entry is malformed")
        weight = choice["weight"]
        if type(choice["expert"]) is not int or choice["expert"] != 96:
            raise ValueError(
                "RankDispatch token references an unadmitted global expert"
            )
        if not is_finite_f32(weight):
            raise ValueError(
                "RankDispatch coefficient is not finite or representable by adapter FP32"
            )
        route_ids.append([choice["expert"]])
        route_weights.append([float(weight)])
    if (
        type(dispatch.get("hidden")) is not int
        or dispatch["hidden"] != 5120
        or type(dispatch.get("rank")) is not int
        or dispatch["rank"] != 0
    ):
        raise ValueError("RankDispatch worker identity or hidden size changed")
    resident_order = dispatch.get("resident_order")
    if (
        type(resident_order) is not list
        or len(resident_order) != 1
        or type(resident_order[0]) is not int
        or resident_order[0] != 0
        or type(dispatch.get("tokens")) is not int
        or dispatch["tokens"] != 2
    ):
        raise ValueError("RankDispatch resident slot order or token count changed")
    slots = dispatch.get("slots")
    if not isinstance(slots, list) or len(slots) != 2:
        raise ValueError("RankDispatch slot coverage is incomplete")
    seen_tokens = set()
    for entry in slots:
        if not isinstance(entry, dict) or set(entry) != {"physical_expert", "slot"}:
            raise ValueError("RankDispatch physical slot is malformed")
        slot = entry["slot"]
        if not isinstance(slot, dict) or set(slot) != {
            "expert",
            "local",
            "rank",
            "slot",
            "token",
            "weight",
        }:
            raise ValueError("RankDispatch logical slot identity is malformed")
        token = slot["token"]
        if type(token) is not int or token not in (0, 1) or token in seen_tokens:
            raise ValueError("RankDispatch token slot is duplicate or out of range")
        seen_tokens.add(token)
        if (
            any(
                type(slot[key]) is not int
                for key in ("expert", "local", "rank", "slot")
            )
            or type(entry["physical_expert"]) is not int
            or entry["physical_expert"] != 0
            or slot["expert"] != 96
            or slot["local"] != 0
            or slot["rank"] != 0
            or slot["slot"] != 0
            or not is_finite_f32(slot["weight"])
            or float(slot["weight"]) != route_weights[token][0]
        ):
            raise ValueError("RankDispatch route and resident slot mapping disagree")
    if seen_tokens != {0, 1}:
        raise ValueError("RankDispatch token slot coverage is incomplete")
    return {
        "route_ids": route_ids,
        "route_weights": route_weights,
        "global_to_local": {96: 0},
        "problem": problem,
        "dispatch": dispatch,
        "stamp": problem_stamp,
    }


def validate_run_id(run_id):
    if not isinstance(run_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{2,127}", run_id
    ):
        raise ValueError("run_id must be 3–128 safe ASCII filename characters")
    return run_id


def validate_dispatch_provenance(provenance, dispatch_sha256):
    """Require the receipt tying the serialized dispatch to a Rust routing run."""
    if not isinstance(provenance, dict):
        raise ValueError("Rust RankDispatch provenance must be an object")
    argv = provenance.get("argv")
    hashes = (
        provenance.get("binary_sha256"),
        provenance.get("stdin_sha256"),
        provenance.get("stdout_sha256"),
    )
    if (
        type(provenance.get("exit_code")) is not int
        or provenance["exit_code"] != 0
        or provenance.get("reproduced_dispatch_matches") is not True
        or provenance.get("pinned_rank_dispatch_sha256") != dispatch_sha256
        or not isinstance(argv, list)
        or len(argv) < 3
        or argv[-2:] != ["routing", "-"]
        or any(not isinstance(part, str) or not part for part in argv)
        or any(
            not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
            for value in hashes
        )
    ):
        raise ValueError("Rust routing receipt does not qualify this RankDispatch")
    return provenance


def default_receipt_dir():
    explicit_root = os.environ.get("VF_H1_RECEIPT_DIR")
    root = Path(explicit_root).resolve(strict=True) if explicit_root else ROOT.resolve()
    return root if explicit_root else root / ".runs/deepseek-v41-qpn2-layer-canary"


def receipt_path(receipt_dir, run_id):
    explicit_root = os.environ.get("VF_H1_RECEIPT_DIR")
    root = Path(explicit_root).resolve(strict=True) if explicit_root else ROOT.resolve()
    candidate = Path(receipt_dir).resolve() / f"{validate_run_id(run_id)}.json"
    try:
        candidate.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"receipt path must stay below the approved root: {root}"
        ) from exc
    return candidate


def classify_receipt(report):
    """Incomplete or unfamiliar states are uncertain and must never be replayed."""
    if not isinstance(report, dict):
        return "UNKNOWN_DO_NOT_REPLAY"
    status = report.get("status")
    if status in TERMINAL_STATUSES:
        return status
    return "UNKNOWN_DO_NOT_REPLAY"


def full_layer_pass_eligible(report):
    """Require both synchronized projections, combine, routing and all oracles."""
    required_true = (
        "full_layer_correct",
        "gate_up_correct",
        "swiglu_correct",
        "down_correct",
        "weighted_combine_correct",
        "rust_rank_dispatch_bound",
    )
    return (
        isinstance(report, dict)
        and all(report.get(name) is True for name in required_true)
        and type(report.get("kernel_launch_count")) is int
        and report["kernel_launch_count"] == 2
        and type(report.get("provider_synchronized_launches")) is int
        and report["provider_synchronized_launches"] == 2
    )


def e2m1(code):
    sign = -1.0 if code & 8 else 1.0
    exponent = (code >> 1) & 3
    mantissa = (code & 1) / 2.0
    magnitude = mantissa if exponent == 0 else (1.0 + mantissa) * 2.0 ** (exponent - 1)
    return math.copysign(magnitude, sign)


def qpn_lane(row):
    return (row & 3) | (((row >> 3) & 3) << 2) | (((row >> 2) & 1) << 4)


def e4m3_scale(code):
    """Decode the unsigned finite E4M3 scale from its mathematical format."""
    if code < 0 or code > 126:
        raise ValueError("scale is not an unsigned finite E4M3 value")
    exponent, mantissa = code >> 3, code & 7
    if exponent == 0:
        return mantissa * 2.0**-9
    return (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)


def decode_dense_qpn2(np, codes, scales, factors, n, k):
    """Temporary CPU-only, vectorized reference matrix for one QPN2 projection."""
    if n <= 0 or k <= 0 or n % 32 or k % 16:
        raise ValueError("QPN2 dimensions must be positive multiples of 32 and 16")
    fragments = n * k // 16
    if (
        len(codes) != fragments * 8
        or len(scales) != fragments
        or len(factors) != n // 32 * 4
    ):
        raise ValueError("QPN2 projection byte lengths differ from its shape")
    packed = np.frombuffer(codes, dtype=np.uint8).reshape(fragments, 8)
    scale_bytes = np.frombuffer(scales, dtype=np.uint8)
    factor_values = np.frombuffer(factors, dtype="<f4")
    if (
        np.any(scale_bytes >= 0x7F)
        or not np.isfinite(factor_values).all()
        or np.any(factor_values <= 0)
    ):
        raise ValueError("invalid QPN2 scale or global factor")

    # This path evaluates decoded E2M1 × E4M3 × global factor in FP64 and
    # rounds each weight once to FP16. The whole projection was separately
    # admitted by the A125 ordered-half oracle before it became this artifact.
    raw_table = np.asarray(
        [e2m1(code) * 2.0**-14 for code in range(16)], dtype=np.float64
    ).astype(np.float16)
    scale_table = np.asarray(
        [e4m3_scale(code) for code in range(127)], dtype=np.float64
    ).astype(np.float16)
    lane_for_row = np.asarray([qpn_lane(row) for row in range(32)], dtype=np.int64)
    slot_for_column = np.argsort(np.asarray(K_ORDER, dtype=np.int64))
    groups = k // 16
    lanes = np.arange(32, dtype=np.int64)
    weights = np.empty((n, k), dtype=np.float16)

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        for tile in range(n // 32):
            frag_indices = (tile * groups + np.arange(groups, dtype=np.int64))[
                :, None
            ] * 32 + lanes[None, :]
            tile_packed = packed[frag_indices]
            nibbles = np.empty((groups, 32, 16), dtype=np.uint8)
            nibbles[:, :, 0::2] = tile_packed & 15
            nibbles[:, :, 1::2] = tile_packed >> 4
            row_codes = nibbles[:, lane_for_row, :][:, :, slot_for_column]
            row_scale_codes = scale_bytes[frag_indices][:, lane_for_row]
            scale_half = scale_table[row_scale_codes]
            gm_half = np.float16(np.float32(factor_values[tile]) * np.float32(16_384.0))
            combined_half = (scale_half * gm_half).astype(np.float16)
            tile_weights = (raw_table[row_codes] * combined_half[:, :, None]).astype(
                np.float16
            )
            weights[tile * 32 : (tile + 1) * 32] = tile_weights.transpose(
                1, 0, 2
            ).reshape(32, k)

    if not np.isfinite(weights).all():
        raise ValueError("QPN2 oracle decoded a nonfinite FP16 weight")
    return weights


def make_input_batches(np, k):
    """Two deterministic dense two-token batches; the second reuses device weights."""
    rng = np.random.Generator(np.random.PCG64(126_1020 + k))
    a = (rng.standard_normal(k) * 0.125).astype(np.float16)
    b = (rng.standard_normal(k) * 0.0625).astype(np.float16)
    sign = np.where(np.arange(k) & 1, -1.0, 1.0).astype(np.float16)
    first = np.stack((a, np.roll(b, 19))).astype(np.float16, copy=False)
    second = np.stack((np.roll(a, -37) * sign, np.flip(b) * sign)).astype(
        np.float16, copy=False
    )
    return (np.ascontiguousarray(first), np.ascontiguousarray(second))


def dense_dot_oracle(np, weights, inputs):
    """FP64 matrix-vector oracle plus Σ|w*x| for the declared FP32 error bound."""
    n, k = weights.shape
    if inputs.ndim != 2 or inputs.shape[1] != k:
        raise ValueError("dense oracle input shape mismatch")
    exact = np.empty((inputs.shape[0], n), dtype=np.float64)
    absolute_products = np.empty_like(exact)
    x64 = inputs.astype(np.float64)
    abs_x64 = np.abs(x64)
    for start in range(0, n, 128):
        stop = min(start + 128, n)
        w64 = weights[start:stop].astype(np.float64)
        exact[:, start:stop] = x64 @ w64.T
        absolute_products[:, start:stop] = abs_x64 @ np.abs(w64).T
    return exact, absolute_products


def dense_magnitude_dot_oracle(np, weights, inputs, row_block=128):
    """Stream Σ|w*x| by output rows without a full FP64 weight copy."""
    if (
        weights.ndim != 2
        or inputs.ndim != 2
        or inputs.shape[1] != weights.shape[1]
        or type(row_block) is not int
        or row_block <= 0
    ):
        raise ValueError("dense magnitude oracle dimensions or block are invalid")
    if not np.isfinite(weights).all() or not np.isfinite(inputs).all():
        raise ValueError("dense magnitude oracle requires finite inputs")
    n, _ = weights.shape
    result = np.empty((inputs.shape[0], n), dtype=np.float64)
    abs_inputs = np.abs(inputs.astype(np.float64))
    for start in range(0, n, row_block):
        stop = min(start + row_block, n)
        abs_weights = weights[start:stop].astype(np.float64)
        np.abs(abs_weights, out=abs_weights)
        result[:, start:stop] = abs_inputs @ abs_weights.T
    return result


def fraction_dot(np, weights, inputs, row):
    """Exact rational dot for an independent small set of full-K outputs."""
    from fractions import Fraction

    total = Fraction(0)
    for weight, value in zip(weights[row], inputs, strict=True):
        total += Fraction.from_float(float(weight)) * Fraction.from_float(float(value))
    return total


def compare_full_output(np, actual, exact, absolute_products, k):
    if actual.shape != exact.shape or exact.shape != absolute_products.shape:
        raise ValueError("GPU output and FP64 oracle shapes differ")
    if not np.isfinite(actual).all() or not np.isfinite(exact).all():
        raise ValueError("nonfinite GPU or oracle output")
    unit = 2.0**-24
    operations = (
        2 * k + 16
    )  # conservative FP32 accumulation count, including split-K merge
    gamma = operations * unit / (1.0 - operations * unit)
    accum_bound = gamma * absolute_products
    half_round_bound = 2.0**-11 * (np.abs(exact) + accum_bound) + 2.0**-25
    bound = accum_bound + half_round_bound
    error = np.abs(actual.astype(np.float64) - exact)
    exceeded = int(np.count_nonzero(error > bound + 2.0**-52))
    ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    rounded_reference = exact.astype(np.float16)
    bit_mismatches = int(
        np.count_nonzero(actual.view(np.uint16) != rounded_reference.view(np.uint16))
    )
    return {
        "values_checked": int(actual.size),
        "max_abs_error_vs_fp64": float(error.max(initial=0.0)),
        "max_abs_bound": float(bound.max(initial=0.0)),
        "max_error_bound_ratio": float(ratio.max(initial=0.0)),
        "outputs_outside_fp32_plus_fp16_rounding_bound": exceeded,
        "bit_mismatches_vs_fp64_rounded_fp16": bit_mismatches,
        "correct": exceeded == 0,
    }


def gpu_preflight(torch, expected_uuid):
    query = "uuid,name,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max,pci.bus_id"
    gpu = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        timeout=15,
        check=False,
    )
    lines = [line.strip() for line in gpu.stdout.splitlines() if line.strip()]
    if gpu.returncode or len(lines) != 1:
        raise RuntimeError("exactly one visible GPU is required on PC3")
    fields = [value.strip() for value in lines[0].split(",")]
    app_lines = [
        line.strip()
        for line in apps.stdout.splitlines()
        if line.strip() and "no running processes" not in line.lower()
    ]
    ready = (
        len(fields) == 9
        and fields[0] == expected_uuid
        and "V100" in fields[1]
        and fields[2] == "0"
        and fields[3] == "0"
        and apps.returncode == 0
        and not app_lines
        and torch.version.cuda == "12.8"
        and not torch.cuda.is_initialized()
    )
    result = {
        "fields": fields,
        "compute_apps": app_lines,
        "torch_cuda": torch.version.cuda,
        "cuda_initialized_before_provider": bool(torch.cuda.is_initialized()),
        "ready": ready,
    }
    if not ready:
        raise RuntimeError(
            "exact UUID/CUDA12.8 idle gate failed: " + json.dumps(result)
        )
    return result


def round_binary16(value):
    """Round one finite scalar to IEEE binary16, independently of Torch."""
    try:
        return struct.unpack("<e", struct.pack("<e", float(value)))[0]
    except OverflowError as exc:
        raise ValueError("binary16 oracle overflow") from exc


def swiglu_binary16_reference(np, gate_up):
    """Scalar SiLU then product, with a binary16 rounding after each op."""
    values = np.asarray(gate_up, dtype=np.float16)
    if values.ndim != 2 or values.shape[1] % 2 or not np.isfinite(values).all():
        raise ValueError("gate/up oracle input must be finite [tokens,2*hidden]")
    hidden = values.shape[1] // 2
    output = np.empty((values.shape[0], hidden), dtype=np.float16)
    for token in range(values.shape[0]):
        for column in range(hidden):
            gate = float(values[token, column])
            up = float(values[token, hidden + column])
            sigmoid = (
                1.0 / (1.0 + math.exp(-gate))
                if gate >= 0.0
                else math.exp(gate) / (1.0 + math.exp(gate))
            )
            silu = round_binary16(gate * sigmoid)
            output[token, column] = round_binary16(silu * up)
    if not np.isfinite(output).all():
        raise ValueError("binary16 SwiGLU oracle produced a nonfinite value")
    return output


def compare_binary16_ulp(np, actual, expected, max_ulps=1.0):
    got = np.asarray(actual, dtype=np.float16)
    reference = np.asarray(expected, dtype=np.float16)
    if got.shape != reference.shape or not np.isfinite(got).all():
        raise ValueError("SwiGLU output shape or finiteness differs")
    upward = np.nextafter(reference, np.float16(np.inf)).astype(np.float64)
    downward = np.nextafter(reference, np.float16(-np.inf)).astype(np.float64)
    ref64 = reference.astype(np.float64)
    ulp = np.maximum(np.abs(upward - ref64), np.abs(ref64 - downward))
    error = np.abs(got.astype(np.float64) - ref64)
    ratio = np.divide(error, ulp, out=np.zeros_like(error), where=ulp > 0)
    exceeded = int(np.count_nonzero(ratio > max_ulps + 1e-12))
    return {
        "values_checked": int(got.size),
        "half_ulp_limit": max_ulps,
        "outputs_over_ulp_limit": exceeded,
        "max_error_half_ulps": float(ratio.max(initial=0.0)),
        "max_abs_error": float(error.max(initial=0.0)),
        "correct": exceeded == 0,
    }


def compare_layer_down(np, weights, activation, reference_activation, actual, k):
    exact_actual, absolute_products = dense_dot_oracle(np, weights, activation)
    exact_reference, _ = dense_dot_oracle(np, weights, reference_activation)
    kernel = compare_full_output(np, actual, exact_actual, absolute_products, k)
    activation_delta = np.abs(
        activation.astype(np.float64) - reference_activation.astype(np.float64)
    )
    propagated = dense_magnitude_dot_oracle(np, weights, activation_delta)
    unit = 2.0**-24
    operations = 2 * k + 16
    gamma = operations * unit / (1.0 - operations * unit)
    accumulation = gamma * absolute_products
    half_round = 2.0**-11 * (np.abs(exact_reference) + accumulation) + 2.0**-25
    bound = accumulation + half_round + propagated
    error = np.abs(actual.astype(np.float64) - exact_reference)
    ratio = np.divide(error, bound, out=np.zeros_like(error), where=bound > 0)
    exceeded = int(np.count_nonzero(error > bound + 2.0**-52))
    return {
        "kernel_vs_actual_activation_oracle": kernel,
        "stagewise_full_down": {
            "values_checked": int(actual.size),
            "max_abs_error": float(error.max(initial=0.0)),
            "max_abs_bound": float(bound.max(initial=0.0)),
            "max_error_bound_ratio": float(ratio.max(initial=0.0)),
            "outputs_outside_bound": exceeded,
            "activation_error_propagated": True,
            "correct": exceeded == 0,
        },
        "exact_reference": exact_reference,
        "bound": bound,
    }


def combine_binary16_reference(np, h2, weights):
    values = np.asarray(h2, dtype=np.float16)
    coefficients = np.asarray(weights, dtype=np.float32)
    if values.ndim != 2 or coefficients.shape != (values.shape[0],):
        raise ValueError("one-coefficient-per-token combine shape required")
    if not np.isfinite(values).all() or not np.isfinite(coefficients).all():
        raise ValueError("combine inputs must be finite")
    return (values.astype(np.float32) * coefficients[:, None]).astype(np.float16)


def _run_canary(args, report, receipt):
    sys.path.insert(0, str(ROOT))
    from scripts.hardware_canary import QPN2Provider, check_cuda_proof

    expected_artifact_identity = {
        "model_revision": args.model_revision,
        "layer": args.layer,
        "rank": 0,
        "epoch": args.artifact_epoch,
        "shape": DEEPSEEK_ARTIFACT_SHAPE,
        "physical_experts": [{"id": 96, "local": 0}],
        "resident_order": [0],
        "payload_bytes": DEEPSEEK_PAYLOAD_BYTES,
    }
    artifact_raw, metadata, resident = read_artifact(
        args.artifact, args.artifact_sha256, expected_artifact_identity
    )
    dispatch_raw = args.dispatch.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", args.dispatch_sha256):
        raise ValueError("RankDispatch SHA-256 must be 64 lowercase hex digits")
    if sha256(dispatch_raw) != args.dispatch_sha256:
        raise ValueError("RankDispatch SHA-256 mismatch")
    provenance_raw = args.dispatch_provenance.read_bytes()
    provenance = validate_dispatch_provenance(
        json.loads(provenance_raw), args.dispatch_sha256
    )
    dispatch_document = json.loads(dispatch_raw)
    dispatch_identity = {
        "run_id": args.run_id,
        "model_revision": args.model_revision,
        "plan_id": args.plan_id,
        "layer": args.layer,
        "batch": args.batch,
        "epoch": args.dispatch_epoch,
    }
    routed = validate_rank_dispatch(
        dispatch_document, expected_identity=dispatch_identity
    )
    problem, dispatch = routed["problem"], routed["dispatch"]
    validate_deepseek_layer_shape(
        rows=dispatch["tokens"],
        hidden=dispatch["hidden"],
        intermediate=2304,
        top_k=1,
    )
    if metadata["model_revision"] != dispatch_identity["model_revision"]:
        raise ValueError("resident artifact and RankDispatch revisions differ")
    if (
        not args.library.is_file()
        or not args.cuda_proof.is_file()
        or not args.cudart.is_file()
    ):
        raise ValueError("SM70 library, CUDA proof, and cudart must exist as files")

    import numpy as np

    inputs = make_input_batches(np, 5120)[0]
    gate_up_weights = decode_dense_qpn2(
        np,
        resident["codes13"],
        resident["scales13"],
        resident["gscales13"],
        4608,
        5120,
    )
    gate_up_exact, gate_up_abs = dense_dot_oracle(np, gate_up_weights, inputs)
    fraction_errors = []
    for token in range(2):
        for row in (0, 31, 32, 2304, 4607):
            exact = fraction_dot(np, gate_up_weights, inputs[token], row)
            fraction_errors.append(abs(float(exact) - gate_up_exact[token, row]))
    if max(fraction_errors, default=0.0) > 2.0**-45:
        raise ValueError("gate/up FP64 oracle differs from exact rational samples")
    down_weights = decode_dense_qpn2(
        np,
        resident["codes2"],
        resident["scales2"],
        resident["gscales2"],
        5120,
        2304,
    )
    report.update(
        model_revision=args.model_revision,
        plan_id=args.plan_id,
        layer=args.layer,
        batch=args.batch,
        artifact_epoch=args.artifact_epoch,
        dispatch_epoch=args.dispatch_epoch,
        resident_artifact_sha256=sha256(artifact_raw),
        resident_artifact_bytes=len(artifact_raw),
        resident_payload_bytes=metadata["payload_bytes"],
        artifact_identity=metadata,
        rank_dispatch_sha256=sha256(dispatch_raw),
        rank_dispatch_provenance_sha256=sha256(provenance_raw),
        rust_dispatch_binary_sha256=provenance["binary_sha256"],
        rank_dispatch_stamp=routed["stamp"],
        route_ids=routed["route_ids"],
        route_weights=routed["route_weights"],
        global_to_local=routed["global_to_local"],
        source_fixture_sha256=FIXTURE_SHA256,
        adapter_entrypoint="moe_apply",
        resident_artifact_load_calls=0,
        rust_rank_dispatch_bound=False,
        gate_up_correct=False,
        swiglu_correct=False,
        down_correct=False,
        weighted_combine_correct=False,
        full_layer_correct=False,
        dense_weight_gpu_allocation=False,
        temporary_dense_cpu_oracle=True,
        temporary_dense_cpu_oracle_bytes=int(
            gate_up_weights.nbytes + down_weights.nbytes
        ),
        temporary_dense_cpu_peak_rss_kib=resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        inputs={"shape": list(inputs.shape), "sha256": sha256(inputs.tobytes())},
        numpy_version=np.__version__,
    )
    torch = __import__("torch")
    preflight = gpu_preflight(torch, args.gpu_uuid)
    library_sha = sha256(args.library.read_bytes())
    check_cuda_proof(args.cuda_proof, library_sha)
    report.update(
        status="REQUEST_ARMED",
        expected_gpu_uuid=args.gpu_uuid,
        library_sha256=library_sha,
        cuda_compile_proof_sha256=sha256(args.cuda_proof.read_bytes()),
        prelaunch_preflight=preflight,
        cuda_executed=False,
        cuda_context_initialized=False,
        kernel_launch_count=0,
        provider_synchronized_launches=0,
        automatic_retry=False,
        replay_allowed=False,
        notes=[
            "This relocatable canary accepts only the exact A130 DeepSeek layer geometry and one local expert copy.",
            "It loads six compressed QPN2 planes once, binds the full Rust RankDispatch identity, and calls adapters.skinny_moe.moe_apply.",
            "CPU QPN2, rational/FP64, binary16 SwiGLU, stagewise down, and weighted-combine oracles gate a PASS; this is not full-model or performance qualification.",
        ],
    )
    receipt.update(report)

    provider = None
    try:
        provider = QPN2Provider(
            args.library,
            args.cudart,
            0,
            args.gpu_uuid,
            ack_hardware_gate=True,
            gate_path=ROOT / "docs/HARDWARE_GATE.md",
        )
        provider.full_expert_enabled = True
        provider.deepseek_probe_enabled = True
        provider.adapter_layer_enabled = True

        def on_progress(event):
            report["cuda_context_initialized"] = True
            report["kernel_launch_count"] = provider.launches
            report["cuda_executed"] = provider.launches > 0
            report["status"] = (
                "GPU_EXECUTION_MAY_HAVE_STARTED"
                if event.get("execution_phase") == "ENQUEUE_PENDING"
                else "GPU_RUNNING"
            )
            report["last_provider_event"] = event
            receipt.update(report)

        provider.progress = on_progress
        report["device"] = provider.identity
        report["cuda_context_initialized"] = True
        report["status"] = "GPU_CONTEXT_INITIALIZED_NO_KERNEL_YET"
        receipt.update(report)

        from adapters import skinny_moe as adapter
        from scripts.hardware_canary import align_reference
        from types import SimpleNamespace

        expert_map = torch.full((97,), -1, dtype=torch.int64, device=provider.device)
        expert_map[96] = 0
        layer = SimpleNamespace(expert_map=expert_map)
        owner = adapter.load_resident_artifact(
            layer,
            args.artifact,
            expected_info=metadata,
            expected_sha256=args.artifact_sha256,
            device=provider.device,
        )
        report["resident_artifact_load_calls"] = 1
        report["resident_compressed_device_bytes"] = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                owner._qc13,
                owner._qs13,
                owner._gs13,
                owner._qc2,
                owner._qs2,
                owner._gs2,
            )
        )
        report["resident_shape"] = list(owner.shape)
        report["resident_artifact_binding"] = list(owner.artifact_binding[:4]) + [
            list(owner.artifact_binding[4])
        ]

        route_ids = routed["route_ids"]
        route_weights = routed["route_weights"]
        ids = torch.tensor(route_ids, dtype=torch.int64, device=provider.device)
        coefficients = torch.tensor(
            route_weights, dtype=torch.float32, device=provider.device
        )
        dispatch_mask = adapter.bind_rank_dispatch(
            layer, dispatch, problem["stamp"], ids, coefficients
        )
        report["rust_rank_dispatch_bound"] = bool(
            dispatch_mask.shape == ids.shape
            and dispatch_mask.dtype == torch.bool
            and bool(dispatch_mask.all())
        )
        if not report["rust_rank_dispatch_bound"]:
            raise ValueError(
                "Rust RankDispatch did not select every validated local route"
            )
        x = torch.from_numpy(inputs).to(provider.device).contiguous()
        recorded_inputs, recorded_outputs = [], []
        original_gemm = provider.gemm_qpn2_moe

        def recording_gemm(activation, *gemm_args):
            recorded_inputs.append(activation.detach().clone())
            result = original_gemm(activation, *gemm_args)
            recorded_outputs.append(result)
            return result

        provider.gemm_qpn2_moe = recording_gemm
        combined = adapter.moe_apply(
            layer,
            x,
            coefficients,
            ids,
            None,
            None,
            kernel=provider,
            align=align_reference,
            selected_mask=dispatch_mask,
        )
        provider.synchronize()
        report["kernel_launch_count"] = provider.launches
        report["provider_synchronized_launches"] = provider.synchronized_launches
        report["cuda_executed"] = provider.launches > 0
        report["activation_h2d_bytes"] = int(inputs.nbytes)
        report["input_activation_is_gpu_resident_for_both_projections"] = True
        if (
            provider.launches != 2
            or provider.synchronized_launches != 2
            or len(recorded_inputs) != 2
            or len(recorded_outputs) != 2
        ):
            raise ValueError("adapter must enqueue exactly gate/up and down once")

        gate_up_actual = recorded_outputs[0].detach().cpu().contiguous().numpy()
        activation_actual = recorded_inputs[1].detach().cpu().contiguous().numpy()
        down_actual = recorded_outputs[1].detach().cpu().contiguous().numpy()
        combined_actual = combined.detach().cpu().contiguous().numpy()
        gate_up_result = compare_full_output(
            np, gate_up_actual, gate_up_exact, gate_up_abs, 5120
        )
        report["gate_up"] = gate_up_result
        report["gate_up_rational_sample_max_abs_error"] = max(
            fraction_errors, default=0.0
        )
        report["gate_up_correct"] = gate_up_result["correct"]

        activation_reference = swiglu_binary16_reference(np, gate_up_actual)
        activation_result = compare_binary16_ulp(
            np, activation_actual, activation_reference, max_ulps=1.0
        )
        report["swiglu"] = activation_result
        report["activation_reference_sha256"] = sha256(activation_reference.tobytes())
        report["activation_actual_sha256"] = sha256(activation_actual.tobytes())
        report["swiglu_correct"] = activation_result["correct"]

        down_result = compare_layer_down(
            np,
            down_weights,
            activation_actual,
            activation_reference,
            down_actual,
            2304,
        )
        down_exact = down_result.pop("exact_reference")
        down_bound = down_result.pop("bound")
        report["down"] = down_result
        report["down_correct"] = down_result["stagewise_full_down"]["correct"]

        coefficients_np = np.asarray(
            [route[0] for route in route_weights], dtype=np.float32
        )
        combine_reference = combine_binary16_reference(np, down_actual, coefficients_np)
        nonzero_combine_mismatches = 0
        for actual_value, expected_value in zip(
            combined_actual.reshape(-1).view(np.uint16),
            combine_reference.reshape(-1).view(np.uint16),
            strict=True,
        ):
            if actual_value != expected_value and not (
                actual_value & 0x7FFF == 0 and expected_value & 0x7FFF == 0
            ):
                nonzero_combine_mismatches += 1
        report["weighted_combine"] = {
            "values_checked": int(combined_actual.size),
            "nonzero_fp16_bit_mismatches": nonzero_combine_mismatches,
            "coefficients": coefficients_np.astype(float).tolist(),
            "coefficient_application_count_per_token": [1, 1],
            "correct": nonzero_combine_mismatches == 0,
        }
        report["weighted_combine_correct"] = nonzero_combine_mismatches == 0

        coeff64 = coefficients_np.astype(np.float64)
        exact_layer_output = down_exact * coeff64[:, None]
        down_to_layer_bound = down_bound * np.abs(coeff64[:, None])
        product64 = down_actual.astype(np.float64) * coeff64[:, None]
        product32 = (down_actual.astype(np.float32) * coefficients_np[:, None]).astype(
            np.float64
        )
        product_round_error = np.abs(product32 - product64)
        output_half_round = (
            2.0**-11
            * (np.abs(exact_layer_output) + down_to_layer_bound + product_round_error)
            + 2.0**-25
        )
        layer_bound = down_to_layer_bound + product_round_error + output_half_round
        layer_error = np.abs(combined_actual.astype(np.float64) - exact_layer_output)
        layer_ratio = np.divide(
            layer_error,
            layer_bound,
            out=np.zeros_like(layer_error),
            where=layer_bound > 0,
        )
        layer_outside = int(np.count_nonzero(layer_error > layer_bound + 2.0**-52))
        report["full_layer_oracle"] = {
            "values_checked": int(combined_actual.size),
            "outputs_outside_bound": layer_outside,
            "max_abs_error": float(layer_error.max(initial=0.0)),
            "max_abs_bound": float(layer_bound.max(initial=0.0)),
            "max_error_bound_ratio": float(layer_ratio.max(initial=0.0)),
            "float32_combine_rounding_max_abs": float(
                product_round_error.max(initial=0.0)
            ),
            "reference": "FP64 down over independently rounded binary16 SwiGLU reference, then the unchanged Rust route coefficient; bounds include measured activation delta and FP16/FP32 rounding",
            "correct": layer_outside == 0,
        }
        report["full_layer_correct"] = (
            report["gate_up_correct"]
            and report["swiglu_correct"]
            and report["down_correct"]
            and report["weighted_combine_correct"]
            and layer_outside == 0
            and report["rust_rank_dispatch_bound"]
            and provider.launches == 2
            and provider.synchronized_launches == 2
        )
        report["outputs_checked_total"] = int(
            gate_up_actual.size
            + activation_actual.size
            + down_actual.size
            + combined_actual.size
        )
        report["temporary_dense_cpu_peak_rss_kib"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        pass_eligible = full_layer_pass_eligible(report)
        report["terminal_pass_eligible"] = pass_eligible
        report["status"] = (
            "A132_DEEPSEEK_QPN2_LAYER_PASS"
            if pass_eligible
            else "A132_DEEPSEEK_QPN2_LAYER_FAILURE_NO_REPLAY"
        )
        report["gpu_correctness"] = (
            "PASS_ADAPTER_LAYER_WITH_STAGEWISE_FP64_AND_FP16_ORACLES"
            if pass_eligible
            else "FAIL_ADAPTER_LAYER_ORACLE"
        )
        report["replay_allowed"] = False
        receipt.update(report)
        print(json.dumps(report, sort_keys=True))
        return 0 if pass_eligible else 2
    except BaseException as exc:
        pending = bool(provider is not None and provider.pending)
        launched = bool(provider is not None and provider.launches > 0)
        context = bool(torch.cuda.is_initialized())
        report["status"] = (
            "UNKNOWN_DO_NOT_REPLAY"
            if pending
            else (
                "TERMINAL_GPU_FAILURE_AFTER_LAUNCH_NO_REPLAY"
                if launched
                else (
                    "TERMINAL_FAILURE_AFTER_CONTEXT_NO_KERNEL_NO_REPLAY"
                    if context
                    else "FAILED_BEFORE_GPU"
                )
            )
        )
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["kernel_launch_count"] = provider.launches if provider is not None else 0
        report["provider_synchronized_launches"] = (
            provider.synchronized_launches if provider is not None else 0
        )
        report["cuda_executed"] = report["kernel_launch_count"] > 0
        report["replay_allowed"] = False
        receipt.update(report)
        print(json.dumps(report, sort_keys=True))
        return 3 if pending else 2


def _parser():
    parser = argparse.ArgumentParser(
        description="One-shot DeepSeek V4.1 QPN2 resident-layer correctness canary."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser(
        "run", help="validate then execute one explicit attempt"
    )
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--artifact-sha256", required=True)
    run.add_argument("--dispatch", type=Path, required=True)
    run.add_argument("--dispatch-sha256", required=True)
    run.add_argument("--dispatch-provenance", type=Path, required=True)
    run.add_argument("--model-revision", required=True)
    run.add_argument("--plan-id", required=True)
    run.add_argument("--layer", type=int, required=True)
    run.add_argument("--batch", type=int, required=True)
    run.add_argument("--artifact-epoch", type=int, required=True)
    run.add_argument("--dispatch-epoch", type=int, required=True)
    run.add_argument("--run-id", required=True)
    run.add_argument("--library", type=Path, required=True)
    run.add_argument("--cuda-proof", type=Path, required=True)
    run.add_argument("--cudart", type=Path, required=True)
    run.add_argument("--gpu-uuid", required=True)
    status = subparsers.add_parser(
        "status", help="classify a prior receipt without retrying its operation"
    )
    status.add_argument("--run-id", required=True)
    return parser


def _reconcile(args):
    path = receipt_path(default_receipt_dir(), args.run_id)
    if path.is_symlink() or not path.is_file():
        report = {
            "run_id": args.run_id,
            "status": "UNKNOWN_DO_NOT_REPLAY",
            "reason": "no safe regular receipt is available; absence is not proof of non-execution",
            "replay_allowed": False,
        }
    else:
        try:
            prior = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            prior = {"status": "UNKNOWN_DO_NOT_REPLAY", "receipt_error": str(exc)}
        report = dict(prior)
        report["reconciled_state"] = classify_receipt(prior)
        report["replay_allowed"] = False
    print(json.dumps(report, sort_keys=True))
    return 0 if classify_receipt(report) in TERMINAL_STATUSES else 3


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        validate_run_id(args.run_id)
        if args.command == "status":
            return _reconcile(args)
        if not re.fullmatch(r"[0-9a-f]{64}", args.artifact_sha256):
            raise ValueError(
                "resident artifact SHA-256 must be 64 lowercase hex digits"
            )
        if not re.fullmatch(r"[0-9a-f]{64}", args.dispatch_sha256):
            raise ValueError("RankDispatch SHA-256 must be 64 lowercase hex digits")
        report = {
            "schema_version": 1,
            "run_id": args.run_id,
            "operation_id": args.run_id,
            "status": "VALIDATING_NO_GPU",
            "resident_artifact_sha256_expected": args.artifact_sha256,
            "rank_dispatch_sha256_expected": args.dispatch_sha256,
            "model_revision": args.model_revision,
            "plan_id": args.plan_id,
            "layer": args.layer,
            "batch": args.batch,
            "artifact_epoch": args.artifact_epoch,
            "dispatch_epoch": args.dispatch_epoch,
            "expected_gpu_uuid": args.gpu_uuid,
            "cuda_executed": False,
            "cuda_context_initialized": False,
            "kernel_launch_count": 0,
            "automatic_retry": False,
            "replay_allowed": False,
        }
        sys.path.insert(0, str(ROOT))
        from scripts.hardware_canary import Receipt

        target = receipt_path(default_receipt_dir(), args.run_id)
        try:
            receipt = Receipt(target, report)
        except FileExistsError:
            if target.is_symlink() or not target.is_file():
                previous = {"status": "UNKNOWN_DO_NOT_REPLAY"}
            else:
                try:
                    previous = json.loads(target.read_text())
                except (OSError, json.JSONDecodeError) as exc:
                    previous = {
                        "status": "UNKNOWN_DO_NOT_REPLAY",
                        "receipt_error": str(exc),
                    }
            result = dict(previous)
            result["refusal"] = "run_id already has a receipt; no replay"
            result["reconciled_state"] = classify_receipt(previous)
            result["replay_allowed"] = False
            print(json.dumps(result, sort_keys=True))
            return 3
        try:
            return _run_canary(args, report, receipt)
        except BaseException as exc:
            report.update(
                status="FAILED_BEFORE_GPU",
                error=f"{type(exc).__name__}: {exc}",
                cuda_executed=False,
                kernel_launch_count=0,
                replay_allowed=False,
            )
            receipt.update(report)
            print(json.dumps(report, sort_keys=True))
            return 2
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
