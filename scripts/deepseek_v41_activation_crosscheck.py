#!/usr/bin/env python3
"""Compare the bounded Rust FP8 activation oracle with the pinned PyTorch CPU path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess

import torch

REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
BLOCK = 32
MAX_ACTIVATION_LEN = 1 << 20
ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_HASHES = {
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
}


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def f32_from_bits(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def verify_pinned_source() -> dict[str, str]:
    source_root = ROOT / "evidence/continuation_a97_model_metadata/upstream_314113f8"
    for name, expected in UPSTREAM_HASHES.items():
        actual = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"pinned official {name} hash mismatch: {actual}")
    kernel = (source_root / "kernel.py").read_text()
    model = (source_root / "model.py").read_text()
    for fragment in (
        "amax_local[i] = T.max(amax_local[i], 1e-4)",
        "fast_round_scale(amax_local[i], fp8_max_inv)",
        "T.clamp(x_local[i, j] / s_local[i], fp8_min, fp8_max)",
        "return fast_pow2(fast_log2_ceil(amax * fp8_max_inv))",
    ):
        if fragment not in kernel:
            raise RuntimeError(f"pinned activation kernel lost contract: {fragment}")
    for fragment in (
        "fp8_block_size = 32",
        'scale_fmt = "ue8m0"',
        "scale_dtype = torch.float8_e8m0fnu",
    ):
        if fragment not in model:
            raise RuntimeError(f"pinned model lost activation format: {fragment}")
    return UPSTREAM_HASHES.copy()


def finite_fp8_values_and_midpoints() -> list[float]:
    codes = torch.arange(0, 127, dtype=torch.uint8)
    positive = codes.view(torch.float8_e4m3fn).to(torch.float32).tolist()
    values: list[float] = []
    for value in positive:
        values.extend((float(value), -float(value)))
    for lower, upper in zip(positive, positive[1:]):
        midpoint = f32((float(lower) + float(upper)) * 0.5)
        values.extend((midpoint, -midpoint))
    return values


def boundary_groups() -> list[list[float]]:
    groups: list[list[float]] = []
    for exponent in range(-22, 120):
        boundary = f32(math.ldexp(448.0, exponent))
        bits = f32_bits(boundary)
        lanes = [boundary, -boundary]
        for neighbor in (bits - 1, bits + 1):
            adjacent = f32_from_bits(neighbor)
            lanes.extend((adjacent, -adjacent))
        groups.append((lanes + [0.0] * BLOCK)[:BLOCK])

    tiny = f32_from_bits(1)
    groups.extend(
        [
            [0.0, -0.0] + [0.0] * (BLOCK - 2),
            [tiny, -tiny] + [0.0] * (BLOCK - 2),
            [f32_from_bits(0x7F7FFFFF), -f32_from_bits(0x7F7FFFFF)]
            + [0.0] * (BLOCK - 2),
        ]
    )

    rng = random.Random(0xA98_20261019)
    for _ in range(128):
        lanes = []
        while len(lanes) < BLOCK:
            bits = rng.getrandbits(32)
            if (bits >> 23) & 0xFF != 0xFF:
                lanes.append(f32_from_bits(bits))
        groups.append(lanes)
    return groups


def torch_reference(values: list[float]) -> tuple[list[int], list[int]]:
    x = torch.tensor(values, dtype=torch.float32, device="cpu")
    encoded: list[int] = []
    scale_codes: list[int] = []
    inverse = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    for block in x.split(BLOCK):
        amax = torch.maximum(
            block.abs().max(), torch.tensor(1.0e-4, dtype=torch.float32)
        )
        scale_probe = amax * inverse
        scale_exponent = math.ceil(math.log2(float(scale_probe.item())))
        scale = torch.tensor(math.ldexp(1.0, scale_exponent), dtype=torch.float32)
        q = (block / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        e8m0 = scale.reshape(1).to(torch.float8_e8m0fnu)
        encoded.extend(int(byte) for byte in q.view(torch.uint8).tolist())
        scale_codes.append(int(e8m0.view(torch.uint8).item()))
    return encoded, scale_codes


def run_cli(binary: Path, input_value: dict) -> tuple[int, dict]:
    completed = subprocess.run(
        [str(binary), "deepseek-v41-ops", "-"],
        input=json.dumps(input_value, separators=(",", ":")).encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
        timeout=60,
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"CLI emitted invalid JSON: {completed.stdout[:300]!r}"
        ) from exc
    return completed.returncode, response


def compare_corpus(binary: Path, values: list[float], label: str):
    if not values or len(values) > MAX_ACTIVATION_LEN or len(values) % BLOCK:
        raise RuntimeError(f"{label}: corpus violates the bounded Rust contract")
    expected_values, expected_scales = torch_reference(values)
    input_bits = [f32_bits(value) for value in values]
    rc, response = run_cli(
        binary,
        {"op": "fp8-act-quant-reference", "input_bits": input_bits},
    )
    if rc != 0 or response.get("ok") is not True:
        raise RuntimeError(
            f"{label}: valid corpus rejected: rc={rc}, response={response}"
        )
    result = response.get("result", {})
    actual_values = result.get("e4m3_bytes")
    actual_scales = result.get("scale_codes")
    if not isinstance(actual_values, list) or len(actual_values) != len(
        expected_values
    ):
        raise RuntimeError(
            f"{label}: Rust E4M3 output length differs from input length"
        )
    if not isinstance(actual_scales, list) or len(actual_scales) != len(
        expected_scales
    ):
        raise RuntimeError(f"{label}: Rust UE8M0 count differs from number of groups")
    if actual_values != expected_values:
        mismatch = next(
            i
            for i, pair in enumerate(zip(actual_values, expected_values))
            if pair[0] != pair[1]
        )
        raise RuntimeError(
            f"{label}: E4M3 mismatch at element {mismatch}: "
            f"input_bits={input_bits[mismatch]:08x}, actual={actual_values[mismatch]}, "
            f"expected={expected_values[mismatch]}"
        )
    if actual_scales != expected_scales:
        mismatch = next(
            i
            for i, pair in enumerate(zip(actual_scales, expected_scales))
            if pair[0] != pair[1]
        )
        raise RuntimeError(
            f"{label}: UE8M0 mismatch at group {mismatch}: "
            f"actual={actual_scales[mismatch]}, expected={expected_scales[mismatch]}"
        )
    if any(not 105 <= code <= 247 for code in actual_scales):
        raise RuntimeError(
            f"{label}: scale code escaped the derived finite admission range"
        )
    capability = result.get("capability", {})
    if (
        result.get("op") != "fp8-act-quant-reference"
        or result.get("input_domain")
        != "finite FP32 bits; exact BF16 values are included"
        or capability.get("gpu_executed") is not False
        or capability.get("cuda_executed") is not False
        or result.get("gemm_executed") is not False
    ):
        raise RuntimeError(f"{label}: capability/input scope missing or overclaimed")
    return input_bits, actual_values, actual_scales


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=os.environ.get("VF_BINARY"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.binary is None:
        parser.error("--binary or VF_BINARY is required")
    binary = args.binary.resolve(strict=True)
    if torch.version.cuda is not None:
        raise RuntimeError("oracle requires the pinned CPU-only PyTorch build")
    if not hasattr(torch, "float8_e8m0fnu"):
        raise RuntimeError("PyTorch lacks float8_e8m0fnu support")
    source_hashes = verify_pinned_source()

    candidates = finite_fp8_values_and_midpoints()
    groups: list[list[float]] = []
    for start in range(0, len(candidates), BLOCK - 1):
        block = [448.0, *candidates[start : start + BLOCK - 1]]
        groups.append((block + [0.0] * BLOCK)[:BLOCK])
    groups.extend(boundary_groups())
    values = [value for group in groups for value in group]
    input_bits, actual_values, actual_scales = compare_corpus(
        binary, values, "FP32 exhaustive format corpus"
    )

    bf16_tensor = (
        torch.tensor(values, dtype=torch.float32).to(torch.bfloat16).to(torch.float32)
    )
    bf16_values = bf16_tensor[torch.isfinite(bf16_tensor)].tolist()
    bf16_values = bf16_values[: len(bf16_values) // BLOCK * BLOCK]
    bf16_bits, bf16_encoded, bf16_scales = compare_corpus(
        binary, bf16_values, "official BF16 activation domain"
    )

    invalid_results = []
    for bits in (0x7FC00000, 0x7F800000, 0xFF800000):
        rc, response = run_cli(
            binary,
            {"op": "fp8-act-quant-reference", "input_bits": [0] * 31 + [bits]},
        )
        if rc == 0 or response.get("ok") is not False:
            raise RuntimeError(f"invalid nonfinite input was accepted: {bits:08x}")
        if response.get("error", {}).get("code") != "NONFINITE_INPUT":
            raise RuntimeError(f"wrong rejection for input {bits:08x}: {response}")
        invalid_results.append({"bits": f"{bits:08x}", "code": "NONFINITE_INPUT"})

    report = {
        "status": "PASS_PINNED_TORCH_CPU_ACT_QUANT_CROSSCHECK",
        "model_revision": REVISION,
        "torch_version": torch.__version__,
        "oracle_device": "cpu",
        "groups": len(actual_scales),
        "elements": len(actual_values),
        "bf16_compatible_elements": len(bf16_encoded),
        "bf16_compatible_groups": len(bf16_scales),
        "input_domain": "finite FP32 bit patterns; exact BF16 values separately checked",
        "upstream_source_sha256": source_hashes,
        "e4m3_all_finite_codes_checked": True,
        "e4m3_all_adjacent_midpoints_checked": True,
        "ue8m0_scale_groups_checked": len(actual_scales),
        "dynamic_scale_boundary_exponents": [-22, 119],
        "signed_zero_and_fp32_subnormal_checked": True,
        "max_finite_fp32_checked": True,
        "invalid_nonfinite_cases": invalid_results,
        "input_sha256": hashlib.sha256(
            b"".join(struct.pack("<I", bits) for bits in input_bits)
        ).hexdigest(),
        "bf16_input_sha256": hashlib.sha256(
            b"".join(struct.pack("<I", bits) for bits in bf16_bits)
        ).hexdigest(),
        "encoded_values_sha256": hashlib.sha256(bytes(actual_values)).hexdigest(),
        "scale_codes_sha256": hashlib.sha256(bytes(actual_scales)).hexdigest(),
        "bf16_encoded_values_sha256": hashlib.sha256(bytes(bf16_encoded)).hexdigest(),
        "bf16_scale_codes_sha256": hashlib.sha256(bytes(bf16_scales)).hexdigest(),
        "gpu_executed": False,
        "cuda_executed": False,
        "gemm_executed": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
