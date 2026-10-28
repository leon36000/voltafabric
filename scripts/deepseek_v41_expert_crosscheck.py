#!/usr/bin/env python3
"""Independent Fraction oracle for one pinned real DeepSeek MXFP4 expert.

Reads the six-tensor selected Safetensors fixture without a ML framework. The
golden values are Python exact rationals with explicit binary64 rounding after
each product and ordered addition. No CUDA or full-model execution is involved.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

REPO = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
ARTIFACT = Path("fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors")
GOLDEN = Path("fixtures/deepseek_v41_layer0_expert0_w1_fraction_samples.json")
ROWS = [0, 1, 17, 1024, 2047, 2303]
INPUT_PATTERN = [0.25, -0.5, 1.0, -2.0, 0.0, 0.125, -1.0]


def e2m1(nibble: int) -> Fraction:
    sign, exponent, mantissa = (nibble >> 3) & 1, (nibble >> 1) & 3, nibble & 1
    half = Fraction(mantissa, 2)
    magnitude = half if exponent == 0 else (1 + half) * Fraction(2) ** (exponent - 1)
    return -magnitude if sign else magnitude


def e8m0(code: int) -> Fraction:
    if not 0 <= code < 255:
        raise ValueError(f"invalid E8M0 scale {code}")
    return Fraction(2) ** (code - 127)


def f64_bits(value: float) -> str:
    return f"{struct.unpack('<Q', struct.pack('<d', value))[0]:016x}"


def tensor_view(blob: bytes, header: dict, name: str) -> tuple[list[int], bytes]:
    item = header[name]
    start, stop = item["data_offsets"]
    data_start = 8 + struct.unpack("<Q", blob[:8])[0]
    payload = blob[data_start + start : data_start + stop]
    if len(payload) != stop - start:
        raise ValueError(f"truncated tensor payload for {name}")
    return item["shape"], payload


def expected_outputs(artifact_path: Path) -> dict:
    blob = artifact_path.read_bytes()
    if len(blob) < 8:
        raise ValueError("artifact is truncated")
    header_len = struct.unpack("<Q", blob[:8])[0]
    if 8 + header_len > len(blob):
        raise ValueError("artifact header exceeds file size")
    header = json.loads(blob[8 : 8 + header_len])
    weight_key = "layers.0.ffn.experts.0.w1.weight"
    scale_key = "layers.0.ffn.experts.0.w1.scale"
    if header[weight_key]["dtype"] != "I8" or header[scale_key]["dtype"] != "F8_E8M0":
        raise ValueError("expert-0 w1 dtype mismatch")
    weight_shape, codes = tensor_view(blob, header, weight_key)
    scale_shape, scales = tensor_view(blob, header, scale_key)
    if weight_shape != [2304, 2560] or scale_shape != [2304, 160]:
        raise ValueError("expert-0 w1 shape mismatch")
    rows, cols = 2304, 5120
    if len(codes) != rows * cols // 2 or len(scales) != rows * cols // 32:
        raise ValueError("expert-0 packed plane length mismatch")

    inputs = [INPUT_PATTERN[col % len(INPUT_PATTERN)] for col in range(cols)]
    expected = {}
    for row in ROWS:
        acc = 0.0
        for col, input_value in enumerate(inputs):
            packed = codes[row * (cols // 2) + col // 2]
            nibble = packed & 15 if col % 2 == 0 else packed >> 4
            scale_code = scales[row * (cols // 32) + col // 32]
            term = float(
                e2m1(nibble) * e8m0(scale_code) * Fraction.from_float(input_value)
            )
            acc = float(Fraction.from_float(acc) + Fraction.from_float(term))
        expected[str(row)] = f64_bits(acc)
    return {
        "schema_version": 1,
        "model": REPO,
        "revision": REVISION,
        "artifact": artifact_path.as_posix(),
        "artifact_sha256": hashlib.sha256(blob).hexdigest(),
        "tensor": weight_key,
        "scale_tensor": scale_key,
        "logical_shape": [rows, cols],
        "input_pattern_f64": INPUT_PATTERN,
        "rows": ROWS,
        "output_f64_bits": expected,
        "oracle": "Python Fraction; E2M1/E8M0 bit-fields; binary64 product rounding then ordered binary64 add rounding",
        "gpu_executed": False,
        "full_expert_forward": False,
        "full_layer_inference": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--write-golden", action="store_true")
    args = parser.parse_args()
    expected = expected_outputs(args.artifact)
    if args.write_golden:
        args.golden.write_text(json.dumps(expected, indent=2) + "\n")
        print(f"wrote {args.golden}")
        return 0
    stored = json.loads(args.golden.read_text())
    if expected != stored:
        raise SystemExit("real-expert exact Fraction oracle differs from pinned golden")
    report = {
        "status": "PASS_REAL_EXPERT_FRACTION_ORACLE",
        "artifact_sha256": expected["artifact_sha256"],
        "tensor": expected["tensor"],
        "rows_checked": len(ROWS),
        "macs_checked": len(ROWS) * 5120,
        "result_bits_exact": True,
        "oracle": expected["oracle"],
        "gpu_executed": False,
        "full_expert_forward": False,
        "full_layer_inference": False,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
