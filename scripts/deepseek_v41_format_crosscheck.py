#!/usr/bin/env python3
"""Cross-check the pinned DeepSeek FP4 byte layout against real expert planes.

This reads the MIT-licensed exporter/kernel snapshot and the small real expert
fixture. It does not import PyTorch, execute a model, or access a GPU.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
ARTIFACT = ROOT / "fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors"
UPSTREAM = ROOT / "evidence/continuation_a97_model_metadata/upstream_314113f8"
EXPECTED_ARTIFACT_SHA256 = (
    "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b"
)
UPSTREAM_HASHES = {
    "convert.py": "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
}
PROJECTIONS = {
    "w1": ([2304, 2560], [2304, 160]),
    "w2": ([5120, 1152], [5120, 72]),
    "w3": ([2304, 2560], [2304, 160]),
}


def e2m1(code: int) -> Fraction:
    sign = (code >> 3) & 1
    exponent = (code >> 1) & 3
    mantissa = Fraction(code & 1, 2)
    magnitude = mantissa if exponent == 0 else (1 + mantissa) * 2 ** (exponent - 1)
    return -magnitude if sign else magnitude


def upstream_fp4_table(path: Path) -> list[Fraction]:
    tree = ast.parse(path.read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "FP4_TABLE"
            for target in node.targets
        )
    )
    raw_values = ast.literal_eval(assignment.value.args[0])
    return [Fraction.from_float(float(value)) for value in raw_values]


def verify_upstream_layout(upstream: Path) -> dict[str, str]:
    for name, expected in UPSTREAM_HASHES.items():
        digest = hashlib.sha256((upstream / name).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"pinned official {name} hash mismatch: {digest}")

    converter = (upstream / "convert.py").read_text()
    kernel = (upstream / "kernel.py").read_text()
    model = (upstream / "model.py").read_text()
    required = (
        (converter, "low = x & 0x0F"),
        (converter, "high = (x >> 4) & 0x0F"),
        (
            converter,
            "torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)",
        ),
        (
            converter,
            "state_dicts[i][name] = state_dicts[i][name].view(torch.float4_e2m1fn_x2)",
        ),
        (kernel, "The FP4 values are packed along the K (last) dimension."),
        (
            kernel,
            "B is stored as [N, K//2] in float4_e2m1fn_x2 (2 FP4 values per byte, packed along K).",
        ),
        (model, "fp4_block_size = 32"),
        (model, "one fp4 scale per 32 elements along K"),
    )
    for text, fragment in required:
        if fragment not in text:
            raise ValueError(
                f"pinned official source no longer states layout contract: {fragment}"
            )

    table = upstream_fp4_table(upstream / "convert.py")
    expected_table = [e2m1(code) for code in range(16)]
    if table != expected_table:
        raise ValueError("official FP4 lookup table differs from E2M1 bit-field values")
    return {name: UPSTREAM_HASHES[name] for name in UPSTREAM_HASHES}


def safetensors_tensors(artifact: Path) -> tuple[bytes, dict]:
    raw = artifact.read_bytes()
    if len(raw) < 8:
        raise ValueError("Safetensors artifact is truncated before header length")
    header_size = struct.unpack("<Q", raw[:8])[0]
    data_start = 8 + header_size
    if data_start > len(raw):
        raise ValueError("Safetensors header exceeds artifact length")
    return raw[data_start:], json.loads(raw[8:data_start])


def verify_artifact(artifact: Path, official_table: list[Fraction]) -> dict:
    raw_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
    if raw_sha256 != EXPECTED_ARTIFACT_SHA256:
        raise ValueError(f"pinned expert artifact hash mismatch: {raw_sha256}")
    payload, header = safetensors_tensors(artifact)

    tensors = {}
    spans = []
    histogram: Counter[int] = Counter()
    scales: Counter[int] = Counter()
    logical_values = 0
    for projection, (weight_shape, scale_shape) in PROJECTIONS.items():
        prefix = f"layers.0.ffn.experts.0.{projection}"
        for suffix, expected_dtype, expected_shape in (
            ("weight", "I8", weight_shape),
            ("scale", "F8_E8M0", scale_shape),
        ):
            name = f"{prefix}.{suffix}"
            tensor = header.get(name)
            if (
                tensor is None
                or tensor.get("dtype") != expected_dtype
                or tensor.get("shape") != expected_shape
            ):
                raise ValueError(f"unexpected pinned tensor metadata for {name}")
            start, stop = tensor["data_offsets"]
            if start < 0 or stop < start or stop > len(payload):
                raise ValueError(f"invalid payload interval for {name}")
            expected_bytes = 1
            for dim in expected_shape:
                expected_bytes *= dim
            if stop - start != expected_bytes:
                raise ValueError(f"payload byte length mismatch for {name}")
            spans.append((start, stop, name))
            data = payload[start:stop]
            if len(data) != expected_bytes:
                raise ValueError(f"truncated payload for {name}")
            if suffix == "weight":
                for byte, count in Counter(data).items():
                    histogram[byte & 0x0F] += count
                    histogram[byte >> 4] += count
                logical_values += 2 * len(data)
                tensors[name] = {
                    "dtype": expected_dtype,
                    "shape": expected_shape,
                    "packed_bytes": len(data),
                    "logical_fp4_values": 2 * len(data),
                }
            else:
                scales.update(data)
                tensors[name] = {
                    "dtype": expected_dtype,
                    "shape": expected_shape,
                    "scale_bytes": len(data),
                }

    spans.sort()
    for previous, current in zip(spans, spans[1:]):
        if previous[1] > current[0]:
            raise ValueError(
                f"overlapping Safetensors payload intervals: {previous[2]} / {current[2]}"
            )
    if scales[255]:
        raise ValueError("real expert contains the E8M0 NaN sentinel 0xff")
    for code in histogram:
        if official_table[code] != e2m1(code):
            raise ValueError(f"official FP4 lookup differs for code {code:#x}")

    return {
        "artifact_sha256": raw_sha256,
        "payload_bytes": len(payload),
        "tensors": tensors,
        "logical_fp4_values_checked": logical_values,
        "all_16_fp4_codes_seen": set(histogram) == set(range(16)),
        "fp4_code_histogram": {f"{code:x}": histogram[code] for code in range(16)},
        "scale_bytes_checked": sum(scales.values()),
        "e8m0_nan_sentinel_count": scales[255],
        "observed_scale_code_range": [min(scales), max(scales)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--upstream-dir", type=Path, default=UPSTREAM)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    upstream_hashes = verify_upstream_layout(args.upstream_dir)
    table = upstream_fp4_table(args.upstream_dir / "convert.py")
    result = {
        "status": "PASS_PINNED_OFFICIAL_FP4_LAYOUT_REAL_EXPERT0",
        "model": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": REVISION,
        "license": "MIT",
        "official_source_sha256": upstream_hashes,
        "layout_contract": {
            "storage_dtype": "I8 byte container viewed as float4_e2m1fn_x2",
            "first_logical_k": "low nibble",
            "second_logical_k": "high nibble",
            "scales": "F8_E8M0 / UE8M0, one per 32 consecutive K values",
            "matrix_layout": "row-major [out, K], packed along K",
        },
        "real_expert": verify_artifact(args.artifact, table),
        "gpu_executed": False,
        "kernel_executed": False,
        "full_expert_forward": False,
        "full_layer_inference": False,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.write_text(output)
    print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
