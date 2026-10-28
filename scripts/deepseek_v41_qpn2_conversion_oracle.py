#!/usr/bin/env python3
"""Independently inspect QPN2 exported from the pinned DeepSeek MXFP4 fixture.

Uses only the Python standard library. Safetensors spans, E2M1/E8M0 source values,
QPN2 fragment addressing, E4M3 power scales, and ordered binary16 rounding are decoded
here independently of the Rust converter. It exports one expert through the real CLI,
never creates dense weights, and never touches CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "fixtures/deepseek_v41_layer0_experts96-192_314113f8.safetensors"
FIXTURE_SHA256 = "e6d39f1821c0e3f7f1cc4827d206391aa428a6f3944731069d45dee4135f8c00"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
EXPERT = 96
K_ORDER = (0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15)
PROJECTIONS = (
    ("w1", 2304, 5120, "codes13", "scales13", "gscales13", 0, 0, 0),
    (
        "w3",
        2304,
        5120,
        "codes13",
        "scales13",
        "gscales13",
        2304 * 5120 // 2,
        2304 * 5120 // 16,
        2304 // 32,
    ),
    ("w2", 5120, 2304, "codes2", "scales2", "gscales2", 0, 0, 0),
)


def sha256(data: bytes | memoryview) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_safetensors(raw: bytes):
    if len(raw) < 8:
        raise ValueError("SafeTensors file is shorter than its header length")
    header_len = struct.unpack_from("<Q", raw)[0]
    data_start = 8 + header_len
    if header_len == 0 or data_start > len(raw):
        raise ValueError("SafeTensors header length is invalid")
    header = json.loads(raw[8:data_start].rstrip(b" "))
    payload = memoryview(raw)[data_start:]
    spans = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets")
        if not isinstance(offsets, list) or len(offsets) != 2:
            raise ValueError(f"{name} has no two-bound payload span")
        start, stop = offsets
        if not isinstance(start, int) or not isinstance(stop, int):
            raise ValueError(f"{name} payload offsets are not integers")
        if start < 0 or stop < start or stop > len(payload):
            raise ValueError(f"{name} payload span is outside the file")
        spans.append((start, stop, name))
    spans.sort()
    if any(left[1] > right[0] for left, right in zip(spans, spans[1:])):
        raise ValueError("SafeTensors payload spans overlap")
    return header, payload


def tensor_span(header, payload, name, dtype, shape):
    tensor = header.get(name)
    if tensor is None or tensor.get("dtype") != dtype or tensor.get("shape") != shape:
        raise ValueError(f"unexpected dtype/shape for {name}")
    start, stop = tensor["data_offsets"]
    elements = math.prod(shape)
    item_bytes = 4 if dtype == "F32" else 1
    if stop - start != elements * item_bytes:
        raise ValueError(f"unexpected byte count for {name}")
    return payload[start:stop]


def encode_safetensors(path: Path, tensors):
    header = {}
    payload = bytearray()
    for name, dtype, shape, data in tensors:
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [start, len(payload)],
        }
    raw_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw_header += b" " * (-len(raw_header) % 8)
    path.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header + payload)


def extract_one_expert(source_header, source_payload, expert, path):
    names = []
    tensors = []
    for projection in ("w1", "w2", "w3"):
        weight_name = f"layers.0.ffn.experts.{expert}.{projection}.weight"
        scale_name = f"layers.0.ffn.experts.{expert}.{projection}.scale"
        weight_shape = (2304, 2560) if projection != "w2" else (5120, 1152)
        scale_shape = (2304, 160) if projection != "w2" else (5120, 72)
        for name, shape, dtype in (
            (weight_name, weight_shape, "I8"),
            (scale_name, scale_shape, "F8_E8M0"),
        ):
            view = tensor_span(source_header, source_payload, name, dtype, list(shape))
            names.append(name)
            tensors.append((name, dtype, shape, view))
    encode_safetensors(path, tensors)
    return tuple(names)


def e2m1(code: int) -> float:
    sign = -1.0 if code & 8 else 1.0
    exponent = (code >> 1) & 3
    mantissa = (code & 1) / 2.0
    magnitude = mantissa if exponent == 0 else (1.0 + mantissa) * 2.0 ** (exponent - 1)
    return math.copysign(magnitude, sign)


def e4m3(byte: int) -> float:
    if byte & 0x80 or byte == 0x7F:
        raise ValueError("QPN2 scale is not a finite unsigned E4M3 value")
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    return (
        mantissa * 2.0**-9 if exponent == 0 else (8 + mantissa) * 2.0 ** (exponent - 10)
    )


def power_e4m3_code(exponent: int) -> int:
    if -9 <= exponent <= -7:
        return 1 << (exponent + 9)
    if -6 <= exponent <= 7:
        return (exponent + 7) << 3
    if exponent == 8:
        return 0x78
    raise ValueError("power-of-two scale is outside finite E4M3 range")


def half_bytes(value: float) -> bytes:
    try:
        return struct.pack("<e", value)
    except OverflowError:
        return b"\x00\x7c"


def half_value(value: float) -> float:
    return struct.unpack("<e", half_bytes(value))[0]


def qpn_lane(row_in_tile: int) -> int:
    return (
        (row_in_tile & 3)
        | (((row_in_tile >> 3) & 3) << 2)
        | (((row_in_tile >> 2) & 1) << 4)
    )


def qpn_fragment(tile: int, lane: int, group: int, groups: int) -> int:
    return (tile * groups + group) * 32 + lane


def qpn_nibble(codes: memoryview, offset: int, fragment: int, slot: int) -> int:
    packed = codes[offset + fragment * 8 + slot // 2]
    return (packed >> (4 * (slot & 1))) & 15


def source_nibble(codes: memoryview, row: int, col: int, columns: int) -> int:
    packed = codes[row * (columns // 2) + col // 2]
    return (packed >> (4 * (col & 1))) & 15


def verify_ordered_half_domain(exponents, factor):
    factor_half = half_value(factor * 16384.0)
    for exponent in exponents:
        scale_code = power_e4m3_code(exponent - round(math.log2(factor)))
        # Match fp8e4m3_to_half2: reinterpret (code << 7) as half, then half * 256.
        rebased = struct.unpack("<e", struct.pack("<H", scale_code << 7))[0]
        scale_half = half_value(rebased * 256.0)
        product_half = half_value(scale_half * factor_half)
        for code in range(16):
            raw_half = half_value(e2m1(code) * 2.0**-14)
            actual = half_bytes(raw_half * product_half)
            expected = half_bytes(e2m1(code) * 2.0**exponent)
            if actual != expected:
                raise ValueError(
                    f"ordered half decode mismatch for E8M0 exponent {exponent}, E2M1 {code}"
                )
    return len(exponents) * 16


def projection_oracle(
    source_header, source_payload, output_header, output_payload, spec
):
    (
        name,
        rows,
        columns,
        codes_name,
        scales_name,
        factors_name,
        code_offset,
        scale_offset,
        factor_offset,
    ) = spec
    source_prefix = f"layers.0.ffn.experts.{EXPERT}.{name}"
    source_codes = tensor_span(
        source_header,
        source_payload,
        source_prefix + ".weight",
        "I8",
        [rows, columns // 2],
    )
    source_scales = tensor_span(
        source_header,
        source_payload,
        source_prefix + ".scale",
        "F8_E8M0",
        [rows, columns // 32],
    )
    target_codes = tensor_span(
        output_header,
        output_payload,
        codes_name,
        "U8",
        [rows * columns // 2 * (2 if name == "w1" or name == "w3" else 1)],
    )
    target_scales = tensor_span(
        output_header,
        output_payload,
        scales_name,
        "U8",
        [rows * columns // 16 * (2 if name == "w1" or name == "w3" else 1)],
    )
    factor_tensor = output_header[factors_name]
    factor_shape = factor_tensor.get("shape")
    expected_factor_width = (4608 if codes_name == "codes13" else 5120) // 32
    if factor_shape != [1, expected_factor_width]:
        raise ValueError(f"invalid factor shape for {factors_name}")
    factor_start, factor_stop = factor_tensor["data_offsets"]
    factors = memoryview(output_payload)[factor_start:factor_stop]
    tiles = rows // 32
    if len(factors) < (factor_offset + tiles) * 4:
        raise ValueError(f"truncated factors for {name}")
    factor_values = [
        struct.unpack_from("<f", factors, (factor_offset + tile) * 4)[0]
        for tile in range(tiles)
    ]
    if not factor_values or any(value != factor_values[0] for value in factor_values):
        raise ValueError(f"projection factor is not uniform: {name}")
    factor = factor_values[0]
    mantissa, exp2 = math.frexp(factor)
    if not math.isfinite(factor) or factor <= 0 or mantissa != 0.5:
        raise ValueError(f"projection factor is not a positive power of two: {name}")
    shift = exp2 - 1
    exponents = set()
    # Check every source E8M0 block and both repeated QPN2 K16 scale bytes.
    # This is bounded scale metadata work; no weight matrix is materialized.
    for row in range(rows):
        tile, lane = divmod(row, 32)[0], qpn_lane(row % 32)
        for block32 in range(columns // 32):
            source_scale = source_scales[row * (columns // 32) + block32]
            if source_scale == 255:
                raise ValueError(f"NaN E8M0 source scale in {name} row {row}")
            exponent = source_scale - 127
            exponents.add(exponent)
            expected_scale = power_e4m3_code(exponent - shift)
            for sub_block in (0, 1):
                fragment = qpn_fragment(
                    tile, lane, block32 * 2 + sub_block, columns // 16
                )
                if target_scales[scale_offset + fragment] != expected_scale:
                    raise ValueError(
                        f"E8M0-to-E4M3 scale mismatch in {name} row {row}, K32 {block32}"
                    )
    coordinates = set()
    columns_to_sample = {0, 1, 7, 15, 16, 31, columns // 2, columns - 2, columns - 1}
    rows_in_tile = (0, 1, 7, 15, 16, 23, 31)
    for tile in range(tiles):
        for row_in_tile in rows_in_tile:
            row = tile * 32 + row_in_tile
            for col in columns_to_sample:
                if col < columns:
                    coordinates.add((row, col))

    for row, col in sorted(coordinates):
        tile = row // 32
        lane = qpn_lane(row % 32)
        group = col // 16
        fragment = qpn_fragment(tile, lane, group, columns // 16)
        slot = K_ORDER.index(col % 16)
        source_code = source_nibble(source_codes, row, col, columns)
        target_code = qpn_nibble(target_codes, code_offset, fragment, slot)
        if source_code != target_code:
            raise ValueError(f"code permutation mismatch at {name}[{row},{col}]")

        scale = source_scales[row * (columns // 32) + col // 32]
        if scale == 255:
            raise ValueError(f"NaN E8M0 source scale at {name}[{row},{col}]")
        exponent = scale - 127
        expected_scale = power_e4m3_code(exponent - shift)
        target_scale = target_scales[scale_offset + fragment]
        if target_scale != expected_scale:
            raise ValueError(f"scale mapping mismatch at {name}[{row},{col}]")
        if e4m3(target_scale) * factor != 2.0**exponent:
            raise ValueError(f"exact real-valued scale mismatch at {name}[{row},{col}]")

    return {
        "shape": [rows, columns],
        "sampled_coordinates": len(coordinates),
        "source_scale_exponents": sorted(exponents),
        "ordered_binary16_code_scale_pairs": verify_ordered_half_domain(
            exponents, factor
        ),
        "global_factor": factor,
        "global_shift": shift,
    }


def run(binary: Path):
    raw = FIXTURE.read_bytes()
    fixture_hash = sha256(raw)
    if fixture_hash != FIXTURE_SHA256:
        raise ValueError("pinned source fixture SHA-256 changed")
    source_header, source_payload = parse_safetensors(raw)
    work = Path(tempfile.mkdtemp(prefix="a125-qpn2-oracle-", dir=ROOT / ".work"))
    try:
        input_path = work / "expert.safetensors"
        output_path = work / "resident.safetensors"
        request_path = work / "request.json"
        source_names = extract_one_expert(
            source_header, source_payload, EXPERT, input_path
        )
        input_sha = sha256(input_path.read_bytes())
        request = {
            "source": {
                "kind": "deepseek-mxfp4-expert",
                "request": {
                    "artifact_path": str(input_path),
                    "expected_sha256": input_sha,
                    "layer": 0,
                    "expert": EXPERT,
                    "rank": 0,
                    "local": 0,
                    "epoch": 125,
                },
            },
            "output": str(output_path),
        }
        request_path.write_text(json.dumps(request, separators=(",", ":")))
        completed = subprocess.run(
            [str(binary), "resident-export", str(request_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"vf resident-export exited {completed.returncode}: {completed.stderr[:2000]}"
            )
        receipt = json.loads(completed.stdout)
        if not receipt.get("ok"):
            raise RuntimeError("vf resident-export returned a refusal")
        exported = output_path.read_bytes()
        output_header, output_payload = parse_safetensors(exported)
        metadata = json.loads(output_header["__metadata__"]["vf_info"])
        if (
            metadata.get("model_revision") != REVISION
            or metadata.get("layer") != 0
            or metadata.get("rank") != 0
            or metadata.get("epoch") != 125
            or metadata.get("shape") != [1, 4608, 5120, 5120, 2304]
            or metadata.get("physical_experts") != [{"id": EXPERT, "local": 0}]
            or receipt["result"].get("cuda_executed") is not False
        ):
            raise ValueError("resident export identity or CPU-only scope differs")
        results = {
            spec[0]: projection_oracle(
                source_header, source_payload, output_header, output_payload, spec
            )
            for spec in PROJECTIONS
        }
        payload_bytes = sum(
            len(tensor_span(output_header, output_payload, name, dtype, shape))
            for name, dtype, shape in (
                ("codes13", "U8", [4608 * 5120 // 2]),
                ("scales13", "U8", [4608 * 5120 // 16]),
                ("gscales13", "F32", [1, 4608 // 32]),
                ("codes2", "U8", [5120 * 2304 // 2]),
                ("scales2", "U8", [5120 * 2304 // 16]),
                ("gscales2", "F32", [1, 5120 // 32]),
            )
        )
        if (
            payload_bytes != 19_907_776
            or metadata.get("payload_bytes") != payload_bytes
        ):
            raise ValueError("exported compressed payload length differs")
        return {
            "status": "A125_INDEPENDENT_CPU_ORACLE_PASS",
            "revision": REVISION,
            "source_fixture": str(FIXTURE.relative_to(ROOT)),
            "source_fixture_sha256": fixture_hash,
            "source_fixture_bytes": len(raw),
            "expert": EXPERT,
            "input_expert_sha256": input_sha,
            "source_tensor_names": source_names,
            "resident_artifact_sha256": sha256(exported),
            "resident_artifact_bytes": len(exported),
            "resident_payload_bytes": payload_bytes,
            "projection_results": results,
            "code_layout_oracle": "independent Python SafeTensors/E2M1/QPN2 fragment decoder",
            "numeric_oracle": "independent Python struct binary16 rounding at each kernel stage",
            "cuda_executed": False,
            "rdma_tested": False,
            "dense_weight_allocation": False,
        }
    finally:
        shutil.rmtree(work)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.binary.resolve(strict=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
