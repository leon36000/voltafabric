#!/usr/bin/env python3
"""Independent CPU oracle for one token through the pinned compressed DeepSeek expert0."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
ARTIFACT = ROOT / "fixtures/deepseek_v41_layer0_expert0_314113f8.safetensors"
GOLDEN = ROOT / "fixtures/deepseek_v41_expert0_forward_bf16_314113f8.json"
UPSTREAM = ROOT / "evidence/continuation_a97_model_metadata/upstream_314113f8"
ARTIFACT_SHA256 = "f1bdb87fe86fcafa8ae72e4154a8b4a0e66b90eaba8366a398676cb2fbe4510b"
UPSTREAM_HASHES = {
    "convert.py": "035028340479145594a81d6084a8424e57363adf83c0d5983914783d95614d76",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
    "model.py": "e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d",
}
BLOCK = 32
INPUT_PATTERN = [
    0.125,
    -0.375,
    0.5,
    -0.25,
    0.0625,
    -0.5,
    0.0,
    0.3125,
    0.75,
    -0.125,
    0.25,
    -0.625,
    -0.1875,
    0.4375,
    -0.0625,
    0.5625,
]
ROUTING_WEIGHT = 0.375
SWIGLU_LIMIT = 10.0


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_upstream() -> None:
    for name, expected in UPSTREAM_HASHES.items():
        if digest(UPSTREAM / name) != expected:
            raise RuntimeError(f"pinned upstream {name} hash mismatch")
    model = (UPSTREAM / "model.py").read_text()
    kernel = (UPSTREAM / "kernel.py").read_text()
    required_model = (
        "fp8_block_size = 32",
        "class Expert(nn.Module):",
        "x = F.silu(gate) * up",
        "return self.w2(x.to(dtype))",
    )
    required_kernel = (
        "C[M, N] = A_fp8[M, K] @ B_fp4[N, K]^T",
        "C_local_accum[i, j] += C_local[i, j] * scale_a_frag[i] * scale_b_frag[j]",
        "act_group_size = act_block_size",
        "weight_group_size = 32",
    )
    if any(fragment not in model for fragment in required_model):
        raise RuntimeError("pinned model expert forward contract changed")
    if any(fragment not in kernel for fragment in required_kernel):
        raise RuntimeError("pinned FP4 GEMM contract changed")


def read_safetensors(path: Path, expected_sha256: str) -> dict[str, tuple[dict, bytes]]:
    raw = path.read_bytes()
    if digest(path) != expected_sha256:
        raise RuntimeError("pinned expert fixture SHA256 mismatch")
    header_size = struct.unpack("<Q", raw[:8])[0]
    header_end = 8 + header_size
    if header_end > len(raw):
        raise RuntimeError("truncated Safetensors header")
    header = json.loads(raw[8:header_end])
    payload = raw[header_end:]
    tensors = {}
    intervals = []
    for name, spec in header.items():
        if name == "__metadata__":
            continue
        start, end = spec["data_offsets"]
        if start < 0 or end < start or end > len(payload):
            raise RuntimeError(f"invalid Safetensors interval for {name}")
        intervals.append((start, end, name))
        tensors[name] = (spec, payload[start:end])
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise RuntimeError("overlapping Safetensors tensors")
    return tensors


def e2m1(code: int) -> float:
    sign = -1.0 if code & 8 else 1.0
    exponent = (code >> 1) & 3
    mantissa = code & 1
    magnitude = (
        mantissa * 0.5
        if exponent == 0
        else (1.0 + mantissa * 0.5) * 2.0 ** (exponent - 1)
    )
    return sign * magnitude


FP4_TABLE = np.asarray([e2m1(code) for code in range(16)], dtype=np.float64)


def tensor_pair(tensors, expert_id: int, projection: str, rows: int, cols: int):
    prefix = f"layers.0.ffn.experts.{expert_id}.{projection}"
    weight_spec, weight_bytes = tensors[f"{prefix}.weight"]
    scale_spec, scale_bytes = tensors[f"{prefix}.scale"]
    expected_weight_shape = [rows, cols // 2]
    expected_scale_shape = [rows, cols // BLOCK]
    if weight_spec["dtype"] != "I8" or weight_spec["shape"] != expected_weight_shape:
        raise RuntimeError(f"unexpected packed shape for {prefix}.weight")
    if scale_spec["dtype"] != "F8_E8M0" or scale_spec["shape"] != expected_scale_shape:
        raise RuntimeError(f"unexpected scale shape for {prefix}.scale")
    packed = np.frombuffer(weight_bytes, dtype=np.uint8).reshape(rows, cols // 2)
    scales = np.frombuffer(scale_bytes, dtype=np.uint8).reshape(rows, cols // BLOCK)
    if np.any(scales == 255):
        raise RuntimeError(f"{prefix} contains invalid UE8M0 NaN scale")
    return packed, scales


def activation_quantize(values: list[float]):
    x = torch.tensor(values, dtype=torch.float32, device="cpu")
    if x.numel() == 0 or x.numel() % BLOCK:
        raise RuntimeError("activation K must be a nonempty multiple of 32")
    qbytes: list[int] = []
    scodes: list[int] = []
    dequantized: list[float] = []
    inverse = torch.tensor(1.0 / 448.0, dtype=torch.float32)
    for block in x.split(BLOCK):
        amax = torch.maximum(
            block.abs().max(), torch.tensor(1.0e-4, dtype=torch.float32)
        )
        exponent = math.ceil(math.log2(float((amax * inverse).item())))
        scale = torch.tensor(math.ldexp(1.0, exponent), dtype=torch.float32)
        quantized = (block / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        stored_scale = scale.reshape(1).to(torch.float8_e8m0fnu)
        qbytes.extend(int(value) for value in quantized.view(torch.uint8).tolist())
        scale_code = int(stored_scale.view(torch.uint8).item())
        scodes.append(scale_code)
        dequantized.extend(
            float(value) for value in (quantized.to(torch.float32) * scale).tolist()
        )
    return qbytes, scodes, dequantized


def compressed_matvec(packed: np.ndarray, scales: np.ndarray, values: list[float]):
    rows, half_cols = packed.shape
    cols = half_cols * 2
    if cols != len(values) or cols % BLOCK:
        raise RuntimeError("compressed matvec shape mismatch")
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros(rows, dtype=np.float64)
    for block in range(cols // BLOCK):
        bytes_block = packed[:, block * (BLOCK // 2) : (block + 1) * (BLOCK // 2)]
        codes = np.empty((rows, BLOCK), dtype=np.uint8)
        codes[:, 0::2] = bytes_block & 15
        codes[:, 1::2] = bytes_block >> 4
        weights = FP4_TABLE[codes]
        scale = np.exp2(scales[:, block].astype(np.int16) - 127)
        weights *= scale[:, None]
        out += weights @ x[block * BLOCK : (block + 1) * BLOCK]
    return out.astype(np.float32)


def forward_reference(tensors, expert_id: int = 0):
    input_values = INPUT_PATTERN * (5120 // len(INPUT_PATTERN))
    input_bf16 = torch.tensor(input_values, dtype=torch.bfloat16, device="cpu")
    input_f32 = input_bf16.to(torch.float32)
    if input_f32.numel() != 5120:
        raise RuntimeError("bad deterministic hidden vector length")

    input_qbytes, input_scales, input_dequant = activation_quantize(input_f32.tolist())
    del input_qbytes, input_scales
    w1, w1s = tensor_pair(tensors, expert_id, "w1", 2304, 5120)
    w3, w3s = tensor_pair(tensors, expert_id, "w3", 2304, 5120)
    w2, w2s = tensor_pair(tensors, expert_id, "w2", 5120, 2304)
    gate_np = compressed_matvec(w1, w1s, input_dequant)
    up_np = compressed_matvec(w3, w3s, input_dequant)
    gate = torch.from_numpy(gate_np).to(torch.float32)
    up = torch.from_numpy(up_np).to(torch.float32)
    gate = torch.clamp(gate, max=SWIGLU_LIMIT)
    up = torch.clamp(up, min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    activation_f32 = F.silu(gate) * up
    activation_f32 = activation_f32 * torch.tensor(ROUTING_WEIGHT, dtype=torch.float32)
    activation_bf16 = activation_f32.to(torch.bfloat16)
    down_qbytes, down_scales, down_dequant = activation_quantize(
        activation_bf16.to(torch.float32).tolist()
    )
    del down_qbytes, down_scales
    output = compressed_matvec(w2, w2s, down_dequant)
    if not np.all(np.isfinite(output)):
        raise RuntimeError("nonfinite result from compressed expert oracle")
    return input_f32, gate, up, activation_bf16, output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--write-golden", action="store_true")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expert-id", type=int, default=0)
    parser.add_argument("--golden", type=Path)
    args = parser.parse_args()
    artifact_path = args.artifact.resolve(strict=True)
    if args.expert_id < 0:
        parser.error("--expert-id must be nonnegative")
    expected_artifact_sha256 = args.expected_sha256
    if expected_artifact_sha256 is None:
        if artifact_path != ARTIFACT.resolve():
            parser.error("--expected-sha256 is required for a non-default artifact")
        expected_artifact_sha256 = ARTIFACT_SHA256
    if len(expected_artifact_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_artifact_sha256
    ):
        parser.error("--expected-sha256 must be 64 lowercase hexadecimal characters")
    golden_path = args.golden or (
        GOLDEN
        if args.expert_id == 0
        else ROOT
        / f"fixtures/deepseek_v41_expert{args.expert_id}_forward_bf16_{REVISION[:8]}.json"
    )
    if torch.version.cuda is not None:
        raise RuntimeError("oracle requires CPU-only PyTorch")
    if torch.get_default_dtype() != torch.float32:
        raise RuntimeError("oracle requires PyTorch default dtype FP32")
    check_upstream()
    tensors = read_safetensors(artifact_path, expected_artifact_sha256)
    input_f32, gate, up, activation_bf16, output = forward_reference(
        tensors, args.expert_id
    )
    data = {
        "schema_version": 1,
        "model_revision": REVISION,
        "artifact_sha256": expected_artifact_sha256,
        "expert_id": args.expert_id,
        "input_dtype": "BF16",
        "routing_weight_f32_bits": f"{struct.unpack('<I', struct.pack('<f', ROUTING_WEIGHT))[0]:08x}",
        "input_f32_bits": [
            f"{value:08x}" for value in input_f32.view(torch.int32).tolist()
        ],
        "output_f32": [float(value) for value in output.tolist()],
    }
    if args.write_golden:
        golden_path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
        golden_mode = "GENERATED_FROM_PINNED_TORCH_CPU_REFERENCE"
    else:
        if not golden_path.is_file():
            raise RuntimeError("golden missing; run once with --write-golden")
        golden = json.loads(golden_path.read_text())
        if golden["input_f32_bits"] != data["input_f32_bits"]:
            raise RuntimeError("fixed BF16 input differs from checked-in oracle")
        if golden["routing_weight_f32_bits"] != data["routing_weight_f32_bits"]:
            raise RuntimeError(
                "fixed routing coefficient differs from checked-in oracle"
            )
        expected = np.asarray(golden["output_f32"], dtype=np.float64)
        observed = output.astype(np.float64)
        max_abs = float(np.max(np.abs(observed - expected)))
        if max_abs > 2.0e-5:
            raise RuntimeError(f"pinned CPU reference drift exceeds 2e-5: {max_abs}")
        golden_mode = "PINNED_GOLDEN_REPRODUCED"

    report = {
        "status": "PASS_SINGLE_TOKEN_COMPRESSED_EXPERT_CPU_ORACLE",
        "golden_mode": golden_mode,
        "model_revision": REVISION,
        "expert_id": args.expert_id,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "fixture_sha256": expected_artifact_sha256,
        "input_shape": [1, 5120],
        "expert_shapes": {"w1": [2304, 5120], "w3": [2304, 5120], "w2": [5120, 2304]},
        "weight_ownership": "compressed uint8 FP4 plus UE8M0 scales; one 32-column NumPy tile transiently decoded at a time",
        "input_quant_groups": 160,
        "down_quant_groups": 72,
        "routing_weight_f32_bits": data["routing_weight_f32_bits"],
        "gate_f32_sha256": hashlib.sha256(gate.numpy().tobytes()).hexdigest(),
        "up_f32_sha256": hashlib.sha256(up.numpy().tobytes()).hexdigest(),
        "pre_down_bf16_sha256": hashlib.sha256(
            activation_bf16.view(torch.uint8).numpy().tobytes()
        ).hexdigest(),
        "output_f32_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
        "output_f32_samples": [
            float(output[index]) for index in (0, 1, 17, 1024, 2047, 5119)
        ],
        "output_max_abs": float(np.max(np.abs(output))),
        "output_elements": int(output.size),
        "gpu_executed": False,
        "cuda_executed": False,
        "rdma_tested": False,
        "performance_claim": "NONE",
        "full_layer_or_model": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
