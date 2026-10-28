#!/usr/bin/env python3
"""Build a tiny three-expert real MXFP4 SafeTensors shard and CPU Torch oracle."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1]
ORACLE_PATH = SOURCE / "scripts/deepseek_v41_expert_forward_oracle.py"
SHARD_PATH = SOURCE / "fixtures/preweighted_mxfp4_tiny.safetensors"
GOLDEN_PATH = SOURCE / "fixtures/preweighted_mxfp4_tiny_golden.json"
EXPERTS = {
    17: {"w1": 2, "w3": 2, "w2": 2},
    93: {"w1": 4, "w3": 1, "w2": 3},
    241: {"w1": 10, "w3": 4, "w2": 9},
}
CODE_VALUES = {1: 0.5, 2: 1.0, 3: 1.5, 4: 2.0, 9: -0.5, 10: -1.0}
WEIGHT_CODE_TO_VALUE = {
    "w1": {17: 1.0, 93: 2.0, 241: -1.0},
    "w3": {17: 1.0, 93: 0.5, 241: 2.0},
    "w2": {17: 1.0, 93: 1.5, 241: -0.5},
}
COEFFICIENTS = [0.0, 0.375, 0.5, 1.0, 2.0]
PATTERN = [-2.0, -0.25, 0.0, 0.5, 4.0, 16.0, 0.125, -16.0]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_oracle():
    spec = importlib.util.spec_from_file_location("deepseek_v41_oracle", ORACLE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned activation oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bits(values: torch.Tensor) -> list[str]:
    return [
        f"{int(v) & 0xFFFFFFFF:08x}"
        for v in values.contiguous().view(torch.int32).tolist()
    ]


def build_shard() -> bytes:
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    for expert_id, projections in EXPERTS.items():
        for projection, code in projections.items():
            packed = bytearray(32 * 16)
            for diagonal in range(32):
                byte_index = diagonal * 16 + diagonal // 2
                if diagonal % 2 == 0:
                    packed[byte_index] |= code
                else:
                    packed[byte_index] |= code << 4
            scales = bytes([127] * 32)
            prefix = f"layers.0.ffn.experts.{expert_id}.{projection}"
            tensors[f"{prefix}.weight"] = ("I8", [32, 16], bytes(packed))
            tensors[f"{prefix}.scale"] = ("F8_E8M0", [32, 1], scales)

    header: dict[str, object] = {
        "__metadata__": {"format": "VoltaFabric tiny MXFP4 oracle fixture"}
    }
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        start = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def run_oracle(oracle, shard_sha: str) -> dict[str, object]:
    tensors = oracle.read_safetensors(SHARD_PATH, shard_sha)
    for expert, projections in EXPERTS.items():
        for projection, expected_code in projections.items():
            packed, scales = oracle.tensor_pair(tensors, expert, projection, 32, 32)
            assert (scales == 127).all()
            for row in range(32):
                for col in range(32):
                    code = (int(packed[row, col // 2]) >> (4 * (col % 2))) & 15
                    assert code == (expected_code if row == col else 0)
    inputs = {
        "pattern": PATTERN * 4,
        "pattern_reversed": list(reversed(PATTERN * 4)),
        "pattern_half": [value * 0.5 for value in PATTERN * 4],
    }
    cases: list[dict[str, object]] = []
    torch.set_num_threads(1)
    for input_name, raw in inputs.items():
        input_bf16 = torch.tensor(raw, dtype=torch.bfloat16, device="cpu")
        input_f32 = input_bf16.to(torch.float32)
        if not torch.equal(input_f32, torch.tensor(raw, dtype=torch.float32)):
            raise AssertionError(f"{input_name} is not exactly BF16 representable")
        q_in, s_in, dequant_in = oracle.activation_quantize(input_f32.tolist())
        x = torch.tensor(dequant_in, dtype=torch.float32, device="cpu")
        for expert_id, projections in EXPERTS.items():
            w1 = torch.tensor(
                [CODE_VALUES[projections["w1"]]] * 32, dtype=torch.float32
            )
            w3 = torch.tensor(
                [CODE_VALUES[projections["w3"]]] * 32, dtype=torch.float32
            )
            w2 = torch.tensor(
                [CODE_VALUES[projections["w2"]]] * 32, dtype=torch.float32
            )
            # The SafeTensors planes are read and validated above. Each projection
            # is diagonal, so each output is exactly one FP32 product, with no sum.
            gate_raw = w1 * x
            up_raw = w3 * x
            gate = torch.clamp(gate_raw, max=10.0)
            up = torch.clamp(up_raw, min=-10.0, max=10.0)
            silu = F.silu(gate)
            silu_up = silu * up
            for coefficient in COEFFICIENTS:
                weighted = silu_up * torch.tensor(coefficient, dtype=torch.float32)
                down_bf16 = weighted.to(torch.bfloat16)
                q_down, s_down, dequant_down = oracle.activation_quantize(
                    down_bf16.to(torch.float32).tolist()
                )
                down_x = torch.tensor(dequant_down, dtype=torch.float32, device="cpu")
                output = w2 * down_x
                cases.append(
                    {
                        "input": input_name,
                        "expert": expert_id,
                        "coefficient": coefficient,
                        "input_activation_e4m3_codes": q_in,
                        "input_activation_ue8m0_codes": s_in,
                        "gate_bits": bits(gate_raw),
                        "up_bits": bits(up_raw),
                        "gate_clamped_bits": bits(gate),
                        "up_clamped_bits": bits(up),
                        "silu_bits": bits(silu),
                        "swiglu_unweighted_bits": bits(silu_up),
                        "pre_down_bits": bits(weighted),
                        "down_bf16_as_f32_bits": bits(down_bf16.to(torch.float32)),
                        "down_activation_e4m3_codes": q_down,
                        "down_activation_ue8m0_codes": s_down,
                        "output_bits": bits(output),
                    }
                )
    return {
        "format": "VoltaFabric tiny MXFP4 diagonal oracle v1",
        "shape": {"hidden": 32, "intermediate": 32, "block": 32},
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": "cpu",
        "torch_num_threads": torch.get_num_threads(),
        "weights_sha256": sha(SHARD_PATH.read_bytes()),
        "oracle_source_path": "source/scripts/deepseek_v41_expert_forward_oracle.py",
        "oracle_source_sha256": sha(ORACLE_PATH.read_bytes()),
        "builder_source_sha256": sha(Path(__file__).read_bytes()),
        "input_vectors_canonical_sha256": sha(
            json.dumps(inputs, separators=(",", ":")).encode()
        ),
        "inputs": {
            name: {
                "values_f32": values,
                "bf16_as_f32_bits": bits(
                    torch.tensor(values, dtype=torch.bfloat16).to(torch.float32)
                ),
            }
            for name, values in inputs.items()
        },
        "projection_codes_e2m1": {str(k): v for k, v in EXPERTS.items()},
        "projection_diagonal_values": {
            projection: {str(expert): value for expert, value in values.items()}
            for projection, values in WEIGHT_CODE_TO_VALUE.items()
        },
        "scale_code_ue8m0": 127,
        "scale_value": 1.0,
        "coefficients_f32": COEFFICIENTS,
        "case_count": len(cases),
        "cases": cases,
        "scope": "Torch CPU oracle: FP8 pack via pinned independent activation_quantize; diagonal FP32 projections; Torch SiLU/clamp/BF16; no Rust candidate, CUDA, or network.",
    }


def main() -> None:
    if torch.version.cuda is not None:
        raise RuntimeError(
            f"CPU-only Torch required, found CUDA build {torch.version.cuda}"
        )
    if torch.get_num_threads() != 1:
        torch.set_num_threads(1)
    oracle = load_oracle()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    raw_shard = build_shard()
    if args.write:
        SHARD_PATH.write_bytes(raw_shard)
    else:
        assert SHARD_PATH.read_bytes() == raw_shard, "compressed fixture bytes changed"
    golden = run_oracle(oracle, sha(raw_shard))
    if args.write:
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")
    else:
        assert json.loads(GOLDEN_PATH.read_text()) == golden, (
            "oracle fixture or provenance changed"
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "experts": len(EXPERTS),
                "cases": golden["case_count"],
                "shard_sha256": sha(raw_shard),
                "golden_sha256": sha(GOLDEN_PATH.read_bytes()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
