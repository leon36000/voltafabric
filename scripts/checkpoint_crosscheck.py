#!/usr/bin/env python3
"""Verify real checkpoint crops against Torch prepack, safetensors and Rust CPU codecs."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file, save_file
from tests.reference.qpn_prepack import _qpn_prepack


def invoke(binary: Path, command: str, value: dict) -> object:
    run = subprocess.run(
        [str(binary), command, "-"],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        timeout=15,
    )
    data = json.loads(run.stdout)
    if run.returncode or not data["ok"]:
        raise AssertionError((run.returncode, data))
    return data["result"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path("fixtures/checkpoint_qwen36_expert0_crop.json"),
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    sample = json.loads(args.sample.read_text())
    assert sample["kind"] == "real_checkpoint_cropped_expert_not_full_model"
    # A CUDA-enabled wheel is fine; every tensor stays on CPU and no context is created.
    assert not torch.cuda.is_initialized(), "CPU oracle must not initialize CUDA"
    tensors = {}
    for name, p in sample["planes"].items():
        n, k = p["rows"], p["columns"]
        codes = torch.tensor(p["codes"], dtype=torch.uint8, device="cpu").reshape(
            n, k // 2
        )
        scale_bytes = torch.tensor(
            p["scales"], dtype=torch.uint8, device="cpu"
        ).reshape(n, k // 16)
        scales = scale_bytes.view(torch.float8_e4m3fn)
        factor = torch.tensor(p["global_dequant"], dtype=torch.float32, device="cpu")
        tensors[name + ".codes"] = codes
        tensors[name + ".scales"] = scales
        tensors[name + ".global"] = factor
    # Export and independently re-read compressed planes; every byte must survive.
    target = args.sample.with_suffix(".safetensors")
    if target.exists():
        saved = load_file(str(target), device="cpu")
        assert set(saved) == set(tensors)
        for name, value in tensors.items():
            assert torch.equal(
                saved[name].reshape(-1).view(torch.uint8),
                value.reshape(-1).view(torch.uint8),
            ), name
    else:
        save_file(
            tensors,
            str(target),
            metadata={
                "source_revision": sample["source"]["revision"],
                "scope": "64x64 crop of one real expert",
            },
        )
        saved = load_file(str(target), device="cpu")
    maximum = 0.0
    count = 0
    for name, p in sample["planes"].items():
        n, k = p["rows"], p["columns"]
        codes = saved[name + ".codes"]
        scale_bytes = saved[name + ".scales"].view(torch.uint8)
        qc, qs = _qpn_prepack(codes, scale_bytes)
        linear = {
            key: p[key]
            for key in ["rows", "columns", "codes", "scales", "global_dequant"]
        }
        packed = invoke(args.binary, "qpn2-pack", linear)
        assert packed["codes"] == qc.tolist() and packed["scales"] == qs.tolist(), name
        unpacked = invoke(args.binary, "qpn2-unpack", packed)
        assert unpacked == linear, name
        nib = torch.stack((codes & 15, codes >> 4), dim=-1).reshape(n, k)
        lut = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float64,
            device="cpu",
        )
        expected = (
            lut[nib.long()]
            * saved[name + ".scales"].double().repeat_interleave(16, dim=1)
            * p["global_dequant"]
        )
        actual = torch.tensor(
            invoke(args.binary, "qpn2-dequant", packed),
            dtype=torch.float64,
            device="cpu",
        )
        linear_actual = torch.tensor(
            invoke(args.binary, "dequant", linear),
            dtype=torch.float64,
            device="cpu",
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(linear_actual, expected, rtol=0, atol=0)
        maximum = max(maximum, float(torch.max(torch.abs(actual - expected))))
        count += n * k
    assert not torch.cuda.is_initialized(), "CPU oracle initialized a CUDA context"
    report = {
        "status": "PASS",
        "evidence_level": "CPU_TESTED_REAL_CHECKPOINT_CROP",
        "source_model": sample["source"]["model"],
        "source_revision": sample["source"]["revision"],
        "source_fixture_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "export_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "exporter": "safetensors.torch save_file/load_file 0.6.2",
        "prepack_reference": "unchanged donor _qpn_prepack executed with torch CPU",
        "planes": len(sample["planes"]),
        "decoded_values": count,
        "max_abs_error": maximum,
        "atol": 0.0,
        "rtol": 0.0,
        "scope": "cropped expert0 layer0, not full model inference",
        "torch_build_cuda": torch.version.cuda,
        "cuda_context_initialized": torch.cuda.is_initialized(),
        "cuda_executed": False,
    }
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
