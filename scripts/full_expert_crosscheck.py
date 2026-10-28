#!/usr/bin/env python3
"""Offline full expert0 Qwen NVFP4 representation oracle; no GPU or dense float weights."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.resident_crosscheck import invoke
from tests.reference.qpn_prepack import _qpn_prepack

REVISION = "1355db6a052410cfd62085d94b58866fd0f2c3c5"
SAMPLE_SHA = "7ede48f5bb49caa171eaf03c691b8d78d11e8644c465e4d008f8bcedf52c1c3a"
REFERENCE_SHA = "cf64be0f9cca470c3e5ab9ed504f8c5b98a903398289a459cad1c0c576c546ee"


def name(plane, suffix):
    return "model.language_model.layers.0.mlp.experts.0." + plane + "." + suffix


def f32(values):
    return b"".join(struct.pack("<f", v) for v in values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "fixtures/checkpoint_qwen36_expert0_full.safetensors",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    # Torch may be CUDA-enabled; this representation oracle must remain CPU-only.
    assert not torch.cuda.is_initialized() and sys.byteorder == "little"
    assert hashlib.sha256(args.sample.read_bytes()).hexdigest() == SAMPLE_SHA
    assert (
        hashlib.sha256(
            (ROOT / "tests/reference/qpn_prepack.py").read_bytes()
        ).hexdigest()
        == REFERENCE_SHA
    )
    with safe_open(str(args.sample), framework="pt", device="cpu") as file:
        meta = file.metadata()
        assert (
            meta["model"] == "nvidia/Qwen3.6-35B-A3B-NVFP4"
            and meta["revision"] == REVISION
        )
        assert (
            meta["license"] == "Apache-2.0" and meta["layer"] == meta["expert"] == "0"
        )
    restored = load_file(str(args.sample), device="cpu")
    planes = ("gate_proj", "up_proj", "down_proj")
    suffixes = ("weight", "weight_scale", "weight_scale_2")
    assert set(restored) == {name(p, s) for p in planes for s in suffixes}
    cropped = json.loads(
        (ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json").read_text()
    )
    packed = {}
    expected = {}
    factors = {}
    for plane in planes:
        codes = restored[name(plane, "weight")]
        scales = restored[name(plane, "weight_scale")]
        global_factor = restored[name(plane, "weight_scale_2")]
        rows, columns = (512, 2048) if plane != "down_proj" else (2048, 512)
        assert (
            codes.device.type == "cpu"
            and codes.dtype == torch.uint8
            and tuple(codes.shape) == (rows, columns // 2)
        )
        assert scales.dtype == torch.float8_e4m3fn and tuple(scales.shape) == (
            rows,
            columns // 16,
        )
        assert global_factor.dtype == torch.float32 and tuple(global_factor.shape) == ()
        scales = scales.view(torch.uint8)
        assert bool((scales < 127).all())
        factor = global_factor.item()
        assert factor > 0 and bool(torch.isfinite(global_factor))
        crop = cropped["planes"][plane]
        assert codes[:64, :32].reshape(-1).tolist() == crop["codes"]
        assert scales[:64, :4].reshape(-1).tolist() == crop["scales"]
        assert f32([factor]) == f32([crop["global_dequant"]])
        packed[plane] = dict(
            rows=rows,
            columns=columns,
            codes=codes.reshape(-1).tolist(),
            scales=scales.reshape(-1).tolist(),
            global_dequant=factor,
        )
        qc, qs = _qpn_prepack(codes, scales)
        expected[plane] = (bytes(qc.tolist()), bytes(qs.tolist()))
        factors[plane] = factor
    request = dict(
        model_revision=REVISION,
        layer=0,
        rank=0,
        registry=dict(epoch=1, copies=[dict(expert=0, rank=0, local=900)]),
        resident_order=[900],
        experts=[
            dict(
                id=0,
                local=900,
                gate=packed["gate_proj"],
                up=packed["up_proj"],
                down=packed["down_proj"],
            )
        ],
    )
    actual = invoke(args.binary, "resident-pack", request)
    assert actual["shape"] == [1, 1024, 2048, 2048, 512]
    assert actual["physical_experts"] == [dict(id=0, local=900)]
    assert actual["resident_order"] == [900]
    assert (
        actual["model_revision"],
        actual["layer"],
        actual["rank"],
        actual["epoch"],
    ) == (REVISION, 0, 0, 1)
    expectations = dict(
        codes13=expected["gate_proj"][0] + expected["up_proj"][0],
        scales13=expected["gate_proj"][1] + expected["up_proj"][1],
        codes2=expected["down_proj"][0],
        scales2=expected["down_proj"][1],
    )
    comparisons = {}
    for key, reference in expectations.items():
        observed = bytes(actual[key])
        assert observed == reference, key
        comparisons[key] = dict(
            bytes=len(reference),
            sha256=hashlib.sha256(reference).hexdigest(),
            equal=True,
        )
    for key, reference in [
        ("gscales13", [factors["gate_proj"]] * 16 + [factors["up_proj"]] * 16),
        ("gscales2", [factors["down_proj"]] * 64),
    ]:
        assert f32(actual[key]) == f32(reference), key
        comparisons[key] = dict(
            elements=len(reference),
            bytes=len(reference) * 4,
            sha256=hashlib.sha256(f32(reference)).hexdigest(),
            equal=True,
        )
    assert (
        actual["payload_bytes"]
        == 1769856
        == sum(c["bytes"] for c in comparisons.values())
    )
    native_request = dict(
        path=str(args.sample.resolve()),
        spec=dict(
            model_revision=REVISION,
            layer=0,
            rank=0,
            registry=request["registry"],
            resident_order=[900],
            experts=[dict(id=0, local=900, source_expert=0)],
            expected_metadata={
                k: meta[k] for k in ("model", "revision", "license", "layer", "expert")
            },
        ),
    )
    imported = invoke(args.binary, "resident-import", native_request)
    assert imported["stack"] == actual
    assert imported["io_bytes"] == imported["file_bytes"] == args.sample.stat().st_size
    # A real sparse local file retains the full expert and adds an unselected
    # 256 MiB tensor. Never decode/materialize/read that tensor.
    source_bytes = args.sample.read_bytes()
    old_n = int.from_bytes(source_bytes[:8], "little")
    header = json.loads(source_bytes[8 : 8 + old_n])
    data = source_bytes[8 + old_n :]
    unselected_bytes = 256 * 1024 * 1024
    header["unselected_sparse_tensor"] = dict(
        dtype="U8",
        shape=[unselected_bytes],
        data_offsets=[len(data), len(data) + unselected_bytes],
    )
    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    header_bytes += b" " * (-len(header_bytes) % 8)
    parent = args.out.resolve().parent if args.out else args.sample.resolve().parent
    with tempfile.TemporaryDirectory(prefix="vf-native-oracle-", dir=parent) as tmp:
        path = Path(tmp) / "partial.safetensors"
        with path.open("wb") as file:
            file.write(len(header_bytes).to_bytes(8, "little"))
            file.write(header_bytes)
            file.write(data)
            file.truncate(8 + len(header_bytes) + len(data) + unselected_bytes)
        partial_request = dict(native_request, path=str(path))
        partial = invoke(args.binary, "resident-import", partial_request)
        assert partial["stack"] == actual
        assert partial["io_bytes"] == 8 + len(header_bytes) + len(data)
        assert partial["file_bytes"] == partial["io_bytes"] + unselected_bytes
        assert partial["io_bytes"] < 2 * 1024 * 1024
    native_evidence = dict(
        full_export_io_bytes=imported["io_bytes"],
        sparse_file_bytes=partial["file_bytes"],
        sparse_io_bytes=partial["io_bytes"],
        unselected_payload_bytes_not_read=unselected_bytes,
        exact_stack_equal=True,
        no_python_weight_lists_at_native_boundary=True,
    )
    assert not torch.cuda.is_initialized(), "CPU oracle initialized a CUDA context"
    report = dict(
        status="PASS",
        native_import=native_evidence,
        evidence_level="CPU_COMPLETE_REAL_EXPERT_COMPRESSED_ONLY",
        model=meta["model"],
        revision=REVISION,
        license=meta["license"],
        layer=0,
        expert=0,
        hidden=2048,
        intermediate=512,
        payload_bytes=actual["payload_bytes"],
        comparisons=comparisons,
        source_safetensors_sha256=SAMPLE_SHA,
        cropped_prefixes_exact=True,
        global_factor_convention="direct F32 weight_scale_2; no reciprocal",
        source_endian="little",
        dequantized_weight_matrices=0,
        torch_build_cuda=torch.version.cuda,
        cuda_context_initialized=torch.cuda.is_initialized(),
        cuda_executed=False,
        full_model_inference=False,
        performance_claim=None,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
