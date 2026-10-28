#!/usr/bin/env python3
"""Compare owned Rust resident stacks to upstream Torch compressed prepack."""

from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.reference.qpn_prepack import _qpn_prepack


def invoke(binary, command, value, ok=True):
    r = subprocess.run(
        [str(binary), command, "-"],
        input=json.dumps(value),
        text=True,
        capture_output=True,
        timeout=30,
    )
    reply = json.loads(r.stdout)
    if ok:
        if r.returncode or not reply["ok"]:
            raise AssertionError((command, r.returncode, reply))
        return reply["result"]
    if r.returncode != 2 or reply["ok"] or "result" in reply:
        raise AssertionError((command, r.returncode, reply))
    return reply["error"]["code"]


def factor_bytes(values):
    return b"".join(struct.pack("<f", v) for v in values)


def expected_stack(experts, order):
    by_local = {e["local"]: e for e in experts}
    q13, s13, g13, q2, s2, g2 = [], [], [], [], [], []
    for local in order:
        e = by_local[local]
        gate, up, down = e["gate"], e["up"], e["down"]
        n, k = gate["rows"], gate["columns"]
        codes = torch.tensor(
            gate["codes"] + up["codes"], dtype=torch.uint8, device="cpu"
        ).reshape(n * 2, k // 2)
        scales = torch.tensor(
            gate["scales"] + up["scales"], dtype=torch.uint8, device="cpu"
        ).reshape(n * 2, k // 16)
        assert codes.device.type == scales.device.type == "cpu"
        qc, qs = _qpn_prepack(codes, scales)
        q13 += qc.tolist()
        s13 += qs.tolist()
        g13 += [gate["global_dequant"]] * (n // 32) + [up["global_dequant"]] * (n // 32)
        n2, k2 = down["rows"], down["columns"]
        qc, qs = _qpn_prepack(
            torch.tensor(down["codes"], dtype=torch.uint8, device="cpu").reshape(
                n2, k2 // 2
            ),
            torch.tensor(down["scales"], dtype=torch.uint8, device="cpu").reshape(
                n2, k2 // 16
            ),
        )
        q2 += qc.tolist()
        s2 += qs.tolist()
        g2 += [down["global_dequant"]] * (n2 // 32)
    return dict(
        codes13=q13, scales13=s13, gscales13=g13, codes2=q2, scales2=s2, gscales2=g2
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument(
        "--sample",
        type=Path,
        default=ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json",
    )
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    # A CUDA-enabled wheel is fine; this oracle must stay on CPU throughout.
    assert not torch.cuda.is_initialized(), "CPU oracle must not initialize CUDA"
    sample = json.loads(args.sample.read_text())
    assert sample["source"]["revision"] == "1355db6a052410cfd62085d94b58866fd0f2c3c5"
    planes = sample["planes"]

    def linear(name):
        return {
            key: copy.deepcopy(planes[name][key])
            for key in ("rows", "columns", "codes", "scales", "global_dequant")
        }

    def expert(global_id, local):
        return dict(
            id=global_id,
            local=local,
            gate=linear("gate_proj"),
            up=linear("up_proj"),
            down=linear("down_proj"),
        )

    experts = [expert(10, 3), expert(99, 900)]
    # A synthetic dyadic change detects accidental gate/up factor sharing.
    experts[0]["up"]["global_dequant"] *= 2
    request = dict(
        model_revision=sample["source"]["revision"],
        layer=0,
        rank=2,
        registry=dict(
            epoch=7,
            copies=[
                dict(expert=10, rank=2, local=3),
                dict(expert=99, rank=2, local=900),
                dict(expert=25, rank=9, local=0),
            ],
        ),
        resident_order=[900, 3],
        experts=experts,
    )
    cases = []
    byte_count = 0
    factors = 0
    for label, order in [
        ("reversed_sparse_order", [900, 3]),
        ("forward_sparse_order", [3, 900]),
    ]:
        request["resident_order"] = order
        actual = invoke(args.binary, "resident-pack", request)
        expected = expected_stack(experts, order)
        assert actual["shape"] == [2, 128, 64, 64, 64]
        assert actual["resident_order"] == order
        assert actual["physical_experts"] == [
            dict(id=next(e["id"] for e in experts if e["local"] == local), local=local)
            for local in order
        ]
        for key in ("codes13", "scales13", "codes2", "scales2"):
            assert actual[key] == expected[key], (label, key)
            byte_count += len(expected[key])
        for key in ("gscales13", "gscales2"):
            assert factor_bytes(actual[key]) == factor_bytes(expected[key]), (
                label,
                key,
            )
            factors += len(expected[key])
        payload = sum(
            len(expected[k]) for k in ("codes13", "scales13", "codes2", "scales2")
        ) + 4 * sum(len(expected[k]) for k in ("gscales13", "gscales2"))
        assert actual["payload_bytes"] == payload
        assert (
            factor_bytes(actual["gscales13"])[16:32]
            != factor_bytes(actual["gscales13"])[0:16]
        ), "synthetic distinct factor sensitivity"
        cases.append(
            dict(name=label, payload_bytes=payload, exact_byte_and_f32_bits=True)
        )
        # Slice each projection out of the owned stack and invert independently.
        for index, local in enumerate(order):
            e = next(e for e in experts if e["local"] == local)
            for name, code_begin, scale_begin, code_len, scale_len in [
                ("gate", index * 4096, index * 512, 2048, 256),
                ("up", index * 4096 + 2048, index * 512 + 256, 2048, 256),
                ("down", index * 2048, index * 256, 2048, 256),
            ]:
                code_key, scale_key = (
                    ("codes2", "scales2") if name == "down" else ("codes13", "scales13")
                )
                fragment = dict(
                    rows=64,
                    columns=64,
                    codes=actual[code_key][code_begin : code_begin + code_len],
                    scales=actual[scale_key][scale_begin : scale_begin + scale_len],
                    global_dequant=e[name]["global_dequant"],
                )
                unpacked = invoke(args.binary, "qpn2-unpack", fragment)
                assert unpacked == {k: e[name][k] for k in fragment}, (
                    label,
                    local,
                    name,
                )
    request["resident_order"] = [900, 3]
    routing = dict(
        hidden=64,
        experts=[10, 25, 99],
        routes=[
            [dict(expert=10, weight=0.25), dict(expert=99, weight=0.5)],
            [dict(expert=99, weight=0.75), dict(expert=25, weight=0.0)],
        ],
        registry=request["registry"],
        stamp=dict(
            run_id="resident-crosscheck",
            model_revision=request["model_revision"],
            plan_id="fixed",
            layer=0,
            batch=1,
            epoch=7,
        ),
        preferred_owners={},
        resident_order={"2": [900, 3], "9": [0], "17": []},
    )
    dispatched = invoke(args.binary, "routing", routing)
    rank = next(d for d in dispatched["ranks"] if d["rank"] == 2)
    invoke(args.binary, "resident-bind", dict(weights=request, dispatch=rank))
    mutants = []
    for field, value in [("model_revision", "foreign"), ("layer", 1), ("epoch", 6)]:
        d = copy.deepcopy(rank)
        d["stamp"][field] = value
        mutants.append(d)
    for field, value in [("rank", 9), ("resident_order", [3, 900])]:
        d = copy.deepcopy(rank)
        d[field] = value
        mutants.append(d)
    d = copy.deepcopy(rank)
    d["slots"][0]["physical_expert"] = 0
    mutants.append(d)
    for d in mutants:
        invoke(
            args.binary, "resident-bind", dict(weights=request, dispatch=d), ok=False
        )
    empty = copy.deepcopy(request)
    empty.update(rank=17, resident_order=[], experts=[])
    packed_empty = invoke(args.binary, "resident-pack", empty)
    assert (
        packed_empty["shape"] == [0, 0, 0, 0, 0] and packed_empty["payload_bytes"] == 0
    )
    idle = next(d for d in dispatched["ranks"] if d["rank"] == 17)
    invoke(args.binary, "resident-bind", dict(weights=empty, dispatch=idle))
    assert not torch.cuda.is_initialized(), "CPU oracle initialized a CUDA context"
    report = dict(
        status="PASS",
        evidence_level="CPU_COMPRESSED_RESIDENT_BUFFERS_ONLY",
        source_revision=sample["source"]["revision"],
        sample_sha256=hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        scope="synthetic IDs10/99 carrying real expert0 crops; up factor for ID10 doubled as declared sensitivity control; not full model",
        reference="unchanged upstream Torch _qpn_prepack, compressed safetensors-derived crop",
        cases=cases,
        packed_bytes_compared=byte_count,
        f32_factors_compared=factors,
        roundtrip_projections=12,
        binding_negative_cases=len(mutants),
        empty_rank=True,
        float_weights_materialized=False,
        torch_build_cuda=torch.version.cuda,
        cuda_context_initialized=torch.cuda.is_initialized(),
        cuda_executed=False,
        performance_claim=None,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
