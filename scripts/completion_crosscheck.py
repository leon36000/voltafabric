#!/usr/bin/env python3
"""CPU bit-exact FP16 receipt and strict collector crosscheck; no kernel is simulated."""

from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.resident_crosscheck import invoke


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    sample = json.loads(
        (ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json").read_text()
    )
    stamp = dict(
        run_id="completion-oracle",
        model_revision=sample["source"]["revision"],
        plan_id="fixed",
        layer=0,
        batch=1,
        epoch=7,
    )

    def expert(id, local):
        def plane(name):
            return {
                k: copy.deepcopy(sample["planes"][name][k])
                for k in ("rows", "columns", "codes", "scales", "global_dequant")
            }

        return dict(
            id=id,
            local=local,
            gate=plane("gate_proj"),
            up=plane("up_proj"),
            down=plane("down_proj"),
        )

    options = dict(splitk13=4, splitk2=4, max_workspace_bytes=128 * 1024 * 1024)

    def launch(weights, dispatch):
        return dict(
            weights=weights,
            dispatch=dispatch,
            expected_stamp=copy.deepcopy(stamp),
            options=options,
        )

    def packet(request, bits):
        d = request["dispatch"]
        return dict(
            launch=request,
            output=dict(
                stamp=copy.deepcopy(stamp),
                rank=d["rank"],
                rows=len(d["slots"]),
                columns=64,
                fp16_bits=bits,
            ),
        )

    # Exhaustive finite binary16 representation, independently reinterpreted by NumPy.
    encodings = np.arange(65536, dtype="<u2")
    half = encodings.view("<f2")
    finite = encodings[np.isfinite(half)]
    assert len(finite) == 63488 and len(finite) % 64 == 0
    rows = len(finite) // 64
    weights = dict(
        model_revision=stamp["model_revision"],
        layer=0,
        rank=2,
        registry=dict(epoch=7, copies=[dict(expert=10, rank=2, local=3)]),
        resident_order=[3],
        experts=[expert(10, 3)],
    )
    d = dict(
        stamp=copy.deepcopy(stamp),
        rank=2,
        hidden=64,
        tokens=rows,
        resident_order=[3],
        slots=[
            dict(
                slot=dict(token=t, slot=0, expert=10, rank=2, local=3, weight=0.25),
                physical_expert=0,
            )
            for t in range(rows)
        ],
    )
    req = packet(launch(weights, d), finite.tolist())
    result = invoke(args.binary, "resident-completions", req)
    assert result["cuda_executed"] is False
    actual = np.asarray(
        [c["value"] for c in result["completions"]], dtype="<f8"
    ).reshape(-1)
    expected = finite.view("<f2").astype("<f8")
    assert np.array_equal(actual.view("<u8"), expected.view("<u8")), (
        "finite FP16 conversion changed f64 bits"
    )
    assert len(result["completions"]) == rows
    for i, c in enumerate(result["completions"]):
        assert (c["token"], c["slot"], c["expert"], c["rank"], c["local"]) == (
            i,
            0,
            10,
            2,
            3,
        )
        assert c["stamp"] == stamp
    negative = 0
    # Keep a small packet for rejected values and identity tests.
    small = copy.deepcopy(req)
    small["launch"]["dispatch"]["tokens"] = 1
    small["launch"]["dispatch"]["slots"] = small["launch"]["dispatch"]["slots"][:1]
    small["output"]["rows"] = 1
    small["output"]["fp16_bits"] = [0] * 64
    for bits in [0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFFFF]:
        bad = copy.deepcopy(small)
        bad["output"]["fp16_bits"][-1] = bits
        assert (
            invoke(args.binary, "resident-completions", bad, ok=False)
            == "NON_FINITE_RESULT"
        )
        negative += 1
    for field in stamp:
        bad = copy.deepcopy(small)
        v = stamp[field]
        bad["output"]["stamp"][field] = v + 1 if isinstance(v, int) else v + "-stale"
        assert (
            invoke(args.binary, "resident-completions", bad, ok=False)
            == "STALE_EXECUTION"
        )
        negative += 1
    for field, value in [
        ("rank", 9),
        ("rows", 0),
        ("rows", 2),
        ("columns", 63),
        ("columns", 0),
        ("fp16_bits", []),
        ("fp16_bits", [0] * 65),
    ]:
        bad = copy.deepcopy(small)
        bad["output"][field] = value
        invoke(args.binary, "resident-completions", bad, ok=False)
        negative += 1
    for bits in [-1, 65536, 1.5]:
        bad = copy.deepcopy(small)
        bad["output"]["fp16_bits"][0] = bits
        assert (
            invoke(args.binary, "resident-completions", bad, ok=False) == "JSON_INPUT"
        )
        negative += 1
    # Real routing selects one owner among replicas. Outputs are explicit synthetic
    # compact activations; weights carry the pinned crop but no GEMM is claimed.
    registry = dict(
        epoch=7,
        copies=[
            dict(expert=10, rank=2, local=3),
            dict(expert=99, rank=2, local=900),
            dict(expert=10, rank=9, local=77),
            dict(expert=99, rank=9, local=5),
        ],
    )
    routes = [
        [dict(expert=10, weight=0.250000000001), dict(expert=99, weight=0.75)],
        [dict(expert=99, weight=0.0), dict(expert=10, weight=1.0)],
        [dict(expert=10, weight=0.5), dict(expert=99, weight=0.5)],
    ]
    problem = dict(
        hidden=64,
        experts=[10, 99],
        routes=routes,
        registry=registry,
        stamp=stamp,
        preferred_owners={"10": 9, "99": 2},
        resident_order={"2": [900, 3], "9": [77, 5], "17": []},
    )
    dispatch = invoke(args.binary, "routing", problem)
    receipts = []
    reference = np.zeros((3, 64), dtype=np.float64)

    def values(token, id):
        return np.asarray(
            (token + 1) * (-1 if id == 10 else 2) + np.arange(64) / 16, dtype="<f2"
        )

    for token, route in enumerate(routes):
        for choice in route:
            reference[token] += (
                values(token, choice["expert"]).astype(np.float64) * choice["weight"]
            )
    for rd in dispatch["ranks"]:
        locals = rd["resident_order"]
        residents = [
            expert(r["expert"], r["local"])
            for r in registry["copies"]
            if r["rank"] == rd["rank"]
        ]
        w = dict(
            model_revision=stamp["model_revision"],
            layer=0,
            rank=rd["rank"],
            registry=registry,
            resident_order=locals,
            experts=residents,
        )
        rd = copy.deepcopy(rd)
        rd["slots"].reverse()
        prepared = invoke(args.binary, "resident-launch", launch(w, rd))
        bits = []
        for slot in prepared["completion_slots"]:
            bits.extend(values(slot["token"], slot["expert"]).view("<u2").tolist())
        converted = invoke(
            args.binary, "resident-completions", packet(launch(w, rd), bits)
        )
        receipts.extend(converted["completions"])
    assert len(receipts) == 6
    receipts.reverse()
    combined = invoke(
        args.binary, "collect-routing", dict(problem=problem, completions=receipts)
    )
    assert np.array_equal(np.asarray(combined), reference)
    # Detect a double coefficient application before collector.
    doubled = copy.deepcopy(receipts)
    for c in doubled:
        weight = routes[c["token"]][c["slot"]]["weight"]
        c["value"] = [v * weight for v in c["value"]]
    wrong = invoke(
        args.binary, "collect-routing", dict(problem=problem, completions=doubled)
    )
    assert not np.array_equal(np.asarray(wrong), reference)
    for mutated, code in [
        (receipts[:-1], "BATCH_INCOMPLETE"),
        (receipts + [receipts[0]], "INVALID_COMPLETION"),
    ]:
        assert (
            invoke(
                args.binary,
                "collect-routing",
                dict(problem=problem, completions=mutated),
                ok=False,
            )
            == code
        )
        negative += 1
    stale = copy.deepcopy(receipts)
    stale[0]["stamp"]["epoch"] -= 1
    assert (
        invoke(
            args.binary,
            "collect-routing",
            dict(problem=problem, completions=stale),
            ok=False,
        )
        == "INVALID_COMPLETION"
    )
    negative += 1
    report = dict(
        status="PASS",
        evidence_level="CPU_COMPACT_OUTPUT_RECEIPTS_ONLY",
        finite_half_encodings_bit_exact=len(finite),
        nonfinite_encoding_count=2048,
        nonfinite_scope="all 2048 rejected in Rust exhaustive test; five packet representatives here",
        negative_cli_cases=negative,
        collector_receipts=6,
        collector_max_abs_error=float(np.max(np.abs(np.asarray(combined) - reference))),
        signed_zero_preserved=True,
        double_weight_sensitivity=True,
        empty_rank=True,
        source_revision=sample["source"]["revision"],
        cuda_executed=False,
        scope="real crop retained compressed; activations are explicit synthetic FP16 outputs, no GEMM/kernel simulated",
        performance_claim=None,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
