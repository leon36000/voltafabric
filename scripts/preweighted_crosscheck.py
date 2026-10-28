#!/usr/bin/env python3
"""Independent binary32 oracle against the actual bounded preweighted CLI."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fp32_oracle", ROOT / "tests/reference/fp32_add_oracle.py"
)
ORACLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORACLE)


def problem(tokens, topk, width, batch):
    experts = [17 + i * 11 for i in range(topk)]
    locals_ = [3 + i * 7 for i in range(topk)]
    return {
        "hidden": width,
        "experts": experts,
        "routes": [
            [{"expert": e, "weight": 0.375} for e in experts] for _ in range(tokens)
        ],
        "registry": {
            "epoch": 11,
            "copies": [
                {"expert": e, "rank": 2, "local": local}
                for e, local in zip(experts, locals_)
            ],
        },
        "stamp": {
            "run_id": "preweighted-fp32-oracle",
            "model_revision": "numeric-corpus-not-a-model",
            "plan_id": "fp32-slot-order-v1",
            "layer": 0,
            "batch": batch,
            "epoch": 11,
        },
        "preferred_owners": {},
        "resident_order": {"2": list(reversed(locals_))},
    }


def call(binary, command, payload, success=True):
    p = subprocess.run(
        [str(binary), command, "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    result = json.loads(p.stdout)
    assert (p.returncode == 0) == success, (p.returncode, p.stdout, p.stderr)
    assert result["ok"] is success, result
    if not success:
        assert "result" not in result, result
    return result


def collect(binary, p, values, success=True):
    contract = call(binary, "preweighted-fp32-contract", p)["result"]
    rows = []
    for token, slots in enumerate(values):
        for slot, bits in enumerate(slots):
            replica = p["registry"]["copies"][slot]
            rows.append(
                {
                    "contract": contract,
                    "stamp": p["stamp"],
                    "token": token,
                    "slot": slot,
                    **replica,
                    "weight_bits": 0x3EC00000,
                    "value_bits": bits,
                }
            )
    # Reverse arrival order without changing original route-slot order.
    return call(
        binary,
        "collect-preweighted-fp32",
        {"problem": p, "completions": list(reversed(rows))},
        success,
    )


def run(binary):
    ORACLE.self_test()
    ORACLE.check_corpus()
    corpus = json.loads((ROOT / "fixtures/preweighted_fp32_sum_cases.json").read_text())
    cases = corpus["cases"] if isinstance(corpus, dict) else corpus
    for batch, case in enumerate(cases):
        status, expected = ORACLE.accumulate(case["terms"])
        assert (status, expected) == (case["status"], case["result"])
        p = problem(1, len(case["terms"]), 1, batch)
        result = collect(
            binary, p, [[[int(b, 16)] for b in case["terms"]]], status == "ok"
        )
        if status == "ok":
            assert result["result"]["fp32_bits"] == [[int(expected, 16)]], case

    rng = random.Random(20261024)
    p = problem(64, 4, 3, 100)
    values = [
        [
            [
                (rng.randrange(2) << 31)
                | (rng.randrange(201) << 23)
                | rng.randrange(1 << 23)
                for _ in range(3)
            ]
            for _ in range(4)
        ]
        for _ in range(64)
    ]
    expected = []
    for slots in values:
        output = []
        for col in range(3):
            status, bits = ORACLE.accumulate([f"{slot[col]:08x}" for slot in slots])
            assert status == "ok"
            output.append(int(bits, 16))
        expected.append(output)
    result = collect(binary, p, values)
    assert result["result"]["fp32_bits"] == expected
    return {
        "status": "PASS",
        "corpus_cases": len(cases),
        "generated_seed": 20261024,
        "generated_input_values": 64 * 4 * 3,
        "generated_output_values": 64 * 3,
        "oracle": "independent exact rational binary32 RNE",
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "gpu_executed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = run(args.binary.resolve(strict=True))
    if args.out:
        with args.out.open("x") as output:
            json.dump(report, output, indent=2)
    print(json.dumps(report))
