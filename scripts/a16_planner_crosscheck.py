#!/usr/bin/env python3
"""Exact binary64 Fraction oracle for all planner weighted scores (CPU only)."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
import random
import struct
import subprocess

MAX = float.fromhex("0x1.fffffffffffffp+1023")
MINSUB = float.fromhex("0x0.0000000000001p-1022")
MINNORMAL = float.fromhex("0x1.0000000000000p-1022")


def payload(costs: list[float], frequencies: list[float]) -> dict:
    shared = len(costs) == 1 and len(frequencies) > 1
    if not shared and len(costs) != len(frequencies):
        raise ValueError("costs must be per-demand or one shared expert cost")
    provenance = {"origin": "synthetic", "source": "A16 exact-dyadic oracle v4"}
    return {
        "topology": {
            "schema_version": 1,
            "origin": "synthetic",
            "source_note": "A16 local exact score oracle; no hardware measurement",
            "devices": [
                {
                    "rank": 0,
                    "host": "synthetic",
                    "capacity_bytes": 18446744073709551615,
                    "reserved_bytes": 0,
                }
            ],
            "links": [],
        },
        "experts": [
            {
                "id": i,
                "resident_bytes": 1,
                "compute_us": c,
                "cost_provenance": provenance,
            }
            for i, c in enumerate(costs)
        ],
        "demands": [
            {"source_rank": 0, "experts": [0 if shared else i], "frequency": w}
            for i, w in enumerate(frequencies)
        ],
        "hidden": 1,
        "wire": {
            "input_bytes": 2,
            "expert_output_bytes": 2,
            "partial_output_bytes": 2,
            "header_bytes": 0,
            "slot_metadata_bytes": 0,
            "alignment": 1,
        },
        "max_iterations": 0,
        "wire_policy": "baseline",
        "max_work_units": 100000,
    }


def exact_mean(costs: list[float], frequencies: list[float]) -> Fraction | None:
    expanded = costs * len(frequencies) if len(costs) == 1 else costs
    numerator = sum(
        (
            Fraction.from_float(w) * Fraction.from_float(c)
            for w, c in zip(frequencies, expanded)
        ),
        Fraction(),
    )
    denominator = sum((Fraction.from_float(w) for w in frequencies), Fraction())
    return numerator / denominator if denominator else None


def invoke(
    binary: Path, costs: list[float], frequencies: list[float]
) -> tuple[float | None, str | None]:
    result = subprocess.run(
        [str(binary), "plan", "-"],
        input=json.dumps(payload(costs, frequencies), allow_nan=False),
        text=True,
        capture_output=True,
        timeout=10,
    )
    data = json.loads(result.stdout)
    if result.returncode != 0 or not data.get("ok"):
        code = data.get("error", {}).get("code")
        if code == "INVALID_INPUT":
            return None, code
        raise AssertionError((result.returncode, data, costs, frequencies))
    return float(data["result"]["modeled_weighted_latency_us"]), None


def scale_exactly(values: list[float], exponent: int) -> list[float] | None:
    scaled = []
    for value in values:
        try:
            candidate = math.ldexp(value, exponent)
            if not math.isfinite(candidate) or candidate <= 0.0:
                return None
            if math.ldexp(candidate, -exponent) != value:
                return None
            scaled.append(candidate)
        except OverflowError:
            return None
    return scaled


def random_positive(rng: random.Random) -> float:
    while True:
        bits = rng.getrandbits(63)
        exponent = (bits >> 52) & 0x7FF
        if bits != 0 and exponent != 0x7FF:
            return struct.unpack(">d", struct.pack(">Q", bits))[0]


def cases() -> list[tuple[list[float], list[float]]]:
    result: list[tuple[list[float], list[float]]] = [
        ([1e-300, 0.0], [1e-20, 1.0]),
        ([MAX / 3.0, 0.0], [MINSUB, 1.0]),
        ([0.0] * 255 + [MAX], [1.0] * 255 + [MINSUB]),
        ([3.0 * MINSUB, 0.0], [1.0, 1.0]),
        ([5.0 * MINSUB, 0.0], [3.0, 7.0]),
        ([3.0 * MINSUB, 3.0 * MINSUB, 3.0 * MINSUB, 0.0], [1.0] * 4),
        ([MAX, 0.0], [MINSUB, MAX]),
        ([MAX, 0.0], [MAX, MAX]),
        ([MAX, MAX / 2.0, 0.0], [MAX, MAX, MINSUB]),
        ([MAX, 3.0], [0.0, 1.0]),
        ([MAX, 3.0, 2.0], [0.0, 1.0, 0.0]),
        ([1.0, MAX], [0.0, 0.0]),
        ([MINNORMAL, 0.0], [(1 << 53) - 2.0, 1.0]),
        ([0.1], [random.Random(20261009).random() or 1.0] * 1024),
    ]
    rng = random.Random(16009)
    for _ in range(50):
        n = rng.randint(2, 8)
        frequencies = [random_positive(rng) for _ in range(n)]
        costs = [
            0.0 if rng.randrange(5) == 0 else random_positive(rng) for _ in range(n)
        ]
        result.append((costs, frequencies))
    if len(result) > 70:
        raise AssertionError(f"case bound exceeded: {len(result)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    base_cases = cases()
    checks = []
    for index, (costs, frequencies) in enumerate(base_cases):
        variants = [
            (costs, frequencies, "original"),
            (costs[::-1], frequencies[::-1], "reversed"),
        ]
        exponent = 128 if index % 2 == 0 else -128
        scaled = scale_exactly(frequencies, exponent)
        if scaled is not None:
            variants.append((costs, scaled, f"common_scale_2^{exponent}"))
        for ordered_costs, ordered_weights, label in variants:
            exact = exact_mean(ordered_costs, ordered_weights)
            wanted = float(exact) if exact is not None else None
            actual, error = invoke(args.binary, ordered_costs, ordered_weights)
            if exact is None:
                if actual is not None or error != "INVALID_INPUT":
                    raise AssertionError(
                        {
                            "case": index,
                            "variant": label,
                            "actual": actual,
                            "error": error,
                            "all_frequencies_zero": True,
                        }
                    )
            elif exact == 0:
                if actual != 0.0 or error is not None:
                    raise AssertionError(
                        {
                            "case": index,
                            "variant": label,
                            "actual": actual,
                            "error": error,
                            "expected": wanted.hex(),
                        }
                    )
            elif wanted == 0.0:
                if actual is not None or error != "INVALID_INPUT":
                    raise AssertionError(
                        {
                            "case": index,
                            "variant": label,
                            "actual": actual,
                            "error": error,
                            "positive_exact_rounds_to_zero": True,
                        }
                    )
            elif error is not None or actual != wanted:
                raise AssertionError(
                    {
                        "case": index,
                        "variant": label,
                        "actual": actual.hex() if actual is not None else None,
                        "expected_exact_round": wanted.hex()
                        if wanted is not None
                        else None,
                        "costs": ordered_costs,
                        "frequencies": ordered_weights,
                    }
                )
            checks.append(
                {
                    "case": index,
                    "variant": label,
                    "n": len(ordered_weights),
                    "actual": actual.hex()
                    if actual is not None
                    else "explicit_invalid_input",
                    "expected_exact_round": wanted.hex()
                    if wanted is not None
                    else None,
                    "exact_fraction_positive": exact > 0 if exact is not None else None,
                    "error": error,
                    "ulp_tolerance": 0,
                }
            )
    report = {
        "status": "PASS",
        "oracle": "Python Fraction over exact binary64 dyadics",
        "base_cases": len(base_cases),
        "evaluations": len(checks),
        "seed": 16009,
        "random_positive_binary64_seed": 16009,
        "maximum_demands": max(len(w) for _, w in base_cases),
        "tolerance": "zero ULP for every returned score; positive exact results rounding to zero must be INVALID_INPUT",
        "scope": "raw nonnegative frequencies (at least one positive for nonempty demands) and finite nonnegative costs across the full binary64 exponent range, permutations, exact power-of-two common rescalings",
        "gpu_claim": None,
        "checks": checks,
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    args.out.write_text(text + "\n")
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "base_cases",
                    "evaluations",
                    "maximum_demands",
                    "random_positive_binary64_seed",
                    "tolerance",
                    "scope",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
