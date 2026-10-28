#!/usr/bin/env python3
"""Independent closed-form CPU checks for two resident remote experts; analytical only."""
from __future__ import annotations
import argparse
from pathlib import Path
import copy
import json
import random
import subprocess

ATOL = 2e-9  # Decimal transfer costs and f64 roundoff; never a GPU/quantization tolerance.

def invoke(binary: Path, payload: dict) -> dict:
    run = subprocess.run([str(binary), "plan", "-"], input=json.dumps(payload),
                         text=True, capture_output=True, timeout=10)
    result = json.loads(run.stdout)
    if run.returncode or not result["ok"]:
        raise AssertionError((run.returncode, result))
    return result["result"]

def payload(hidden: int, compute: float, forward_latency: float, reverse_latency: float,
            forward_bw: float, reverse_bw: float, shared: bool, input_bytes: int,
            output_bytes: int, partial_bytes: int) -> dict:
    provenance = {"origin": "synthetic", "source": "closed-form adversarial CPU fixture"}
    return {
        "topology": {
            "schema_version": 1, "origin": "synthetic",
            "source_note": "Closed-form assumptions, not hardware measurements",
            "devices": [{"rank": 0, "host": "synthetic", "capacity_bytes": 1, "reserved_bytes": 0},
                        {"rank": 1, "host": "synthetic", "capacity_bytes": 4, "reserved_bytes": 0}],
            "links": [
                {"id": "forward", "src": 0, "dst": 1, "bandwidth_bytes_per_s": forward_bw,
                 "latency_us": forward_latency, "enabled": True, "transport": "synthetic",
                 "resources": ["bus"] if shared else [], "cost_provenance": provenance},
                {"id": "reverse", "src": 1, "dst": 0, "bandwidth_bytes_per_s": reverse_bw,
                 "latency_us": reverse_latency, "enabled": True, "transport": "synthetic",
                 "resources": ["bus"] if shared else [], "cost_provenance": provenance}],
        },
        "experts": [{"id": expert, "resident_bytes": 2, "compute_us": compute,
                     "cost_provenance": provenance} for expert in (10, 99)],
        "demands": [{"source_rank": 0, "experts": [10, 99], "frequency": 1.0}],
        "hidden": hidden,
        "wire": {"input_bytes": input_bytes, "expert_output_bytes": output_bytes,
                 "partial_output_bytes": partial_bytes, "header_bytes": 0,
                 "slot_metadata_bytes": 0, "alignment": 1},
        "max_iterations": 0,
    }

def expected(p: dict) -> tuple[float, float]:
    h = p["hidden"]; wire = p["wire"]; forward, reverse = p["topology"]["links"]
    a = forward["latency_us"] + h * wire["input_bytes"] / forward["bandwidth_bytes_per_s"] * 1e6
    b = reverse["latency_us"] + h * wire["expert_output_bytes"] / reverse["bandwidth_bytes_per_s"] * 1e6
    bg = reverse["latency_us"] + h * wire["partial_output_bytes"] / reverse["bandwidth_bytes_per_s"] * 1e6
    c = p["experts"][0]["compute_us"]
    # Each direction serializes its messages. Both expert calculations share one GPU.
    second_compute_start = max(a + c, 2 * a)
    if forward["resources"]:
        baseline = second_compute_start + max(c, b) + b
    else:
        baseline = max(second_compute_start + c, a + c + b) + b
    return baseline, a + 2 * c + bg

def assert_close(actual: float, wanted: float) -> float:
    difference = abs(actual - wanted)
    if difference > ATOL:
        raise AssertionError((actual, wanted, difference, ATOL))
    return difference

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cases = [
        payload(4096, 100., 0., 0., 1e9, 1e9, False, 2, 2, 4),
        payload(4096, 100., 100., 100., 1e9, 1e9, True, 2, 2, 4),
    ]
    rng = random.Random(90210)
    for _ in range(64):
        cases.append(payload(rng.choice([1, 17, 4096]), rng.choice([0., 3., 100., 1000.]),
                             rng.choice([0., 2., 100.]), rng.choice([0., 7., 200.]),
                             rng.choice([1e7, 1e9, 1e10]), rng.choice([1e7, 1e9, 1e10]),
                             rng.choice([False, True]), rng.choice([2, 4]),
                             rng.choice([2, 4]), rng.choice([2, 4, 8])))
    maximum = 0.
    count = 0
    for p in cases:
        baseline, grouped = expected(p)
        for policy, wanted in [("baseline", baseline), ("grouped", grouped),
                               ("adaptive", min(baseline, grouped))]:
            candidate = copy.deepcopy(p); candidate["wire_policy"] = policy
            result = invoke(args.binary, candidate)
            maximum = max(maximum, assert_close(result["modeled_weighted_latency_us"], wanted))
            comparison = result["demand_comparisons"][0]
            assert_close(comparison["baseline"]["modeled_latency_us"], baseline)
            assert_close(comparison["grouped"]["modeled_latency_us"], grouped)
            assert result["result_kind"] == "analytical_resident_placement_not_gpu_measurement"
            count += 1
    try:
        assert_close(cases[0]["experts"][0]["compute_us"] + 1., cases[0]["experts"][0]["compute_us"])
    except AssertionError:
        sensitivity = "detected deliberate scalar corruption"
    else:
        raise AssertionError("oracle comparison is insensitive")
    report = {"status": "PASS", "evidence_level": "CPU_TESTED_ANALYTICAL",
              "cases": count, "seed": 90210, "max_abs_error": maximum, "atol": ATOL,
              "scope": "two equal-cost remote experts; asymmetric directed transfers; optional shared bus",
              "sensitivity_control": sensitivity, "gpu_claim": None}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
