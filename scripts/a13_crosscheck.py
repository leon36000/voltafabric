#!/usr/bin/env python3
"""Independent A13 CPU regressions. Exact small PP enumeration; no GPU or speed claim."""

import argparse
import itertools
import json
import math
from pathlib import Path
import random
import subprocess

import torch


def call(binary, command, payload):
    run = subprocess.run(
        [str(binary), command, "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(run.stdout)
    assert (run.returncode == 0) == bool(data["ok"]), (command, data)
    return data


def pp_oracle(problem):
    layers, stages = problem["layers"], problem["stages"]
    candidates = []
    for cuts in itertools.combinations(range(1, len(layers)), len(stages) - 1):
        ends = (0, *cuts, len(layers))
        values = []
        for stage, (start, end) in enumerate(zip(ends, ends[1:])):
            if (
                sum(l["bytes"] for l in layers[start:end])
                > stages[stage]["available_bytes"]
            ):
                break
            # Arbitrary precision integer sums, independent of Rust f64 accumulation.
            values.append(
                sum(l["compute_us_by_stage"][stage] for l in layers[start:end])
            )
        else:
            costs = [v + s["boundary_after_us"] for v, s in zip(values, stages)]
            candidates.append((max(costs), ends, values, sum(costs)))
    return candidates


def check_pp(binary, p):
    choices = pp_oracle(p)
    r = call(binary, "pipeline", p)
    if not choices:
        assert not r["ok"] and r["error"]["code"] == "PIPELINE_INFEASIBLE", r
        return
    assert r["ok"], r
    result = r["result"]
    ends = tuple([s["first_layer"] for s in result["segments"]] + [len(p["layers"])])
    chosen = next(c for c in choices if c[1] == ends)
    assert chosen[0] == min(c[0] for c in choices), (p, result, choices)
    assert result["bottleneck_proxy_us"] == chosen[0], (p, result, chosen)
    assert result["serial_latency_us"] == chosen[3], (p, result, chosen)
    assert [s["compute_us"] for s in result["segments"]] == chosen[2], (
        p,
        result,
        chosen,
    )
    assert [s["rank"] for s in result["segments"]] == [s["rank"] for s in p["stages"]]


def collision(first, group, policy):
    single = 2
    links = [
        dict(
            id=f"{a}-{b}",
            src=a,
            dst=b,
            bandwidth_bytes_per_s=1e6,
            latency_us=1,
            enabled=True,
            transport="synthetic",
            resources=[],
        )
        for a, b in [(0, group), (group, 0), (0, single), (single, 0)]
    ]
    return dict(
        topology=dict(
            schema_version=1,
            origin="synthetic",
            source_note="A13 CPU oracle",
            devices=[
                dict(rank=r, host="synthetic", capacity_bytes=c, reserved_bytes=0)
                for r, c in [(0, 1), (group, 20), (single, 10)]
            ],
            links=links,
        ),
        experts=[
            dict(id=e, resident_bytes=10, compute_us=1) for e in range(first, first + 3)
        ],
        demands=[
            dict(source_rank=0, experts=list(range(first, first + 3)), frequency=1)
        ],
        hidden=64,
        wire=dict(
            input_bytes=2,
            expert_output_bytes=2,
            partial_output_bytes=4,
            header_bytes=0,
            slot_metadata_bytes=0,
            alignment=1,
        ),
        max_iterations=0,
        max_work_units=1000000,
        wire_policy=policy,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    counts = dict(pipeline=0, placement=0, unsigned_e4m3=0)
    p = dict(
        layers=[
            dict(bytes=1, compute_us_by_stage=c)
            for c in [[1, 10**20], [1, 10], [1, 20]]
        ],
        stages=[dict(rank=i, available_bytes=3, boundary_after_us=0) for i in range(2)],
    )
    check_pp(args.binary, p)
    counts["pipeline"] += 1
    rng = random.Random(131313)
    for _ in range(192):
        n = rng.randrange(2, 8)
        m = rng.randrange(1, min(n, 3) + 1)
        layers = [
            dict(
                bytes=rng.randrange(1, 8),
                compute_us_by_stage=[
                    10**20 if i < stage else rng.randrange(0, 65) for stage in range(m)
                ],
            )
            for i in range(n)
        ]
        total = sum(l["bytes"] for l in layers)
        p = dict(
            layers=layers,
            stages=[
                dict(
                    rank=stage * 13 + 2,
                    available_bytes=rng.randrange(1, total + 1),
                    boundary_after_us=rng.randrange(0, 5) if stage + 1 < m else 0,
                )
                for stage in range(m)
            ],
        )
        check_pp(args.binary, p)
        counts["pipeline"] += 1
    for first in (1, 10, 100, 4294967293):
        for policy in ("baseline", "grouped", "adaptive"):
            outputs = []
            for group in (first, 17):
                r = call(args.binary, "plan", collision(first, group, policy))
                assert r["ok"], r
                result = r["result"]
                assert result["owners"] == {
                    str(first): 2,
                    str(first + 1): group,
                    str(first + 2): group,
                }
                assert result["work_units_used"] <= 1000000
                outputs.append(result)
                counts["placement"] += 1
            for field in (
                "modeled_weighted_latency_us",
                "initial_latency_us",
                "demand_comparisons",
            ):
                assert outputs[0][field] == outputs[1][field], (
                    first,
                    policy,
                    field,
                    outputs,
                )
    for byte in range(256):
        value = (
            torch.tensor([byte], dtype=torch.uint8)
            .view(torch.float8_e4m3fn)
            .float()
            .item()
        )
        expected_valid = math.isfinite(value) and math.copysign(1, value) > 0
        p = dict(
            rows=32,
            columns=64,
            codes=[0x22] * 1024,
            scales=[byte] * 128,
            global_dequant=1,
        )
        r = call(args.binary, "dequant", p)
        assert r["ok"] == expected_valid, (byte, value, r)
        if expected_valid:
            assert all(v == value for row in r["result"] for v in row), (byte, value)
        counts["unsigned_e4m3"] += 1
    report = dict(
        status="PASS",
        cases=counts,
        total=sum(counts.values()),
        pp_reference="enumeration and arbitrary precision integer sums",
        scale_reference="Torch F8_E4M3 with independent signbit/finite classification",
        placement_reference="identity-renaming controls with forced owners",
        cuda_executed=False,
        performance_claim=None,
    )
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
