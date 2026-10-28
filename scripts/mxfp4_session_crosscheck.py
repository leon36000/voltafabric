#!/usr/bin/env python3
"""Real persistent MXFP4 sessions versus cold sessions; exact independent CPU bits."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "fixtures/preweighted_mxfp4_tiny_golden.json"
ARTIFACT = ROOT / "fixtures/preweighted_mxfp4_tiny.safetensors"
SPEC = importlib.util.spec_from_file_location(
    "fp32", ROOT / "tests/reference/fp32_add_oracle.py"
)
FP32 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FP32)


def fixture(binary, workers=None):
    golden = json.loads(GOLDEN.read_text())
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == golden["weights_sha256"]
    maps = {2: [(241, 9), (17, 7), (93, 1)], 9: [(93, 30), (17, 70), (241, 3)]}
    sources = []
    for rank, mapping in maps.items():
        argv = (workers or {}).get(str(rank), [str(binary.parent / "vf-rank-session")])
        sources.append(
            dict(
                endpoint=dict(rank=rank, argv=argv),
                prefer_bf16_activations=True,
                artifact_path=str(ARTIFACT),
                binding=dict(
                    artifact_sha256=golden["weights_sha256"],
                    model_revision="r12-tiny-torch",
                    rank=rank,
                    layer=0,
                    epoch=31,
                    experts=[dict(expert=e, local=local_id) for e, local_id in mapping],
                ),
            )
        )
    pattern_names = ["pattern", "pattern_half", "pattern_reversed"]
    routes = [
        [(17, 0.375), (93, 0.5)],
        [(241, 1.0), (17, 0.375)],
        [(93, 0.0), (241, 2.0)],
    ]
    cases = {(c["input"], c["expert"], c["coefficient"]): c for c in golden["cases"]}
    batches = []
    expected = []
    for batch in range(3):
        names = pattern_names[batch:] + pattern_names[:batch]
        problem = dict(
            hidden=32,
            experts=[17, 93, 241],
            routes=[[dict(expert=e, weight=w) for e, w in row] for row in routes],
            registry=dict(
                epoch=31,
                copies=[
                    dict(expert=e, rank=r, local=local_id)
                    for r, mapping in maps.items()
                    for e, local_id in mapping
                ],
            ),
            stamp=dict(
                run_id="r12-mxfp4-crosscheck",
                model_revision="r12-tiny-torch",
                plan_id="selected-replicas-two-ranks",
                layer=0,
                batch=batch,
                epoch=31,
            ),
            preferred_owners={"17": 9, "93": 2, "241": 9},
            resident_order={
                str(r): [local_id for _, local_id in mapping]
                for r, mapping in maps.items()
            },
        )
        inputs = [
            dict(
                token=t,
                value_bits=[
                    int(x, 16) for x in golden["inputs"][name]["bf16_as_f32_bits"]
                ],
            )
            for t, name in enumerate(names)
        ]
        batches.append(dict(problem=problem, inputs=inputs))
        output = []
        for token, name in enumerate(names):
            row = []
            for col in range(32):
                terms = [
                    cases[name, e, w]["output_bits"][col] for e, w in routes[token]
                ]
                status, bits = FP32.accumulate(terms)
                assert status == "ok"
                row.append(int(bits, 16))
            output.append(row)
        expected.append(output)
    return dict(workers=sources, batches=batches, timeout_ms=10000), expected


def execute(binary, payload, expected, label, out=None):
    argv = [str(binary), "session-run-mxfp4", "-"]
    data = json.dumps(payload)
    start = time.perf_counter()
    p = subprocess.run(argv, input=data, capture_output=True, text=True, timeout=120)
    elapsed = time.perf_counter() - start
    if out:
        for suffix, content in [
            ("input.json", data),
            ("stdout.json", p.stdout),
            ("stderr.log", p.stderr),
        ]:
            with (out.parent / (out.stem + "." + label + "." + suffix)).open("x") as f:
                f.write(content)
    assert p.returncode == 0, (label, p.returncode, p.stdout, p.stderr)
    envelope = json.loads(p.stdout)
    assert envelope["ok"], envelope
    result = envelope["result"]
    assert result["ok"] and result["error"] is None, result
    assert [b["output_bits"] for b in result["batches"]] == expected
    assert all(c["exit_code"] == 0 and c["error"] is None for c in result["closes"])
    assert len(result["loads"]) == 2 and all(
        load["transferred"] for load in result["loads"]
    )
    assert all(
        t[1]["loads_transferred"] == 1 and t[1]["execs"] == len(expected)
        for t in result["data_phase_traffic"]
    )
    counters = dict(
        weight_bytes=sum(
            t[1]["weight_bytes_sent"] for t in result["data_phase_traffic"]
        ),
        bytes_sent=sum(t[1]["bytes_sent"] for t in result["data_phase_traffic"]),
        bytes_received=sum(
            t[1]["bytes_received"] for t in result["data_phase_traffic"]
        ),
    )
    return dict(
        argv=argv,
        exit_code=p.returncode,
        seconds=elapsed,
        **counters,
        peak_worker_rss_kib=[c["peak_rss_kib"] for c in result["closes"]],
        output_sha256=hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
    )


def run(binary, workers=None, out=None):
    payload, expected = fixture(binary, workers)
    persistent = execute(binary, payload, expected, "persistent", out)
    cold = []
    for index, batch in enumerate(payload["batches"]):
        single = copy.deepcopy(payload)
        single["batches"] = [copy.deepcopy(batch)]
        single["batches"][0]["problem"]["stamp"]["run_id"] += f"-cold-{index}"
        cold.append(execute(binary, single, [expected[index]], f"cold{index}", out))
    cold_weights = sum(x["weight_bytes"] for x in cold)
    assert cold_weights == 3 * persistent["weight_bytes"]
    reversed_payload = copy.deepcopy(payload)
    for batch in reversed_payload["batches"]:
        batch["problem"]["stamp"]["run_id"] += "-reverse"
        batch["faults"] = {"2": {"kind": "reverse"}, "9": {"kind": "reverse"}}
    reversed_receipts = execute(binary, reversed_payload, expected, "reversed", out)
    return dict(
        status="PASS",
        scope="CPU persistent sessions, not GPU or service throughput",
        binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
        worker_argv=[w["endpoint"] for w in payload["workers"]],
        exact_output_values=3 * 3 * 32,
        oracle="independent Torch + rational FP32 slot sum",
        persistent=persistent,
        reversed=reversed_receipts,
        cold=cold,
        avoided_weight_request_bytes=cold_weights - persistent["weight_bytes"],
        avoided_compressed_payload_bytes=4 * ARTIFACT.stat().st_size,
        gpu_executed=False,
        rdma_tested=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--worker-map", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    workers = json.loads(args.worker_map.read_text()) if args.worker_map else None
    report = run(args.binary.resolve(strict=True), workers, args.out)
    if args.out:
        with args.out.open("x") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report))
