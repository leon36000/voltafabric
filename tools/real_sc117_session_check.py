#!/usr/bin/env python3
"""One-shot local SC117 CPU session diagnostic; no replay, GPU or network worker."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import uuid

INDEX_SHA = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "index", "bridge", "worker", "oracle", "work-root", "digest-0", "digest-1"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    for path in (args.pack, args.index, args.bridge, args.worker, args.oracle, args.work_root):
        if not os.path.isabs(path):
            ap.error("all paths must be absolute")
    digests = [args.digest_0, args.digest_1]
    if any(len(d) != 64 or any(c not in "0123456789abcdef" for c in d) for d in digests):
        ap.error("digests must be lowercase SHA-256 values supplied by the operator")
    work = Path(tempfile.mkdtemp(prefix="real-session-", dir=args.work_root))
    ledger = work / "ledger"
    ledger.mkdir()
    references = []
    for expert in (0, 1):
        ref = work / f"reference-{expert}.bin"
        command = [args.oracle, args.pack, str(expert), str(ref)]
        subprocess.run(command, check=True, timeout=30, stdout=subprocess.DEVNULL)
        raw = ref.read_bytes()
        if len(raw) != 2560 * (4 + 8):
            raise ValueError("invalid oracle output length")
        references.append((struct.unpack("<2560f", raw[:10240]), struct.unpack("<2560d", raw[10240:])))
    if references[0][0] != references[1][0]:
        raise ValueError("oracle inputs differ")
    run_id = "sc117-real-cpu-" + uuid.uuid4().hex
    stamp = {"run_id": run_id, "model_revision": "sc117-local-observation-r399",
             "plan_id": "two-expert-local-cpu", "layer": 0, "batch": 0, "epoch": 1}
    replicas = [{"expert": e, "rank": 2 + e, "local": 0} for e in (0, 1)]
    bindings = [{"model_revision": stamp["model_revision"], "rank": 2 + e, "layer": 0,
                 "expert": e, "local": 0, "epoch": 1, "index_sha256": INDEX_SHA,
                 "blob_sha256": digests[e]} for e in (0, 1)]
    init = {"template": {"hidden": 2560, "experts": [0, 1],
                         "routes": [[{"expert": 0, "weight": 0.5}, {"expert": 1, "weight": 0.5}]],
                         "registry": {"epoch": 1, "copies": replicas}, "stamp": stamp,
                         "preferred_owners": {}, "resident_order": {"2": [0], "3": [0]}},
            "model_experts": [0, 1], "index": Path(args.index).read_text(), "pack_path": args.pack,
            "ledger_path": str(ledger), "endpoints": [{"rank": 2 + e, "argv": [args.worker]} for e in (0, 1)],
            "bindings": bindings, "resident_payload_limit": 5_000_000, "timeout_ms": 30_000}
    batch = {"op": "batch", "batch": 1, "tokens": 1, "top_k": 2,
             "inputs": references[0][0], "ids": [0, 1], "kinds": [-1, -1]}
    payload = "".join(json.dumps(value, allow_nan=False) + "\n" for value in (init, batch, {"op": "shutdown"}))
    child = subprocess.Popen([args.bridge], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = child.communicate(payload, timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.communicate()
        raise RuntimeError("session timed out; outcome unknown and must not be replayed")
    if child.returncode != 0:
        raise RuntimeError("bridge failed; do not replay: " + stderr[-1000:])
    replies = [json.loads(line) for line in stdout.splitlines()]
    if len(replies) != 3 or replies[0].get("state") != "ready" or replies[2].get("state") != "closed":
        raise ValueError("incomplete or unexpected bridge conversation")
    completed = replies[1]
    if completed.get("state") != "completed" or completed.get("batch") != 1:
        raise ValueError("batch not complete")
    rows = completed.get("cpu_rows", [])
    if len(rows) != 2 or [r.get("position") for r in rows] != [0, 1]:
        raise ValueError("unexpected row coverage")
    max_abs = 0.0
    for expert, row in enumerate(rows):
        values = row.get("value", [])
        if len(values) != 2560:
            raise ValueError("unexpected output width")
        for got, want in zip(values, references[expert][1]):
            if not math.isfinite(got) or not math.isfinite(want):
                raise ValueError("nonfinite result")
            error = abs(got - want)
            max_abs = max(max_abs, error)
            if error > 1e-7 + 1e-3 * abs(want):
                raise ValueError("session result differs from scalar reference")
    print(json.dumps({"status": "ok", "run_id": run_id, "experts": [0, 1],
                      "rows": len(rows), "columns": 2560, "max_abs_error": max_abs,
                      "work_dir": str(work), "scope": "local CPU, unweighted FP32"}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print("diagnostic failed: " + str(error), file=sys.stderr)
        sys.exit(1)
