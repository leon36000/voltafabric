Warning: truncated output (original token count: 3065)
Total output lines: 283

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import socket
import struct
import subprocess
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "adapters" / "strata_native_bridge"))
import voltafabric_launch as launcher  # noqa: E402

LAYERS = 48
HIDDEN = 2560
PACK_BYTES = 50_292_326_400
EXPERT = 8
EPOCH = 80
FIRST_BATCH = 9_651_021
FIRST_ATTEMPT = 51
RUN_ID = "r651-cpp-48-session-fleet-v3"
MODEL_REVISION = "r651-synthetic-sc117-v3"
PLAN_ID = "r651-synthetic-cpu-plan-v3"
RECEIPT_ROOT = ROOT / "research" / "R651" / "receipts"
LEDGER_ROOT = ROOT / ".runs" / "r651-v3-local-ledgers"
GUARD = RECEIPT_ROOT / "RUN_GUARD_V3.json"


def _write_json_exclusive(path: pathlib.Path, value: dict) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _receipt_snapshot() -> list[dict]:
    result: list[dict] = []
    for layer in range(LAYERS):
        folder = LEDGER_ROOT / f"layer-{layer:02d}"
        for path in sorted(folder.glob("*.claim.json")):
            result.append(json.loads(path.read_text(encoding="utf-8")))
    return sorted(result, key=lambda claim: (
        claim["stamp"]["layer"], claim["stamp"]["batch"], claim["attempt"]
    ))


def _capture(status: str, detail: str) -> None:
    claims = _receipt_snapshot()
    (RECEIPT_ROOT / "local-fleet-claims-v3.json").write_text(
        json.dumps({"claims": claims, "count": len(claims)}, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    GUARD.write_text(
        json.dumps({
            "run_id": RUN_ID,
            "status": status,
            "detail": detail,
            "claims_captured": len(claims),
            "replay": "forbidden",
        }, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _make_synthetic_payload(gate_type: int, down_type: int, span_bytes: int) -> bytes:
    # Reuse the already-qualified R650 IQ3_XXS/IQ4_NL vector for every layer
    # with that layout; other supported layouts use zero-code payloads.
    if (gate_type, down_type, span_bytes) != (18, 20, 2_176_000):
        return bytes(span_bytes)
    payload = bytearray(span_bytes)
    gate_up_bytes = (HIDDEN // 256 * 98) * 640 * 2
    for offset in range(0, gate_up_bytes, 98):
        payload[offset + 1] = 0x28
    for offset in range(gate_up_bytes, len(payload), 18):
        payload[offset + 1] = 0x28
    return bytes(payload)


def main() -> int:
    bridge_path = pathlib.Path(os.environ["VF_R651_BRIDGE"]).resolve(strict=True)
    rank_path = pathlib.Path(os.environ["VF_R651_RANK_SESSION"]).resolve(strict=True)
    callback_path = pathlib.Path(os.environ["VF_R651_CPP_CALLBACK"]).resolve(strict=True)
    temp_root = pathlib.Path(os.environ["VF_R651_TMP_ROOT"]).resolve(strict=True)
    if not all(path.is_file() and os.access(path, os.X_OK)
               for path in (bridge_path, rank_path, callback_path)):
        raise RuntimeError("all three native executables must be regular executable files")
    if not temp_root.is_dir() or not RECEIPT_ROOT.is_dir():
        raise RuntimeError("temporary and receipt roots must already exist")
    if GUARD.exists():
        raise RuntimeError("this identified run was already started; never replay it")
    if LEDGER_ROOT.exists() and any(LEDGER_ROOT.iterdir()):
        raise RuntimeError("local ledger is not empty; never reuse it")
    LEDGER_ROOT.mkdir(parents=True, exist_ok=True)
    ledgers = []
    for layer in range(LAYERS):
        ledger = LEDGER_ROOT / f"layer-{layer:02d}"
        ledger.mkdir()
        ledgers.append(ledger)

    index_path = ROOT / "fixtures" / "strata_sc117_native_experts_v3.txt"
    index = index_path.read_text(encoding="utf-8")
    entries: dict[int, tuple[int, int, int, int]] = {}
    for line in index.splitlines():
        if not line or not line[0].isdigit():
            continue
        fields = line.split()
        if len(fields) < 5:
            raise RuntimeError("native index has a malformed layer entry")
        layer, gate_type, down_type, offset, span_bytes = map(int, fields[:5])
        if layer in entries or not 0 <= layer < LAYERS:
            raise RuntimeError("native index duplicates or exceeds the 48-layer layout")
        entries[layer] = (gate_type, down_type, offset, span_bytes)
    if sorted(entries) != list(range(LAYERS)):
        raise RuntimeError("native index no longer describes exactly 48 ordered layers")
    index_sha = hashlib.sha256(index.encode()).hexdigest()

    identity = {
        "run_id": RUN_ID,
        "model_revision": MODEL_REVISION,
        "plan_id": PLAN_ID,
        "epoch": EPOCH,
    }
    sessions: list[launcher.BridgeSession] = []
    config: dict = {}
    cpp_started = False
    cpp_finished = False
    graceful = False

    try:
        with tempfile.TemporaryDirectory(prefix="r651-sparse-pack-", dir=temp_root) as td:
            pack = pathlib.Path(td) / "synthetic-sc117.gguf"
            blobs: dict[int, bytes] = {}
            with pack.open("w+b") as stream:
                stream.truncate(PACK_BYTES)
                for layer in range(LAYERS):
                    gate_type, down_type, offset, span_bytes = entries[layer]
                    if gate_type not in (18, 21, 22, 23) or down_type not in (20, 42):
                        raise RuntimeError("native index contains an unsupported GGML layout")
                    blob = _make_synthetic_payload(gate_type, down_type, span_bytes)
                    blobs[layer] = blob
                    # Keep a sparse 50.3 GB logical pack with bounded physical
                    # allocation. Only the selected expert span is materialized.
                    stream.seek(offset + EXPERT * span_bytes)
              …65 tokens truncated…024:
                    raise RuntimeError("filesystem allocation exceeded the 128 MiB fixture cap")

            sessions_cfg = []
            for layer in range(LAYERS):
                gate_type, down_type, offset, span_bytes = entries[layer]
                binding = {
                    "model_revision": MODEL_REVISION,
                    "rank": 0,
                    "layer": layer,
                    "expert": EXPERT,
                    "local": 0,
                    "epoch": EPOCH,
                    "index_sha256": index_sha,
                    "blob_sha256": hashlib.sha256(blobs[layer]).hexdigest(),
                }
                stamp = {**identity, "layer": layer, "batch": 0}
                template = {
                    "hidden": HIDDEN,
                    "experts": [EXPERT],
                    "routes": [[{"expert": EXPERT, "weight": 0.25}]],
                    "registry": {"epoch": EPOCH, "copies": [{"expert": EXPERT, "rank": 0, "local": 0}]},
                    "stamp": stamp,
                    "preferred_owners": {},
                    "resident_order": {"0": [0]},
                }
                init = {
                    "template": template,
                    "model_experts": [EXPERT],
                    "index": index,
                    "pack_path": str(pack),
                    "ledger_path": str(ledgers[layer]),
                    "endpoints": [{"rank": 0, "argv": [str(rank_path)]}],
                    "bindings": [binding],
                    "resident_payload_limit": span_bytes,
                    "timeout_ms": 30_000,
                }
                sessions_cfg.append({"layer": layer, "init": init, "expected_ranks": [0]})

            config = {
                "bridge_executable": str(bridge_path),
                "working_directory": str(temp_root),
                "startup_timeout_ms": 300_000,
                "shutdown_timeout_ms": 30_000,
                "epoch": EPOCH,
                "sessions": sessions_cfg,
            }
            # The fixed identity is permanently consumed from this point even
            # if startup, a request, or shutdown later becomes uncertain.
            _write_json_exclusive(GUARD, {
                **identity,
                "first_batch": FIRST_BATCH,
                "first_attempt": FIRST_ATTEMPT,
                "status": "RUN_STARTED_DO_NOT_REPLAY",
                "replay": "forbidden",
            })
            deadline = time.monotonic() + 300.0
            for session in sessions_cfg:
                sessions.append(launcher._start_bridge(config, session, deadline))
            fds = [session.channel.fileno() for session in sessions]
            if len(set(fds)) != LAYERS or any(fd < 3 for fd in fds):
                raise RuntimeError("launcher did not produce 48 distinct inherited sockets")

            args = [
                str(callback_path), RUN_ID, MODEL_REVISION, PLAN_ID, str(EPOCH),
                str(FIRST_BATCH), str(FIRST_ATTEMPT), *(str(fd) for fd in fds),
            ]
            cpp_started = True
            completed = subprocess.run(args, cwd=temp_root, pass_fds=tuple(fds),
                                       capture_output=True, text=True, timeout=300, check=False)
            cpp_finished = True
            if completed.returncode != 0:
                raise RuntimeError(
                    f"C++ fleet callback exited {completed.returncode}: {completed.stderr[-2000:]}"
                )
            if completed.stdout.strip() != (
                "R651_CPP_POOL_PASS layers=48 batches=2 callbacks=96 "
                "terminal_refusal=PASS shutdown=PASS"
            ):
                raise RuntimeError("unexpected C++ fleet result")
            # PoolFleetOwner sent the VFB3 terminal request to every Rust
            # coordinator. Match launch(): close inherited streams and wait;
            # do not send a second coordinator shutdown frame.
            graceful = False
        if not launcher._finish_sessions(sessions, config, graceful=graceful):
            raise RuntimeError("coordinator cleanup failed; do not replay this run")
        sessions.clear()

        claims = _receipt_snapshot()
        expected = []
        for layer in range(LAYERS):
            expected.extend([
                (layer, FIRST_BATCH, FIRST_ATTEMPT),
                (layer, FIRST_BATCH + 1, FIRST_ATTEMPT + 1),
            ])
        actual = sorted(
            (c["stamp"]["layer"], c["stamp"]["batch"], c["attempt"]) for c in claims
        )
        if actual != expected or len(claims) != LAYERS * 2:
            raise RuntimeError("the 96 durable layer/batch claims did not match the requested identities")
        if any(c["status"] != "RESERVED_UNKNOWN_NO_REPLAY" for c in claims):
            raise RuntimeError("a local claim has an unexpected state")
        _capture("CONSUMED_DO_NOT_REPLAY", "96 C++ callbacks and graceful shutdown verified")
        print(json.dumps({
            "result": "PASS",
            "bridge_sessions": LAYERS,
            "rank_sessions": LAYERS,
            "pool_owner": "real C++ PoolFleetOwner::create_from_fds",
            "callbacks": len(claims),
            "batches_per_layer": 2,
            "synthetic_pack_logical_bytes": PACK_BYTES,
            "synthetic_pack_physical_bytes_capped": True,
            "failure_latching": "invalid layer refused without partial write",
            "shutdown": "graceful C++ workers and Python coordinators",
            "claims": len(claims),
        }, sort_keys=True))
        return 0
    except Exception:
        if sessions:
            launcher._finish_sessions(sessions, config, graceful=False)
            sessions.clear()
        if GUARD.exists():
            state = "UNKNOWN_NO_REPLAY" if cpp_started and not cpp_finished else "FAILED_NO_REPLAY"
            _capture(state, "failure after the identified run was started")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
