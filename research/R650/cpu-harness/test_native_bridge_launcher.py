from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import socket
import struct
import sys
import tempfile
import time

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "adapters" / "strata_native_bridge"))
import voltafabric_launch as launcher  # noqa: E402

HIDDEN = 2560
PACK_BYTES = 50_292_326_400
EXPERT = 8
EPOCH = 77
BATCH = 9_650_001
ATTEMPT = 30
RUN_ID = "r650-actual-bridge-launcher"
MODEL_REVISION = "r650-synthetic-sc117"
PLAN_ID = "r650-synthetic-plan"


def read_exact(channel: socket.socket, size: int, deadline: float) -> bytes:
    data = bytearray()
    while len(data) < size:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("VFR3 response deadline")
        channel.settimeout(remaining)
        chunk = channel.recv(size - len(data))
        if not chunk:
            raise RuntimeError("coordinator closed before complete VFR3")
        data.extend(chunk)
    return bytes(data)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def main() -> int:
    bridge_path = pathlib.Path(os.environ["VF_R650_BRIDGE"]).resolve(strict=True)
    rank_path = pathlib.Path(os.environ["VF_R650_RANK_SESSION"]).resolve(strict=True)
    temp_root = pathlib.Path(os.environ["VF_R650_TMP_ROOT"]).resolve(strict=True)
    ledger = pathlib.Path(os.environ["VF_R650_LEDGER_DIR"]).resolve(strict=True)
    if not bridge_path.is_file() or not os.access(bridge_path, os.X_OK):
        raise RuntimeError("bridge must be a regular executable")
    if not rank_path.is_file() or not os.access(rank_path, os.X_OK):
        raise RuntimeError("rank session must be a regular executable")
    if not temp_root.is_dir() or not ledger.is_dir() or next(ledger.iterdir(), None) is not None:
        raise RuntimeError("temporary root/empty fresh ledger required")

    index_path = ROOT / "fixtures" / "strata_sc117_native_experts_v3.txt"
    index = index_path.read_text(encoding="utf-8")
    line = next(line for line in index.splitlines() if line and line[0].isdigit())
    layer, gate_type, down_type, layer_offset, span_bytes = map(int, line.split()[:5])
    if layer != 0:
        raise RuntimeError("synthetic fixture must start at layer zero")
    span_offset = layer_offset + EXPERT * span_bytes
    gate_up_bytes = (HIDDEN // 256 * 98) * 640 * 2
    if gate_type != 18 or down_type != 20 or span_bytes != 2_176_000:
        raise RuntimeError("fixture expert geometry changed; refuse stale synthetic builder")

    payload = bytearray(span_bytes)
    for offset in range(0, gate_up_bytes, 98):
        payload[offset + 1] = 0x28
    for offset in range(gate_up_bytes, len(payload), 18):
        payload[offset + 1] = 0x28
    blob_sha = hashlib.sha256(payload).hexdigest()
    index_sha = hashlib.sha256(index.encode("utf-8")).hexdigest()

    with tempfile.TemporaryDirectory(prefix="r650-sparse-pack-", dir=temp_root) as td:
        pack = pathlib.Path(td) / "synthetic-sparse.gguf"
        with pack.open("w+b") as stream:
            stream.truncate(PACK_BYTES)
            if pack.stat().st_size != PACK_BYTES:
                raise RuntimeError("sparse fixture has wrong declared size")
            if pack.stat().st_blocks * 512 > 8 * 1024 * 1024:
                raise RuntimeError("filesystem did not preserve sparse pack; refusing large allocation")
            stream.seek(span_offset)
            stream.write(payload)
            stream.flush()
        if pack.stat().st_blocks * 512 > 16 * 1024 * 1024:
            raise RuntimeError("synthetic expert unexpectedly allocated too much disk")

        stamp = {
            "run_id": RUN_ID,
            "model_revision": MODEL_REVISION,
            "plan_id": PLAN_ID,
            "layer": 0,
            "batch": 0,
            "epoch": EPOCH,
        }
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
            "ledger_path": str(ledger),
            "endpoints": [{"rank": 0, "argv": [str(rank_path)]}],
            "bindings": [{
                "model_revision": MODEL_REVISION, "rank": 0, "layer": 0,
                "expert": EXPERT, "local": 0, "epoch": EPOCH,
                "index_sha256": index_sha, "blob_sha256": blob_sha,
            }],
            "resident_payload_limit": span_bytes,
            "timeout_ms": 10_000,
        }
        session_config = {"layer": 0, "init": init, "expected_ranks": [0]}
        config = {
            "bridge_executable": str(bridge_path),
            "working_directory": str(temp_root),
            "startup_timeout_ms": 30_000,
            "shutdown_timeout_ms": 5_000,
            "epoch": EPOCH,
            "sessions": [session_config],
        }

        sessions = []
        graceful = False
        try:
            sessions.append(launcher._start_bridge(config, session_config, time.monotonic() + 30.0))
            hidden = [f32(math.sin(f32(f32(i) * f32(0.17))) / 16.0) for i in range(HIDDEN)]
            body = b"".join(struct.pack("<f", value) for value in hidden)
            body += struct.pack("<I", EXPERT) + struct.pack("<i", -1)
            run, model, plan = RUN_ID.encode(), MODEL_REVISION.encode(), PLAN_ID.encode()
            header = struct.pack(
                "<4sIQIIIQHHHHQ", b"VFB3", 1, BATCH, 1, 1, 0, EPOCH,
                len(run), len(model), len(plan), 0, ATTEMPT,
            )
            sessions[0].channel.sendall(header + run + model + plan + body)
            reply_header = read_exact(sessions[0].channel, 52, time.monotonic() + 30.0)
            fields = struct.unpack("<4sIQIIIQHHHHQ", reply_header)
            magic, status, batch, rows, columns, layer, epoch, run_len, model_len, plan_len, reserved, attempt = fields
            if (magic, status, batch, rows, columns, layer, epoch, reserved, attempt) != (
                b"VFR3", 0, BATCH, 1, HIDDEN, 0, EPOCH, 0, ATTEMPT
            ):
                raise RuntimeError(f"invalid native response metadata: {fields!r}")
            identity = read_exact(sessions[0].channel, run_len + model_len + plan_len, time.monotonic() + 30.0)
            if identity != run + model + plan:
                raise RuntimeError("native VFR3 identity mismatch")
            row = read_exact(sessions[0].channel, 4 + HIDDEN * 4, time.monotonic() + 30.0)
            position = struct.unpack_from("<I", row, 0)[0]
            values = struct.unpack_from(f"<{HIDDEN}f", row, 4)
            if position != 0 or not all(math.isfinite(value) for value in values) or not any(value != 0.0 for value in values):
                raise RuntimeError("native row shape or finite-value check failed")
            graceful = True
        finally:
            if sessions:
                cleaned = launcher._finish_sessions(sessions, config, graceful=graceful)
                if not cleaned:
                    raise RuntimeError("coordinator/session shutdown failed; ticket remains consumed")

    claims = list(ledger.glob("*.claim.json"))
    if len(claims) != 1:
        raise RuntimeError(f"expected one clean local reservation, got {len(claims)}")
    claim = json.loads(claims[0].read_text(encoding="utf-8"))
    expected = (RUN_ID, MODEL_REVISION, PLAN_ID, 0, EPOCH, BATCH, ATTEMPT)
    actual = (
        claim["stamp"]["run_id"], claim["stamp"]["model_revision"], claim["stamp"]["plan_id"],
        claim["stamp"]["layer"], claim["stamp"]["epoch"], claim["stamp"]["batch"], claim["attempt"],
    )
    if actual != expected or claim["status"] != "RESERVED_UNKNOWN_NO_REPLAY":
        raise RuntimeError(f"reservation identity/status mismatch: {actual!r}")
    print(json.dumps({
        "result": "PASS",
        "bridge": "real vf-strata-native-bridge",
        "worker": "real local vf-rank-session",
        "pack": "sparse 50,292,326,400-byte file; one 2,176,000-byte synthetic expert",
        "request": {"run_id": RUN_ID, "batch": BATCH, "attempt": ATTEMPT},
        "response": "VFR3 identity, dimensions, finite nonzero row",
        "shutdown": "graceful",
        "ledger_status": claim["status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
