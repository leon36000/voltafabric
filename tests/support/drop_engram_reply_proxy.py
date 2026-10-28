#!/usr/bin/env python3
"""Test-only stream proxy: let the worker answer one Engram lookup, then lose that reply."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def read_exact(stream, length: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < length:
        block = stream.read(length - len(chunks))
        if not block:
            raise EOFError("worker stream ended inside a declared raw body")
        chunks.extend(block)
    return bytes(chunks)


def main() -> int:
    worker, receipt_path = sys.argv[1:]
    child = subprocess.Popen(
        [worker], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr
    )
    assert child.stdin is not None and child.stdout is not None
    try:
        while True:
            request_line = sys.stdin.buffer.readline()
            if not request_line:
                break
            request = json.loads(request_line)
            child.stdin.write(request_line)
            request_op = request.get("op")
            if request_op == "load_engram_shard_raw":
                child.stdin.write(read_exact(sys.stdin.buffer, int(request["artifact_bytes"])))
            elif request_op == "fetch_engram_rows_varint":
                child.stdin.write(read_exact(sys.stdin.buffer, int(request["row_id_bytes"])))
            child.stdin.flush()

            reply_line = child.stdout.readline()
            if not reply_line:
                break
            reply = json.loads(reply_line)
            raw_body = b""
            if reply.get("reply") == "engram_rows_raw":
                raw_body = read_exact(child.stdout, int(reply["frame_bytes"]))
            if request_op in {"fetch_engram_rows", "fetch_engram_rows_varint"}:
                Path(receipt_path).write_text(
                    json.dumps(
                        {
                            "request": "fetch_engram_rows",
                            "wire_request": request_op,
                            "attempt": request["attempt"],
                            "worker_reply_consumed": reply.get("reply") == "engram_rows_raw",
                            "reply_bytes_discarded": len(raw_body),
                            "reply": reply.get("reply"),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                # The worker completed and formed a valid response. Drop only the response
                # at the transport boundary so the coordinator must classify the result as
                # uncertain and refuse to publish/cache a partial batch.
                return 0
            sys.stdout.buffer.write(reply_line)
            sys.stdout.buffer.write(raw_body)
            sys.stdout.buffer.flush()
    finally:
        if child.poll() is None:
            child.terminate()
        try:
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
