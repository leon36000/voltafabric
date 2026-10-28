"""Actual CLI must reject bounded aggregate input before EOF or a blocked open."""

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "r21_cross", ROOT / "scripts/mxfp4_session_crosscheck.py"
)
cross = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cross)
BINARY = Path(os.environ.get("VF_BINARY", ROOT / "target/release/vf")).resolve()
MIB = 1024 * 1024


@unittest.skipUnless(BINARY.is_file(), "native CLI required")
class RemainingBudget(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("TMPDIR", ROOT / "evidence/adapter-temp"))
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="r21-budget-", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def prepare(self, size):
        # Sparse, non-expert U8 padding tests resource bounds; it is not model data.
        raw = cross.ARTIFACT.read_bytes()
        n = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + n])
        body = raw[8 + n :]
        padding = size - len(raw)
        for _ in range(10):
            header["r21_padding"] = dict(
                dtype="U8",
                shape=[padding],
                data_offsets=[len(body), len(body) + padding],
            )
            encoded = json.dumps(header, separators=(",", ":")).encode()
            encoded += b" " * ((-len(encoded)) % 8)
            wanted = size - 8 - len(encoded) - len(body)
            if wanted == padding:
                break
            padding = wanted
        self.assertEqual(8 + len(encoded) + len(body) + padding, size)
        path = self.directory / "large-a.safetensors"
        with path.open("wb") as stream:
            stream.write(len(encoded).to_bytes(8, "little") + encoded + body)
            stream.truncate(size)
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        alias = self.directory / "large-b.safetensors"
        alias.symlink_to(path)
        fifo = self.directory / "third.safetensors"
        os.mkfifo(fifo)
        payload, _ = cross.fixture(BINARY)
        third = copy.deepcopy(payload["workers"][0])
        third["binding"]["rank"] = third["endpoint"]["rank"] = 77
        for expert in third["binding"]["experts"]:
            expert["local"] += 100
        payload["workers"].append(third)
        marker = self.directory / "forbidden-worker"
        for worker, source in zip(payload["workers"], [path, alias, fifo]):
            worker["artifact_path"] = str(source)
            worker["binding"]["artifact_sha256"] = digest
            worker["endpoint"]["argv"] = [
                sys.executable,
                "-c",
                "from pathlib import Path;Path(" + repr(str(marker)) + ").touch()",
            ]
        for batch in payload["batches"]:
            p = batch["problem"]
            p["preferred_owners"]["241"] = 77
            p["registry"]["copies"].extend(
                dict(rank=77, expert=e["expert"], local=e["local"])
                for e in third["binding"]["experts"]
            )
            p["resident_order"]["77"] = [
                e["local"] for e in third["binding"]["experts"]
            ]
        return path, fifo, payload, marker

    def start(self, payload):
        return subprocess.Popen(
            [str(BINARY), "session-run-mxfp4", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_remaining_budget_rejects_before_stream_eof(self):
        path, fifo, payload, marker = self.prepare(60 * MIB)
        prefix = 8 * MIB + 1
        sent, release = threading.Event(), threading.Event()
        failures = []

        def writer():
            fd = None
            try:
                deadline = time.monotonic() + 30
                while fd is None:
                    try:
                        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("coordinator did not open third input")
                        time.sleep(0.005)
                os.set_blocking(fd, True)
                with os.fdopen(fd, "wb") as sink, path.open("rb") as source:
                    fd = None
                    left = prefix
                    while left:
                        value = source.read(min(65536, left))
                        self.assertTrue(value)
                        sink.write(value)
                        left -= len(value)
                    sink.flush()
                    sent.set()
                    release.wait(10)
            except Exception as error:
                failures.append(repr(error))
                sent.set()
            finally:
                if fd is not None:
                    os.close(fd)

        q = self.start(payload)
        thread = threading.Thread(target=writer)
        thread.start()
        try:
            q.stdin.write(json.dumps(payload).encode())
            q.stdin.close()
            q.stdin = None
            self.assertTrue(sent.wait(30), "prefix writer did not finish")
            self.assertEqual(failures, [])
            try:
                stdout, stderr = q.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                self.fail("aggregate admission waited for EOF beyond remaining+1 bytes")
            self.assertEqual(q.returncode, 2, stderr)
            self.assertEqual(json.loads(stdout)["error"]["code"], "ARTIFACT_BUDGET")
            self.assertFalse(marker.exists())
        finally:
            release.set()
            thread.join(timeout=5)
            if q.poll() is None:
                try:
                    q.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    q.kill()
                    q.communicate(timeout=5)
        self.assertFalse(thread.is_alive())

    def test_zero_remaining_budget_refuses_before_opening_fifo(self):
        _, _, payload, marker = self.prepare(64 * MIB)
        q = self.start(payload)
        try:
            try:
                stdout, stderr = q.communicate(
                    input=json.dumps(payload).encode(), timeout=4
                )
            except subprocess.TimeoutExpired:
                self.fail("exhausted aggregate budget attempted a blocking file open")
            self.assertEqual(q.returncode, 2, stderr)
            self.assertEqual(json.loads(stdout)["error"]["code"], "ARTIFACT_BUDGET")
            self.assertFalse(marker.exists())
        finally:
            if q.poll() is None:
                q.kill()
                q.communicate(timeout=5)

    def test_existing_snapshot_remains_usable_after_budget_exhaustion(self):
        _, _, payload, _ = self.prepare(64 * MIB)
        payload["workers"][2]["artifact_path"] = payload["workers"][0]["artifact_path"]
        worker = BINARY.parent / "vf-rank-session"
        if not worker.is_file():
            worker = BINARY.parent / "cpu_worker"
        self.assertTrue(worker.is_file(), "CPU worker required for valid control")
        for source in payload["workers"]:
            source["endpoint"]["argv"] = [str(worker)]
        result = subprocess.run(
            [str(BINARY), "session-run-mxfp4", "-"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)["result"]
        self.assertTrue(report["ok"], report["error"])
        _, expected = cross.fixture(BINARY)
        self.assertEqual([b["output_bits"] for b in report["batches"]], expected)
        self.assertEqual(len(report["loads"]), 3)
        self.assertTrue(all(c["exit_code"] == 0 for c in report["closes"]))
        self.assertTrue(
            all(
                t[1]["loads_transferred"] == 1 and t[1]["execs"] == 3
                for t in report["data_phase_traffic"]
            )
        )


if __name__ == "__main__":
    unittest.main()
