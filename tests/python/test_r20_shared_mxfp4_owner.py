"""Per-series immutable file snapshot, independent bindings and distinct-path checks."""

import copy
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
    "r20_crosscheck", ROOT / "scripts/mxfp4_session_crosscheck.py"
)
cross = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cross)
BINARY = Path(os.environ.get("VF_BINARY", ROOT / "target/release/vf")).resolve()
WORKER = BINARY.parent / "vf-rank-session"


@unittest.skipUnless(BINARY.is_file() and WORKER.is_file(), "native binaries required")
class SharedSnapshot(unittest.TestCase):
    def setUp(self):
        base = Path(os.environ.get("TMPDIR", ROOT / "evidence/adapter-temp"))
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="r20-owner-", dir=base)
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.payload, self.expected = cross.fixture(BINARY)

    def invoke(self):
        return subprocess.run(
            [str(BINARY), "session-run-mxfp4", "-"],
            input=json.dumps(self.payload),
            text=True,
            capture_output=True,
            timeout=20,
        )

    def require_atomic_rejection(self):
        marker = self.directory / "spawned"
        for worker in self.payload["workers"]:
            worker["endpoint"]["argv"] = [
                sys.executable,
                "-c",
                "from pathlib import Path;Path(" + repr(str(marker)) + ").touch()",
            ]
        result = self.invoke()
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertFalse(json.loads(result.stdout)["ok"])
        self.assertFalse(marker.exists(), "no worker may start on admission failure")

    def test_same_path_is_one_immutable_snapshot_with_distinct_bindings(self):
        fifo = self.directory / "one-read.safetensors"
        os.mkfifo(fifo)
        raw = cross.ARTIFACT.read_bytes()
        failures = []

        def write_once():
            deadline = time.monotonic() + 5
            fd = None
            try:
                while fd is None:
                    try:
                        fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise TimeoutError("reader did not open FIFO")
                        time.sleep(0.005)
                # The coordinator already owns its read FD. Remove the path before
                # completing the first read, so reopening it deterministically fails.
                fifo.unlink()
                os.set_blocking(fd, True)
                with os.fdopen(fd, "wb") as stream:
                    fd = None
                    stream.write(raw)
            except Exception as error:
                failures.append(repr(error))
            finally:
                if fd is not None:
                    os.close(fd)

        for worker in self.payload["workers"]:
            worker["artifact_path"] = str(fifo)
        writer = threading.Thread(target=write_once)
        writer.start()
        try:
            result = self.invoke()
        finally:
            writer.join(timeout=6)
        self.assertFalse(writer.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)["result"]
        self.assertTrue(report["ok"], report["error"])
        self.assertEqual([b["output_bits"] for b in report["batches"]], self.expected)
        self.assertEqual(len(report["loads"]), 2)
        self.assertTrue(all(c["exit_code"] == 0 for c in report["closes"]))
        self.assertTrue(
            all(
                t[1]["loads_transferred"] == 1 and t[1]["execs"] == 3
                for t in report["data_phase_traffic"]
            )
        )

    def test_different_path_with_same_declared_digest_must_be_read_and_rejected(self):
        bad = self.directory / "corrupt.safetensors"
        raw = bytearray(cross.ARTIFACT.read_bytes())
        raw[-1] ^= 1
        bad.write_bytes(raw)
        self.payload["workers"][1]["artifact_path"] = str(bad)
        self.require_atomic_rejection()

    def test_shared_path_cannot_reuse_a_foreign_digest(self):
        self.payload["workers"][1]["binding"]["artifact_sha256"] = "0" * 64
        self.require_atomic_rejection()

    def test_shared_path_still_validates_every_expert_mapping(self):
        binding = copy.deepcopy(self.payload["workers"][1]["binding"])
        binding["experts"][0]["expert"] = 999999
        self.payload["workers"][1]["binding"] = binding
        self.require_atomic_rejection()


if __name__ == "__main__":
    unittest.main()
