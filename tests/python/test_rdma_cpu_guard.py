"""Real harmless processes prove fail-closed helper admission, including Python -O."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".work/rdma-guard-tests"
SCRIPT = ROOT / "scripts/rdma_cpu_guard.py"


class CpuRdmaGuard(unittest.TestCase):
    def probe(self, *, optimized=False, wrong_hash=False, mode="cpu"):
        WORK.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=WORK) as temp:
            root = Path(temp)
            marker = root / "executed"
            receipt = root / "guard.json"
            config = dict(
                mode=mode,
                argv=["/usr/bin/touch", str(marker)],
                receipt=str(receipt),
                hashes={
                    "/usr/bin/touch": "0" * 64
                    if wrong_hash
                    else hashlib.sha256(Path("/usr/bin/touch").read_bytes()).hexdigest()
                },
            )
            q = subprocess.run(
                [
                    sys.executable,
                    *(["-O"] if optimized else []),
                    "-B",
                    str(SCRIPT),
                    "--config",
                    json.dumps(config),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return q, json.loads(receipt.read_text()), marker.exists()

    def test_control_executes_once(self):
        q, receipt, executed = self.probe()
        self.assertEqual(q.returncode, 0, q.stderr)
        self.assertTrue(executed)
        self.assertEqual(receipt["status"], "WORKER_EXITED")
        self.assertEqual(receipt["worker_exit_code"], 0)

    def test_wrong_hash_refused_before_spawn(self):
        q, receipt, executed = self.probe(wrong_hash=True)
        self.assertEqual(q.returncode, 2)
        self.assertFalse(executed)
        self.assertNotIn("worker_pid", receipt)
        self.assertIn("identity mismatch", receipt["error"])

    def test_optimized_mode_cannot_bypass_hash(self):
        q, receipt, executed = self.probe(optimized=True, wrong_hash=True)
        self.assertEqual(q.returncode, 2)
        self.assertFalse(executed)
        self.assertNotIn("worker_pid", receipt)
        self.assertIn("optimized Python", receipt["error"])

    def test_gpu_mode_refused_before_inventory_or_execution(self):
        q, receipt, executed = self.probe(mode="gpu")
        self.assertEqual(q.returncode, 2)
        self.assertFalse(executed)
        self.assertNotIn("worker_pid", receipt)
        self.assertIn("CPU RDMA guard only", receipt["error"])


if __name__ == "__main__":
    unittest.main()
