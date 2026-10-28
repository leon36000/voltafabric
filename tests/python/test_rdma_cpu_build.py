"""Actual compiler/loader checks; no NIC, GPU or network operation."""

import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / ".work/rdma-build-tests"
SCRIPT = ROOT / "scripts/build_rdma_cpu.py"


class RdmaNativeBuild(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=WORK)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, target, *args):
        return subprocess.run(
            [sys.executable, "-B", str(SCRIPT), "--out", str(target), *args],
            capture_output=True,
            text=True,
            timeout=35,
        )

    def test_native_control_and_relocated_loader(self):
        out = self.root / "build"
        q = self.build(out)
        self.assertEqual(q.returncode, 0, q.stderr)
        receipt = json.loads((out / "build.json").read_text())
        self.assertEqual(receipt["status"], "COMPILED_NATIVE_ONLY")
        self.assertEqual(receipt["machine"], platform.machine())
        self.assertFalse(receipt["sanitized"])
        self.assertEqual(
            receipt["source_sha256"],
            hashlib.sha256(
                (ROOT / "scripts/rdma_cpu_bridge.c").read_bytes()
            ).hexdigest(),
        )
        moved = self.root / "different path with spaces"
        out.rename(moved)
        binary = moved / receipt["binary"]
        self.assertEqual(
            hashlib.sha256(binary.read_bytes()).hexdigest(), receipt["binary_sha256"]
        )
        q = subprocess.run(
            [str(binary)], cwd=moved, capture_output=True, text=True, timeout=3
        )
        self.assertEqual(q.returncode, 2)
        self.assertIn("usage role bind-ip port cookie worker", q.stderr)

    def test_existing_output_directory_is_preserved(self):
        out = self.root / "existing"
        out.mkdir()
        (out / "marker").write_bytes(b"do not replace")
        q = self.build(out)
        self.assertEqual(q.returncode, 2)
        self.assertIn("output already exists", q.stderr)
        self.assertEqual((out / "marker").read_bytes(), b"do not replace")
        self.assertEqual([p.name for p in out.iterdir()], ["marker"])

    def test_symlink_output_cannot_overwrite_target(self):
        target = self.root / "target"
        target.mkdir()
        (target / "marker").write_bytes(b"preserve")
        alias = self.root / "alias"
        alias.symlink_to(target, target_is_directory=True)
        q = self.build(alias)
        self.assertEqual(q.returncode, 2)
        self.assertIn("output already exists", q.stderr)
        self.assertEqual((target / "marker").read_bytes(), b"preserve")
        self.assertEqual([p.name for p in target.iterdir()], ["marker"])

    def test_compiler_failure_is_terminal_not_a_build(self):
        out = self.root / "failed"
        q = self.build(out, "--cc", "/usr/bin/false")
        self.assertEqual(q.returncode, 2)
        receipt = json.loads((out / "build.json").read_text())
        self.assertEqual(receipt["status"], "COMPILE_FAILED")
        self.assertNotEqual(receipt["exit_code"], 0)
        self.assertFalse((out / "vf-rsocket-cpu").exists())
        self.assertTrue((out / "rdma_cpu_bridge.c").is_file())
        self.assertEqual(self.build(out).returncode, 2)

    def test_sanitizer_is_explicit_and_loader_works(self):
        out = self.root / "sanitized"
        q = self.build(out, "--sanitized")
        self.assertEqual(q.returncode, 0, q.stderr)
        receipt = json.loads((out / "build.json").read_text())
        self.assertTrue(receipt["sanitized"])
        self.assertIn("-fsanitize=address,undefined", receipt["argv"])
        q = subprocess.run(
            [str(out / "vf-rsocket-cpu")], capture_output=True, text=True, timeout=3
        )
        self.assertEqual(q.returncode, 2, q.stderr)
        self.assertNotIn("AddressSanitizer", q.stderr)
        self.assertNotIn("runtime error:", q.stderr)


if __name__ == "__main__":
    unittest.main()
