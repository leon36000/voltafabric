"""CPU synthetic fixtures. These tests never launch R81 or a model."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import pinned_validator as P

ROOT = Path(__file__).resolve().parent


def fixture():
    n, _ = P.load_pinned()
    test_path = P.R88.parent / "test_nominal_contract.py"
    # This test fixture is synthetic and is not part of runtime admission.
    namespace = {"__name__": "r96_fixture", "__file__": str(test_path)}
    with mock.patch.dict(sys.modules, {"nominal_contract": n}):
        exec(compile(test_path.read_bytes(), str(test_path), "exec"), namespace)
    rows, golden = namespace["synthetic"]()
    return rows, golden


def log(rows):
    return b"\n".join(json.dumps(row).encode() for row in rows) + b"\n"


class Tests(unittest.TestCase):
    def test_full_synthetic_rows_and_real_golden(self):
        rows, _ = fixture()
        result, proof = P.validate_rows_log(log(rows))
        self.assertEqual(result["exact_Collector_values"], 2944)
        self.assertEqual(result["status"], "R81_ROWS_ADMITTED_NOT_COMPLETE_HARDWARE_QUALIFICATION")
        self.assertTrue(result["guard_exit_and_postflight_still_required"])
        self.assertEqual(proof["validator_sha256"], P.R88_SHA)

    def test_reject_changed_counter_type_or_output(self):
        rows, _ = fixture()
        bad = copy.deepcopy(rows)
        bad[6]["execs"] = 2.0
        with self.assertRaises(AssertionError):
            P.validate_rows_log(log(bad))
        bad = copy.deepcopy(rows)
        bad[2]["output_bits"][0][0] ^= 1
        with self.assertRaises(AssertionError):
            P.validate_rows_log(log(bad))

    def test_oracle_computation_agrees_with_existing_r88(self):
        actual, _ = P.load_pinned()
        reference = {}
        raw = P.read_pinned(P.R88, P.R88_SHA)
        reference["__file__"] = str(P.R88)
        exec(compile(raw, str(P.R88), "exec"), reference)
        g = json.loads(P.read_pinned(P.GOLDEN, P.GOLDEN_SHA))
        self.assertEqual(actual.expected_collector(g), reference["expected_collector"](g))

    def test_source_corruption_and_symlink_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            d = Path(d)
            target = d / "v.py"
            target.write_bytes(P.R88.read_bytes() + b"\n")
            with mock.patch.object(P, "R88", target):
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    P.load_pinned()
            target.write_bytes(P.R88.read_bytes())
            link = d / "link.py"
            link.symlink_to(target)
            with mock.patch.object(P, "R88", link):
                with self.assertRaises(OSError):
                    P.load_pinned()

    def test_exec_uses_snapshot_after_path_replaced(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            for name in ("R88", "R55", "R24"):
                with self.subTest(source=name):
                    target = Path(d) / (name + ".py")
                    target.write_bytes(getattr(P, name).read_bytes())
                    read = P.read_pinned
                    def replace_after_read(path, expected, limit=2 * 1024 * 1024):
                        raw = read(path, expected, limit)
                        if path == target:
                            target.write_text("raise RuntimeError('replacement executed')\n")
                        return raw
                    with mock.patch.object(P, name, target), mock.patch.object(P, "read_pinned", side_effect=replace_after_read):
                        validator, proof = P.load_pinned()
                        self.assertEqual(proof["validator_sha256"], P.R88_SHA)
                        self.assertEqual(validator.OWNER_COUNTERS["kernel_launches"], 736)
                        g = json.loads(P.read_pinned(P.GOLDEN, P.GOLDEN_SHA))
                        self.assertEqual(len(validator.expected_collector(g)), 46)

    def test_optimized_python_is_refused(self):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        process = subprocess.run([sys.executable, "-O", "-c", "import pinned_validator"],
                                 cwd=ROOT, env=env, capture_output=True, timeout=10)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"Python optimization is forbidden", process.stderr)

    def test_manifest_rational_or_golden_corruption_rejected(self):
        rows, _ = fixture()
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            d = Path(d)
            for name in ("R55_MANIFEST", "R55", "R24", "GOLDEN"):
                with self.subTest(name=name):
                    f = d / name
                    f.write_bytes(getattr(P, name).read_bytes() + b"x")
                    with mock.patch.object(P, name, f):
                        with self.assertRaisesRegex(ValueError, "digest mismatch"):
                            P.validate_rows_log(log(rows), golden_path=f if name == "GOLDEN" else P.GOLDEN)

    def test_read_bound_and_duplicate_manifest_key(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p = Path(d) / "x"
            raw = b"1234"
            p.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "cap"):
                P.read_pinned(p, hashlib.sha256(raw).hexdigest(), 3)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            P.strict_json(b'{"files":{},"files":{}}')


if __name__ == "__main__":
    unittest.main()
