"""Exercise the real verify.py subprocess boundary; no full CPU suite or GPU.

Extract its try statement unchanged except the test's short 0.3s timeout. This
does not replace the production 600s limit. Subprocesses and pipes are real.
"""
import ast
import json
import locale
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent
VARIANT = os.environ["VF_R90_TEST_VARIANT"]
SOURCE = ROOT / VARIANT / "verify.py"


def extracted_boundary(path):
    tree = ast.parse(path.read_text())
    candidates = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        for child in ast.walk(node):
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute)
                    and isinstance(child.func.value, ast.Name)
                    and child.func.value.id == "subprocess" and child.func.attr == "run"):
                matches = [k for k in child.keywords if k.arg == "timeout"]
                if matches and isinstance(matches[0].value, ast.Constant) and matches[0].value.value == 600:
                    candidates.append((node, matches[0]))
    if len(candidates) != 1:
        raise AssertionError(f"expected exactly one verify boundary, found {len(candidates)}")
    node, timeout = candidates[0]
    timeout.value = ast.Constant(value=0.3)  # sole test-only AST change
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")


def run_boundary(command, cwd):
    ns = dict(subprocess=subprocess, locale=locale, command=command, root=Path(cwd),
              env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    exec(extracted_boundary(SOURCE), ns)
    return ns["rc"], ns["text"]


class VerifyBoundary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="r90-real-subprocess-")
        self.cwd = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_success_uses_actual_combined_stdout_and_status(self):
        rc, text = run_boundary([sys.executable, "-B", "-c",
                                 "import os;os.write(1,b'out\\n');os.write(2,b'err\\n')"], self.cwd)
        self.assertEqual(rc, 0)
        self.assertEqual(text, "out\nerr\n")

    def test_timeout_preserves_bytes_from_stdout_and_redirected_stderr(self):
        code = "import os,time;os.write(1,b'partial-stdout\\n');os.write(2,b'partial-stderr\\n');time.sleep(4)"
        rc, text = run_boundary([sys.executable, "-B", "-c", code], self.cwd)
        self.assertEqual(rc, -1)
        self.assertIn("partial-stdout\npartial-stderr\n", text)
        self.assertIn("TimeoutExpired:", text)

    def test_timeout_invalid_bytes_do_not_hide_the_diagnostic(self):
        code = "import os,time;os.write(1,b'invalid-\\xff\\n');time.sleep(4)"
        rc, text = run_boundary([sys.executable, "-B", "-c", code], self.cwd)
        self.assertEqual(rc, -1)
        self.assertIn("invalid-\\xff\n", text)
        self.assertIn("TimeoutExpired:", text)

    def test_timeout_with_unterminated_output_separates_diagnostic(self):
        code = "import os,time;os.write(1,b'no-newline');time.sleep(4)"
        rc, text = run_boundary([sys.executable, "-B", "-c", code], self.cwd)
        self.assertEqual(rc, -1)
        self.assertIn("no-newline\nTimeoutExpired:", text)

    def test_timeout_string_output_is_supported(self):
        exception = subprocess.TimeoutExpired(cmd=["hypothetical"], timeout=0.3, output="text-output")
        with mock.patch.object(subprocess, "run", side_effect=exception):
            rc, text = run_boundary(["not-started"], self.cwd)
        self.assertEqual(rc, -1)
        self.assertIn("text-output\nTimeoutExpired:", text)

    def test_child_nonzero_exit_is_not_reclassified(self):
        rc, text = run_boundary([sys.executable, "-B", "-c", "import sys;print('failed');sys.exit(7)"], self.cwd)
        self.assertEqual(rc, 7)
        self.assertEqual(text, "failed\n")

    def test_timeout_without_output_is_failure(self):
        rc, text = run_boundary([sys.executable, "-B", "-c", "import time;time.sleep(4)"], self.cwd)
        self.assertEqual(rc, -1)
        self.assertTrue(text.startswith("TimeoutExpired:"), repr(text))

    def test_missing_executable_retains_original_oserror_behavior(self):
        rc, text = run_boundary([str(ROOT / "missing-executable-never-created")], self.cwd)
        self.assertEqual(rc, -1)
        self.assertTrue(text.startswith("FileNotFoundError:"), repr(text))


if __name__ == "__main__":
    unittest.main(verbosity=2)
