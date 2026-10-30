"""Check the actual adjacent verify.py try and write_text statement on disk.

Only the timeout literal is shortened to 0.3 seconds in this extracted AST.
"""
import ast
import locale
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "candidate/verify.py"


def run_and_write(command, out):
    tree = ast.parse(SOURCE.read_text())
    found = []
    for loop in ast.walk(tree):
        if not isinstance(loop, ast.For):
            continue
        for index, node in enumerate(loop.body[:-1]):
            following = loop.body[index + 1]
            if not (isinstance(node, ast.Try) and isinstance(following, ast.Expr)
                    and isinstance(following.value, ast.Call)
                    and isinstance(following.value.func, ast.Attribute)
                    and following.value.func.attr == "write_text"):
                continue
            calls = [x for x in ast.walk(node) if isinstance(x, ast.Call)
                     and isinstance(x.func, ast.Attribute) and isinstance(x.func.value, ast.Name)
                     and x.func.value.id == "subprocess" and x.func.attr == "run"]
            if not calls:
                continue
            timeout = next(x for x in calls[0].keywords if x.arg == "timeout")
            assert isinstance(timeout.value, ast.Constant) and timeout.value.value == 600
            timeout.value = ast.Constant(value=0.3)
            found.append((node, following))
    assert len(found) == 1
    program = compile(ast.fix_missing_locations(ast.Module(body=list(found[0]), type_ignores=[])), str(SOURCE), "exec")
    ns = dict(subprocess=subprocess, locale=locale, command=command, root=out, out=out,
              label="step", env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    exec(program, ns)
    return ns["rc"], (out / "step.log").read_text()


class PersistLog(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT / "tmp", prefix="r90-written-log-")
        self.out = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_timeout_really_writes_partial_stdout_and_stderr(self):
        code = "import os,time;os.write(1,b'persist-stdout\\n');os.write(2,b'persist-stderr\\n');time.sleep(4)"
        rc, log = run_and_write([sys.executable, "-B", "-c", code], self.out)
        self.assertEqual(rc, -1)
        self.assertIn("persist-stdout\npersist-stderr\n", log)
        self.assertIn("TimeoutExpired:", log)

    def test_missing_executable_really_writes_error_log(self):
        rc, log = run_and_write([str(ROOT / "missing-executable-never-created")], self.out)
        self.assertEqual(rc, -1)
        self.assertTrue(log.startswith("FileNotFoundError:"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
