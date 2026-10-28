"""A53-R36: when a guard_schemata stage-one run hits --timeout, every process of that run
must be stopped, not only the direct child.

subprocess.run(timeout=...) kills only the process it started. The cargo test binaries,
session workers and helpers below it kept running into the following mutant runs, taking
their CPU and memory and pushing them past --timeout in turn.
"""

import importlib.util
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "guard_schemata", ROOT / "scripts" / "guard_schemata.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

FAKE_CARGO = """#!/bin/sh
sleep 60 &
echo $! > "$GRANDCHILD_PID"
sleep 60
"""


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


class TimeoutCleanup(unittest.TestCase):
    def test_a_timed_out_run_leaves_no_process_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "bin").mkdir()
            cargo = tmp / "bin" / "cargo"
            cargo.write_text(FAKE_CARGO)
            cargo.chmod(0o755)
            pid_file = tmp / "grandchild.pid"
            env = dict(os.environ)
            env.update(
                PATH=f"{tmp / 'bin'}:/usr/bin:/bin", GRANDCHILD_PID=str(pid_file)
            )
            run = guard.run_rust(tmp, [], env, 1, tmp / "log.txt", 2)
            self.assertTrue(run["timed_out"])
            self.assertIsNone(run["exit_code"])
            pid = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            while alive(pid) and time.monotonic() < deadline:
                time.sleep(0.1)
            survived = alive(pid)
            if survived:
                os.kill(pid, signal.SIGKILL)
            self.assertFalse(
                survived, "a process of the timed-out run was left running"
            )


if __name__ == "__main__":
    unittest.main()
