"""A53-R33: guard_schemata's stage-one verdict must not count a timed-out run as a kill.

A run that hits --timeout records exit_code None, and the verdict was `exit_code != 0`, so
a mutant whose run executed no test at all was reported killed and the sweep could end
ALL_KILLED. The tool's contract: "a mutant dies only if some test reaches that refusal and
checks its code or message".
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "guard_schemata", ROOT / "scripts" / "guard_schemata.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def run(exit_code, timed_out=False):
    return {"exit_code": exit_code, "timed_out": timed_out}


class StageOneVerdict(unittest.TestCase):
    def test_a_timed_out_run_is_not_a_kill(self):
        self.assertFalse(guard.stage_one_killed(run(None, timed_out=True)))

    def test_a_finished_failing_run_is_a_kill(self):
        self.assertTrue(guard.stage_one_killed(run(101)))
        self.assertTrue(guard.stage_one_killed(run(1)))

    def test_a_passing_run_is_not_a_kill(self):
        self.assertFalse(guard.stage_one_killed(run(0)))


if __name__ == "__main__":
    unittest.main()
