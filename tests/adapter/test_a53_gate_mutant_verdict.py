"""A53-R34: rust_mutations --gate-python must not report as a survivor a mutant that no test
ran on.

verify.py stops at its first failing step, and rustfmt and clippy run before any test step.
A mutant whose edit is not rustfmt- or clippy-clean stopped there; that cosmetic failure was
filtered out, so the mutant was reported compiled and surviving although no test, oracle or
build step ran. Three mutants of the current CASES (R02, P01, Q04) are not rustfmt-clean.
"""

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "rust_mutations", ROOT / "scripts" / "rust_mutations.py"
)
mutations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mutations)


class GateMutantVerdict(unittest.TestCase):
    def test_a_gate_stopped_at_a_cosmetic_step_tested_nothing(self):
        for steps in (["rustfmt"], ["clippy"], ["python_format"], ["python_lint"]):
            self.assertEqual(mutations.gate_outcome(steps, []), "untested", steps)

    def test_a_failed_semantic_step_kills(self):
        self.assertEqual(mutations.gate_outcome(["rust_debug"], []), "killed")
        self.assertEqual(mutations.gate_outcome(["independent_oracle"], []), "killed")

    def test_a_gate_that_passed_every_step_is_a_survivor(self):
        self.assertEqual(mutations.gate_outcome([], []), "survived")

    def test_an_environment_error_is_not_a_kill(self):
        # Unchanged: a step that failed on a missing import proves nothing.
        self.assertEqual(
            mutations.gate_outcome(["quantized_adapter"], ["quantized_adapter"]),
            "survived",
        )

    def test_compilation_is_read_from_the_step_logs(self):
        self.assertFalse(
            mutations.logs_compiled(["ok", "error[E0425]: cannot find value"])
        )
        self.assertFalse(
            mutations.logs_compiled(["error: could not compile `voltafabric`"])
        )
        self.assertTrue(mutations.logs_compiled(["test result: ok. 3 passed"]))


if __name__ == "__main__":
    unittest.main()
