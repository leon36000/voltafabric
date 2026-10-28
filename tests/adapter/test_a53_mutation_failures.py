"""A53-R37: rust_mutations must record which tests killed each mutant, in every mode.

The failing-test list was read from the tool's own stdout with cargo's pattern only, so:
in --python-tests mode the unittest list was computed, then overwritten (always empty, for
the 36 Python mutants of the catalogue); in --gate-python mode cargo and unittest report in
verify.py's per-step logs, not its stdout (always empty there too); and a cargo name holding
spaces (a doctest, "src/x.rs - item (line N)") never matched `^test (\\S+) ... FAILED`.
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

CARGO = """running 2 tests
test stale_registry_rejected ... FAILED
test other_case ... ok
test src/session.rs - session::CompressedArtifact (line 2191) - compile fail ... FAILED
"""
UNITTEST = """FAIL: test_live_requires_authorized_spend (test_a43.MainRefusalTests.test_live)
ERROR: test_private_path (test_a43.PathRefusalTests.test_private_path)
"""


class MutationFailures(unittest.TestCase):
    def test_cargo_names_are_read_including_doctests(self):
        self.assertEqual(
            mutations.failed_tests([CARGO]),
            [
                "src/session.rs - session::CompressedArtifact (line 2191) - compile fail",
                "stale_registry_rejected",
            ],
        )

    def test_unittest_names_are_read(self):
        self.assertEqual(
            mutations.failed_tests([UNITTEST]),
            ["test_live_requires_authorized_spend", "test_private_path"],
        )

    def test_names_are_gathered_across_step_logs(self):
        self.assertEqual(
            mutations.failed_tests(["no failure here", CARGO, UNITTEST]),
            sorted(
                mutations.failed_tests([CARGO]) + mutations.failed_tests([UNITTEST])
            ),
        )


if __name__ == "__main__":
    unittest.main()
