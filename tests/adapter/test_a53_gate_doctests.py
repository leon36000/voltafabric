"""A53-R30: the CPU gate (scripts/verify.py) must run the crate's doctests.

`cargo test --all-targets` never runs doctests, so the compile_fail blocks that keep the
fields of CompressedArtifact private (A53-R20) were outside the gate: a change making a
field public again passed every gate step.
"""

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cargo_steps():
    """(label, argv) of every `cargo` step written in verify.py."""
    tree = ast.parse((ROOT / "scripts" / "verify.py").read_text())
    steps = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[1], ast.List)
        ):
            try:
                label, argv = ast.literal_eval(node)
            except ValueError:
                continue
            if isinstance(label, str) and argv and argv[0] == "cargo":
                steps.append((label, argv))
    return steps


class GateDoctests(unittest.TestCase):
    def test_the_gate_runs_the_doctests(self):
        doc = [
            (label, argv)
            for label, argv in cargo_steps()
            if argv[:2] == ["cargo", "test"] and "--doc" in argv
        ]
        self.assertEqual(
            doc, [("rust_doctests", ["cargo", "test", "--doc", "--locked"])]
        )

    def test_the_doctests_the_step_runs_exist(self):
        # Without them the step would pass on nothing: one compile_fail block per field.
        text = (ROOT / "src" / "session.rs").read_text()
        self.assertEqual(text.count("```compile_fail"), 3)


if __name__ == "__main__":
    unittest.main()
