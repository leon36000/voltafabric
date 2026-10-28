"""A53-R35: every Rust mutant of scripts/rust_mutations.py must be rustfmt-clean.

verify.py stops at its first failing step and runs rustfmt before any test step, so
`rust_mutations --gate-python` could only report a mutant whose edit rustfmt rejects as
untested (A53-R34). R02, P01 and Q04 were such mutants. Their clippy cleanliness, the
other cosmetic step, is checked by r35/mutant_hygiene.py (one clippy build per mutant).
"""

import ast
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rust_cases():
    tree = ast.parse((ROOT / "scripts" / "rust_mutations.py").read_text())
    cases = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign) and node.targets[0].id == "CASES"
    )
    return [case for case in cases if case[1].endswith(".rs")]


class MutantHygiene(unittest.TestCase):
    def test_every_rust_mutant_is_rustfmt_clean(self):
        rustfmt = shutil.which("rustfmt")
        self.assertIsNotNone(
            rustfmt, "rustfmt is required, as for the gate's rustfmt step"
        )
        unclean = []
        with tempfile.TemporaryDirectory() as work:
            for case in rust_cases():
                name, rel, clause = case[0], case[1], case[2]
                base = (ROOT / rel).read_text()
                self.assertEqual(base.count(clause), 1, name)
                path = Path(work) / f"{name}.rs"
                path.write_text(base.replace(clause, case[5] if len(case) > 5 else ""))
                run = subprocess.run(
                    [rustfmt, "--edition", "2021", "--check", str(path)],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if run.returncode:
                    unclean.append(name)
        self.assertEqual(unclean, [])


if __name__ == "__main__":
    unittest.main()
