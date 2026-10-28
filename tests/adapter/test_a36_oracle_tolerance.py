"""The oracles' declared tolerance must be the tolerance they apply.

A36 found that scripts/crosscheck.py could compare at 1e-4 while its receipt kept
claiming {'atol': 2e-11, 'rtol': 2e-11} and its own sensitivity control still
passed, because that control corrupts at 1e-3. Nothing in the 30-step gate
noticed. These tests tie each oracle's claim to its measured behaviour.
"""

import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import crosscheck  # noqa: E402
import planner_crosscheck  # noqa: E402

DOCUMENTED_MOE_TOLERANCE = 2e-11
DOCUMENTED_PLANNER_ATOL = 2e-9


class OracleToleranceTests(unittest.TestCase):
    def test_moe_oracle_applies_exactly_the_tolerance_it_reports(self):
        source = (ROOT / "scripts/crosscheck.py").read_text()
        applied = re.search(
            r"math\.isclose\(x, y, rel_tol=([0-9.e-]+), abs_tol=([0-9.e-]+)\)", source
        )
        self.assertIsNotNone(applied, "the comparison line changed shape")
        reported = re.search(
            r"'tolerance':\{'atol':([0-9.e-]+),'rtol':([0-9.e-]+)\}", source
        )
        self.assertIsNotNone(reported, "the reported tolerance changed shape")
        applied_values = {float(applied.group(1)), float(applied.group(2))}
        reported_values = {float(reported.group(1)), float(reported.group(2))}
        self.assertEqual(
            applied_values,
            reported_values,
            "the receipt would claim a tolerance the oracle does not apply",
        )
        self.assertEqual(applied_values, {DOCUMENTED_MOE_TOLERANCE})

    def test_moe_oracle_rejects_an_error_above_its_tolerance(self):
        # A29 measured that a 1e-9 relative perturbation is detected; pin it, so a
        # loosened tolerance cannot pass unnoticed.
        self.assertEqual(crosscheck.close([[1.0]], [[1.0]]), 0.0)
        with self.assertRaises(AssertionError):
            crosscheck.close([[1.0]], [[1.0 + 1e-9]])
        with self.assertRaises(AssertionError):
            crosscheck.close([[1.0]], [[1.001]])
        # Just inside the declared tolerance must still be accepted, so the test
        # pins a threshold rather than demanding exactness.
        self.assertGreater(crosscheck.close([[1.0]], [[1.0 + 1e-12]]), 0.0)

    def test_moe_oracle_rejects_a_nonfinite_output(self):
        with self.assertRaises(AssertionError):
            crosscheck.close([[float("nan")]], [[float("nan")]])

    def test_planner_oracle_atol_matches_its_documented_value_and_behaviour(self):
        self.assertEqual(planner_crosscheck.ATOL, DOCUMENTED_PLANNER_ATOL)
        self.assertEqual(planner_crosscheck.assert_close(1.0, 1.0), 0.0)
        with self.assertRaises(AssertionError):
            planner_crosscheck.assert_close(1.0, 1.0 + 1e-6)
        self.assertLessEqual(
            planner_crosscheck.assert_close(1.0, 1.0 + 1e-10),
            planner_crosscheck.ATOL,
        )

    def test_checkpoint_oracle_is_bit_exact_in_claim_and_application(self):
        # A37: this oracle could compare at 1e-6 while its receipt still reported
        # atol 0.0, and the reverse drift was equally unnoticed.
        source = (ROOT / "scripts/checkpoint_crosscheck.py").read_text()
        applied = re.findall(
            r"assert_close\([^)]*rtol=([0-9.e-]+),\s*atol=([0-9.e-]+)\)", source
        )
        self.assertTrue(applied, "the checkpoint comparison changed shape")
        for rtol, atol in applied:
            self.assertEqual(
                (float(rtol), float(atol)),
                (0.0, 0.0),
                "the checkpoint codec is compared bit for bit",
            )
        reported = dict(re.findall(r'"(atol|rtol)":\s*([0-9.e-]+)', source))
        self.assertEqual(
            {k: float(v) for k, v in reported.items()},
            {"atol": 0.0, "rtol": 0.0},
            "the receipt would claim a tolerance the oracle does not apply",
        )

    def test_no_structurally_exact_oracle_introduces_a_nonzero_tolerance(self):
        """The oracles that compare structurally declare exactness. Any nonzero
        numeric tolerance appearing in one of them is a silent loosening, which is
        what A36 and A37 showed the gate cannot see."""
        exact = (
            "resident_crosscheck.py",
            "completion_crosscheck.py",
            "launch_crosscheck.py",
            "inspection_crosscheck.py",
            "a13_crosscheck.py",
            "a14_crosscheck.py",
            "a16_planner_crosscheck.py",
            "full_expert_crosscheck.py",
            "checkpoint_crosscheck.py",
        )
        for name in exact:
            source = (ROOT / "scripts" / name).read_text()
            with self.subTest(oracle=name):
                for keyword, value in re.findall(
                    r"\b(atol|rtol)[\"']?[:=]\s*([0-9.e-]+)", source
                ):
                    self.assertEqual(
                        float(value),
                        0.0,
                        f"{name} declares a nonzero {keyword}",
                    )


if __name__ == "__main__":
    unittest.main()
