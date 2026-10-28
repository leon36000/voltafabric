"""CPU contract tests for grouped adapter guard flags."""

import unittest

import torch

from adapters.skinny_moe import _collect_any_flags


class GroupedGuardFlagTests(unittest.TestCase):
    def test_collects_each_predicate_without_changing_its_truth(self):
        flags = _collect_any_flags(
            torch.tensor([False, False]),
            torch.tensor([False, True]),
            torch.tensor([False]),
        )

        self.assertEqual(flags, (False, True, False))

    def test_empty_predicates_are_false(self):
        flags = _collect_any_flags(
            torch.tensor([], dtype=torch.bool),
            torch.tensor([False, False]),
        )

        self.assertEqual(flags, (False, False))


if __name__ == "__main__":
    unittest.main()
