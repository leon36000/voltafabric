"""Route validation preserves identities without discarding a compact dispatch."""

import copy
import itertools
import unittest
from types import SimpleNamespace

import torch

from adapters import skinny_moe as adapter


def binding_fixture():
    stamp = dict(
        run_id="a143", model_revision="m", plan_id="p", layer=0, batch=1, epoch=9
    )
    layer = SimpleNamespace(skinny_moe_shape=(2, 128, 64, 64, 64), expert_map=None)
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5]])
    rank = dict(
        stamp=stamp, rank=0, hidden=64, tokens=2, resident_order=[10, 20], slots=[]
    )
    for token in range(2):
        for slot in range(2):
            expert = int(ids[token, slot])
            rank["slots"].append(
                dict(
                    physical_expert=expert,
                    slot=dict(
                        token=token,
                        slot=slot,
                        expert=expert,
                        rank=0,
                        local=[10, 20][expert],
                        weight=float(weights[token, slot]),
                    ),
                )
            )
    return layer, rank, stamp, ids, weights


class RouteValidationTests(unittest.TestCase):
    def test_binding_does_not_compute_unused_compact_slot_positions(self):
        fixture = binding_fixture()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            mask = adapter.bind_rank_dispatch(*fixture)
        operations = {event.key: event.count for event in prof.key_averages()}
        self.assertEqual(mask.tolist(), [[True, True], [True, True]])
        self.assertEqual(operations.get("aten::nonzero", 0), 0, operations)
        self.assertEqual(operations.get("aten::div", 0), 0, operations)

    def test_empty_dispatch_still_rejects_invalid_unselected_routes(self):
        layer, rank, stamp, ids, weights = binding_fixture()
        rank["slots"] = []
        for invalid in ([[0, 0], [1, 0]], [[-1, 1], [1, 0]], [[0, 2], [1, 0]]):
            with self.subTest(ids=invalid), self.assertRaises(ValueError):
                adapter.bind_rank_dispatch(
                    layer, rank, stamp, torch.tensor(invalid), weights
                )
        layer.expert_map = torch.tensor([0, 0])
        with self.assertRaisesRegex(ValueError, "alias"):
            adapter.bind_rank_dispatch(layer, rank, stamp, ids, weights)

    def test_slot_permutations_and_empty_dispatch_keep_the_exact_mask(self):
        layer, rank, stamp, ids, weights = binding_fixture()
        for slots in (
            rank["slots"],
            list(reversed(rank["slots"])),
            rank["slots"][::2],
            [],
        ):
            dispatch = copy.deepcopy(rank)
            dispatch["slots"] = slots
            expected = [[False, False], [False, False]]
            for routed in slots:
                expected[routed["slot"]["token"]][routed["slot"]["slot"]] = True
            self.assertEqual(
                adapter.bind_rank_dispatch(
                    layer, dispatch, stamp, ids, weights
                ).tolist(),
                expected,
            )

    def test_sparse_compaction_matches_an_independent_enumerated_slot_oracle(self):
        mapping = torch.full((9,), -1, dtype=torch.int64)
        mapping[2], mapping[8] = 1, 0
        physical = {2: 1, 4: -1, 8: 0}
        for first, second in itertools.product(
            itertools.permutations(physical, 2), repeat=2
        ):
            rows = [first, second]
            ids = torch.tensor(rows)
            for bits in itertools.product((False, True), repeat=4):
                selected = torch.tensor(bits).reshape(2, 2)
                if any(
                    chosen and physical[expert] < 0
                    for expert, chosen in zip(first + second, bits)
                ):
                    with self.assertRaisesRegex(ValueError, "no local resident"):
                        adapter._select_local_slots(ids, mapping, 2, selected)
                    continue
                expected = [
                    (i, i // 2, physical[expert])
                    for i, (expert, chosen) in enumerate(zip(first + second, bits))
                    if chosen
                ]
                result = adapter._select_local_slots(ids, mapping, 2, selected)
                self.assertEqual(
                    list(zip(*(tensor.tolist() for tensor in result))), expected
                )


if __name__ == "__main__":
    unittest.main()
