"""Load-once ownership and failure semantics for compact QPN2 EXEC batches."""

import copy
import struct
import unittest
from unittest.mock import patch

import torch

from adapters import skinny_moe as adapter
from adapters.persistent_gpu_owner import (
    OwnerOutput,
    OwnerRejected,
    OwnerUnknown,
    PersistentQPN2Owner,
)
from scripts import hardware_canary as canary
from tests.adapter.test_a152_compact_rank_dispatch import BINARY, dispatch, stamp
from tests.adapter.test_skinny_adapter import (
    UniformPackedCPUOracle,
    align_reference,
    layer,
)


def resident_info(epoch=11):
    return {
        "model_revision": "a152-model",
        "layer": 3,
        "rank": 2,
        "epoch": epoch,
        "payload_bytes": 13872,
        "physical_experts": [{"id": 99, "local": 700}, {"id": 10, "local": 300}],
        "resident_order": [700, 300],
        "shape": [2, 128, 64, 64, 64],
    }


class SyncCPUOracle(UniformPackedCPUOracle):
    """CPU numerical oracle with the provider completion contract."""

    def __init__(self, *, fail_at=None, reject_before_admission_at=None):
        super().__init__()
        self.device = torch.device("cpu")
        self.admissions = 0
        self.launches = 0
        self.requests = 0
        self.synchronized_launches = 0
        self.weight_addresses = []
        self.fail_at = fail_at
        self.reject_before_admission_at = reject_before_admission_at

    def gemm_qpn2_moe(self, x, codes, scales, global_scales, *args):
        self.requests += 1
        if self.requests == self.reject_before_admission_at:
            self.reject_before_admission_at = None
            raise ValueError("injected provider preflight rejection")
        self.weight_addresses.append(
            (codes.data_ptr(), scales.data_ptr(), global_scales.data_ptr())
        )
        self.admissions += 1
        if self.requests == self.fail_at:
            raise RuntimeError("injected failure after provider admission")
        result = super().gemm_qpn2_moe(x, codes, scales, global_scales, *args)
        self.launches += 1
        return result

    def synchronize(self):
        self.synchronized_launches = self.launches

    def close(self):
        self.synchronize()


def loaded_owner(provider, *, epoch=11, info=None):
    info = resident_info(epoch) if info is None else copy.deepcopy(info)
    template = layer()
    owner_map = torch.full((100,), -1, dtype=torch.int64)
    owner_map[99], owner_map[10] = 0, 1
    calls = []

    def install(target, path, *, expected_info, expected_sha256, device):
        calls.append((path, copy.deepcopy(expected_info), expected_sha256, device))
        target._skinny_moe_resident = adapter._ResidentWeights(
            template.skinny_moe_qc13,
            template.skinny_moe_qs13,
            template.skinny_moe_gs13,
            template.skinny_moe_qc2,
            template.skinny_moe_qs2,
            template.skinny_moe_gs2,
            tuple(expected_info["shape"]),
            4,
            4,
            (
                expected_info["model_revision"],
                expected_info["layer"],
                expected_info["rank"],
                expected_info["epoch"],
                tuple(expected_info["resident_order"]),
            ),
        )
        return target._skinny_moe_resident

    with patch(
        "adapters.persistent_gpu_owner.skinny_moe.load_resident_artifact",
        side_effect=install,
    ):
        owner = PersistentQPN2Owner.load(
            artifact_path="/tmp/a152-artifact.safetensors",
            expected_info=info,
            expected_sha256="a" * 64,
            provider=provider,
            expert_map=owner_map,
        )
    return owner, calls


class PersistentQPN2OwnerTests(unittest.TestCase):
    def test_real_owner_exec_output_crosses_rust_completion_and_collector(self):
        problem = canary.routing_payload(2)
        problem["experts"] = [10, 99]
        problem["routes"] = [
            [dict(expert=99, weight=0.5), dict(expert=10, weight=0.25)],
            [dict(expert=10, weight=0.75), dict(expert=99, weight=0.25)],
        ]
        problem["registry"] = {
            "epoch": 7,
            "copies": [
                dict(expert=99, rank=2, local=900),
                dict(expert=10, rank=2, local=3),
            ],
        }
        problem["preferred_owners"] = {"99": 2}
        problem["resident_order"] = {"2": [900, 3]}
        info = {
            "model_revision": problem["stamp"]["model_revision"],
            "layer": problem["stamp"]["layer"],
            "rank": 2,
            "epoch": problem["stamp"]["epoch"],
            "payload_bytes": 13872,
            "physical_experts": [
                {"id": 99, "local": 900},
                {"id": 10, "local": 3},
            ],
            "resident_order": [900, 3],
            "shape": [2, 128, 64, 64, 64],
        }
        provider = SyncCPUOracle()
        owner, loads = loaded_owner(provider, epoch=7, info=info)
        self.addCleanup(owner.close)
        activations = torch.tensor([[2**-6] * 64, [-(2**-7)] * 64], dtype=torch.float16)
        local_by_expert = {99: 900, 10: 3}

        for attempt in (1, 2):
            with self.subTest(attempt=attempt):
                current = copy.deepcopy(problem)
                current["stamp"]["batch"] = attempt
                context = canary.prepare_rust_canary(BINARY, current)[0]
                # The small Rust fixture uses the same packed codes, group scales,
                # and gate/up factors as the owner's CPU resident. Match its down
                # global factors to the fixture owner before deriving the plan.
                for expert in context["launch"]["weights"]["experts"]:
                    expert["down"]["global_dequant"] = 1.0
                context["stack"] = canary.rust_call(
                    BINARY,
                    "resident-pack",
                    context["launch"]["weights"],
                )
                context["plan"] = canary.rust_call(
                    BINARY, "resident-launch", context["launch"]
                )
                dispatch_value = context["launch"]["dispatch"]
                token_ids = sorted(
                    {entry["slot"]["token"] for entry in dispatch_value["slots"]}
                )
                output = owner.execute(
                    dispatch_value,
                    token_ids=token_ids,
                    activations=activations,
                    attempt=attempt,
                    align=align_reference,
                )
                expected_ids = tuple(
                    (
                        item["token"],
                        item["slot"],
                        item["expert"],
                        item["rank"],
                        item["local"],
                    )
                    for item in context["plan"]["completion_slots"]
                )
                raw = output.as_raw_output(expected_ids)
                if attempt == 1:
                    stale_raw = copy.deepcopy(raw)
                    stale_raw["stamp"]["epoch"] += 1
                    with self.assertRaisesRegex(RuntimeError, "STALE_EXECUTION"):
                        canary.rust_call(
                            BINARY,
                            "resident-completions",
                            {"launch": context["launch"], "output": stale_raw},
                        )
                converted = canary.rust_call(
                    BINARY,
                    "resident-completions",
                    {"launch": context["launch"], "output": raw},
                )
                packed = output.as_raw_output_bytes(expected_ids)
                converted_binary = canary.rust_call(
                    BINARY,
                    "resident-completions",
                    {
                        "launch": context["launch"],
                        "output": {
                            key: packed[key]
                            for key in ("stamp", "rank", "rows", "columns")
                        },
                    },
                    raw_body=packed["fp16_le_bytes"],
                )
                self.assertEqual(
                    converted_binary["completions"], converted["completions"]
                )
                collected = canary.rust_call(
                    BINARY,
                    "collect-routing",
                    {
                        "problem": current,
                        "completions": converted["completions"],
                    },
                )
                routed_raw = {
                    **raw,
                    "attempt": output.attempt,
                    "fp16_bits": list(raw["fp16_bits"]),
                    "slots": [
                        {
                            "token": token,
                            "slot": slot,
                            "expert": expert,
                            "local": local,
                        }
                        for token, slot, expert, _rank, local in expected_ids
                    ],
                }
                direct = canary.rust_call(
                    BINARY,
                    "collect-fp16",
                    {
                        "problem": current,
                        "expected_attempt": attempt,
                        "outputs": [routed_raw],
                    },
                )

                # Independent binary16 decode and route-weight accumulation. The
                # CPU collector must apply each original coefficient exactly once.
                raw_values = {}
                for row, identity in enumerate(expected_ids):
                    token, slot, expert, rank, local = identity
                    self.assertEqual(local, local_by_expert[expert])
                    raw_values[(token, slot, expert, rank, local)] = [
                        struct.unpack(
                            "<e",
                            struct.pack(
                                "<H", raw["fp16_bits"][row * raw["columns"] + column]
                            ),
                        )[0]
                        for column in range(raw["columns"])
                    ]
                oracle = []
                for token, routes in enumerate(current["routes"]):
                    total = [0.0] * current["hidden"]
                    for slot, route in enumerate(routes):
                        values = raw_values[
                            (
                                token,
                                slot,
                                route["expert"],
                                2,
                                local_by_expert[route["expert"]],
                            )
                        ]
                        for column, value in enumerate(values):
                            total[column] += route["weight"] * value
                    oracle.append(total)
                self.assertEqual(collected, oracle)
                self.assertEqual(direct["output"], oracle)
                self.assertEqual(direct["output"], collected)
                self.assertEqual(direct["accepted_rows"], len(expected_ids))
                self.assertEqual(
                    direct["fp16_payload_bytes"],
                    len(raw["fp16_bits"]) * 2,
                )
                self.assertEqual(
                    direct["fp16_payload_bytes"] * 4,
                    sum(len(item["value"]) * 8 for item in converted["completions"]),
                )
                self.assertIs(direct["cuda_executed"], False)

                if attempt == 1:
                    wrong_identity = copy.deepcopy(routed_raw)
                    wrong_identity["slots"][0]["local"] += 1
                    stale_epoch = copy.deepcopy(routed_raw)
                    stale_epoch["stamp"]["epoch"] += 1
                    stale_attempt = copy.deepcopy(routed_raw)
                    stale_attempt["attempt"] += 1
                    nonfinite = copy.deepcopy(routed_raw)
                    nonfinite["fp16_bits"][-1] = 0x7C00
                    duplicate = copy.deepcopy(routed_raw)
                    duplicate["slots"][1] = copy.deepcopy(duplicate["slots"][0])
                    incomplete = copy.deepcopy(routed_raw)
                    incomplete["rows"] -= 1
                    incomplete["slots"].pop()
                    del incomplete["fp16_bits"][-raw["columns"] :]
                    for bad, code in (
                        (wrong_identity, "INVALID_COMPLETION"),
                        (stale_epoch, "STALE_EXECUTION"),
                        (stale_attempt, "STALE_EXECUTION"),
                        (nonfinite, "NON_FINITE_RESULT"),
                        (duplicate, "INVALID_COMPLETION"),
                        (incomplete, "BATCH_INCOMPLETE"),
                    ):
                        with self.subTest(rejected=code):
                            with self.assertRaisesRegex(RuntimeError, code):
                                canary.rust_call(
                                    BINARY,
                                    "collect-fp16",
                                    {
                                        "problem": current,
                                        "expected_attempt": attempt,
                                        "outputs": [bad],
                                    },
                                )
                    with self.assertRaisesRegex(RuntimeError, "INVALID_INPUT"):
                        canary.rust_call(
                            BINARY,
                            "collect-fp16",
                            {
                                "problem": current,
                                "expected_attempt": 0,
                                "outputs": [routed_raw],
                            },
                        )

        self.assertEqual(len(loads), 1)
        self.assertEqual(len(provider.weight_addresses), 4)
        self.assertEqual(provider.weight_addresses[0], provider.weight_addresses[2])
        self.assertEqual(provider.weight_addresses[1], provider.weight_addresses[3])
        self.assertEqual(provider.synchronized_launches, provider.launches)

    def test_owner_output_rejects_missing_duplicate_and_malformed_identity_sets(self):
        identities = ((1, 0, 10, 2, 300), (4, 1, 99, 2, 700))
        output = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=1,
            identities=identities,
            rows=2,
            columns=2,
            fp16_le_bytes=struct.pack("<4H", 0x3C00, 0x3C00, 0x4000, 0x4000),
        )
        with self.assertRaises(OwnerRejected):
            output.as_raw_output(identities[:1])
        with self.assertRaises(OwnerRejected):
            output.as_raw_output((identities[0], identities[0]))
        with self.assertRaises(OwnerRejected):
            output.as_raw_output((identities[0], (4, 1, 99, 2, True)))
        raw = output.as_raw_output(tuple(reversed(identities)))
        self.assertEqual(raw["fp16_bits"], (0x4000, 0x4000, 0x3C00, 0x3C00))

    def test_owner_output_exports_identity_ordered_little_endian_fp16_bytes(self):
        identities = ((1, 0, 10, 2, 300), (4, 1, 99, 2, 700))
        original = struct.pack("<4H", 0x3C00, 0x0001, 0x8000, 0x7BFF)
        output = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=1,
            identities=identities,
            rows=2,
            columns=2,
            fp16_le_bytes=original,
        )

        packed = output.as_raw_output_bytes(tuple(reversed(identities)))
        self.assertEqual(
            packed["fp16_le_bytes"], struct.pack("<4H", 0x8000, 0x7BFF, 0x3C00, 0x0001)
        )
        self.assertEqual(packed["rows"], 2)
        self.assertEqual(packed["columns"], 2)
        self.assertEqual(output.fp16_bits, (0x3C00, 0x0001, 0x8000, 0x7BFF))
        with self.assertRaises(OwnerRejected):
            output.as_raw_output_bytes((identities[0], identities[0]))
        malformed = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=2,
            identities=identities,
            rows=2,
            columns=2,
            fp16_le_bytes=original[:-1],
        )
        with self.assertRaises(OwnerRejected):
            malformed.as_raw_output_bytes(identities)
        nonfinite = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=2,
            identities=identities,
            rows=2,
            columns=2,
            fp16_le_bytes=struct.pack("<4H", 0x3C00, 0x7C00, 0x8000, 0x7BFF),
        )
        with self.assertRaises(OwnerRejected):
            nonfinite.as_raw_output_bytes(identities)
        empty = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=2,
            identities=(),
            rows=0,
            columns=64,
            fp16_le_bytes=b"",
        )
        self.assertEqual(empty.as_raw_output_bytes(())["fp16_le_bytes"], b"")
        invalid_empty_shape = OwnerOutput(
            stamp=stamp(),
            rank=2,
            attempt=2,
            identities=(),
            rows=0,
            columns=65_537,
            fp16_le_bytes=b"",
        )
        with self.assertRaises(OwnerRejected):
            invalid_empty_shape.as_raw_output_bytes(())

    def test_one_install_two_execs_reuse_compressed_planes_and_keep_raw_identity(self):
        provider = SyncCPUOracle()
        owner, loads = loaded_owner(provider)
        x1 = torch.stack(
            (torch.full((64,), 0.015625), torch.full((64,), -0.0078125))
        ).to(torch.float16)
        first = owner.execute(
            dispatch(),
            token_ids=[1, 4],
            activations=x1,
            attempt=1,
            align=align_reference,
        )
        x2 = torch.stack(
            (torch.full((64,), -0.03125), torch.full((64,), 0.00390625))
        ).to(torch.float16)
        second_dispatch = dispatch(stamp=stamp(batch=8, epoch=11))
        second = owner.execute(
            second_dispatch,
            token_ids=[1, 4],
            activations=x2,
            attempt=2,
            align=align_reference,
        )

        self.assertEqual(len(loads), 1)
        self.assertEqual(len(provider.calls), 4)
        self.assertEqual(provider.synchronized_launches, provider.launches)
        self.assertEqual(provider.weight_addresses[0], provider.weight_addresses[2])
        self.assertEqual(provider.weight_addresses[1], provider.weight_addresses[3])
        self.assertEqual(
            first.identities,
            ((1, 1, 99, 2, 700), (1, 2, 10, 2, 300), (4, 0, 10, 2, 300)),
        )
        self.assertEqual(first.rows, 3)
        self.assertEqual(first.columns, 64)
        self.assertEqual(len(first.fp16_bits), 3 * 64)
        self.assertEqual(second.attempt, 2)
        self.assertEqual(second.stamp["batch"], 8)
        self.assertEqual(owner.state, "READY")

        # Independent scalar SwiGLU reference. Returned half bit patterns are
        # unweighted; coefficients remain for Rust's collector.
        decoded = (
            torch.tensor(first.fp16_bits, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.float16)
            .reshape(3, 64)
        )
        expected = []
        row_by_token = {1: x1[0], 4: x1[1]}
        gate_scales = {99: 0.25, 10: 0.5}
        for token, _, expert, _, _ in first.identities:
            gate_scale = gate_scales[expert]
            h1 = (row_by_token[token].float().sum() * gate_scale).half()
            # Uniform oracle's up plane is twice the gate plane for these fixtures.
            up = (row_by_token[token].float().sum() * gate_scale * 2).half()
            activation = (torch.nn.functional.silu(h1) * up).half()
            h2 = (64 * activation).half()
            expected.append(torch.full((64,), float(h2), dtype=torch.float16))
        torch.testing.assert_close(decoded, torch.stack(expected), rtol=0, atol=0)

    def test_rebind_requires_new_epoch_and_old_epoch_exec_is_refused(self):
        provider = SyncCPUOracle()
        owner, _ = loaded_owner(provider, epoch=8)
        owner.rebind_epoch(12)
        self.assertEqual(owner.epoch, 12)
        with self.assertRaises(OwnerRejected):
            owner.rebind_epoch(12)
        with self.assertRaises(OwnerRejected):
            owner.rebind_epoch(11)
        old = dispatch(stamp=stamp(epoch=11))
        with self.assertRaises(OwnerRejected):
            owner.execute(
                old,
                token_ids=[1, 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=1,
                align=align_reference,
            )
        self.assertEqual(provider.calls, [])
        valid = dispatch(stamp=stamp(epoch=12))
        result = owner.execute(
            valid,
            token_ids=[1, 4],
            activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
            attempt=1,
            align=align_reference,
        )
        self.assertEqual(result.stamp["epoch"], 12)

    def test_failure_after_provider_admission_is_unknown_and_poisoned_without_replay(
        self,
    ):
        provider = SyncCPUOracle(fail_at=2)
        owner, _ = loaded_owner(provider)
        with self.assertRaises(OwnerUnknown):
            owner.execute(
                dispatch(),
                token_ids=[1, 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=1,
                align=align_reference,
            )
        self.assertEqual(owner.state, "POISONED_UNKNOWN")
        calls = len(provider.calls)
        with self.assertRaises(OwnerUnknown):
            owner.execute(
                dispatch(stamp=stamp(batch=8)),
                token_ids=[1, 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=2,
                align=align_reference,
            )
        self.assertEqual(len(provider.calls), calls)
        owner.close()
        self.assertEqual(owner.state, "POISONED_UNKNOWN_CLOSED")
        with self.assertRaises(OwnerUnknown):
            owner.execute(
                dispatch(stamp=stamp(batch=9)),
                token_ids=[1, 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=3,
                align=align_reference,
            )
        self.assertEqual(len(provider.calls), calls)

    def test_empty_dispatch_reuses_owner_without_entering_provider(self):
        provider = SyncCPUOracle()
        owner, loads = loaded_owner(provider)
        empty = dispatch(slots=[])
        result = owner.execute(
            empty,
            token_ids=[],
            activations=torch.empty((0, 64), dtype=torch.float16),
            attempt=1,
            align=align_reference,
        )
        self.assertEqual(len(loads), 1)
        self.assertEqual(result.identities, ())
        self.assertEqual((result.rows, result.columns), (0, 64))
        self.assertEqual(result.fp16_bits, ())
        self.assertEqual(provider.admissions, 0)
        self.assertEqual(provider.calls, [])

    def test_provider_preflight_rejection_is_known_and_does_not_poison_owner(self):
        provider = SyncCPUOracle(reject_before_admission_at=1)
        owner, _ = loaded_owner(provider)
        with self.assertRaises(OwnerRejected):
            owner.execute(
                dispatch(),
                token_ids=[1, 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=1,
                align=align_reference,
            )
        self.assertEqual(provider.admissions, 0)
        self.assertEqual(owner.state, "READY")
        result = owner.execute(
            dispatch(stamp=stamp(batch=8)),
            token_ids=[1, 4],
            activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
            attempt=2,
            align=align_reference,
        )
        self.assertEqual(result.attempt, 2)
        self.assertEqual(provider.admissions, 2)

    def test_unhashable_token_ids_are_rejected_before_attempt_admission(self):
        provider = SyncCPUOracle()
        owner, _ = loaded_owner(provider)
        with self.assertRaises(OwnerRejected):
            owner.execute(
                dispatch(),
                token_ids=[[1], 4],
                activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
                attempt=1,
                align=align_reference,
            )
        self.assertEqual(provider.admissions, 0)
        result = owner.execute(
            dispatch(),
            token_ids=[1, 4],
            activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
            attempt=1,
            align=align_reference,
        )
        self.assertEqual(result.attempt, 1)

    def test_invalid_duplicate_route_is_rejected_before_provider_and_attempt_consumed(
        self,
    ):
        provider = SyncCPUOracle()
        owner, _ = loaded_owner(provider)
        bad = dispatch()
        bad["slots"].append(copy.deepcopy(bad["slots"][0]))
        bad["slots"][-1]["slot"]["slot"] = 3
        with self.assertRaises(OwnerRejected):
            owner.execute(
                bad,
                token_ids=[1, 4],
                activations=torch.ones((2, 64), dtype=torch.float16),
                attempt=1,
                align=align_reference,
            )
        self.assertEqual(provider.calls, [])
        good = owner.execute(
            dispatch(),
            token_ids=[1, 4],
            activations=torch.full((2, 64), 2**-6, dtype=torch.float16),
            attempt=2,
            align=align_reference,
        )
        self.assertEqual(good.attempt, 2)


if __name__ == "__main__":
    unittest.main()
