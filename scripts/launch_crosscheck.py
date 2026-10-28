#!/usr/bin/env python3
"""Crosscheck Rust compact launch plans with independent row semantics and real C validator."""

from __future__ import annotations
import argparse
import copy
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.hardware_canary import Params
from scripts.resident_crosscheck import invoke


def semantic(plan, dispatch):
    rows = sorted(
        dispatch["slots"], key=lambda r: (r["slot"]["token"], r["slot"]["slot"])
    )
    assert plan["completion_slots"] == [r["slot"] for r in rows]
    assert plan["source_tokens"] == [r["slot"]["token"] for r in rows]
    seen = []
    for block, expert in enumerate(plan["block_expert"]):
        assert expert >= 0
        for index in plan["sorted_slots"][block * 8 : block * 8 + 8]:
            if index < len(rows):
                assert rows[index]["physical_expert"] == expert
                seen.append(index)
            else:
                assert index == len(rows)
    assert sorted(seen) == list(range(len(rows)))
    assert plan["ntpp"] == len(plan["sorted_slots"]) == 8 * len(plan["block_expert"])
    assert plan["block_expert"] == sorted(plan["block_expert"])


def params(projection, plan):
    p = Params()
    p.abi_version = 1
    p.struct_size = C.sizeof(Params)
    for key in (
        "experts",
        "n",
        "k",
        "slots",
        "input_rows",
        "in_div",
        "splitk",
        "nacc",
        "codes_bytes",
        "scales_bytes",
        "gscales_count",
        "x_count",
        "y_count",
    ):
        setattr(p, key, projection[key])
    blocks = (C.c_int32 * len(plan["block_expert"]))(*plan["block_expert"])
    indices = (C.c_int32 * len(plan["sorted_slots"]))(*plan["sorted_slots"])
    p.block_expert_count = p.block_expert_host_count = len(blocks)
    p.sorted_slots_count = p.sorted_slots_host_count = len(indices)
    p.ntpp_count = 1
    p.ntpp_host = plan["ntpp"]
    p.block_expert_host = blocks
    p.sorted_slots_host = indices
    types = dict(Params._fields_)
    # Opaque, aligned, disjoint addresses only. The CPU validator never dereferences them.
    address = 0x10000000
    for name, size in [
        ("codes", p.codes_bytes),
        ("scales", p.scales_bytes),
        ("gscales", p.gscales_count * 4),
        ("x", p.x_count * 2),
        ("y", p.y_count * 2),
        ("block_expert", len(blocks) * 4),
        ("sorted_slots", len(indices) * 4),
        ("ntpp", 4),
    ]:
        setattr(p, name, C.cast(C.c_void_p(address), types[name]))
        address = (address + size + 4095) & ~4095
    return p, (blocks, indices)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary", type=Path, required=True)
    ap.add_argument("--validator", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    library = C.CDLL(str(args.validator.resolve()))
    validate = library.vf_qpn2_moe_validate
    validate.argtypes = [C.POINTER(Params)]
    validate.restype = C.c_int
    sample = json.loads(
        (ROOT / "fixtures/checkpoint_qwen36_expert0_crop.json").read_text()
    )
    planes = sample["planes"]

    def plane(name):
        return {
            k: copy.deepcopy(planes[name][k])
            for k in ("rows", "columns", "codes", "scales", "global_dequant")
        }

    weights = dict(
        model_revision=sample["source"]["revision"],
        layer=0,
        rank=2,
        registry=dict(
            epoch=7,
            copies=[
                dict(expert=10, rank=2, local=3),
                dict(expert=99, rank=2, local=900),
            ],
        ),
        resident_order=[900, 3],
        experts=[
            dict(
                id=i,
                local=l,
                gate=plane("gate_proj"),
                up=plane("up_proj"),
                down=plane("down_proj"),
            )
            for i, l in [(10, 3), (99, 900)]
        ],
    )
    stamp = dict(
        run_id="launch-oracle",
        model_revision=weights["model_revision"],
        plan_id="fixed",
        layer=0,
        batch=1,
        epoch=7,
    )
    rng = random.Random(731902)
    validator_calls = 0
    negative = 0
    mapping_sensitivity = False
    cases = []
    for case in range(66):
        tokens = case if case < 2 else rng.randrange(2, 65)
        slots = []
        for token in range(tokens):
            for slot, (expert, local, physical) in enumerate(
                [(10, 3, 1), (99, 900, 0)]
            ):
                if rng.randrange(4) != 0 or case == 1:
                    slots.append(
                        dict(
                            slot=dict(
                                token=token,
                                slot=slot,
                                expert=expert,
                                rank=2,
                                local=local,
                                weight=rng.choice([0.0, 0.25, 0.250000000001, 1.0]),
                            ),
                            physical_expert=physical,
                        )
                    )
        rng.shuffle(slots)
        d = dict(
            stamp=stamp,
            rank=2,
            hidden=64,
            tokens=tokens,
            resident_order=[900, 3],
            slots=slots,
        )
        request = dict(
            weights=weights,
            dispatch=d,
            expected_stamp=copy.deepcopy(stamp),
            options=dict(splitk13=4, splitk2=4, max_workspace_bytes=128 * 1024 * 1024),
        )
        plan = invoke(args.binary, "resident-launch", request)
        semantic(plan, d)
        assert plan["cuda_executed"] is False
        assert plan["skip"] == (len(slots) == 0)
        assert plan["stamp"] == stamp
        if not slots:
            assert plan["workspace_bytes"] == 0 and plan["projection13"] is None
            cases.append(dict(slots=0, skip=True))
            continue
        expected_bytes = (
            len(slots) * (64 + 128 + 64 + 64) * 2 + len(plan["block_expert"]) * 36 + 4
        )
        assert plan["workspace_bytes"] == expected_bytes
        exact = copy.deepcopy(request)
        exact["options"]["max_workspace_bytes"] = expected_bytes
        assert invoke(args.binary, "resident-launch", exact) == plan
        exact["options"]["max_workspace_bytes"] -= 1
        assert (
            invoke(args.binary, "resident-launch", exact, ok=False)
            == "CAPACITY_EXCEEDED"
        )
        negative += 1
        for key in ("projection13", "projection2"):
            proj = plan[key]
            assert proj["slots"] == proj["input_rows"] == len(slots)
            assert proj["in_div"] == proj["nacc"] == 1
            assert proj["x_count"] == len(slots) * proj["k"]
            assert proj["y_count"] == len(slots) * proj["n"]
            p, keepalive = params(proj, plan)
            assert validate(C.byref(p)) == 0
            validator_calls += 1
            corrupt = copy.deepcopy(plan)
            valid = [i for i, v in enumerate(corrupt["sorted_slots"]) if v < len(slots)]
            if len(valid) > 1:
                corrupt["sorted_slots"][valid[1]] = corrupt["sorted_slots"][valid[0]]
                bad, hold = params(proj, corrupt)
                assert validate(C.byref(bad)) == -6
                validator_calls += 1
            corrupt = copy.deepcopy(plan)
            corrupt["sorted_slots"][valid[0]] = len(slots)
            bad, hold = params(proj, corrupt)
            assert validate(C.byref(bad)) == -6
            validator_calls += 1
            corrupt = copy.deepcopy(plan)
            corrupt["block_expert"][0] = proj["experts"]
            bad, hold = params(proj, corrupt)
            assert validate(C.byref(bad)) == -5
            validator_calls += 1
        if not mapping_sensitivity and len(set(plan["block_expert"])) == 2:
            corrupt = copy.deepcopy(plan)
            corrupt["block_expert"][0] ^= 1
            p, keepalive = params(plan["projection13"], corrupt)
            assert (
                validate(C.byref(p)) == 0
            )  # C coverage cannot know routing ownership.
            validator_calls += 1
            try:
                semantic(corrupt, d)
            except AssertionError:
                mapping_sensitivity = True
            else:
                raise AssertionError(
                    "semantic oracle failed to detect wrong valid expert"
                )
        for field in stamp:
            bad = copy.deepcopy(request)
            value = stamp[field]
            bad["expected_stamp"][field] = (
                value + 1 if isinstance(value, int) else value + "-stale"
            )
            assert (
                invoke(args.binary, "resident-launch", bad, ok=False)
                == "STALE_EXECUTION"
            )
            negative += 1
        cases.append(dict(slots=len(slots), skip=False, workspace_bytes=expected_bytes))
    assert mapping_sensitivity
    report = dict(
        status="PASS",
        evidence_level="CPU_FFI_LAUNCH_METADATA_ONLY",
        cases=len(cases),
        ffi_validator_calls=validator_calls,
        negative_cli_cases=negative,
        semantic_mapping_sensitivity=mapping_sensitivity,
        abi_size=C.sizeof(Params),
        validator_sha256=hashlib.sha256(args.validator.read_bytes()).hexdigest(),
        real_crop_source_revision=sample["source"]["revision"],
        scope="synthetic routes/IDs with real expert0 crops; opaque addresses checked by CPU C validator only",
        cuda_executed=False,
        device_addresses_verified=False,
        performance_claim=None,
    )
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
