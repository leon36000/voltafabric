#!/usr/bin/env python3
"""A14 exact CPU half-scale admission oracle. No GPU or performance claim."""

import argparse
import json
from pathlib import Path
import random
import struct
import subprocess
import torch
from adapters.skinny_moe import _validate_sm70_scale_domain


def f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def check(binary, scales, factors, k=64):
    raw = torch.tensor(scales, dtype=torch.uint8)
    globals_ = torch.tensor(factors, dtype=torch.float32)
    # Independent literal kernel steps: decode every byte with Torch's FP8 codec,
    # round each scaled global to half, then each product to half. Never weights.
    decoded = raw.view(torch.float8_e4m3fn).float().reshape(-1, k * 2)
    gm = (globals_ * 16384.0).half()
    product = (decoded * gm.float().reshape(-1, 1)).half()
    accepted = bool(torch.isfinite(gm).all() and torch.isfinite(product).all())
    result = subprocess.run(
        [str(binary), "sm70-scale-domain", "-"],
        input=json.dumps(dict(scales=scales, gscales=factors, k=k)),
        capture_output=True,
        text=True,
        timeout=10,
    )
    data = json.loads(result.stdout)
    assert (result.returncode == 0) == data["ok"], data
    assert data["ok"] == accepted, (factors, scales[:8], data, accepted)
    if accepted:
        assert data["result"]["cuda_executed"] is False
        assert data["result"]["scope"] == "CPU_SCALE_ADMISSION_ONLY"
    else:
        assert data["error"]["code"] == "SM70_SCALE_RANGE", data
    try:
        _validate_sm70_scale_domain(raw, globals_, k)
        python_accepted = True
    except ValueError:
        python_accepted = False
    assert python_accepted == accepted, (factors, scales[:8], accepted)
    return 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    factors = [
        f32(1),
        f32(0x007FFFFF),
        f32(0x00800000),
        2**-39,
        2**-10,
        3.2152449421118945e-5,
        0.5,
        1.0,
        4.0,
        65504.0 / 16384.0,
        65520.0 / 16384.0,
        146.125 / 16384.0,
        146.25 / 16384.0,
    ]
    tie = struct.unpack("<I", struct.pack("<f", 65520.0 / 16384.0))[0]
    factors.extend([f32(tie - 1), f32(tie + 1), f32(0x7F7FFFFF)])
    cases = 0
    for encoded in range(127):
        for factor in factors:
            cases += check(args.binary, [encoded] * 128, [factor])
    rng = random.Random(1409)
    for _ in range(96):
        bits = rng.randrange(1, 0x7F800000)
        cases += check(
            args.binary, [rng.randrange(127) for _ in range(256)], [f32(bits), 2**-10]
        )
    # Distinct factors by tile, K boundaries and empty resident planes.
    cases += check(args.binary, [0] * 128 + [126] * 128, [1.0, 2**-10])
    cases += check(args.binary, [126] * 512, [2**-10, 2**-10], k=128)
    cases += check(args.binary, [], [])
    print(
        json.dumps(
            dict(
                status="PASS",
                cases=cases,
                scope="CPU_SCALE_ADMISSION_ONLY",
                cuda_executed=False,
                tolerance="exact acceptance",
            )
        )
    )


if __name__ == "__main__":
    main()
