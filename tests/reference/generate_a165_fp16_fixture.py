#!/usr/bin/env python3
"""Regenerate the direct-f64 binary16 oracle fixture with Python's struct 'e'."""

import json
import math
import random
import struct
from pathlib import Path


def cases():
    values = [
        0.0,
        -0.0,
        math.ldexp(1.0, -25),
        math.nextafter(math.ldexp(1.0, -25), 0.0),
        math.nextafter(math.ldexp(1.0, -25), math.inf),
        math.ldexp(1.0, -24),
        -math.ldexp(1.0, -24),
        3 * math.ldexp(1.0, -25),
        1023 * math.ldexp(1.0, -24),
        math.ldexp(1.0, -14) - math.ldexp(1.0, -25),
        math.nextafter(math.ldexp(1.0, -14) - math.ldexp(1.0, -25), 0.0),
        math.nextafter(math.ldexp(1.0, -14) - math.ldexp(1.0, -25), math.inf),
        math.ldexp(1.0, -14),
        -math.ldexp(1.0, -14),
        1.0,
        -1.0,
        1.00048828125,
        1.00146484375,
        65504.0,
        math.nextafter(65504.0, math.inf),
        math.nextafter(65520.0, 0.0),
        65520.0,
        math.nextafter(65520.0, math.inf),
        -65504.0,
        -65520.0,
        float("nan"),
        float("inf"),
        float("-inf"),
        1e308,
        -1e308,
    ]
    # These adjacent binary64 values expose double-rounding differences in an
    # implementation that first narrows f64 to f32 before making the half value.
    for midpoint in (1.00048828125, 1.00146484375, -1.00048828125, -1.00146484375):
        values.extend(
            [
                math.nextafter(midpoint, -math.inf),
                midpoint,
                math.nextafter(midpoint, math.inf),
            ]
        )

    rng = random.Random(0xA165)
    exponents = [-1020, -100, -40, -26, -25, -24, -15, -14, -13, -1, 0, 1, 10, 14, 15, 16]
    for _ in range(160):
        exponent = rng.choice(exponents)
        significand = 1.0 + rng.getrandbits(52) / (1 << 52)
        value = math.ldexp(significand, exponent)
        values.append(-value if rng.getrandbits(1) else value)
    return values


def encode_case(value):
    input_bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    try:
        half = struct.unpack("<H", struct.pack("<e", value))[0]
        expected = None if (half & 0x7C00) == 0x7C00 else f"{half:04x}"
    except OverflowError:
        expected = None
    return {"input_f64_bits": f"{input_bits:016x}", "expected_fp16_bits": expected}


if __name__ == "__main__":
    output = Path(__file__).resolve().parents[1] / "fixtures" / "a165_fp16_rounding.json"
    output.write_text(json.dumps([encode_case(value) for value in cases()], indent=2) + "\n")
    print(f"wrote {output}")
