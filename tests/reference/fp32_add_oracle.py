#!/usr/bin/env python3
"""Independent exact-rational oracle for sequential IEEE-754 binary32 addition.

Inputs and outputs use 8-digit hexadecimal bit strings. No NumPy or Rust code is
used. Any non-finite intermediate (including rounded overflow) fails the entire
case, matching the proposed batch-level fail-closed contract.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
CORPUS = HERE.parents[1] / "fixtures" / "preweighted_fp32_sum_cases.json"
MASK = 0xFFFFFFFF


def decode(bits: int) -> Fraction:
    sign = -1 if bits >> 31 else 1
    exp = (bits >> 23) & 0xFF
    frac = bits & 0x7FFFFF
    if exp == 0xFF:
        raise ValueError("non-finite binary32 input")
    if exp == 0:
        value = Fraction(frac, 1 << 149)
    else:
        significand = (1 << 23) | frac
        power = exp - 127 - 23
        value = (
            Fraction(significand << power, 1)
            if power >= 0
            else Fraction(significand, 1 << -power)
        )
    return sign * value


def round_even(value: Fraction) -> int:
    q, r = divmod(value.numerator, value.denominator)
    twice = 2 * r
    return q + (twice > value.denominator or (twice == value.denominator and q & 1))


def encode(value: Fraction, zero_sign: int = 0) -> int:
    """Correctly round an exact rational to binary32, RNE; raise on infinity."""
    if value == 0:
        return 0x80000000 if zero_sign else 0
    sign = 0x80000000 if value < 0 else 0
    x = abs(value)
    # Determine floor(log2(x)) using integer comparisons only.
    e = x.numerator.bit_length() - x.denominator.bit_length()
    if e >= 0:
        if x < (1 << e):
            e -= 1
    elif x < Fraction(1, 1 << -e):
        e -= 1
    if e < -126:
        mant = round_even(x * (1 << 149))
        if mant == 0:
            return sign
        if mant >= (1 << 23):
            return sign | (1 << 23)  # minimum normal
        return sign | mant
    shift = 23 - e
    scaled = x * (1 << shift) if shift >= 0 else x / (1 << -shift)
    mant = round_even(scaled)
    if mant == (1 << 24):
        mant >>= 1
        e += 1
    if e > 127:
        raise OverflowError("binary32 rounded to infinity")
    return sign | ((e + 127) << 23) | (mant - (1 << 23))


def add_bits(a: int, b: int) -> int:
    av, bv = decode(a), decode(b)
    total = av + bv
    if total == 0:
        # RNE exact cancellation is +0. Only -0 + -0 preserves a negative zero.
        neg_zero = av == 0 and bv == 0 and (a >> 31) and (b >> 31)
        return encode(total, int(bool(neg_zero)))
    return encode(total)


def accumulate(terms: list[str]) -> tuple[str, str | None]:
    acc = 0  # contract starts at positive zero
    try:
        for text in terms:
            bits = int(text, 16)
            decode(bits)
            acc = add_bits(acc, bits)
        return "ok", f"{acc:08x}"
    except OverflowError:
        return "overflow", None


CASES = [
    {
        "id": "single_roundtrip",
        "terms": ["3f800000"],
        "status": "ok",
        "result": "3f800000",
    },
    {
        "id": "half_ulp_even_down",
        "terms": ["3f800000", "33800000"],
        "status": "ok",
        "result": "3f800000",
    },
    {
        "id": "half_ulp_even_up",
        "terms": ["3f800001", "33800000"],
        "status": "ok",
        "result": "3f800002",
    },
    {
        "id": "large_small_cancel_fixed",
        "terms": ["4b800000", "3f800000", "cb800000"],
        "status": "ok",
        "result": "00000000",
    },
    {
        "id": "large_small_cancel_permuted",
        "terms": ["4b800000", "cb800000", "3f800000"],
        "status": "ok",
        "result": "3f800000",
    },
    {
        "id": "subnormal_min_plus_min",
        "terms": ["00000001", "00000001"],
        "status": "ok",
        "result": "00000002",
    },
    {
        "id": "subnormal_halfway_to_zero",
        "terms": ["00000001", "80000001"],
        "status": "ok",
        "result": "00000000",
    },
    {
        "id": "signed_negative_zero_inputs",
        "terms": ["80000000", "80000000"],
        "status": "ok",
        "result": "00000000",
    },
    {
        "id": "negative_zero_plus_negative_value_then_cancel",
        "terms": ["80000000", "bf800000", "3f800000"],
        "status": "ok",
        "result": "00000000",
    },
    {
        "id": "max_finite_plus_zero",
        "terms": ["7f7fffff", "00000000"],
        "status": "ok",
        "result": "7f7fffff",
    },
    {
        "id": "overflow_positive",
        "terms": ["7f7fffff", "7f7fffff"],
        "status": "overflow",
        "result": None,
    },
    {
        "id": "overflow_after_prior_lane_work_fails_case",
        "terms": ["3f800000", "7f7fffff", "7f7fffff"],
        "status": "overflow",
        "result": None,
    },
    {
        "id": "three_quarters_order_a",
        "terms": ["3f000000", "3e800000", "be000000"],
        "status": "ok",
        "result": "3f200000",
    },
    {
        "id": "three_quarters_order_b",
        "terms": ["3f000000", "be000000", "3e800000"],
        "status": "ok",
        "result": "3f200000",
    },
]


def self_test() -> None:
    # Exact dyadic decoding and tie-to-even checks use Fraction/integer arithmetic.
    assert decode(0x3F800000) == Fraction(1)
    assert decode(0x33800000) == Fraction(1, 1 << 24)
    assert encode(Fraction(1) + Fraction(1, 1 << 24)) == 0x3F800000
    assert encode(decode(0x3F800001) + Fraction(1, 1 << 24)) == 0x3F800002
    assert encode(Fraction(1, 1 << 149)) == 1
    assert encode(Fraction(1, 1 << 150)) == 0  # half minimum subnormal, even zero
    assert add_bits(0x80000000, 0x80000000) == 0x80000000
    assert add_bits(0x00000001, 0x80000001) == 0
    try:
        encode(decode(0x7F7FFFFF) * 2)
    except OverflowError:
        pass
    else:
        raise AssertionError("overflow must be rejected")
    assert accumulate(["4b800000", "3f800000", "cb800000"]) == ("ok", "00000000")
    assert accumulate(["4b800000", "cb800000", "3f800000"]) == ("ok", "3f800000")


def check_corpus() -> None:
    doc = json.loads(CORPUS.read_text())
    assert doc["format"] == "voltafabric-fp32-sequential-add-v1"
    assert doc["cases"] == CASES, (
        "embedded regression constants differ from the fixture"
    )
    for case in doc["cases"]:
        status, result = accumulate(case["terms"])
        assert (status, result) == (case["status"], case["result"]), case["id"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-corpus", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.write_corpus:
        for case in CASES:
            status, result = accumulate(case["terms"])
            assert (status, result) == (case["status"], case["result"]), case["id"]
        CORPUS.write_text(
            json.dumps(
                {
                    "format": "voltafabric-fp32-sequential-add-v1",
                    "semantics": "Each term is an already-preweighted finite FP32 contribution. Start at +0; add in listed order with one binary32 RNE rounding after each addition. Any rounded infinity fails the entire case.",
                    "cases": CASES,
                },
                indent=2,
            )
            + "\n"
        )
    check_corpus()
    print(f"oracle self-tests passed; {len(CASES)} corpus cases verified")


if __name__ == "__main__":
    main()
