#!/usr/bin/env python3
"""A58 DeepSeek-V4.1-Flash CPU format/operator oracle against the built `vf` CLI.

MXFP4: exact rationals from E2M1/E8M0 bit fields (not the Rust table) plus Torch's
float8_e8m0fnu codec for scales. Torch FP4 execution is NOT used (no CPU copy kernel).
Activation: Torch FP32 clamp/silu reference of the official Expert.forward pre-cast.
Tolerance fixed before results: <=8 FP32 ULP, or abs<=1e-7, or rel<=1e-6.
No GPU, full-expert or full-model claim.
"""

import argparse
from fractions import Fraction
import json
from pathlib import Path
import random
import struct
import subprocess

import torch
import torch.nn.functional as F

ULP, ABS, REL = 8, 1e-7, 1e-6


def run(binary, payload):
    res = subprocess.run(
        [str(binary), "deepseek-v41-ops", "-"],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=30,
    )
    data = json.loads(res.stdout)
    assert (res.returncode == 0) == data["ok"], data
    return data


def e2m1(n):
    sign, exp, man = (n >> 3) & 1, (n >> 1) & 3, n & 1
    half = Fraction(man, 2)
    mag = half if exp == 0 else (1 + half) * Fraction(2) ** (exp - 1)
    return sign, mag


def e8m0(b):
    assert b != 255
    return Fraction(2) ** (b - 127)


def f64_bits(hexbits):
    return struct.unpack("<d", bytes.fromhex(hexbits)[::-1])[0]


def f64_from_bits(bits):
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def f64_to_bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def check_matrix(binary, rows, cols, codes, scales, positions):
    req = dict(op="mxfp4-values", rows=rows, cols=cols, codes=codes, scales=scales)
    data = run(binary, {**req, "positions": positions})
    assert data["ok"], data
    r = data["result"]
    cap = r["capability"]
    assert cap["scope"] == "CPU_FORMAT_OPERATOR_ONLY" and cap["gpu_executed"] is False
    assert cap["full_model"] is False and cap["full_expert_forward"] is False
    s = r["summary"]
    elements = rows * cols
    assert (s["code_bytes"], s["scale_bytes"]) == (elements // 2, elements // 32)
    for (row, col), got in zip(positions, r["values"], strict=True):
        byte = codes[row * cols // 2 + col // 2]
        nib = byte & 15 if col % 2 == 0 else byte >> 4
        sc = scales[row * cols // 32 + col // 32]
        sign, mag = e2m1(nib)
        want = mag * e8m0(sc)
        value = f64_bits(got["f64_bits"])
        assert Fraction(value) == (-want if sign else want), (row, col, got)
        # Sign bit checked separately so signed zeros are exact.
        assert (struct.pack("<d", value)[7] >> 7) == sign, (row, col, got)
        assert value == got["value"] and (got["nibble"], got["scale_code"]) == (nib, sc)
    return len(positions)


def refused(binary, payload, code):
    data = run(binary, payload)
    assert not data["ok"] and data["error"]["code"] == code, (payload.get("rows"), data)
    return 1


def matrix_cases(binary):
    n = 0
    # Torch E8M0 codec agrees with rational 2^(b-127) for every valid byte; 255 is NaN.
    decoded = torch.arange(256, dtype=torch.uint8).view(torch.float8_e8m0fnu).double()
    for b in range(255):
        assert Fraction(decoded[b].item()) == e8m0(b), b
    assert torch.isnan(decoded[255])
    # All 16 nibbles x all 255 valid scales (row r -> scale r), two bounded requests.
    row = [((2 * j) % 16) | (((2 * j + 1) % 16) << 4) for j in range(16)]
    codes = row * 255
    scales = list(range(255))
    pos = [[r, c] for r in range(255) for c in range(32)]
    n += check_matrix(binary, 255, 32, codes, scales, pos[:4096])
    n += check_matrix(binary, 255, 32, codes, scales, pos[4096:])
    # Low nibble first, adjacent groups and rows.
    codes = [0x10, 0x32] + [0x98] * 30 + [0x7F] * 32
    pos = [[r, c] for r in range(2) for c in range(64)]
    n += check_matrix(binary, 2, 64, codes, [127, 128, 0, 254], pos)
    rng = random.Random(58)
    for _ in range(8):
        rows, cols = rng.randint(1, 6), 32 * rng.randint(1, 4)
        codes = [rng.randrange(256) for _ in range(rows * cols // 2)]
        scales = [rng.randrange(255) for _ in range(rows * cols // 32)]
        pos = [[r, c] for r in range(rows) for c in range(cols)]
        n += check_matrix(binary, rows, cols, codes, scales, pos)
    base = dict(op="mxfp4-values", rows=1, cols=32, codes=[0] * 16, scales=[1])
    base["positions"] = [[0, 0]]
    n += refused(binary, {**base, "scales": [255]}, "MXFP4_SCALE_NAN")
    for bad in (
        {"codes": [0] * 15},
        {"codes": [0] * 17},
        {"scales": []},
        {"scales": [1, 1]},
        {"codes": [1], "scales": [0] * 16},
        {"cols": 48, "codes": [0] * 24, "scales": [1, 1]},
        {"rows": 0},
        {"rows": 2**64 - 1, "codes": [], "scales": []},
        {"rows": 65536, "cols": 65536, "codes": [], "scales": []},
        {"positions": [[0, 32]]},
        {"positions": [[1, 0]]},
        {"positions": []},
        {"positions": [[0, 0]] * 4097},
    ):
        n += refused(binary, {**base, **bad}, "INVALID_INPUT")
    n += refused(binary, {**base, "global_scale": 1.0}, "JSON_INPUT")
    return n


def matvec_cases(binary):
    """Independent Fraction oracle for ordered FP64 accumulation over MXFP4 planes."""
    rng = random.Random(5801)
    checks = 0
    cases = [(2, 64, [0, 128, 126, 254], None)]
    cases += [(rng.randint(1, 4), 32 * rng.randint(1, 4), None, None) for _ in range(4)]
    cases += [
        (1, 32, [0], [2.0**-900] * 32),
        (1, 32, [0], [5e-324] * 32),
    ]
    dyadics = [0.0, -0.0, 0.25, -0.5, 1.0, -2.0, 3.0, 2.0**-40, -(2.0**20)]
    for case, (rows, cols, fixed_scales, fixed_inputs) in enumerate(cases):
        codes = [rng.randrange(256) for _ in range(rows * cols // 2)]
        scales = fixed_scales or [
            rng.choice([0, 1, 126, 127, 128, 129, 253, 254])
            for _ in range(rows * cols // 32)
        ]
        inputs = fixed_inputs or [
            dyadics[(i * 7 + case) % len(dyadics)] for i in range(cols)
        ]
        payload = dict(
            op="mxfp4-matvec",
            rows=rows,
            cols=cols,
            codes=codes,
            scales=scales,
            input_bits=[f64_to_bits(x) for x in inputs],
        )
        data = run(binary, payload)
        assert data["ok"], data
        result = data["result"]
        assert result["capability"]["scope"] == "CPU_FORMAT_OPERATOR_ONLY"
        assert result["capability"]["gpu_executed"] is False
        assert result["capability"]["full_expert_forward"] is False
        expected = []
        for row in range(rows):
            acc = 0.0
            for col, x in enumerate(inputs):
                packed = codes[row * (cols // 2) + col // 2]
                nibble = packed & 15 if col % 2 == 0 else packed >> 4
                scale_code = scales[row * (cols // 32) + col // 32]
                sign, magnitude = e2m1(nibble)
                weight = -magnitude if sign else magnitude
                # Fraction conversion performs a separately rounded binary64 multiply,
                # then a separately rounded add in increasing K order.
                term = float(weight * e8m0(scale_code) * Fraction.from_float(x))
                acc = float(Fraction.from_float(acc) + Fraction.from_float(term))
            expected.append(acc)
        got_bits = result["output_f64_bits"]
        want_bits = [f"{f64_to_bits(x):016x}" for x in expected]
        assert got_bits == want_bits, (case, got_bits, want_bits)
        checks += rows

    # The vector length is part of the projection contract; no truncation or padding.
    bad = dict(
        op="mxfp4-matvec", rows=1, cols=32, codes=[0] * 16, scales=[127], input_bits=[]
    )
    refused(binary, bad, "INVALID_INPUT")
    return checks


def bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def from_bits(b):
    return struct.unpack("<f", struct.pack("<I", b))[0]


def ordered(b):
    return b if b < 0x80000000 else 0x80000000 - (b & 0x7FFFFFFF)


def torch_ref(gate, up, weight):
    g = torch.tensor(gate, dtype=torch.float32)
    u = torch.tensor(up, dtype=torch.float32)
    u = torch.clamp(u, min=-10.0, max=10.0)
    g = torch.clamp(g, max=10.0)
    x = F.silu(g) * u
    if weight is not None:
        x = torch.tensor([weight], dtype=torch.float32) * x
    return [bits(v) for v in x.tolist()]


def check_act(binary, gate, up, weight=None, exact_zero=True):
    payload = dict(op="pre-down-activation")
    payload["gate_bits"] = [bits(v) for v in gate]
    payload["up_bits"] = [bits(v) for v in up]
    if weight is not None:
        payload["routing_weight_bits"] = bits(weight)
    data = run(binary, payload)
    assert data["ok"], data
    r = data["result"]
    assert r["capability"]["scope"] == "CPU_FORMAT_OPERATOR_ONLY"
    assert r["bf16_cast"] is False and r["down_gemm"] is False
    want = torch_ref(gate, up, weight)
    worst = 0.0
    for i, (got, ref) in enumerate(zip(r["activation_bits"], want, strict=True)):
        a, b = from_bits(got), from_bits(ref)
        err = abs(a - b)
        ulp = abs(ordered(got) - ordered(ref))
        ok = ulp <= ULP or err <= ABS or err <= REL * abs(b)
        assert ok, (gate[i], up[i], weight, a, b)
        # Every reference zero, including FP32 underflow, needs exact sign bits.
        if exact_zero and b == 0.0:
            assert got == ref, ("signed zero", gate[i], up[i], weight, hex(got))
        worst = max(worst, err)
    return len(gate), worst


def activation_cases(binary):
    n, worst = 0, 0.0
    nxt = lambda x, d: from_bits(bits(x) + d)  # noqa: E731
    edge = [0.0, -0.0, 1.0, -1.0, 9.5, 10.0, -10.0, 20.0, -20.0, 1e30, -1e30]
    edge += [3.4e38, -3.4e38, -88.0, -104.0, 1e-40, -1e-40]
    edge += [nxt(10.0, 1), nxt(10.0, -1), nxt(-10.0, 1), nxt(-10.0, -1)]
    gate = [g for g in edge for _ in edge]
    up = [u for _ in edge for u in edge]
    for weight in (None, 0.0, 1.0, 1.5, 2.0**-130):
        k, w = check_act(binary, gate, up, weight)
        n, worst = n + k, max(worst, w)
    rng = random.Random(4158)
    for weight in (None, 0.75):
        gate = [rng.uniform(-30, 30) for _ in range(4000)]
        up = [rng.uniform(-30, 30) for _ in range(4000)]
        k, w = check_act(binary, gate, up, weight)
        n, worst = n + k, max(worst, w)

    def finite():
        positive = rng.randrange(0x7F800000)
        return from_bits(rng.choice([positive, positive | 0x80000000]))

    gate = [finite() for _ in range(4000)]
    up = [finite() for _ in range(4000)]
    k, w = check_act(binary, gate, up, None)
    n, worst = n + k, max(worst, w)
    # gate=-20 must not equal a symmetric clamp's silu(-10).
    sym = torch_ref([-10.0], [1.0], None)[0]
    req = dict(op="pre-down-activation", gate_bits=[bits(-20.0)], up_bits=[bits(1.0)])
    got = run(binary, req)
    assert got["result"]["activation_bits"][0] != sym
    base = dict(op="pre-down-activation", gate_bits=[bits(1.0)], up_bits=[bits(1.0)])
    for special in (0x7FC00000, 0x7F800000, 0xFF800000):
        n += refused(binary, {**base, "gate_bits": [special]}, "NONFINITE_INPUT")
        n += refused(binary, {**base, "up_bits": [special]}, "NONFINITE_INPUT")
    # Native FP32 underflow: Torch gives exactly -0 (0x80000000) here.
    under = [-105.0, -100.0, -95.0, -90.0, -89.0, -88.75]
    k, _ = check_act(binary, under, [1.0] * len(under), None)
    req = dict(op="pre-down-activation", gate_bits=[bits(g) for g in under])
    req["up_bits"] = [bits(1.0)] * len(under)
    assert run(binary, req)["result"]["activation_bits"] == [0x80000000] * len(under)
    n += k
    # Finite coefficient -0 (>= 0) and empty equal shapes are valid.
    k, _ = check_act(binary, [2.0, 2.0, -90.0], [3.0, -3.0, 1.0], -0.0)
    n += k + check_act(binary, [], [], None)[0] + check_act(binary, [], [], 0.5)[0] + 1
    for w in (-1.0, -(2.0**-149)):
        n += refused(binary, {**base, "routing_weight_bits": bits(w)}, "INVALID_INPUT")
    for special in (0x7FC00000, 0x7F800000):
        n += refused(binary, {**base, "routing_weight_bits": special}, "INVALID_INPUT")
    n += refused(binary, {**base, "up_bits": []}, "INVALID_INPUT")
    big = dict(op="pre-down-activation", gate_bits=[bits(10.0)], up_bits=[bits(10.0)])
    big["routing_weight_bits"] = 0x7F7FFFFF  # FLT_MAX: 99.95 * FLT_MAX overflows
    n += refused(binary, big, "NUMERICAL_OVERFLOW")
    return n, worst


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    matrix = matrix_cases(args.binary)
    matvec = matvec_cases(args.binary)
    act, worst = activation_cases(args.binary)
    report = dict(
        scope="CPU_FORMAT_OPERATOR_ONLY",
        mxfp4_exact_checks=matrix,
        mxfp4_matvec_exact_checks=matvec,
        activation_checks=act,
        activation_max_abs_error=worst,
        tolerance=dict(ulp=ULP, abs=ABS, rel=REL),
        torch=torch.__version__,
        torch_fp4_executed=False,
        gpu_executed=False,
        full_model=False,
    )
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
