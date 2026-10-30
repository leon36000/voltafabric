#!/usr/bin/env python3
"""Independent scalar Q8_0/Q8_K packing oracle; GGML is not imported."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

source, prefix = map(Path, sys.argv[1:])
raw = source.read_bytes()
if len(raw) != 10240:
    raise ValueError('input length')
x = struct.unpack('<2560f', raw)
if not all(math.isfinite(v) for v in x):
    raise ValueError('nonfinite input')
def f32(v):
    return struct.unpack('<f', struct.pack('<f', v))[0]
def away(v):
    return math.floor(v + .5) if v >= 0 else math.ceil(v - .5)
def q80(values, rounding=away):
    out = bytearray()
    for start in range(0, len(values), 32):
        block = values[start:start+32]
        d = f32(max(map(abs, block))/127.0)
        inv = f32(1.0/d) if d else 0.0
        q = [rounding(f32(v*inv)) for v in block]
        if any(v < -128 or v > 127 for v in q): raise ValueError('q8_0 code out of range')
        out += struct.pack('<e32b', d, *q)
    return bytes(out)
def q8k(values):
    out = bytearray()
    for start in range(0, len(values), 256):
        block = values[start:start+256]
        signed_max = 0.0
        amax = 0.0
        for v in block:
            if abs(v) > amax:
                amax, signed_max = abs(v), v
        if amax == 0:
            out += bytes(292)
            continue
        inv = f32(-127.0/signed_max)
        d = f32(1.0/inv)
        q = [min(127, round(f32(inv*v))) for v in block]
        if any(v < -128 or v > 127 for v in q): raise ValueError('q8_k code out of range')
        sums = [sum(q[i:i+16]) for i in range(0,256,16)]
        out += struct.pack('<f256b16h', d, *q, *sums)
    return bytes(out)
summary = {'input_sha256': hashlib.sha256(raw).hexdigest()}
for name, oracle in (('q80',q80(x)), ('q8k',q8k(x))):
    ref = Path(str(prefix)+'-'+name+'-ref.bin').read_bytes()
    fast = Path(str(prefix)+'-'+name+'-fast.bin').read_bytes()
    if len(ref)!=len(oracle) or len(fast)!=len(oracle): raise ValueError('packed length')
    mismatches_ref = sum(a!=b for a,b in zip(oracle,ref,strict=True))
    mismatches_fast = sum(a!=b for a,b in zip(oracle,fast,strict=True))
    nearest_even = q80(x, round) if name == 'q80' else oracle
    summary[name] = {'bytes':len(oracle), 'nearest_even_mismatch_bytes':sum(a!=b for a,b in zip(nearest_even,fast,strict=True)), 'reference_mismatch_bytes':mismatches_ref,
                     'optimized_mismatch_bytes':mismatches_fast,
                     'reference_sha256':hashlib.sha256(ref).hexdigest(),
                     'optimized_sha256':hashlib.sha256(fast).hexdigest()}
    if mismatches_ref: raise ValueError(name+' scalar reference mismatch')
print(json.dumps(summary,sort_keys=True))
