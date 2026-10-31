"""Python binary16 oracle for the isolated R428 expert-255 contribution."""
import hashlib
import math
import struct
import sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("usage: independent_fp16.py PC1_F32 PC6_F32")
rows = []
for name in sys.argv[1:]:
    raw = Path(name).read_bytes()
    if len(raw) != 10240:
        raise ValueError("wrong row size")
    values = struct.unpack("<2560f", raw)
    if not all(math.isfinite(x) for x in values):
        raise ValueError("non-finite row")
    rows.append(values)
    print("input_sha256", hashlib.sha256(raw).hexdigest())

def bits(x):
    return struct.pack("<e", x)

def decoded(x):
    return struct.unpack("<e", bits(x))[0]

x, y = rows
weight = 0.04014204069972038
print("fp16_different", sum(bits(a) != bits(b) for a, b in zip(x, y)))
print("collected_max", max(abs(weight * decoded(a) - weight * decoded(b)) for a, b in zip(x, y)))
