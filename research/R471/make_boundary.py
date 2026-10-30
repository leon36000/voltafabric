#!/usr/bin/env python3
"""Make deterministic representable half-step and signed-max blocks."""
import struct
import sys
v = [127.0, 0.5, -0.5, 1.5, -1.5, 2.5, -2.5, 126.5, -126.5] + [0.0]*23
with open(sys.argv[1], 'xb') as f:
    f.write(struct.pack('<2560f', *(v*80)))
