#!/usr/bin/env python3
"""Independent IQ4_NL and Q8_0 scalar projection; never imports GGML."""
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

pack, prefix = map(Path, sys.argv[1:])
with pack.open('rb') as f:
    f.seek(2390092800)
    blob = f.read(1971200)
if len(blob) != 1971200 or hashlib.sha256(blob).hexdigest() != '8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5':
    raise ValueError('expert blob differs from R428 local observation')
weights = Path(str(prefix)+'-weights.bin').read_bytes()
if len(weights) != 2560*360 or blob[-len(weights):] != weights:
    raise ValueError('down weight slice differs')
levels = (-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113)
def f32(v): return struct.unpack('<f',struct.pack('<f',v))[0]
results = {}
for mode in ('smooth','boundary'):
    inp = Path(str(prefix)+'-'+mode+'-input.f32').read_bytes()
    rawq = Path(str(prefix)+'-'+mode+'-quant.bin').read_bytes()
    out = Path(str(prefix)+'-'+mode+'-output.f32').read_bytes()
    if len(inp)!=2560 or len(rawq)!=680 or len(out)!=10240: raise ValueError('geometry')
    values = struct.unpack('<640f',inp)
    if any(not math.isfinite(v) for v in values): raise ValueError('nonfinite')
    independently_packed = bytearray()
    blocks = []
    for i in range(20):
        x = values[i*32:(i+1)*32]
        d = f32(max(map(abs,x))/127.0)
        inv = f32(1/d) if d else 0.0
        q = [round(f32(v*inv)) for v in x]
        independently_packed += struct.pack('<e32b',d,*q)
        blocks.append((struct.unpack_from('<e',rawq,i*34)[0],q))
    if bytes(independently_packed) != rawq: raise ValueError('optimized activation packing differs from nearest-even oracle')
    got = struct.unpack('<2560f',out)
    max_abs = 0.0; violations = 0
    for row in range(2560):
        total = 0.0
        for block in range(20):
            pos = row*360 + block*18
            dw = struct.unpack_from('<e',weights,pos)[0]
            qbytes = weights[pos+2:pos+18]
            da, qa = blocks[block]
            dot = sum(levels[qbytes[j]&15]*qa[j] + levels[qbytes[j]>>4]*qa[j+16] for j in range(16))
            total += float(dw)*float(da)*dot
        if not math.isfinite(total) or not math.isfinite(got[row]): raise ValueError('nonfinite projection')
        error = abs(got[row]-total)
        max_abs = max(max_abs,error)
        violations += error > 1e-6 + 2e-5*abs(total)
    results[mode] = {'max_abs':max_abs,'violations':violations,'rows':2560,
                     'activation_sha256':hashlib.sha256(inp).hexdigest(),
                     'quant_sha256':hashlib.sha256(rawq).hexdigest(),
                     'output_sha256':hashlib.sha256(out).hexdigest()}
    if violations: raise ValueError(mode+' independent IQ4 projection tolerance')
print(json.dumps({'scope':'synthetic intermediate; independent scalar IQ4_NL decode and Q8_0 quantization',
                  'blob_sha256':hashlib.sha256(blob).hexdigest(), 'weights_sha256':hashlib.sha256(weights).hexdigest(),
                  'tolerance':'1e-6 + 2e-5 * abs(reference)', 'results':results},sort_keys=True))
