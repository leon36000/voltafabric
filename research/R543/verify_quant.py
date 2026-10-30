"""Independent float32 Q8_K and Q8_0 packing checks for actual R537 inputs.

Q8_K is adapted from R473's independent_iq2.py. The Q8_0 nearest-even rule
and format follow the observed x86 GGML CPU implementation; this is not an
independent format specification or a cross-architecture claim.
"""
import hashlib,json,math,struct,sys
from pathlib import Path
if len(sys.argv) not in (3,4) or (len(sys.argv)==4 and sys.argv[3] not in ('--negative','--negative-q8')):
    raise SystemExit('usage: verify_quant TRACE QUANT [--negative]')
trace=Path(sys.argv[1]).read_bytes(); data=bytearray(Path(sys.argv[2]).read_bytes())
if len(trace)!=92*37380*4 or hashlib.sha256(trace).hexdigest()!='8787cdb8133d0962c23e273507ea471dc4d45014ac7c4f2d323705764da2c54e': raise ValueError('trace identity')
if len(data)!=92*(2920+6528) or hashlib.sha256(data).hexdigest()!='a12b7cd214277039e94458a44c824645823505fd1057ae9a311534ccd6e887c5': raise ValueError('quantized data identity')
if len(sys.argv)==4:
    data[91*(2920+6528)+(4 if sys.argv[3]=='--negative' else 2920+2)]^=1
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def q8k(values):
    packed=bytearray()
    for i in range(0,len(values),256):
        block=values[i:i+256]; maxv=0.0; amax=0.0
        for v in block:
            if abs(v)>amax: amax,maxv=abs(v),v
        if amax==0:
            packed+=bytes(292); continue
        inv=f32(-127.0/maxv); d=f32(1.0/inv)
        q=[min(127,round(f32(inv*v))) for v in block]
        if any(n < -128 or n > 127 for n in q): raise ValueError('Q8_K range')
        packed+=struct.pack('<f256b16h',d,*q,*[sum(q[j:j+16]) for j in range(0,256,16)])
    return bytes(packed)
def q8(values):
    packed=bytearray()
    for i in range(0,len(values),32):
        block=values[i:i+32]; amax=max(map(abs,block)); d=f32(amax/127.0); inv=f32(127.0/amax) if amax else 0.0
        q=[round(f32(v*inv)) for v in block]
        if any(n < -128 or n > 127 for n in q): raise ValueError('Q8_0 range')
        packed+=struct.pack('<e32b',d,*q)
    return bytes(packed)
for pos in range(92):
    row=struct.unpack_from('<37380f',trace,pos*37380*4)
    x=row[:2560]; y=row[28676:34820]
    if not all(math.isfinite(v) for v in (*x,*y)): raise ValueError('nonfinite activation')
    start=pos*(2920+6528); got_k=bytes(data[start:start+2920]);got_0=bytes(data[start+2920:start+2920+6528])
    want_k=q8k(x);want_0=q8(y)
    if got_k!=want_k:
        i=next(i for i,(a,b) in enumerate(zip(got_k,want_k,strict=True)) if a!=b)
        raise ValueError(f'Q8_K mismatch position={pos} byte={i} got={got_k[i]} expected={want_k[i]}')
    if got_0!=want_0:
        i=next(i for i,(a,b) in enumerate(zip(got_0,want_0,strict=True)) if a!=b)
        raise ValueError(f'Q8_0 mismatch position={pos} byte={i} got={got_0[i]} expected={want_0[i]}')
print(json.dumps({'status':'PASS','positions':92,'q8k_blocks':920,'q8_0_blocks':17664,'quant_bytes':len(data),'trace_sha256':hashlib.sha256(trace).hexdigest(),'quant_sha256':hashlib.sha256(data).hexdigest()},sort_keys=True))
