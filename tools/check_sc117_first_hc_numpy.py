#!/usr/bin/env python3
"""Independent NumPy comparison for the SC117 first hyperconnection CPU reference."""
import argparse
import json
import subprocess
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
for name in ("dense","embedding","binary","norm-sha","down-sha","up-sha","inject-sha"):
    ap.add_argument("--"+name,required=True)
a=ap.parse_args()
result=json.loads(subprocess.check_output([a.binary,a.dense,a.embedding,a.norm_sha,a.down_sha,a.up_sha,a.inject_sha]))
x=np.fromfile(a.embedding,dtype='<f4')
if x.size!=2560: raise SystemExit('unexpected embedding size')
with open(a.dense,'rb') as f:
    def read(offset,n,dtype):
        f.seek(offset); data=f.read(n)
        if len(data)!=n: raise ValueError('short tensor read')
        return np.frombuffer(data,dtype=dtype).copy()
    gamma=read(22410240,40960,'<f4'); down=read(15774720,6553600,'<u2')
    up=read(22451200,6553600,'<u2'); inj=read(22328320,81920,'<u2')
def expand(a): return (a.astype(np.uint32)<<16).view(np.float32)
def bf16(x):
    bits=x.astype(np.float32).view(np.uint32)
    return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
r=np.tile(x,4); rms=np.sqrt(np.mean(x.astype(np.float64)**2)+float(np.float32(1e-6)))
xn=(r.astype(np.float64)/rms*gamma.astype(np.float64)).astype(np.float32); xq=bf16(xn)
lo=(expand(down).reshape(320,10240).astype(np.float64)@xq.astype(np.float64)).astype(np.float32)/np.float32(4)
lo=bf16(lo/(np.float32(1)+np.exp(-lo)))
gate=(expand(up).reshape(10240,320).astype(np.float64)@lo.astype(np.float64)).astype(np.float32)
gated=xn/(np.float32(1)+np.exp(-gate)); mixed=np.zeros(2560,np.float32)
for stream in range(4): mixed+=gated[stream*2560:(stream+1)*2560]
mixed/=np.float32(4)
injection=(expand(inj).reshape(4,10240).astype(np.float64)@xq.astype(np.float64)).astype(np.float32)
if len(result['mixed'])!=2560 or len(result['inject'])!=4: raise SystemExit('shape mismatch')
a1=float(np.max(np.abs(np.array(result['mixed'])-mixed))); a2=float(np.max(np.abs(np.array(result['inject'])-injection)))
if a1>1e-5 or a2>1e-5: raise SystemExit('hyperconnection mismatch')
print(json.dumps({'status':'ok','mixed_max_abs':a1,'inject_max_abs':a2,'embedding_norm':float(np.linalg.norm(x)),
                  'mixed_norm':float(np.linalg.norm(mixed))},sort_keys=True))
