#!/usr/bin/env python3
"""NumPy reference for zero-history SC117 QKV convolution and Q/K L2 norm."""
import argparse
import hashlib
import json
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
for name in ('dense','qkv','output'): ap.add_argument('--'+name,required=True)
a=ap.parse_args()
with open(a.dense,'rb') as f:
    f.seek(42726592); raw=f.read(163840)
if len(raw)!=163840: raise SystemExit('short convolution tensor')
w=np.frombuffer(raw,dtype='<f4').reshape(10240,4)
x=np.fromfile(a.qkv,dtype='<f4'); got=np.fromfile(a.output,dtype='<f4')
if x.size!=10240 or got.size!=10240: raise SystemExit('invalid vector length')
s=(x*w[:,3]).astype(np.float32); expected=(s/(np.float32(1)+np.exp(-s))).astype(np.float32)
qk=expected[:4096].reshape(32,128)
squares=(qk.astype(np.float64)**2).sum(axis=1)
inv=(1/np.sqrt(squares+float(np.float32(1e-6)))).astype(np.float32)
qk*=inv[:,None]
if not np.isfinite(expected).all() or not np.isfinite(got).all(): raise SystemExit('nonfinite result')
diff=np.abs(got.astype(np.float64)-expected.astype(np.float64)); maximum=float(diff.max())
if maximum>1e-6: raise SystemExit('preprocess differs')
print(json.dumps({'status':'ok','values':len(got),'max_abs':maximum,'qkv_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
                  'output_sha256':hashlib.sha256(got.tobytes()).hexdigest()},sort_keys=True))
