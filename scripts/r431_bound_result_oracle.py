"""Validate layer-2 collector against scalar expert references and shared against scalar GGML references."""
import ctypes,hashlib,json,sys
from pathlib import Path
import numpy as np
if len(sys.argv)!=2: raise SystemExit('usage: oracle WORK_DIR')
p=Path(sys.argv[1]);raw=(p/'result.json').read_bytes()
if hashlib.sha256(raw).hexdigest()!='e71d9b301b7b653bf312f6db0620f8f0dff7f759844744fbeb055f6513811408':raise ValueError('result identity')
r=json.loads(raw); ids=[405,143,190,42,232,231,315,223,294,470]
weights=np.asarray(r['weights'],dtype=np.float32).astype(np.float64)
ref=[]
for expert in ids:
 b=(p/f'ref-{expert}.bin').read_bytes()
 if len(b)!=30720:raise ValueError('expert reference length')
 ref.append(np.frombuffer(b,dtype='<f8',offset=10240).astype(np.float32).astype(np.float16).astype(np.float64))
expected=np.zeros(2560,dtype=np.float64)
for w,row in zip(weights,ref):expected+=w*row
actual=np.asarray(r['collector'],dtype=np.float64)
diff=np.abs(actual-expected)
print('collector_max_abs',float(diff.max()))
if not np.isfinite(actual).all() or (diff>1e-5).any():raise ValueError('collector discrepancy')
shared=np.asarray(r['shared'],dtype=np.float32); shared_ref=np.fromfile(p/'shared-reference.f32',dtype='<f4')
err=np.abs(shared.astype(np.float64)-shared_ref.astype(np.float64));print('shared_max_abs',float(err.max()))
if len(shared_ref)!=2560 or not np.isfinite(shared_ref).all() or (err>1e-7+1e-3*np.abs(shared_ref)).any():raise ValueError('shared discrepancy')
# Reconstruct the final residual independently, with f32 rounding at each operation.
combined=(actual+shared.astype(np.float64)).astype(np.float32);post=np.asarray(r['post'],dtype=np.float32).reshape(4,2560);inj=np.asarray(r['injection'],dtype=np.float32)
libm=ctypes.CDLL("libm.so.6"); libm.expf.argtypes=[ctypes.c_float]; libm.expf.restype=ctypes.c_float
weight=[]
for v in inj:
 x=np.float32(v/np.float32(4)); e=np.float32(libm.expf(float(-x if x>=0 else x)))
 sig=np.float32(np.float32(1)/np.float32(np.float32(1)+e)) if x>=0 else np.float32(e/np.float32(np.float32(1)+e))
 weight.append(np.float32(np.float32(2)*sig))
weight=np.asarray(weight,dtype=np.float32)
res=(post+combined[None,:]*weight[:,None]).astype('<f4');h=hashlib.sha256(res.tobytes()).hexdigest()
print('residual_reference_sha256',h,'actual_sha256',r['residual_sha'])
if h!=r['residual_sha']:raise ValueError('residual hash differs; inspect f32 rounding')
