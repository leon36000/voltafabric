"""Check two local collectors and final residuals against staged scalar references.
Expert 255 token 0 uses vector-kernel intermediate quantization; see R428 report.
"""
import ctypes,hashlib,json,sys
from pathlib import Path
import numpy as np
p=Path(sys.argv[1]);raw=(p/'result.json').read_bytes()
if hashlib.sha256(raw).hexdigest()!='31800dc58ccac050049ce417dd0bdb61143e3d022c8ab9cae210f02ccfec8cd4':raise ValueError('result identity')
r=json.loads(raw);libm=ctypes.CDLL('libm.so.6');libm.expf.argtypes=[ctypes.c_float];libm.expf.restype=ctypes.c_float
for t in r['tokens']:
 i=t['token'];weights=np.asarray(t['weights'],dtype=np.float32).astype(np.float64);expected=np.zeros(2560,np.float64)
 for w,e in zip(weights,t['ids']):
  b=(p/f'ref-{i}-{e}.bin').read_bytes()
  if len(b)!=30720:raise ValueError('reference length')
  expected+=w*np.frombuffer(b,dtype='<f8',offset=10240).astype(np.float32).astype(np.float16).astype(np.float64)
 actual=np.asarray(t['collector'],dtype=np.float64);error=np.max(np.abs(actual-expected))
 shared=np.asarray(t['shared'],dtype=np.float32);ref=np.fromfile(p/f'shared{i}.f32',dtype='<f4');serr=np.max(np.abs(shared.astype(np.float64)-ref.astype(np.float64)))
 if error>1e-5 or (np.abs(shared.astype(np.float64)-ref)>1e-7+1e-3*np.abs(ref)).any():raise ValueError('collector/shared mismatch')
 combined=(actual+shared.astype(np.float64)).astype(np.float32);post=np.asarray(t['post'],dtype=np.float32).reshape(4,2560);inj=np.asarray(t['injection'],dtype=np.float32);weight=[]
 for v in inj:
  x=np.float32(v/np.float32(4));e=np.float32(libm.expf(float(-x if x>=0 else x)))
  sig=np.float32(np.float32(1)/np.float32(np.float32(1)+e)) if x>=0 else np.float32(e/np.float32(np.float32(1)+e));weight.append(np.float32(np.float32(2)*sig))
 result=(post+combined[None,:]*np.asarray(weight,dtype=np.float32)[:,None]).astype('<f4');sha=hashlib.sha256(result.tobytes()).hexdigest()
 print('token',i,'collector_max_abs',float(error),'shared_max_abs',float(serr),'residual_match',sha==t['residual_sha'])
 if sha!=t['residual_sha']:raise ValueError('residual mismatch')
