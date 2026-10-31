"""Independent f32-step reconstruction from retained R428 fields and R604 collection."""
import ctypes,hashlib,json,math,os,struct
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=root/'research/R605';w=root/'.runs/vf-r605-work';plan=json.loads((p/'PLAN.json').read_text());h=lambda x:hashlib.sha256(x).hexdigest()
for path,digest in plan['inputs'].items():assert h((root/path).read_bytes())==digest
v=json.loads((w/'r428.json').read_bytes())['tokens'][0]; f32=lambda x:struct.unpack('<f',struct.pack('<f',x))[0]
post=[f32(x) for x in v['post']];shared=[f32(x) for x in v['shared']];inject=[f32(x) for x in v['injection']];assert (len(post),len(shared),len(inject))==(10240,2560,4)
lib=ctypes.CDLL('libm.so.6');expf=lib.expf;expf.argtypes=[ctypes.c_float];expf.restype=ctypes.c_float
weights=[]
for z in inject:
 x=f32(z/4.0)
 if x>=0: sig=f32(1.0/f32(1.0+expf(f32(-x))))
 else:
  e=expf(x);sig=f32(e/f32(1.0+e))
 weights.append(f32(2.0*sig))
def expected(row):
 combined=[f32(r+float(s)) for r,s in zip(row,shared)]; assert len(combined)==2560
 return struct.pack('<10240f',*(f32(post[j*2560+i]+f32(combined[i]*weights[j])) for j in range(4) for i in range(2560)))
old=v['collector'];data=(w/'historical.f32').read_bytes();want=expected(old);assert data==want and h(data)==v['residual_sha']
new=struct.unpack('<2560d',(root/'.runs/vf-r604-work/collector.f64').read_bytes());newdata=(w/'scalar.f32').read_bytes();expected_new=expected(new)
if newdata!=expected_new:
 a=struct.unpack('<10240f',newdata);b=struct.unpack('<10240f',expected_new);n=sum(x!=y for x,y in zip(a,b));m=max(abs(x-y) for x,y in zip(a,b));raise SystemExit(f'FAIL: {n} values differ, maximum {m}')
a=struct.unpack('<10240f',newdata);b=struct.unpack('<10240f',data);diff=sum(x!=y for x,y in zip(a,b));maxdiff=max(abs(x-y) for x,y in zip(a,b));assert all(math.isfinite(x) for x in a)
result={'historical_residual_sha256':h(data),'scalar_residual_sha256':h(newdata),'exact_python_values':10240,'values_differ_from_historical':diff,'max_absolute_difference':maxdiff,'offline_reassembly':True,'new_layer_execution':False}
with open(p/'RESULT.json','x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print('PASS Python verified 10240 f32 values and historical residual; scalar SHA-256',result['scalar_residual_sha256'],'different values',diff,'max abs',maxdiff)
