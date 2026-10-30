import base64,hashlib,json,math,struct,sys
p=json.loads(sys.stdin.buffer.read(4000000));expected={'scalars':'bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2','scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa','codes':'c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993'}
if set(p)!=set(expected):raise ValueError('payload set mismatch')
b={k:base64.b64decode(v,validate=True) for k,v in p.items()}
for k,v in b.items():
 if hashlib.sha256(v).hexdigest()!=expected[k]:raise ValueError('payload sha mismatch')
import torch
torch.set_num_threads(1)
if torch.version.cuda is not None:raise ValueError('CPU-only torch required')
g=struct.unpack('<6f',b['scalars']); x=[((i%31)-15)/16.0 for i in range(2048)]; shapes=[(2048,512),(512,2048),(512,2048)]; factors=[g[1],g[3],g[5]]
def e4(v):
 if v&128 or v==127:raise ValueError('invalid scale')
 e=(v>>3)&15; m=v&7
 return math.ldexp(m,-9) if e==0 else math.ldexp(8+m,e-10)
def e2(v):
 z=v&7;mag=z*0.5 if z<2 else math.ldexp(1.0+(z&1)*0.5,(z>>1)-1)
 return -mag if v&8 else mag
codes=[memoryview(b['codes'])[i*524288:(i+1)*524288] for i in range(3)];scales=[memoryview(b['scales'])[i*65536:(i+1)*65536] for i in range(3)]
def scalar_mv(k,vector):
 rows,cols=shapes[k]; c=codes[k];s=scales[k];factor=factors[k]
 def w(row,col):
  z=c[row*(cols//2)+col//2]; nib=(z>>4) if col%2 else (z&15)
  return e2(nib)*e4(s[row*(cols//16)+col//16])*factor
 return [math.fsum(w(row,col)*vector[col] for col in range(cols)) for row in range(rows)]
def torch_mv(k,vector):
 rows,cols=shapes[k]; c=torch.frombuffer(bytearray(codes[k]),dtype=torch.uint8).reshape(rows,cols//2);ix=torch.stack((c&15,c>>4),dim=-1).reshape(rows,cols).long();lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.],dtype=torch.float64);s=torch.frombuffer(bytearray(scales[k]),dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).reshape(rows,cols//16).repeat_interleave(16,dim=1);matrix=lut[ix]*s*factors[k];return torch.mv(matrix,torch.tensor(vector,dtype=torch.float64)).tolist()
def close(a,b):
 if len(a)!=len(b):raise ValueError('length differs')
 return all(math.isfinite(u) and math.isfinite(v) and abs(u-v)<=1e-10+1e-10*abs(u) for u,v in zip(a,b))
gate=scalar_mv(1,x);up=scalar_mv(2,x);tg= torch_mv(1,x);tu=torch_mv(2,x)
if not close(gate,tg) or not close(up,tu):raise ValueError('gate/up differ outside predeclared tolerance')
h=[(z/(1+math.exp(-z)))*u for z,u in zip(gate,up)];th=(torch.nn.functional.silu(torch.tensor(tg,dtype=torch.float64))*torch.tensor(tu,dtype=torch.float64)).tolist()
if not close(h,th):raise ValueError('activation differs outside tolerance')
y=[.375*v for v in scalar_mv(0,h)];ty=[.375*v for v in torch_mv(0,th)]
if not close(y,ty):raise ValueError('expert differs outside tolerance')
print(json.dumps({'status':'ANALYTICAL_CPU_EXPERT_DOUBLE_TWO_IMPLEMENTATIONS_AGREE','scalar_count':len(y),'max_abs_diff':max(abs(a-b) for a,b in zip(y,ty)),'tolerance_abs':1e-10,'tolerance_rel':1e-10,'coefficient':0.375,'input':'((i%31)-15)/16','output_f64_sha256':hashlib.sha256(struct.pack('<%dd'%len(y),*y)).hexdigest(),'output_first_4':y[:4],'sample_sha256':expected,'python':sys.version.split()[0],'torch':torch.__version__,'gpu':False,'limitations':'FP64 analytical model; no FP16 CUDA rounding, no native prepack, no GPU, full model or shard LFS hash; model integration not proven.'},sort_keys=True))
