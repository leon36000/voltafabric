import base64,hashlib,json,struct,sys,math
p=json.loads(sys.stdin.buffer.read(500000));ex={'scalars':'bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2','scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa'}
if set(p)!=set(ex):raise ValueError('bad payload')
b={k:base64.b64decode(v,validate=True) for k,v in p.items()}
for k in ex:
 if hashlib.sha256(b[k]).hexdigest()!=ex[k]:raise ValueError('hash mismatch')
import torch
torch.set_num_threads(1)
if torch.version.cuda is not None:raise ValueError('CPU torch required')
factors=struct.unpack('<6f',b['scalars']);uniq=sorted(set(b['scales']));assert len(uniq)==65 and all(s<127 for s in uniq)
def frombits(bits):return struct.unpack('<e',struct.pack('<H',bits))[0]
def half(v):return struct.unpack('<e',struct.pack('<e',v))[0]
def bits(v):return struct.unpack('<H',struct.pack('<e',v))[0]
def py_value(s,n,g):
 fp=half(frombits(s<<7)*256.0);gm=half(g*16384.0);sc=half(fp*gm);q=frombits(((n&8)<<12)|((n&7)<<9));return bits(half(q*sc))
scales=[];ns=[];gs=[];expected=[]
for fac in (factors[1],factors[3],factors[5]):
 for s in uniq:
  for n in range(16):scales.append(s);ns.append(n);gs.append(fac);expected.append(py_value(s,n,fac))
rawscale=torch.tensor([s<<7 for s in scales],dtype=torch.int32).to(torch.int16).view(torch.float16);sf=(rawscale*torch.tensor(256.0,dtype=torch.float16)).to(torch.float16);g=(torch.tensor(gs,dtype=torch.float32)*16384.0).to(torch.float16);sc=(sf*g).to(torch.float16);q=torch.tensor([((n&8)<<12)|((n&7)<<9) for n in ns],dtype=torch.int32).to(torch.int16).view(torch.float16);actual=(q*sc).view(torch.int16).to(torch.int32).bitwise_and(65535).tolist();mismatches=sum(a!=b for a,b in zip(expected,actual))
if mismatches:raise ValueError('FP16 bit mismatch count '+str(mismatches))
nonzero=sum((x&0x7fff)!=0 for x in expected);signed_zero=sum(x==0x8000 for x in expected);packed=struct.pack('<%dH'%len(expected),*expected)
print(json.dumps({'status':'FP16_DEQUANT_PYTHON_TORCH_CPU_BIT_EXACT','pairs':len(expected),'scale_codes':len(uniq),'factors':3,'bit_mismatches':mismatches,'nonzero_results':nonzero,'negative_zeros':signed_zero,'output_bits_sha256':hashlib.sha256(packed).hexdigest(),'torch':torch.__version__,'gpu':False,'limits':'Source-derived CPU emulation of dequant only; does not execute CUDA or model accumulation; complete shard LFS SHA remains unverified.'},sort_keys=True))
