import base64,hashlib,json,math,struct,sys
p=json.loads(sys.stdin.buffer.read(4000000));ex={'scalars':'bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2','scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa','codes':'c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993'}
if set(p)!=set(ex):raise ValueError('bad payload')
b={k:base64.b64decode(v,validate=True) for k,v in p.items()}
for k in ex:
 if hashlib.sha256(b[k]).hexdigest()!=ex[k]:raise ValueError('hash mismatch')
import torch
torch.set_num_threads(1)
if torch.version.cuda is not None:raise ValueError('CPU-only Torch required')
g=struct.unpack('<6f',b['scalars']);fac=[g[1],g[3],g[5]];shapes=[(2048,512),(512,2048),(512,2048)];names=['down','gate','up'];lut4=[0.,.5,1.,1.5,2.,3.,4.,6.,-0.,-.5,-1.,-1.5,-2.,-3.,-4.,-6.]
def frombits(v):return struct.unpack('<e',struct.pack('<H',v))[0]
def half(v):return struct.unpack('<e',struct.pack('<e',v))[0]
def intermediate(s,n,fac):
 fp=half(frombits(s<<7)*256.0);gm=half(fac*16384.0);sc=half(fp*gm);q=frombits(((n&8)<<12)|((n&7)<<9));return half(q*sc)
results=[]
for i,(rows,cols) in enumerate(shapes):
 codes=torch.frombuffer(bytearray(b['codes'][i*524288:(i+1)*524288]),dtype=torch.uint8).reshape(rows,cols//2);q=torch.stack((codes&15,codes>>4),dim=-1).reshape(rows,cols).long();sb=b['scales'][i*65536:(i+1)*65536];raw=torch.frombuffer(bytearray(sb),dtype=torch.uint8).reshape(rows,cols//16);sc=raw.long().repeat_interleave(16,dim=1);used=set(sb)
 if any(x>=127 for x in used):raise ValueError('invalid scale')
 l=torch.zeros((128,16),dtype=torch.float64)
 for s in used:
  for n in range(16):l[s,n]=intermediate(s,n,fac[i])
 half_values=l[sc,q];ideal_sc=raw.view(torch.float8_e4m3fn).to(torch.float64).repeat_interleave(16,dim=1);ideal=torch.tensor(lut4,dtype=torch.float64)[q]*ideal_sc*fac[i];diff=(half_values-ideal).abs();delta=(diff!=0);underflow=(half_values==0)&(ideal!=0)
 results.append({'projection':names[i],'values':rows*cols,'nonzero_difference_count':int(delta.sum().item()),'max_abs_difference':float(diff.max().item()),'underflow_to_zero_count':int(underflow.sum().item()),'max_ideal_abs':float(ideal.abs().max().item())})
out={'status':'FULL_REAL_WEIGHT_QPN2_FP16_VS_IDEAL_CPU_CHARACTERIZED','projections':results,'total_values':sum(x['values'] for x in results),'gpu':False,'limits':'CPU emulation derived from source, ideal weights only, no actual CUDA, no activation/forward or full shard hash.'};print(json.dumps(out,sort_keys=True))
