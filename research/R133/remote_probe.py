import base64,hashlib,inspect,json,math,struct,sys,types
payload=json.loads(sys.stdin.buffer.read(4000000))
expected={'adapter':'fa6a5459d3b913594104618076973390c28c8c252cc2edf35998e7b047339abd','scalars':'bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2','block_scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa','codes':'c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993'}
if set(payload)!=set(expected):raise ValueError('payload set differs')
raw={k:base64.b64decode(v,validate=True) for k,v in payload.items()}
for k,v in raw.items():
 if hashlib.sha256(v).hexdigest()!=expected[k]:raise ValueError('sha mismatch '+k)
assert len(raw['scalars'])==24 and len(raw['block_scales'])==196608 and len(raw['codes'])==1572864
module=types.ModuleType('vf_r133');sys.modules[module.__name__]=module;exec(compile(raw['adapter'],'<r133-adapter>','exec'),module.__dict__);torch=module.torch
torch.set_num_threads(1)
if torch.version.cuda is not None:raise ValueError('expected explicit CPU-only Torch')
vals=struct.unpack('<6f',raw['scalars']);dc,gc,uc=(torch.frombuffer(bytearray(raw['codes'][i*524288:(i+1)*524288]),dtype=torch.uint8) for i in range(3));ds,gs,us=(torch.frombuffer(bytearray(raw['block_scales'][i*65536:(i+1)*65536]),dtype=torch.uint8) for i in range(3))
layer=types.SimpleNamespace(w13_weight=torch.cat((gc.reshape(512,1024),uc.reshape(512,1024))).reshape(1,1024,1024),w13_weight_scale=torch.cat((gs.reshape(512,128),us.reshape(512,128))).reshape(1,1024,128),w13_weight_scale_2=torch.tensor([[vals[3],vals[5]]],dtype=torch.float32),w2_weight=dc.reshape(1,2048,256),w2_weight_scale=ds.reshape(1,2048,32),w2_weight_scale_2=torch.tensor([vals[1]],dtype=torch.float32))
calls=[]
def fake_identity_prepack(c,s):
 calls.append({'codes_shape':list(c.shape),'scales_shape':list(s.shape)});return c.clone(),s.clone()
module.prepack_experts(layer,prepack=fake_identity_prepack);owner=layer._skinny_moe_resident
assert tuple(owner.shape)==(1,1024,2048,2048,512) and len(calls)==2
assert torch.equal(owner._gs13[0,:16],torch.full((16,),vals[3])) and torch.equal(owner._gs13[0,16:],torch.full((16,),vals[5])) and torch.equal(owner._gs2[0],torch.full((64,),vals[1]))
# Python bit formula versus PyTorch native float8_e4m3fn conversion, every distinct observed scale.
uniq=sorted(set(raw['block_scales']));tb=torch.tensor(uniq,dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float64).tolist()
def e4m3(v):
 if v&128 or (v&127)==127:raise ValueError('invalid scale')
 exp=(v>>3)&15;mant=v&7
 return math.ldexp(mant,-9) if exp==0 else math.ldexp(8+mant,exp-10)
ref=[e4m3(v) for v in uniq]
if ref!=tb:raise ValueError('float8 decode mismatch')
print(json.dumps({'status':'REAL_SAMPLE_CPU_SHAPE_AND_SCALE_ADMITTED_WITH_FAKE_PREPACK','source_sha256':expected,'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'owner_shape':owner.shape,'prepack':'explicit CPU fake identity clone; not CUDA or actual QPN2 permutation','prepack_calls':calls,'distinct_scale_codes_checked':len(uniq),'float8_formula_vs_torch':'exact','global_scales_gate_up_equal_in_real_sample':vals[3]==vals[5],'limitations':'No GPU, native prepack, output forward, full shard SHA or full model. Byte sample remains independently unverified against complete shard.'},sort_keys=True))
