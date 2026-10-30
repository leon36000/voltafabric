import ast,base64,hashlib,json,struct,sys,types,copy
p=json.loads(sys.stdin.buffer.read(4000000));raw={k:base64.b64decode(v,validate=True) for k,v in p['files'].items()}
for k,v in raw.items():assert hashlib.sha256(v).hexdigest()==p['plan']['hashes'][k]
import torch
torch.set_num_threads(1);assert torch.version.cuda is None
m=types.ModuleType('vf_r155_adapter');sys.modules[m.__name__]=m;exec(compile(raw['adapter'],'<pinned-adapter>','exec'),m.__dict__)
t=ast.parse(raw['donor']);f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_qpn_prepack');tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[]);ast.fix_missing_locations(tree);g={'torch':torch};exec(compile(tree,'<donor-prepack>','exec'),g)
vals=struct.unpack('<6f',raw['scalars']);cs=[torch.frombuffer(bytearray(raw['codes'][i*524288:(i+1)*524288]),dtype=torch.uint8) for i in range(3)];ss=[torch.frombuffer(bytearray(raw['scales'][i*65536:(i+1)*65536]),dtype=torch.uint8) for i in range(3)]
def matrix(i,rows,cols,factor):
 c=cs[i].reshape(rows,cols//2);n=torch.stack((c&15,c>>4),dim=-1).reshape(rows,cols).to(torch.int32);rb=ss[i].reshape(rows,cols//16).to(torch.int32).repeat_interleave(16,dim=1);sc=((rb<<7).to(torch.int16).view(torch.float16)*torch.tensor(256.,dtype=torch.float16)).to(torch.float16);gm=(torch.tensor(factor,dtype=torch.float32)*16384.).to(torch.float16);sc=(sc*gm).to(torch.float16);q=(((n&8)<<12)|((n&7)<<9)).to(torch.int16).view(torch.float16);return (q*sc).to(torch.float16).to(torch.float64)
mat13=torch.cat((matrix(1,512,2048,vals[3]),matrix(2,512,2048,vals[5])),dim=0);mat2=matrix(0,2048,512,vals[1])
class Provider:
 def __init__(self):self.calls=0
 def gemm_qpn2_moe(self,x,codes,scales,global_scales,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.calls+=1;assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1];mat=mat13 if self.calls==1 else mat2;assert tuple(mat.shape)==(n,k);return torch.mv(mat,x[0].to(torch.float64)).to(torch.float16).reshape(1,n)
def align(ids,block,e):assert ids.tolist()==[[0]] and block==8 and e==1;return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
def layer(expert):
 mp=torch.full((2,),-1,dtype=torch.int64);mp[expert]=0
 l=types.SimpleNamespace(w13_weight=torch.cat((cs[1].reshape(512,1024),cs[2].reshape(512,1024))).reshape(1,1024,1024),w13_weight_scale=torch.cat((ss[1].reshape(512,128),ss[2].reshape(512,128))).reshape(1,1024,128),w13_weight_scale_2=torch.tensor([[vals[3],vals[5]]],dtype=torch.float32),w2_weight=cs[0].reshape(1,2048,256),w2_weight_scale=ss[0].reshape(1,2048,32),w2_weight_scale_2=torch.tensor([vals[1]],dtype=torch.float32),apply_router_weight_on_input=False,expert_map=mp)
 m.prepack_experts(l,prepack=g['_qpn_prepack']);return l
x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);out=[];neg=[]
for rank in p['routing']['result']['ranks']:
 slot=rank['slots'][0]['slot'];l=layer(slot['expert']);provider=Provider();ids,y=m.moe_apply_dispatch_local(l,x,rank,p['stamp'],token_ids=[0],kernel=provider,align=align);assert provider.calls==2 and ids==[(slot['token'],slot['slot'],slot['expert'],slot['rank'],slot['local'])] and torch.isfinite(y).all();out.append({'identity':ids[0],'row':base64.b64encode(y[0].numpy().tobytes()).decode()});bad=copy.deepcopy(rank);bad['stamp']['epoch']=0;pv=Provider()
 try:m.moe_apply_dispatch_local(l,x,bad,p['stamp'],token_ids=[0],kernel=pv,align=align);raise AssertionError('stale accepted')
 except ValueError:assert pv.calls==0;neg.append('stale-before-provider')
print(json.dumps({'status':'TRUE_ADAPTER_REAL_SAMPLE_DONOR_PREPACK_CPU_MATMUL','rows':out,'provider_calls':4,'stale_refusals':len(neg),'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'gpu':False,'limits':'Torch CPU matmul reconstructs dense FP16 weights from original sample rather than decoding passed packed buffers; no native GPU computation, actual CUDA order or full model.'},sort_keys=True))
