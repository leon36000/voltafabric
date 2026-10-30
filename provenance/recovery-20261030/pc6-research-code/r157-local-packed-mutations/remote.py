import ast,base64,hashlib,json,struct,sys,types,copy
p=json.loads(sys.stdin.buffer.read(4000000));raw={k:base64.b64decode(v,validate=True) for k,v in p['files'].items()}
for k,v in raw.items():assert hashlib.sha256(v).hexdigest()==p['plan']['hashes'][k]
import torch
torch.set_num_threads(1);assert torch.version.cuda is None
m=types.ModuleType('vf_r155_adapter');sys.modules[m.__name__]=m;exec(compile(raw['adapter'],'<pinned-adapter>','exec'),m.__dict__)
t=ast.parse(raw['donor']);f=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='_qpn_prepack');tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),f],type_ignores=[]);ast.fix_missing_locations(tree);g={'torch':torch};exec(compile(tree,'<donor-prepack>','exec'),g)
vals=struct.unpack('<6f',raw['scalars']);cs=[torch.frombuffer(bytearray(raw['codes'][i*524288:(i+1)*524288]),dtype=torch.uint8) for i in range(3)];ss=[torch.frombuffer(bytearray(raw['scales'][i*65536:(i+1)*65536]),dtype=torch.uint8) for i in range(3)]
def unpack_qpn(codes, scales, global_scales, n, k):
 tiles,groups=n//32,k//16
 qc=codes.reshape(tiles,groups,32,8);qs=scales.reshape(tiles,groups,32)
 lane=torch.arange(32);col=((lane>>2)&3)*8+(lane&3)+((lane&16)>0).long()*4
 invcol=torch.argsort(col);order=torch.tensor([0,2,4,6,1,3,5,7,8,10,12,14,9,11,13,15]);invk=torch.argsort(order)
 nib=torch.stack((qc&15,qc>>4),dim=-1).reshape(tiles,groups,32,16)
 nib=nib[:,:,invcol,:][:,:,:,invk].permute(0,2,1,3).reshape(n,k).to(torch.int32)
 sf=qs[:,:,invcol].permute(0,2,1).reshape(n,groups).to(torch.int32).repeat_interleave(16,dim=1)
 scale=((sf<<7).to(torch.int16).view(torch.float16)*torch.tensor(256.,dtype=torch.float16)).to(torch.float16)
 gs=global_scales[0].repeat_interleave(32).reshape(n,1);gm=(gs*16384.).to(torch.float16);scale=(scale*gm).to(torch.float16)
 q=(((nib&8)<<12)|((nib&7)<<9)).to(torch.int16).view(torch.float16)
 return (q*scale).to(torch.float16).to(torch.float64)
class Provider:
 def __init__(self,mutation=None):self.calls=0;self.mutation=mutation
 def gemm_qpn2_moe(self,x,codes,scales,global_scales,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.calls+=1;assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1]
  if self.calls==2 and self.mutation=='codes':
   codes=codes.clone();codes.reshape(n//32,k//16,32,8)[0,:,0,:]=0
  if self.calls==2 and self.mutation=='scales':
   scales=scales.clone();scales.reshape(n//32,k//16,32)[0,:,0]=0
  mat=unpack_qpn(codes,scales,global_scales,n,k)
  return torch.mv(mat,x[0].to(torch.float64)).to(torch.float16).reshape(1,n)
def align(ids,block,e):assert ids.tolist()==[[0]] and block==8 and e==1;return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
def layer(expert):
 mp=torch.full((2,),-1,dtype=torch.int64);mp[expert]=0
 l=types.SimpleNamespace(w13_weight=torch.cat((cs[1].reshape(512,1024),cs[2].reshape(512,1024))).reshape(1,1024,1024),w13_weight_scale=torch.cat((ss[1].reshape(512,128),ss[2].reshape(512,128))).reshape(1,1024,128),w13_weight_scale_2=torch.tensor([[vals[3],vals[5]]],dtype=torch.float32),w2_weight=cs[0].reshape(1,2048,256),w2_weight_scale=ss[0].reshape(1,2048,32),w2_weight_scale_2=torch.tensor([vals[1]],dtype=torch.float32),apply_router_weight_on_input=False,expert_map=mp)
 m.prepack_experts(l,prepack=g['_qpn_prepack']);return l
x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);out=[];neg=[];mutations=[]
for rank in p['routing']['result']['ranks']:
 slot=rank['slots'][0]['slot'];l=layer(slot['expert']);provider=Provider();ids,y=m.moe_apply_dispatch_local(l,x,rank,p['stamp'],token_ids=[0],kernel=provider,align=align);assert provider.calls==2 and ids==[(slot['token'],slot['slot'],slot['expert'],slot['rank'],slot['local'])] and torch.isfinite(y).all();out.append({'identity':ids[0],'row':base64.b64encode(y[0].numpy().tobytes()).decode()});
 if slot['rank']==2:
  for kind in ('codes','scales'):
   mp=Provider(kind);mi,my=m.moe_apply_dispatch_local(l,x,rank,p['stamp'],token_ids=[0],kernel=mp,align=align);changed=torch.nonzero(y[0]!=my[0],as_tuple=False).reshape(-1).tolist();assert mp.calls==2 and mi==ids and changed==[0] and y[0,0]!=0 and my[0,0]==0;mutations.append({'kind':kind,'changed_indices':changed,'nominal_row0':float(y[0,0]),'mutated_row0':float(my[0,0])})
 bad=copy.deepcopy(rank);bad['stamp']['epoch']=0;pv=Provider()
 try:m.moe_apply_dispatch_local(l,x,bad,p['stamp'],token_ids=[0],kernel=pv,align=align);raise AssertionError('stale accepted')
 except ValueError:assert pv.calls==0;neg.append('stale-before-provider')
print(json.dumps({'status':'TRUE_ADAPTER_QPN2_LOCAL_ROW_MUTATIONS','rows':out,'provider_calls':8,'stale_refusals':len(neg),'mutations':mutations,'torch':torch.__version__,'torch_cuda_build':torch.version.cuda,'gpu':False,'limits':'CPU inverse permutation and FP16 decode consumes passed packed buffers; no native GPU computation, actual CUDA order or full model.'},sort_keys=True))
