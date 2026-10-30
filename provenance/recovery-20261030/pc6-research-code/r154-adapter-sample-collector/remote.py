import sys,json,base64,hashlib,types,copy
p=json.loads(sys.stdin.buffer.read(200000));source=base64.b64decode(p['adapter'],validate=True);raw=base64.b64decode(p['row'],validate=True);assert hashlib.sha256(source).hexdigest()==p['plan']['adapter_sha256'];assert hashlib.sha256(raw).hexdigest()==p['plan']['row_fp16_sha256']
import torch
torch.set_num_threads(1);assert torch.version.cuda is None
m=types.ModuleType('vf_r154_adapter');sys.modules[m.__name__]=m;exec(compile(source,'<pinned-vf-adapter>','exec'),m.__dict__)
from types import SimpleNamespace as NS
row=torch.frombuffer(bytearray(raw),dtype=torch.float16).clone();assert len(row)==2048
class Provider:
 def __init__(self):self.calls=0
 def gemm_qpn2_moe(self,x,codes,scales,global_scales,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.calls+=1;assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1]
  if self.calls==1:assert (n,k)==(1024,2048);return torch.zeros((1,n),dtype=torch.float16)
  assert self.calls==2 and (n,k)==(2048,512);return row.reshape(1,-1).clone()
def align(ids,block,e):assert ids.tolist()==[[0]] and block==8 and e==1;return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
def layer(expert):
 mapping=torch.full((2,),-1,dtype=torch.int64);mapping[expert]=0
 return NS(skinny_moe_shape=(1,1024,2048,2048,512),skinny_moe_qc13=torch.zeros(1048576,dtype=torch.uint8),skinny_moe_qs13=torch.full((131072,),56,dtype=torch.uint8),skinny_moe_gs13=torch.ones((1,32),dtype=torch.float32),skinny_moe_qc2=torch.zeros(524288,dtype=torch.uint8),skinny_moe_qs2=torch.full((65536,),56,dtype=torch.uint8),skinny_moe_gs2=torch.ones((1,64),dtype=torch.float32),skinny_moe_splitk13=8,skinny_moe_splitk2=8,apply_router_weight_on_input=False,expert_map=mapping)
x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16); outputs=[];negative=[]
for rank in p['routing']['result']['ranks']:
 slot=rank['slots'][0]['slot'];l=layer(slot['expert']);provider=Provider();ids,y=m.moe_apply_dispatch_local(l,x,rank,p['stamp'],token_ids=[0],kernel=provider,align=align);assert provider.calls==2 and torch.equal(y[0],row);assert ids==[(slot['token'],slot['slot'],slot['expert'],slot['rank'],slot['local'])];outputs.append({'identity':ids[0],'row':base64.b64encode(y[0].numpy().tobytes()).decode()})
 bad=copy.deepcopy(rank);bad['stamp']['epoch']=0;pv=Provider()
 try:m.moe_apply_dispatch_local(l,x,bad,p['stamp'],token_ids=[0],kernel=pv,align=align);raise AssertionError('stale accepted')
 except ValueError:assert pv.calls==0;negative.append('stale-before-provider')
print(json.dumps({'status':'REAL_ADAPTER_CPU_FAKE_PROVIDER_ROWS','outputs':outputs,'negative':negative,'calls':4,'gpu':False,'scope':'Real adapter, synthetic planes and explicit provider returning precomputed R153 CPU sample proxy; no actual MoE arithmetic or CUDA.'},sort_keys=True))
