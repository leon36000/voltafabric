import sys,json,base64,hashlib,pathlib,types,ast,copy
payload=json.loads(sys.stdin.buffer.read(7000000));raw={k:base64.b64decode(v,validate=True) for k,v in payload['files'].items()}
for k,v in raw.items():assert hashlib.sha256(v).hexdigest()==payload['sha256'][k]
import torch
torch.set_num_threads(1);assert torch.version.cuda is None
adapters=types.ModuleType('adapters');adapters.__path__=[];sys.modules['adapters']=adapters;scripts=types.ModuleType('scripts');scripts.__path__=[];sys.modules['scripts']=scripts
for name,key in [('adapters.skinny_moe','adapter'),('scripts.resident_artifact','artifact_reader'),('adapters.persistent_gpu_owner','owner')]:
 mod=types.ModuleType(name);sys.modules[name]=mod;exec(compile(raw[key],'<verified-'+key+'>','exec'),mod.__dict__)
 if name=='adapters.skinny_moe':adapters.skinny_moe=mod
owner=sys.modules['adapters.persistent_gpu_owner'];tree=ast.parse(raw['provider']);func=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='unpack_qpn');sub=ast.Module(body=[func],type_ignores=[]);ast.fix_missing_locations(sub);g={'torch':torch};exec(compile(sub,'<r156-packed-cpu-provider>','exec'),g)
class Provider:
 def __init__(self):self.device=torch.device('cpu');self.admissions=0;self.launches=0;self.synchronized_launches=0;self.fail_next=False;self.ptrs=[]
 def gemm_qpn2_moe(self,x,codes,scales,gs,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.admissions+=1;self.ptrs.append((codes.data_ptr(),scales.data_ptr(),gs.data_ptr()))
  if self.fail_next:self.fail_next=False;raise RuntimeError('synthetic CPU failure after admission')
  assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1];mat=g['unpack_qpn'](codes,scales,gs,n,k);out=torch.mv(mat,x[0].to(torch.float64)).to(torch.float16).reshape(1,n);self.launches+=1;return out
 def synchronize(self):self.synchronized_launches=self.launches
 def close(self):pass
def align(ids,block,e):assert ids.tolist()==[[0]] and block==8 and e==1;return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
root=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs/r158-real-resident-owner-cpu-20261028');root.mkdir(mode=0o700);results=[];x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16)
for rank in payload['routing']['result']['ranks']:
 rid=rank['rank'];artifact=raw['artifact'+str(rid)];path=root/('resident-'+str(rid)+'.safetensors')
 with path.open('xb') as f:f.write(artifact)
 info=payload['infos'][str(rid)];provider=Provider();mapping=torch.full((2,),-1,dtype=torch.int64);mapping[rank['slots'][0]['slot']['expert']]=0
 o=owner.PersistentQPN2Owner.load(artifact_path=str(path),expected_info=info,expected_sha256=payload['sha256']['artifact'+str(rid)],provider=provider,expert_map=mapping,device='cpu');assert o.state=='READY'
 outputs=[]
 for attempt in (1,2):
  out=o.execute(rank,token_ids=[0],activations=x,attempt=attempt,align=align);assert out.attempt==attempt and out.rows==1 and out.columns==2048;outputs.append(out)
 assert outputs[0].fp16_le_bytes==outputs[1].fp16_le_bytes and provider.admissions==provider.launches==provider.synchronized_launches==4;assert provider.ptrs[0]==provider.ptrs[2] and provider.ptrs[1]==provider.ptrs[3]
 try:o.execute(rank,token_ids=[0],activations=x,attempt=2,align=align);raise AssertionError('duplicate attempt accepted')
 except owner.OwnerRejected:pass
 result={'rank':rid,'identity':outputs[0].identities[0],'row':base64.b64encode(outputs[0].fp16_le_bytes).decode(),'two_exec_equal':True,'counts_after_success':[provider.admissions,provider.launches,provider.synchronized_launches],'attempt_rejected':True}
 if rid==2:
  provider.fail_next=True
  try:o.execute(rank,token_ids=[0],activations=x,attempt=3,align=align);raise AssertionError('failure accepted')
  except owner.OwnerUnknown:pass
  assert o.state=='POISONED_UNKNOWN'
  try:o.execute(rank,token_ids=[0],activations=x,attempt=4,align=align);raise AssertionError('poison ignored')
  except owner.OwnerUnknown:pass
  o.close();assert o.state=='POISONED_UNKNOWN_CLOSED';result['injected_cpu_unknown_poisoned']=True
 else:o.close();assert o.state=='CLOSED';result['normal_close']=True
 results.append(result)
print(json.dumps({'status':'REAL_RUST_RESIDENT_LOAD_OWNER_TWO_CPU_EXECS','results':results,'gpu':False,'limits':'CPU-only reference provider; injected synthetic UNKNOWN not replayed; no model or GPU.'},sort_keys=True))
