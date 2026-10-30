"""Private CPU multipart load endpoint using the real persistent QPN2 owner.

Research protocol: it is not yet the Rust Session protocol or production transport.
"""
import ast,hashlib,os,pathlib,sys
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path[:0]=[str(R),str(R/'research/R162')]
import torch,wire
from adapters.resident_multipart import ResidentMultipart,MAX_CHUNK_BYTES
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from adapters.qpn2_cpu_reference import unpack_exact
from scripts.resident_artifact import load_stack
assert torch.version.cuda is None;torch.set_num_threads(1)
provider_path=R/'research/R156/remote.py';tree=ast.parse(provider_path.read_bytes());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unpack_qpn');g={'torch':torch};exec(compile(ast.Module(body=[func],type_ignores=[]),'<pinned-cpu-provider>','exec'),g)
class Provider:
 def __init__(self, fail=False):self.device=torch.device('cpu');self.admissions=self.launches=self.synchronized_launches=0;self.fail=fail;self.ptrs=[]
 def gemm_qpn2_moe(self,x,codes,scales,gs,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.admissions+=1;self.ptrs.append((codes.data_ptr(),scales.data_ptr(),gs.data_ptr()))
  if self.fail:
   marker=pathlib.Path(sys.argv[2]);payload=b'{"event":"injected_after_admission","gpu":false}\n'
   with marker.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
   raise RuntimeError('synthetic CPU failure after admission')
  count=codes.numel()//(n*k//2);out=torch.empty((s,n),dtype=torch.float16);cc=codes.reshape(count,-1);ss=scales.reshape(count,-1);gg=gs.reshape(count,-1);matrices={};covered=set()
  assert int(ntpp)==len(blocks)*8 and slots.numel()==len(blocks)*8
  for block,expert in enumerate(blocks.tolist()):
   assert 0<=expert<count
   if expert not in matrices:matrices[expert]=unpack_exact(cc[expert],ss[expert],gg[expert:expert+1],n,k).double()
   for value in slots[block*8:block*8+8].tolist():
    if value>=s:continue
    assert value>=0 and value not in covered;covered.add(value);out[value]=torch.mv(matrices[expert],x[value].to(torch.float64)).to(torch.float16)
  assert covered==set(range(s));self.launches+=1;return out
 def synchronize(self):self.synchronized_launches=self.launches
 def close(self):pass
def align(ids,block,e):
 assert block==8 and ids.ndim==2 and ids.shape[1]==1
 values=ids.flatten().tolist();s=len(values);permuted=[];blocks=[]
 for expert in range(e):
  rows=[index for index,x in enumerate(values) if x==expert]
  for base in range(0,len(rows),block):
   chunk=rows[base:base+block];permuted.extend(chunk+[s]*(block-len(chunk)));blocks.append(expert)
 assert all(0<=x<e for x in values)
 return torch.tensor(permuted),torch.tensor(blocks),torch.tensor(len(permuted))
stream=sys.stdin.buffer;writer=sys.stdout.buffer
hello=wire.read_header(stream)
if set(hello)!={'op','protocol','rank'} or hello['op']!='hello' or hello['protocol']!=1 or type(hello['rank']) is not int:raise ValueError('invalid hello')
rank=hello['rank'];wire.write_frame(writer,{'reply':'hello','hello':{'protocol':1,'rank':rank,'arch':'aarch64','os':'linux','endian':'little','build':'r267-exact-scale-cpu','capabilities':['cpu_f64_dense_moe_v1','cpu_qpn2_multipart_v1','cpu_qpn2_resident_v1','cpu_exec_fp16_raw_v1','cpu_completion_batch_fp16_raw_v1'],'max_message_bytes':16777216,'pid':os.getpid()}})
begin=wire.read_header(stream)
if set(begin)!={'op','identity','artifact_bytes','artifact_sha256'} or begin['op']!='begin_multipart':raise ValueError('invalid begin')
id=begin['identity']
if type(id)!=dict or id.get('rank')!=rank:raise ValueError('rank mismatch')
receiver=ResidentMultipart(pathlib.Path(sys.argv[1]),id,begin['artifact_bytes'],begin['artifact_sha256']);wire.write_frame(writer,{'reply':'multipart_ready','offset':0})
offset=0
while offset<begin['artifact_bytes']:
 h=wire.read_header(stream)
 if set(h)!={'op','offset','bytes'} or h['op']!='multipart_chunk' or type(h['bytes']) is not int or not 0<h['bytes']<=MAX_CHUNK_BYTES:raise ValueError('invalid chunk header')
 data=wire.read_exact(stream,h['bytes']);receiver.append(h['offset'],data);offset+=len(data);wire.write_frame(writer,{'reply':'chunk_accepted','offset':offset})
finish=wire.read_header(stream)
if finish!={'op':'multipart_finish'}:raise ValueError('invalid finish')
info_holder={}
def validate(path,bound):
 artifact=load_stack(path,expected_sha256=begin['artifact_sha256']);info=artifact.metadata
 if any(info[k]!=bound[k] for k in ('model_revision','layer','epoch','rank')):raise ValueError('resident binding mismatch')
 info_holder.update(info)
path=receiver.finish(validate);mapping=torch.full((256,),-1,dtype=torch.int64)
for physical,expert in enumerate(info_holder['physical_experts']):
 if expert['id']>=len(mapping):raise ValueError('expert outside model domain')
 mapping[expert['id']]=physical
provider=Provider(fail=True);owner=PersistentQPN2Owner.load(artifact_path=str(path),expected_info=info_holder,expected_sha256=begin['artifact_sha256'],provider=provider,expert_map=mapping,device='cpu')
assert owner.state=='READY';wire.write_frame(writer,{'reply':'loaded','artifact_sha256':begin['artifact_sha256'],'epoch':id['epoch'],'experts':len(info_holder['physical_experts']),'transferred':True})
execs=0; previous_attempt=0
while True:
 req=wire.read_header(stream)
 if req['op']=='shutdown':
  assert execs >= 0
  owner.close();assert owner.state=='CLOSED' and provider.admissions==provider.launches==provider.synchronized_launches==2*execs
  wire.write_frame(writer,{'reply':'bye','execs':execs});break
 assert req['op']=='exec_raw_fp16' and req['artifact_sha256']==begin['artifact_sha256'] and req['raw_completion_fp16_batch']
 body=wire.read_exact(stream,req['activation_bytes']);tokens,values=wire.decode_fp16_inputs(req,body);d=req['dispatch'];assert d['rank']==rank and d['stamp']['run_id']==id['run_id'] and len(d['slots'])>0;slots=[entry['slot'] for entry in d['slots']];identities=tuple(tuple(slot[k] for k in ('token','slot','expert','rank','local')) for slot in slots);stamp=d['stamp'];attempt=req['attempt'];assert attempt>previous_attempt;previous_attempt=attempt
 def event(kind,**extra):wire.write_frame(writer,{'reply':'event','event':{'stamp':stamp,'rank':rank,'attempt':attempt,'event':{'kind':kind,**extra}}})
 event('started');result=owner.execute(d,token_ids=tokens,activations=torch.tensor(values,dtype=torch.float16),attempt=attempt,align=align);packed=result.as_raw_output_bytes(identities);wire.write_frame(writer,{'reply':'completion_batch_fp16_raw','stamp':stamp,'completion_stamp':packed['stamp'],'rank':rank,'attempt':attempt,'hidden':packed['columns'],'values':packed['rows']*packed['columns'],'completions':[{k:slot[k] for k in ('token','slot','expert','local')} for slot in slots]},packed['fp16_le_bytes']);event('finished',slots=len(slots));execs+=1
