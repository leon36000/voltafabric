"""Private CPU multipart load endpoint using the real persistent QPN2 owner.

Research protocol: it is not yet the Rust Session protocol or production transport.
"""
import ast,hashlib,os,pathlib,sys
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path[:0]=[str(R),str(R/'research/R162')]
import torch,wire
from adapters.resident_multipart import ResidentMultipart,MAX_CHUNK_BYTES
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from scripts.resident_artifact import load_stack
assert torch.version.cuda is None;torch.set_num_threads(1)
provider_path=R/'research/R156/remote.py';tree=ast.parse(provider_path.read_bytes());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unpack_qpn');g={'torch':torch};exec(compile(ast.Module(body=[func],type_ignores=[]),'<pinned-cpu-provider>','exec'),g)
class Provider:
 def __init__(self):self.device=torch.device('cpu');self.admissions=self.launches=self.synchronized_launches=0
 def gemm_qpn2_moe(self,*args,**kwargs):raise RuntimeError('execution not enabled in this load-only endpoint')
 def synchronize(self):self.synchronized_launches=self.launches
 def close(self):pass
stream=sys.stdin.buffer;writer=sys.stdout.buffer
hello=wire.read_header(stream)
if set(hello)!={'op','protocol','rank'} or hello['op']!='hello' or hello['protocol']!=1 or type(hello['rank']) is not int:raise ValueError('invalid hello')
rank=hello['rank'];wire.write_frame(writer,{'reply':'hello','hello':{'protocol':1,'rank':rank,'arch':'aarch64','os':'linux','endian':'little','build':'r259-cpu-load-only','capabilities':['cpu_f64_dense_moe_v1','cpu_qpn2_multipart_v1'],'max_message_bytes':16777216,'pid':os.getpid()}})
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
provider=Provider();owner=PersistentQPN2Owner.load(artifact_path=str(path),expected_info=info_holder,expected_sha256=begin['artifact_sha256'],provider=provider,expert_map=mapping,device='cpu')
assert owner.state=='READY';wire.write_frame(writer,{'reply':'loaded','artifact_sha256':begin['artifact_sha256'],'epoch':id['epoch'],'experts':len(info_holder['physical_experts']),'transferred':True})
if wire.read_header(stream)!={'op':'shutdown'}:raise ValueError('shutdown expected')
owner.close();assert owner.state=='CLOSED';wire.write_frame(writer,{'reply':'bye','execs':0})
