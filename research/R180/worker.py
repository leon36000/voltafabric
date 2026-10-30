"""One-batch CPU research endpoint: real owner, synthetic scalar provider; not production."""
import ast,hashlib,json,os,pathlib,struct,sys
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository'; P=pathlib.Path(__file__).resolve().parent
sys.path[:0]=[str(R),str(R/'research/R162')]
import wire,torch
from adapters.persistent_gpu_owner import PersistentQPN2Owner
from scripts.resident_artifact import _decode_json
assert torch.version.cuda is None;torch.set_num_threads(1)
provider_path=R/'research/R156/remote.py';tree=ast.parse(provider_path.read_bytes());func=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='unpack_qpn');g={'torch':torch};exec(compile(ast.Module(body=[func],type_ignores=[]),'<pinned-cpu-provider>','exec'),g)
class Provider:
 def __init__(self, fail=False):self.device=torch.device('cpu');self.admissions=self.launches=self.synchronized_launches=0;self.fail=fail;self.ptrs=[]
 def gemm_qpn2_moe(self,x,codes,scales,gs,blocks,slots,ntpp,s,n,k,in_div,splitk,nacc):
  self.admissions+=1;self.ptrs.append((codes.data_ptr(),scales.data_ptr(),gs.data_ptr()))
  if self.fail: raise RuntimeError('synthetic CPU failure after admission')
  assert s==1 and blocks.tolist()==[0] and slots.tolist()==[0,1,1,1,1,1,1,1];mat=g['unpack_qpn'](codes,scales,gs,n,k);out=torch.mv(mat,x[0].to(torch.float64)).to(torch.float16).reshape(1,n);self.launches+=1;return out
 def synchronize(self):self.synchronized_launches=self.launches
 def close(self):pass
def align(ids,block,e):
 assert ids.tolist()==[[0]] and block==8 and e==1
 return torch.tensor([0,1,1,1,1,1,1,1]),torch.tensor([0]),torch.tensor(8)
stream=sys.stdin.buffer;writer=sys.stdout.buffer;hello=wire.read_header(stream);assert hello['op']=='hello' and hello['protocol']==1;rank=hello['rank'];assert rank in (2,9);run_id=sys.argv[1];assert run_id.startswith('r180-') and len(run_id)==37
caps=['cpu_f64_dense_moe_v1','cpu_qpn2_resident_v1','cpu_qpn2_resident_raw_v1','cpu_exec_fp16_raw_v1','cpu_completion_batch_fp16_raw_v1'];wire.write_frame(writer,{'reply':'hello','hello':{'rank':rank,'protocol':1,'pid':os.getpid(),'os':'linux','arch':'aarch64','endian':'little','build':'real-owner-cpu-test','capabilities':caps,'max_message_bytes':16777216}})
load=wire.read_header(stream);assert load['op']=='load_artifact_raw' and load['epoch']==1;artifact=wire.read_exact(stream,load['artifact_bytes']);digest=hashlib.sha256(artifact).hexdigest();allowed={2:'8f1563c9a1c9308b2b63ebc58ceef94689393402a878386e2f32bdf3166b2ff7',9:'1d3eafe7c7e37f96a03b5bfa1cbce04619adcb50f061c91648a2643555611c6d'};assert digest==load['artifact_sha256']==allowed[rank]
header_len=struct.unpack('<Q',artifact[:8])[0];header=_decode_json(artifact[8:8+header_len]);info=json.loads(header['__metadata__']['vf_info']);assert info['rank']==rank and info['epoch']==1
path=P/(run_id+'-rank'+str(rank)+'.safetensors')
with path.open('xb') as f:f.write(artifact)
expert=info['physical_experts'][0]['id'];mapping=torch.full((2,),-1,dtype=torch.int64);mapping[expert]=0;provider=Provider(fail=len(sys.argv)>2 and sys.argv[2]=='fail-after-admission' and rank==2);owner=PersistentQPN2Owner.load(artifact_path=str(path),expected_info=info,expected_sha256=digest,provider=provider,expert_map=mapping,device='cpu');assert owner.state=='READY';wire.write_frame(writer,{'reply':'loaded','artifact_sha256':digest,'epoch':1,'experts':1,'transferred':True})
execs=0; previous_attempt=0
while True:
 req=wire.read_header(stream)
 if req['op']=='shutdown':
  assert execs==2 and provider.ptrs[0]==provider.ptrs[2] and provider.ptrs[1]==provider.ptrs[3]
  owner.close();assert owner.state=='CLOSED' and provider.admissions==provider.launches==provider.synchronized_launches==2*execs
  wire.write_frame(writer,{'reply':'bye','execs':execs});break
 assert req['op']=='exec_raw_fp16' and req['artifact_sha256']==digest and req['raw_completion_fp16_batch']
 body=wire.read_exact(stream,req['activation_bytes']);tokens,values=wire.decode_fp16_inputs(req,body);d=req['dispatch'];assert d['rank']==rank and d['stamp']['run_id']==run_id and len(d['slots'])==1;slot=d['slots'][0]['slot'];identity=tuple(slot[k] for k in ('token','slot','expert','rank','local'));stamp=d['stamp'];attempt=req['attempt'];assert attempt>previous_attempt;previous_attempt=attempt
 def event(kind,**extra):wire.write_frame(writer,{'reply':'event','event':{'stamp':stamp,'rank':rank,'attempt':attempt,'event':{'kind':kind,**extra}}})
 event('started');result=owner.execute(d,token_ids=tokens,activations=torch.tensor(values,dtype=torch.float16),attempt=attempt,align=align);packed=result.as_raw_output_bytes((identity,));wire.write_frame(writer,{'reply':'completion_batch_fp16_raw','stamp':stamp,'completion_stamp':packed['stamp'],'rank':rank,'attempt':attempt,'hidden':packed['columns'],'values':packed['rows']*packed['columns'],'completions':[{k:slot[k] for k in ('token','slot','expert','local')}]},packed['fp16_le_bytes']);event('finished',slots=1);execs+=1
