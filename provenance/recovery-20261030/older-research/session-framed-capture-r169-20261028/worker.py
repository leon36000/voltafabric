"""CPU-only captured-output endpoint; never runs an owner or GPU."""
import ast,base64,copy,hashlib,json,os,pathlib,struct,sys
from dataclasses import dataclass
from typing import Any
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R/'research/R162'));import wire
src=(R/'adapters/persistent_gpu_owner.py').read_bytes();assert hashlib.sha256(src).hexdigest()=='e12945ae9167843f7f056bcc12734ee222af5f361def1f108138960168d823f3';tree=ast.parse(src);nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('OwnerRejected','OwnerOutput')];env={'dataclass':dataclass,'Any':Any,'deepcopy':copy.deepcopy,'struct':struct,'_MAX_RAW_OUTPUT_VALUES':16777216};exec(compile(ast.Module(body=nodes,type_ignores=[]),'<pinned-owner>','exec'),env);OwnerOutput=env['OwnerOutput']
stream=sys.stdin.buffer;writer=sys.stdout.buffer
hello=wire.read_header(stream);assert hello['op']=='hello' and hello['protocol']==1;rank=hello['rank'];assert rank in (2,9)
caps=['cpu_f64_dense_moe_v1','cpu_qpn2_resident_v1','cpu_qpn2_resident_raw_v1','cpu_exec_fp16_raw_v1','cpu_completion_batch_fp16_raw_v1'];wire.write_frame(writer,{'reply':'hello','hello':{'rank':rank,'protocol':1,'pid':os.getpid(),'os':'linux','arch':'aarch64','endian':'little','build':'cpu-captured-test','capabilities':caps,'max_message_bytes':16777216}})
load=wire.read_header(stream);assert load['op']=='load_artifact_raw' and load['epoch']==1;artifact=wire.read_exact(stream,load['artifact_bytes']);assert hashlib.sha256(artifact).hexdigest()==load['artifact_sha256'];wire.write_frame(writer,{'reply':'loaded','artifact_sha256':load['artifact_sha256'],'epoch':1,'experts':1,'transferred':True})
req=wire.read_header(stream);assert req['op']=='exec_raw_fp16' and req['artifact_sha256']==load['artifact_sha256'] and req['raw_completion_fp16_batch'];data=wire.read_exact(stream,req['activation_bytes']);tokens,values=wire.decode_fp16_inputs(req,data);assert tokens==[0] and values==[[((i%31)-15)/16 for i in range(2048)]]
d=req['dispatch'];assert d['rank']==rank and len(d['slots'])==1;slot=d['slots'][0]['slot'];identity=tuple(slot[k] for k in ('token','slot','expert','rank','local'));stamp=d['stamp'];attempt=req['attempt']
capture=(B/'.runs/r158-real-resident-owner-cpu/remote.stdout').read_bytes();assert hashlib.sha256(capture).hexdigest()=='2667891667f5d05d2b7a9eb28c3f0a8f88c201a6c464a3583038d438687958f6';item=next(x for x in json.loads(capture)['results'] if x['rank']==rank);assert tuple(item['identity'])==identity
body=base64.b64decode(item['row'],validate=True);packed=OwnerOutput(stamp=stamp,rank=rank,attempt=attempt,identities=(identity,),rows=1,columns=2048,fp16_le_bytes=body).as_raw_output_bytes((identity,))
def event(kind,**extra):wire.write_frame(writer,{'reply':'event','event':{'stamp':stamp,'rank':rank,'attempt':attempt,'event':{'kind':kind,**extra}}})
if len(sys.argv)>1 and sys.argv[1]=='bad' and rank==2: slot=copy.deepcopy(slot);slot['local']=99
event('started');wire.write_frame(writer,{'reply':'completion_batch_fp16_raw','stamp':stamp,'completion_stamp':stamp,'rank':rank,'attempt':attempt,'hidden':2048,'values':2048,'completions':[{k:slot[k] for k in ('token','slot','expert','local')}]},packed['fp16_le_bytes']);event('finished',slots=1)
end=wire.read_header(stream);assert end['op']=='shutdown';wire.write_frame(writer,{'reply':'bye','execs':1})
