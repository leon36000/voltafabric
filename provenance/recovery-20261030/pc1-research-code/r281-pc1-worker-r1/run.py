import json,uuid,pathlib,torch,os
from safetensors import safe_open
from safetensors.torch import save_file
from adapters.session_bridge import DurableSession
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r281-layer1-session-r1';root.mkdir();(root/'receiver').mkdir();p=json.loads((base/'r281-pc1-worker-r1/template.json').read_text());manifest=json.loads((base/'r280-layer1-r1/owners/manifest.json').read_text());p['artifacts']=[manifest['ranks'][str(i)]['path'] for i in range(16)];p['routing']['stamp']={'run_id':'r281-'+uuid.uuid4().hex,'model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r281-layer1-pc1','layer':1,'batch':0,'epoch':1};p['routing']['experts']=list(range(256));p['routing']['registry']={'epoch':1,'copies':[{'expert':i,'rank':i//16,'local':i%16} for i in range(256)]};p['routing']['resident_order']={str(i):list(range(16)) for i in range(16)}
py=str(base/'r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python');worker=str(base/'r281-pc1-worker-r1/worker.py');p['workers']=[{'rank':i,'argv':[py,'-B',worker,str(root/'receiver')]} for i in range(16)]
torch.set_num_threads(1);torch.manual_seed(281);x=(torch.randn(2,2048)*.1).half();checkpoint=base/'r188-qwen-checkpoint-pc1-20261028-r1/model/model-00001-of-00003.safetensors'
with safe_open(str(checkpoint),framework='pt',device='cpu') as f:gate=f.get_tensor('model.language_model.layers.1.mlp.gate.weight').half()
logits=(x.float()@gate.float().T).half().float();top,ids=logits.topk(8,-1);weights=top.softmax(-1);p['routing']['routes']=[[{'expert':int(e),'weight':float(w)} for e,w in zip(ids[0],weights[0])]];p['inputs']=[x[0].tolist()];(root/'input.json').write_text(json.dumps(p,indent=2)+'\n');save_file({'x':x,'ids':ids,'weights':weights},str(root/'capture.safetensors'))
s=DurableSession(base/'r280-rust-x86-r1/target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=180)
try:
 for b in range(2):
  out=s(None,x[b:b+1],weights[b:b+1],ids[b:b+1],None,None);assert out.shape==(1,2048)
 closed=s.close();assert len(closed['close'])==16 and not torch.cuda.is_initialized()
 result={'status':'PASS','run_id':p['routing']['stamp']['run_id'],'ledger':str(s.directory),'routes':ids.tolist(),'owners':16,'gpu_initialized':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except BaseException as e:
 (root/'failure.json').write_text(json.dumps({'status':'UNKNOWN','error':str(e),'run_id':p['routing']['stamp']['run_id'],'pid':s.proc.pid})+'\n');raise
