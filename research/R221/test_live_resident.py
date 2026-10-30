"""Exercise the resident method against a real Session using a base-class stand-in.

Actual 1Cat subclass/constructor compatibility was separately checked in R220.
"""
import hashlib,json,pathlib,sys,types,uuid,torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';D=B/'.runs/r221-live-resident-20261028';sys.path[:0]=[str(R),str(R/'research/R201')]
# PC6 CPU runtime has no 1Cat installation. Stand in only for the abstract base;
# the method under test and the session/owners are the actual VoltaFabric code.
for name in ('vllm','vllm.model_executor','vllm.model_executor.layers','vllm.model_executor.layers.fused_moe'):
 m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
m=types.ModuleType('vllm.model_executor.layers.fused_moe.fused_moe_method_base')
class Base:
 def __init__(self,moe):self.moe=moe;self.moe_quant_config=None;self.moe_kernel=None
m.FusedMoEMethodBase=Base;sys.modules[m.__name__]=m
from adapters.session_bridge import DurableSession
from adapters.onecat_resident_method import ResidentMoEMethod,ResidentAdmissionError
from verify import accumulate
from nvfp4_oracle import expert_forward

def main():
 torch.set_num_threads(1);assert torch.version.cuda is None
 q=B/'.runs/r204-real-experts-20261028';source=q/'experts.safetensors';assert hashlib.sha256(source.read_bytes()).hexdigest()=='bb24ca67eef13e2eb2000caa2c19f64381135d1748c77a7deccc8e51bbce45dd';run='r202-'+uuid.uuid4().hex
 p=json.loads((R/'research/R159/input.json').read_bytes());stamp=p['routing']['stamp'];stamp.update(run_id=run,model_revision='1355db6a052410cfd62085d94b58866fd0f2c3c5',plan_id='r221-live-resident');p['routing']['experts']=list(range(8));p['routing']['routes']=[[{'expert':e,'weight':.125} for e in range(8)]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if e<4 else 9,'local':e%4} for e in range(8)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(q/'session/owners'/f'rank{r}.safetensors') for r in (2,9)];p['fp16_completions']=True;p['workers']=[{'rank':rank,'argv':[str(B/'source/.work/a92-cpu-venv/bin/python'),'-B',str(R/'research/R202/worker.py'),run,str(D/'hashes.json'),str(D)]} for rank in (2,9)]
 bridge=DurableSession(B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session',p,D/'ledger',deadline_seconds=30);layer=types.SimpleNamespace(layer_name='model.layers.0.mlp.experts');method=ResidentMoEMethod(None,bridge,model_revision=stamp['model_revision'],plan_id=stamp['plan_id'],layer_id=stamp['layer'],epoch=stamp['epoch'],layer_name=layer.layer_name)
 try:method.create_weights(layer,8,2048,512,torch.float16,global_num_experts=256)
 except ResidentAdmissionError:pass
 else:raise AssertionError('partial expert coverage admitted')
 method.create_weights(layer,8,2048,512,torch.float16,global_num_experts=8);assert method.get_fused_moe_quant_config(layer) is None
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)]],dtype=torch.float16);ids=torch.tensor([[5,0,7,2,4,1,6,3]]);weights=torch.tensor([[.125]*8]);errs=[]
 for _ in range(2):
  out=method.apply(layer,x,weights,ids,None,None);routes=[[{'expert':int(e),'weight':float(w)} for e,w in zip(ids[0].tolist(),weights[0].tolist())]];expected=accumulate(routes,[x[0]],lambda expert,row:expert_forward(source,0,expert,row));error=float((out.double()-expected).abs().max());assert error<=.0001;errs.append(error)
 closed=bridge.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close'])
 try:method.get_fused_moe_quant_config(layer)
 except ResidentAdmissionError:pass
 else:raise AssertionError('closed session admitted')
 result={'status':'SUCCESS','run_id':run,'batches':2,'actual_session':True,'actual_onecat_base':False,'coverage_refused':True,'closed_refused':True,'max_abs_errors':errs,'gpu':False,'ledger':str(bridge.directory)};(D/'receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
