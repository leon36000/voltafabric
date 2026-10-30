"""Live CPU integration with distinct checkpoint experts and independent oracle."""
import hashlib,json,pathlib,sys,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';D=B/'.runs/r215-general-session-bridge-20261028';sys.path[:0]=[str(R),str(R/'research/R201')]
from adapters.session_bridge import DurableSession,DurableHookError
from verify import accumulate
from nvfp4_oracle import expert_forward

def main():
 torch.set_num_threads(1);assert torch.version.cuda is None
 q=B/'.runs/r206-gate-route-20261028'; ids=json.loads((q/'routes.json').read_text())['selected_experts']; source=q/'selected.safetensors';assert hashlib.sha256(source.read_bytes()).hexdigest()=='0e597ecf47a568d6066ac2490bb09b37f615a1f725f9bdac9364cab4c195928b'
 run='r202-'+uuid.uuid4().hex;p=json.loads((R/'research/R159/input.json').read_bytes());p['routing']['stamp'].update(run_id=run,model_revision='1355db6a052410cfd62085d94b58866fd0f2c3c5',plan_id='r215-general-session-bridge');p['routing']['experts']=ids;p['routing']['routes']=[[{'expert':e,'weight':.125} for e in ids]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if i<4 else 9,'local':i%4} for i,e in enumerate(ids)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[str(q/'owners'/f'rank{rank}.safetensors') for rank in (2,9)];p['fp16_completions']=True
 p['workers']=[{'rank':rank,'argv':[str(B/'source/.work/a92-cpu-venv/bin/python'),'-B',str(B/'.runs/r207-gate-runner-20261028/worker.py'),run,str(D/'hashes.json'),str(D)]} for rank in (2,9)]
 bridge=DurableSession(B/'.runs/r181-target/debug/examples/r181_persistent_cpu_session',p,D/'ledger',deadline_seconds=30)
 x=torch.tensor([[((i%31)-15)/16 for i in range(2048)], [((i%17)-8)/32 for i in range(2048)]],dtype=torch.float16);routes=torch.tensor([ids,list(reversed(ids))]);weights=torch.tensor([[.125]*8,[.0625]*8],dtype=torch.float32)
 for bad in (torch.tensor([ids,[ids[0]]*8]),torch.tensor([ids,[999]*8])):
  try:bridge(None,x,weights,bad,None,None)
  except DurableHookError:pass
  else:raise AssertionError('invalid route was admitted')
 assert bridge.batch==0 and not bridge.terminal
 errors=[]
 for _ in range(2):
  out=bridge(None,x,weights,routes,None,None);r=[[{'expert':int(e),'weight':float(w)} for e,w in zip(row,ww)] for row,ww in zip(routes.tolist(),weights.tolist())];expected=accumulate(r,list(x),lambda expert,row:expert_forward(source,0,expert,row));err=float((out.to(torch.float64)-expected).abs().max());assert err<=.0001;errors.append(err)
 closed=bridge.close();assert all(c['exit_code']==0 and c['bye_execs']==2 for c in closed['close']);receipt={'status':'SUCCESS','run_id':run,'batches':2,'tokens':2,'rejected_routes':2,'max_abs_errors':errors,'gpu':False,'ledger':str(bridge.directory)};(D/'receipt.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n');print(json.dumps(receipt))
if __name__=='__main__':main()
