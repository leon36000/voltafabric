import json,pathlib,uuid,torch,time
from safetensors.torch import load_file
from adapters.session_bridge import DurableSession,DurableHookError
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');root=B/'.runs/r279-attributed-owner-failure-r1';root.mkdir();(root/'receiver').mkdir();p=json.loads((B/'.runs/r272-layer3-session-20261028-r2/input.json').read_text());p['routing']['stamp']['run_id']='r279-'+uuid.uuid4().hex;p['routing']['stamp']['plan_id']='r279-fault-attribution';capture=load_file(str(B/'.runs/r272-layer3-owners-20261028/capture.safetensors'));route=json.loads((B/'.runs/r271-layer3-selection-20261028/receipt.json').read_text())['routes'][0]
for worker in p['workers']:
 worker['argv'][-1]=str(root/'receiver')
 if worker['rank']==0:worker['argv'][2]=str(B/'.runs/gitlab-import-r113-20261028/repository/research/R279/failing_worker.py');worker['argv'].append(str(root/'fault-marker.json'))
(root/'input.json').write_text(json.dumps(p,indent=2)+'\n');s=DurableSession(B/'.runs/r181-target/debug/examples/r261_multipart_cpu_session',p,root/'ledger',deadline_seconds=45);ids=torch.tensor([route['ids']],dtype=torch.int64);weights=torch.tensor([route['weights']],dtype=torch.float32);x=capture['batch0.input'];assert 46 in route['ids']
try:s(None,x,weights,ids,None,None)
except DurableHookError as e:failure=str(e)
else:raise AssertionError('injected failure returned output')
assert s.terminal and (root/'fault-marker.json').exists();assert (s.directory/'batch-00000000000000000000.claim.json').exists() and not (s.directory/'batch-00000000000000000000.done.json').exists() and not (s.directory/'batch-00000000000000000000.fp16').exists()
try:s(None,x,weights,ids,None,None)
except DurableHookError:refused=True
else:raise AssertionError('terminal session allowed replay')
try:exitcode=s.proc.wait(timeout=30)
except BaseException:exitcode=None
out={'status':'PASS','run_id':p['routing']['stamp']['run_id'],'marker':json.loads((root/'fault-marker.json').read_text()),'error':failure,'session_terminal':s.terminal,'replay_refused':refused,'coordinator_exit':exitcode,'ledger':str(s.directory),'gpu':False};assert exitcode is not None;(root/'receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
