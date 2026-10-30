import copy,json,pathlib,subprocess,sys,torch,uuid,os,hashlib
from adapters.session_bridge import DurableSession
from nvfp4_oracle import expert_forward
base=pathlib.Path('/run/media/pc1/Infrastructure/Naos/60_travaux/VoltaFabric-offline-20261007-cd798a1c/.runs');root=base/'r288-nine-session-r1';root.mkdir();(root/'receiver').mkdir();code=base/'r288-code-r1';python=base/'r178-isolated-1cat-cuda128-20261028-r1/venv/bin/python';sock='/run/media/pc1/Infrastructure/Naos/.r288-owner-r1.sock';stderr=(root/'daemon-stderr.txt').open('wb');daemon=subprocess.Popen([str(python),'-B',str(code/'multiplex.py'),sock,str(root/'receiver'),'9'],stdout=subprocess.PIPE,stderr=stderr,bufsize=0);assert daemon.stdout.readline()==b'READY\n';(root/'daemon-pid.json').write_text(json.dumps({'pid':daemon.pid})+'\n')
p=json.loads((base/'r281-layer1-session-r1/input.json').read_text());sessions={};run='r288-'+uuid.uuid4().hex;torch.set_num_threads(1);torch.manual_seed(285);x=(torch.randn(1,2048)*.1).half();ids=torch.tensor([[5,7]],dtype=torch.int64);weights=torch.tensor([[.625,.375]],dtype=torch.float32);p['inputs']=[x[0].tolist()]
try:
 for layer in range(9):
  q=copy.deepcopy(p);manifest=json.loads((base/'r283-four-layer-owners-r1'/'layer0'/'manifest.json').read_text());q['artifacts']=[manifest['ranks']['0']['path']];q['workers']=[{'rank':0,'argv':['/usr/bin/python3','-B',str(code/'proxy.py'),sock]}];q['routing']['experts']=list(range(64));q['routing']['registry']={'epoch':1,'copies':[{'expert':i,'rank':0,'local':i} for i in range(64)]};q['routing']['resident_order']={'0':list(range(64))};q['routing']['stamp'].update(run_id=run,plan_id=f'r288-multiplex-{layer}',layer=0,batch=0);q['routing']['routes']=[[{'expert':5,'weight':.625},{'expert':7,'weight':.375}]];(root/f'input{layer}.json').write_text(json.dumps(q,indent=2)+'\n');sessions[layer]=DurableSession(base/'r280-rust-x86-r1/target/debug/examples/r261_multipart_cpu_session',q,root/f'ledger{layer}',deadline_seconds=180)
 results=[]
 for batch in (0,):
  for layer in range(9):
   out=sessions[layer](None,x,weights,ids,None,None);source=base/'r262-full-layer-r1'/'experts.safetensors';expect=(expert_forward(source,0,5,x[0]).double()*.625+expert_forward(source,0,7,x[0]).double()*.375).half();err=(out[0].float()-expect.float()).abs().max().item();assert err<1e-4,(batch,layer,err);results.append({'batch':batch,'session':layer,'actual_layer':0,'max_abs':err})
 for layer in range(9):assert sessions[layer].close()['state']=='closed'
 assert daemon.stdout.readline()==b'CLOSED\n' and daemon.wait(timeout=30)==0 and not torch.cuda.is_initialized()
 result={'status':'PASS','run_id':run,'daemon_pid':daemon.pid,'distinct_sessions':9,'owners_in_process':9,'model_layers':1,'cases':results,'gpu':False};(root/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
except BaseException as e:
 (root/'failure.json').write_text(json.dumps({'status':'UNKNOWN','run_id':run,'error':str(e),'daemon_pid':daemon.pid})+'\n');raise
finally:stderr.close()
