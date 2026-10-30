"""CPU pipeline: compact checkpoint -> resident owners -> Rust -> independent oracle."""
import hashlib,json,pathlib,struct,subprocess,sys,uuid
import torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path[:0]=[str(R/'research/R200'),str(R/'research/R201')];from pack_checkpoint import pack;from verify import verify
CLI=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';PY=B/'source/.work/a92-cpu-venv/bin/python'
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def run(plan_path,root):
 plan_path=pathlib.Path(plan_path);plan_raw=plan_path.read_bytes();plan=json.loads(plan_raw);root=pathlib.Path(root);root.mkdir();state={'status':'STARTED','operation_id':plan['operation_id'],'plan_sha256':hashlib.sha256(plan_raw).hexdigest()}
 def publish():
  p=root/'progress.tmp';p.write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');p.replace(root/'progress.json')
 publish()
 try:
  assert plan['gpu'] is False and plan['experts']==list(range(8)) and plan['layer']==0 and 0<plan['oracle_atol']<=.01
  state['status']='PACKING';publish();packed=pack(source=plan['source'],source_sha=plan['source_sha256'],output=root/'owners',revision=plan['revision'],layer=plan['layer'],epoch=1,groups={2:[0,1,2,3],9:[4,5,6,7]},cli=CLI,cli_sha=plan['cli_sha256'])
  hashes={r:v['sha256'] for r,v in packed.items()};(root/'hashes.json').write_text(json.dumps(hashes,sort_keys=True)+'\n');run_id='r202-'+uuid.uuid4().hex;state.update(status='SESSION_STARTING',run_id=run_id,hashes=hashes);publish()
  p=json.loads((R/'research/R159/input.json').read_bytes());p['routing']['stamp'].update(run_id=run_id,plan_id=plan['operation_id'],model_revision=plan['revision']);p['routing']['experts']=list(range(8));p['routing']['routes']=[[{'expert':e,'weight':.125} for e in [5,0,7,2,4,1,6,3]],[{'expert':e,'weight':.0625} for e in [3,6,1,4,2,7,0,5]]];p['routing']['registry']['copies']=[{'expert':e,'rank':2 if e<4 else 9,'local':e%4} for e in range(8)];p['routing']['resident_order']={'2':list(range(4)),'9':list(range(4))};p['artifacts']=[packed[str(r)]['path'] for r in (2,9)];p['workers']=[{'rank':r,'argv':[str(PY),'-B',str(R/'research/R202/worker.py'),run_id,str(root/'hashes.json'),str(root)]} for r in (2,9)];bits=[struct.unpack('<H',struct.pack('<e',v))[0] for v in p.pop('inputs')[0]];p['inputs_fp16_bits']=[bits,[0]*2048];p.update(fp16_completions=True,tolerance_abs=.01,timeout_ms=60000);raw=json.dumps(p,separators=(',',':')).encode();(root/'input.json').write_bytes(raw)
  result=subprocess.run([str(CLI),'session-run-compressed','-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=120);(root/'output.json').write_bytes(result.stdout);(root/'stderr.txt').write_bytes(result.stderr)
  if result.returncode:raise RuntimeError('session failed, output retained; no replay')
  state.update(status='VERIFYING',session_exit=result.returncode);publish();torch.set_num_threads(1);assert torch.version.cuda is None
  comparison=verify(source=plan['source'],source_sha=plan['source_sha256'],input_path=root/'input.json',input_sha=hashlib.sha256(raw).hexdigest(),output_path=root/'output.json',output_sha=hashlib.sha256(result.stdout).hexdigest(),atol=plan['oracle_atol']);(root/'oracle.json').write_text(json.dumps(comparison,sort_keys=True,indent=2)+'\n');state.update(status='SUCCEEDED',oracle=comparison)
 except BaseException as e:
  state.update(status='FAILED_OR_UNKNOWN_NO_REPLAY',error_type=type(e).__name__,error=str(e)[:500]);publish();raise
 finally:
  publish()
 (root/'receipt.json').write_text(json.dumps(state,sort_keys=True,indent=2)+'\n');return state
