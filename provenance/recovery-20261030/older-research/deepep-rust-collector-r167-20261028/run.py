import copy,hashlib,json,pathlib,struct,subprocess
base=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012'); out=pathlib.Path(__file__).resolve().parent
cli=base/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli'; cli_sha=hashlib.sha256(cli.read_bytes()).hexdigest(); assert cli_sha=='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4'
stamp={'run_id':'r167-cpu-synthetic','model_revision':'deepep-oracle-only','plan_id':'r167-three-rows','layer':0,'batch':0,'epoch':1}
problem={'hidden':1,'experts':[0,1,2,3],'routes':[[{'expert':i,'weight':1.0} for i in range(3)]],'registry':{'epoch':1,'copies':[{'expert':i,'rank':i//2,'local':i%2} for i in range(4)]},'stamp':stamp,'preferred_owners':{},'resident_order':{'0':[0,1],'1':[0,1]}}
outputs=[]
for i,v in enumerate([2048.,1.,-2048.]):
 outputs.append({'stamp':stamp,'rank':i//2,'attempt':1,'rows':1,'columns':1,'slots':[{'token':0,'slot':i,'expert':i,'local':i%2}],'fp16_bits':[struct.unpack('<H',struct.pack('<e',v))[0]]})
base_req={'problem':problem,'expected_attempt':1,'outputs':outputs}
results={}
for name,req in [('forward',base_req),('reverse',{**base_req,'outputs':list(reversed(outputs))}),('missing',{**base_req,'outputs':outputs[:-1]})]:
 body=json.dumps(req,separators=(',',':')).encode(); p=subprocess.run([str(cli),'collect-fp16','-'],input=body,capture_output=True,timeout=15); payload=json.loads(p.stdout); results[name]={'exit':p.returncode,'payload':payload,'input_sha256':hashlib.sha256(body).hexdigest(),'stderr':p.stderr.decode()}; (out/(name+'-input.json')).write_bytes(body)
assert results['forward']['exit']==results['reverse']['exit']==0,results
assert results['forward']['payload']['result']['output']==results['reverse']['payload']['result']['output']==[[1.0]],results
assert results['missing']['exit']!=0 and not results['missing']['payload']['ok'],results
receipt={'cli_sha256':cli_sha,'cases':results,'gpu':False}; (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
