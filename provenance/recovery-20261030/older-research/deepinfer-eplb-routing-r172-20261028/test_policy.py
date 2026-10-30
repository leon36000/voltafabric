import ast,copy,hashlib,json,pathlib,subprocess,torch
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012'); P=pathlib.Path(__file__).resolve().parent; src=B/'.runs/deepinfer-source-audit-r163-20261028/source/deepinfer/model_executor/eplb/policy.py'; raw=src.read_bytes();nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('_batch_greedy_indices','rebalance_experts_global')];assert len(nodes)==2; scope={'torch':torch};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(src),'exec'),scope)
torch.set_num_threads(1); assert torch.version.cuda is None
cli=B/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli'; assert hashlib.sha256(cli.read_bytes()).hexdigest()=='11b8a4a515f582de169ffe9abf35e13811e1ad93d52586e607dafac3687f87c4'
results={}
for name,counts in [('balanced',[10,10,10,10]),('hot',[100,20,10,5])]:
 physical,logical,n=scope['rebalance_experts_global'](torch.tensor([[counts]],dtype=torch.float32),6,3); physical=physical.tolist()[0];logical=logical.tolist()[0];n=n.tolist()[0]; assert len(physical)==6 and set(physical)==set(range(4))
 copies=[{'expert':e,'rank':i//3,'local':i%3} for i,e in enumerate(physical)];stamp={'run_id':'r172-'+name,'model_revision':'synthetic','plan_id':'deepinfer-eplb-cpu','layer':0,'batch':0,'epoch':1};problem={'hidden':1,'experts':[0,1,2,3],'routes':[[{'expert':0,'weight':0.25},{'expert':1,'weight':0.75}]],'registry':{'epoch':1,'copies':copies},'stamp':stamp,'preferred_owners':{},'resident_order':{'0':[0,1,2],'1':[0,1,2]}}
 body=json.dumps(problem,separators=(',',':')).encode();p=subprocess.run([str(cli),'routing','-'],input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15);reply=json.loads(p.stdout);(P/(name+'-input.json')).write_bytes(body);(P/(name+'-output.json')).write_bytes(p.stdout);results[name]={'counts':counts,'physical_to_logical':physical,'logical_to_physical':logical,'copies_per_expert':n,'exit':p.returncode,'reply':reply}
 assert (p.returncode==0 and reply['ok']) if name=='balanced' else (p.returncode!=0 and not reply['ok']), results[name]
 if name=='balanced':
  slots=[x['slot'] for rank in reply['result']['ranks'] for x in rank['slots']]; assert sorted((s['expert'],s['weight']) for s in slots)==[(0,0.25),(1,0.75)]
 else: assert len({(c['expert'],c['rank']) for c in copies})<len(copies)
receipt={'upstream_sha256':hashlib.sha256(raw).hexdigest(),'upstream_git_commit':'553741557a57df44a46fb3ab72b6aafeaf945337','cli_sha256':hashlib.sha256(cli.read_bytes()).hexdigest(),'gpu':False,'cases':results};(P/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:{'physical':v['physical_to_logical'],'exit':v['exit'],'error':v['reply'].get('error')} for k,v in results.items()}))
