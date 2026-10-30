import pathlib,json,hashlib,struct,base64,subprocess,copy
b=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');p=b/'.runs/r161-fp16-owner-collector';prev=b/'.runs/r158-real-resident-owner-cpu';pr=(p/'plan.json').read_bytes();assert hashlib.sha256(pr).hexdigest()=='728b49510d7c1c0d233c042e1631538d9273ebd1ad1bbfb8cb11ff29715e0d4c';plan=json.loads(pr);raw=(prev/'remote.stdout').read_bytes();problem=(prev/'problem.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==plan['owner_stdout_sha256'] and hashlib.sha256(problem).hexdigest()==plan['routing_problem_sha256'];o=json.loads(raw);assert o['status']=='REAL_RUST_RESIDENT_LOAD_OWNER_TWO_CPU_EXECS';problem=json.loads(problem);outputs=[]
for a in reversed(o['results']):
 assert a['two_exec_equal'] and a['counts_after_success']==[4,4,4];row=base64.b64decode(a['row'],validate=True);assert len(row)==4096;token,slot,expert,rank,local=a['identity'];outputs.append({'stamp':problem['stamp'],'rank':rank,'attempt':1,'rows':1,'columns':2048,'slots':[{'token':token,'slot':slot,'expert':expert,'local':local}],'fp16_bits':list(struct.unpack('<2048H',row))})
cli=b/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';assert hashlib.sha256(cli.read_bytes()).hexdigest()==plan['cli_sha256'];cases=[]
def call(name,value):
 body=json.dumps(value,separators=(',',':')).encode();proc=subprocess.run([str(cli),'collect-fp16','-'],input=body,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20);(p/(name+'.json')).write_bytes(proc.stdout);result=json.loads(proc.stdout);cases.append({'case':name,'exit':proc.returncode,'input_sha256':hashlib.sha256(body).hexdigest(),'stdout_sha256':hashlib.sha256(proc.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(proc.stderr).hexdigest(),'ok':result.get('ok')});return proc,result
base={'problem':problem,'expected_attempt':1,'outputs':outputs};r,res=call('nominal',base);assert r.returncode==0 and res['ok'];actual=res['result'];a=struct.unpack('<2048e',base64.b64decode(o['results'][0]['row']));assert actual['output']==[[.75*x for x in a]] and actual['accepted_rows']==2 and actual['fp16_payload_bytes']==8192
for name in plan['failure_mutants']:
 x=copy.deepcopy(base)
 if name=='attempt':x['outputs'][0]['attempt']=2
 elif name=='slot':x['outputs'][0]['slots'][0]['local']=99
 elif name=='duplicate':x['outputs'][1]=copy.deepcopy(x['outputs'][0])
 elif name=='nonfinite':x['outputs'][0]['fp16_bits'][0]=0x7c00
 elif name=='missing':x['outputs'].pop()
 r,res=call(name,x);assert r.returncode!=0 and res['ok'] is False
out={'status':'CAPTURED_REAL_OWNER_FP16_ROWS_TO_RUST_COLLECTOR','accepted_rows':2,'values':2048,'payload_bytes':8192,'cases':cases,'gpu':False,'limits':'Offline consumption of R158 captured CPU-owner rows; reconstructed metadata from associated routing; no live Session bridge, GPU, full model or independent tensor producer.'};(p/'result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'cases':[(x['case'],x['exit']) for x in cases]}))
