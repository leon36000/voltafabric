import hashlib,json,math,pathlib,struct,subprocess
b=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');p=b/'.runs/r153-real-sample-cpu-collector';planraw=(p/'plan.json').read_bytes();assert hashlib.sha256(planraw).hexdigest()=='4795a472e6f5847ed8e9cb0d3db1943069bb15f21e9adddab041288e2bbe1521';plan=json.loads(planraw);s=b/'.runs/r132-qwen-expert-sample';vals={n:(s/f).read_bytes() for n,f in [('codes','codes.bin'),('scales','block_scales.bin'),('scalars','scalars.bin')]}
for n,raw in vals.items():assert hashlib.sha256(raw).hexdigest()==plan['sample_sha256'][n]
cli=b/'.runs/gpu-rdma-case-contract-r61/assets/bin/aarch64/vf_cli';assert hashlib.sha256(cli.read_bytes()).hexdigest()==plan['cli_sha256']; factors=struct.unpack('<6f',vals['scalars']);shapes=[(2048,512),(512,2048),(512,2048)];factors=[factors[1],factors[3],factors[5]];codes=[memoryview(vals['codes'])[i*524288:(i+1)*524288] for i in range(3)];scales=[memoryview(vals['scales'])[i*65536:(i+1)*65536] for i in range(3)]
def half(x):return struct.unpack('<e',struct.pack('<e',x))[0]
def bits(x):return struct.unpack('<e',struct.pack('<H',x))[0]
def deq(scale,nibble,factor):
 a=half(bits(scale<<7)*256.0);gm=half(factor*16384.0);sc=half(a*gm);q=bits(((nibble&8)<<12)|((nibble&7)<<9));return half(q*sc)
luts=[[[deq(s,n,f) for n in range(16)] for s in range(127)] for f in factors]
def mv(k,vector):
 rows,cols=shapes[k];c=codes[k];sc=scales[k];lut=luts[k]
 def w(row,col):
  z=c[row*(cols//2)+col//2];n=(z>>4) if col%2 else (z&15);return lut[sc[row*(cols//16)+col//16]][n]
 return [math.fsum(w(row,col)*vector[col] for col in range(cols)) for row in range(rows)]
x=[((i%31)-15)/16 for i in range(2048)];gate=mv(1,x);up=mv(2,x);act=[(a/(1+math.exp(-a)))*z for a,z in zip(gate,up)];y=[half(v) for v in mv(0,act)];assert len(y)==2048 and all(math.isfinite(v) for v in y)
stamp={'run_id':'r153-cpu','model_revision':'qwen-r132-test-copy','plan_id':'duplicate-sample-fixture','layer':0,'batch':0,'epoch':1};problem={'hidden':2048,'experts':[0,1],'routes':[plan['routes']],'registry':{'epoch':1,'copies':[{'expert':0,'rank':2,'local':0},{'expert':1,'rank':9,'local':0}]},'stamp':stamp,'preferred_owners':{},'resident_order':{'2':[0],'9':[0]}}
records=[]
def call(cmd,obj):
 raw=json.dumps(obj,separators=(',',':')).encode();r=subprocess.run([str(cli),cmd,'-'],input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=20);records.append({'command':cmd,'exit':r.returncode,'input_sha256':hashlib.sha256(raw).hexdigest(),'stdout_sha256':hashlib.sha256(r.stdout).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr).hexdigest()});return r,json.loads(r.stdout)
r,route=call('routing',problem);assert r.returncode==0 and route['ok'];slots=route['result']['slots'];assert len(slots)==2
completions=[{'stamp':stamp,**{k:q[k] for k in ['token','slot','expert','rank','local']},'value':y} for q in slots];r,out=call('collect-routing',{'problem':problem,'completions':list(reversed(completions))});assert r.returncode==0 and out['ok'];actual=out['result'];assert len(actual)==1 and len(actual[0])==len(y);expected=[0.75*v for v in y];assert actual[0]==expected
for name,rows in [('missing',completions[:1]),('duplicate',[completions[0],completions[0]]),('stale',[dict(completions[0],stamp=dict(stamp,epoch=0)),completions[1]])]:
 r,o=call('collect-routing',{'problem':problem,'completions':rows});assert r.returncode!=0 or o.get('ok') is False;records[-1]['negative_case']=name
out={'status':'REAL_SAMPLE_CPU_PROXY_TO_RUST_COLLECTOR_VERIFIED','plan_sha256':hashlib.sha256(planraw).hexdigest(),'output_fp16_sha256':hashlib.sha256(struct.pack('<2048e',*y)).hexdigest(),'collector_values':len(y),'real_sample_expert_count':1,'synthetic_identical_experts':2,'weights':[0.25,0.5],'completion_order':'reverse','negative_cases':['missing','duplicate','stale'],'records':records,'limits':'CPU f64 scalar forward proxy with FP16 dequant and output, two synthetic copies of one sampled expert; no Skinny adapter, native kernel, GPU, full shard hash or model.'};(p/'result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
