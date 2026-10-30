import json,hashlib,pathlib,struct,math,os
b=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');p=b/'.runs/r159-sampled-rust-session-cpu';plan=json.loads((p/'plan.json').read_bytes());rc=json.loads((p/'receipt.json').read_bytes());raw=(p/'stdout.json').read_bytes();assert rc['status']=='TERMINAL' and rc['exit']==0 and hashlib.sha256(raw).hexdigest()==rc['stdout_sha256'];r=json.loads(raw)['result'];assert r['ok'] and r['exact_batches']==2 and r['batches_ok']==2 and r['exec_requests']==4 and r['error'] is None and len(r['artifacts'])==2 and len(r['close'])==2
for t in r['traffic']:assert t['traffic']['loads_transferred']==1 and t['traffic']['execs']==2 and t['traffic']['rebinds']==0
for c in r['close']:
 assert c['exit_code']==0 and c['bye_execs']==2 and not c['killed'] and c['error'] is None
 try:os.kill(c['pid'],0)
 except ProcessLookupError:pass
 else:raise AssertionError('owned worker PID still exists')
s=b/'.runs/r132-qwen-expert-sample';expected={'codes':'c1bbdf2d6c0c15d7da34352689d65530b4daf981aa6392bfb17d17010dd7e993','scales':'98aa1d1823f373ff28fc078a0443c1de7fa45594a9cfee1751e6d3f4cacc85aa','scalars':'bcf1e13241cb2960c903be775ca03e1a2776d3cc560b394e92137a7fb7b715b2'};data={k:(s/n).read_bytes() for k,n in [('codes','codes.bin'),('scales','block_scales.bin'),('scalars','scalars.bin')]}
for k,v in data.items():assert hashlib.sha256(v).hexdigest()==expected[k]
f=struct.unpack('<6f',data['scalars']);factors=[f[1],f[3],f[5]];shapes=[(2048,512),(512,2048),(512,2048)];codes=[memoryview(data['codes'])[i*524288:(i+1)*524288] for i in range(3)];scales=[memoryview(data['scales'])[i*65536:(i+1)*65536] for i in range(3)]
def e2(n):
 m=n&7;v=.5*m if m<2 else math.ldexp(1.+.5*(m&1),(m>>1)-1);return -v if n&8 else v
def e4(n):
 assert n<127;e=n>>3;m=n&7;return math.ldexp(m,-9) if e==0 else math.ldexp(8+m,e-10)
def mv(j,x):
 rows,cols=shapes[j];c=codes[j];s=scales[j];f=factors[j]
 def w(row,col):
  v=c[row*(cols//2)+col//2];return e2(v>>4 if col&1 else v&15)*e4(s[row*(cols//16)+col//16])*f
 return [math.fsum(w(row,col)*x[col] for col in range(cols)) for row in range(rows)]
max_error=0.;records=r['recorded'];assert [a['batch'] for a in records]==[0,1]
for rec in records:
 batch=rec['batch'];x=[((i%31)-15)/16*(1+((7*batch+i)%11)/64) for i in range(2048)];gate=mv(1,x);up=mv(2,x);a=[(v/(1+math.exp(-v)))*u for v,u in zip(gate,up)];y=mv(0,a);want=[.75*v for v in y];got=rec['output'][0];assert len(got)==2048
 for z,w in zip(got,want):
  err=abs(z-w);max_error=max(max_error,err);assert err<=1e-10+1e-10*abs(w)
out={'status':'RUST_COMPRESSED_SESSION_SAMPLED_EXPERT_CPU_VERIFIED','batches':2,'owners':2,'exec_requests':4,'load_transfers':[1,1],'bye_execs':[2,2],'worker_exits':[0,0],'recorded_values':4096,'scalar_reference_max_abs_error':max_error,'reference_tolerance_abs':1e-10,'reference_tolerance_rel':1e-10,'stdout_sha256':rc['stdout_sha256'],'plan_sha256':hashlib.sha256((p/'plan.json').read_bytes()).hexdigest(),'gpu':False,'limitations':'Rust compressed CPU sessions and Python scalar oracle; no Python PersistentQPN2Owner bridge, no GPU, no full model or full shard hash.'};(p/'result.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))
