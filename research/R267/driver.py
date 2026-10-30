import json,pathlib,struct,subprocess,sys
from safetensors.torch import load_file
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');root=pathlib.Path(sys.argv[1]) if len(sys.argv)>1 else B/'.runs/r261-multipart-exec-20261028';p=json.loads((root/'input.json').read_text());captures=load_file(str(root/'capture.safetensors'));binary=B/'.runs/r181-target/debug/examples/r261_multipart_cpu_session'
with (root/'coordinator-stderr.txt').open('wb') as err:
 proc=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,bufsize=0)
 def send(x):proc.stdin.write(json.dumps(x,separators=(',',':'),allow_nan=False).encode()+b'\n');proc.stdin.flush()
 def recv(label):
  b=proc.stdout.readline();(root/(label+'.json')).write_bytes(b)
  if not b:raise RuntimeError('coordinator ended; inspect receipt, no replay')
  value=json.loads(b)
  if not value.get('ok'):raise RuntimeError('coordinator failed '+label)
  return value
 send(p);assert recv('ready')['state']=='ready'
 for b in range(2):
  x=captures[f'batch{b}.input'].contiguous().numpy().tobytes();bits=list(struct.unpack('<2048H',x));ids=captures[f'batch{b}.ids'][0].tolist();weights=captures[f'batch{b}.weights'][0].tolist();routes=[[{'expert':int(e),'weight':float(w)} for e,w in zip(ids,weights)]];send({'batch':b,'routes':routes,'inputs_fp16_bits':[bits]});assert recv('batch'+str(b))['batch']==b
 send({'op':'shutdown'});assert recv('close')['state']=='closed';proc.stdin.close();code=proc.wait(timeout=30)
 if code:raise RuntimeError('coordinator nonzero exit')
(root/'driver-success.json').write_text(json.dumps({'status':'PASS','run_id':p['routing']['stamp']['run_id'],'batches':2,'gpu':False})+'\n');print('PASS')
