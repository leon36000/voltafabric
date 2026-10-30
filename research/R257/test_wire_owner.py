import hashlib,json,pathlib,subprocess,sys,uuid
B=pathlib.Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012');R=B/'.runs/gitlab-import-r113-20261028/repository';sys.path.insert(0,str(R/'research/R162'));import wire
root=B/'.runs'/('r257-owner-'+uuid.uuid4().hex);root.mkdir(mode=0o700)
artifact=B/'.runs/r255-artifact-frame-20261028/ten/rank2.safetensors';expected='b97f59a4bf9e9720a97d3b546e53f1134d613c49a70bcc47ef15852b82ce0e46';payload=artifact.read_bytes();assert hashlib.sha256(payload).hexdigest()==expected
worker=R/'research/R257/worker.py';python=B/'source/.work/a92-cpu-venv/bin/python'
def start():return subprocess.Popen([str(python),'-B',str(worker),str(root)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,bufsize=0)
def identity():return {'run_id':'r257-'+uuid.uuid4().hex,'model_revision':'1355db6a052410cfd62085d94b58866fd0f2c3c5','plan_id':'r257-load','layer':0,'epoch':1,'rank':2,'attempt':1}
def hello(proc):
 wire.write_frame(proc.stdin,{'op':'hello','protocol':1,'rank':2});h=wire.read_header(proc.stdout);assert h['reply']=='hello' and 'cpu_qpn2_multipart_v1' in h['capabilities'];return h
p=start();h=hello(p);bind=identity();wire.write_frame(p.stdin,{'op':'begin_multipart','identity':bind,'artifact_bytes':len(payload),'artifact_sha256':expected});assert wire.read_header(p.stdout)=={'reply':'multipart_ready','offset':0}
for off in range(0,len(payload),h['max_chunk_bytes']):
 chunk=payload[off:off+h['max_chunk_bytes']];wire.write_frame(p.stdin,{'op':'multipart_chunk','offset':off,'bytes':len(chunk)},chunk);assert wire.read_header(p.stdout)=={'reply':'chunk_accepted','offset':off+len(chunk)}
wire.write_frame(p.stdin,{'op':'multipart_finish'});loaded=wire.read_header(p.stdout);assert loaded=={'reply':'loaded','artifact_sha256':expected,'rank':2,'experts':10};wire.write_frame(p.stdin,{'op':'shutdown'});assert wire.read_header(p.stdout)=={'reply':'bye','execs':0};p.stdin.close();code=p.wait(timeout=30);err=p.stderr.read().decode();assert code==0,(code,err)
# New identity: terminate after an incomplete first chunk; no owner is published.
q=start();hello(q);bad=identity();wire.write_frame(q.stdin,{'op':'begin_multipart','identity':bad,'artifact_bytes':len(payload),'artifact_sha256':expected});wire.read_header(q.stdout);wire.write_frame(q.stdin,{'op':'multipart_chunk','offset':1,'bytes':1},b'x');q.stdin.close();badcode=q.wait(timeout=30);assert badcode!=0
key=hashlib.sha256(json.dumps(bad,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest();assert not (root/key/'resident.bin').exists()
result={'status':'PASS','identity':bind,'artifact_sha256':expected,'bytes':len(payload),'capability':'cpu_qpn2_multipart_v1','owner':'PersistentQPN2Owner READY then CLOSED','refusal':'wrong offset exits without resident','gpu':False,'root':str(root)};(root/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result))
