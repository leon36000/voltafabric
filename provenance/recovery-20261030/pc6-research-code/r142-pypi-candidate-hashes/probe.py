import concurrent.futures,hashlib,json,os,threading,urllib.request,urllib.parse
from pathlib import Path
root=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r142-pypi-candidate-hashes');source=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gitlab-import-r113-20261028/repository/research/R141/parsed-candidates.json');raw=source.read_bytes();planraw=(root/'plan.json').read_bytes();plan=json.loads(planraw)
if hashlib.sha256(planraw).hexdigest()!='53d088b1b8ecbe6ec92cb6430fc0ae9b60976a915efd20aa9ecbd2c1d7107a0c' or hashlib.sha256(raw).hexdigest()!=plan['source_sha256']:raise ValueError('plan or source mismatch')
a=json.loads(raw)
if len(a)!=plan['source_count']:raise ValueError('count mismatch')
remote=[x for x in a if urllib.parse.urlsplit(x['link']).scheme=='https'];local=[x for x in a if urllib.parse.urlsplit(x['link']).scheme=='file']
if len(remote)!=188 or len(local)!=1 or any(urllib.parse.urlsplit(x['link']).hostname!='files.pythonhosted.org' for x in remote):raise ValueError('unexpected link set')
(root/'responses').mkdir(mode=0o700);(root/'intent.json').write_text(json.dumps({'status':'STARTED_NO_RETRY','requests':len(remote),'plan_sha256':hashlib.sha256(planraw).hexdigest()})+'\n');lock=threading.Lock();budget=[0]
def task(item):
 name=item['name'];version=item['version'];url='https://pypi.org/pypi/'+urllib.parse.quote(name,safe='')+'/'+urllib.parse.quote(version,safe='')+'/json';result={'name':name,'version':version,'selected_link':item['link']}
 try:
  req=urllib.request.Request(url,headers={'User-Agent':'VoltaFabric-dependency-admission'})
  with urllib.request.urlopen(req,timeout=plan['network_timeout_seconds']) as f:
   chunks=[];n=0
   while True:
    chunk=f.read(min(65536,plan['per_response_cap']+1-n))
    if not chunk:break
    n+=len(chunk)
    with lock:
     budget[0]+=len(chunk)
     if budget[0]>plan['aggregate_cap']:raise ValueError('aggregate cap exceeded')
    if n>plan['per_response_cap']:raise ValueError('per-response cap exceeded')
    chunks.append(chunk)
  data=b''.join(chunks);obj=json.loads(data);wanted=urllib.parse.urlsplit(item['link']); matches=[x for x in obj.get('urls',[]) if urllib.parse.urlsplit(x.get('url','')).path==wanted.path and urllib.parse.urlsplit(x.get('url','')).hostname==wanted.hostname]
  if len(matches)!=1:raise ValueError('selected file missing or ambiguous')
  f=matches[0];sha=f.get('digests',{}).get('sha256');size=f.get('size')
  if not isinstance(sha,str) or len(sha)!=64 or type(size)!=int or size<0:raise ValueError('hash or size absent')
  result.update({'status':'HASH_METADATA_FOUND','sha256':sha,'size':size,'filename':f['filename'],'yanked':f.get('yanked'),'release_metadata_sha256':hashlib.sha256(data).hexdigest(),'requires_python':f.get('requires_python')})
  path=root/'responses'/(hashlib.sha256((name+'=='+version).encode()).hexdigest()+'.json')
  with path.open('xb') as z:z.write(data);z.flush();os.fsync(z.fileno())
 except Exception as e:result.update({'status':'FAILED','error_class':type(e).__name__})
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=list(pool.map(task,remote))
results.sort(key=lambda x:x['name']);out={'status':'RELEASE_METADATA_HASH_SURVEY','plan_sha256':hashlib.sha256(planraw).hexdigest(),'remote_count':len(results),'successes':sum(x['status']=='HASH_METADATA_FOUND' for x in results),'failures':sum(x['status']!='HASH_METADATA_FOUND' for x in results),'metadata_bytes':budget[0],'compressed_artifact_bytes_known':sum(x.get('size',0) for x in results),'local_1cat':{'sha256':'8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6','bytes':245787536,'previously_verified':'R125'},'candidates':results,'limitations':'PyPI release metadata is not exact wheel PEP658 dependency metadata or actual wheel download verification; no transitive validation or 0.1.10 override.'}
with (root/'terminal.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:out[k] for k in ('status','remote_count','successes','failures','metadata_bytes','compressed_artifact_bytes_known')}));print('failures',[(x['name'],x.get('error_class')) for x in results if x['status']!='HASH_METADATA_FOUND'])
