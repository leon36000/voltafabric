import concurrent.futures,email,hashlib,json,os,threading,urllib.request,urllib.parse
from pathlib import Path
from packaging.utils import canonicalize_name
from packaging.version import Version
root=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r143-pep658-metadata');source=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/gitlab-import-r113-20261028/repository/research/R142/terminal.json');raw=source.read_bytes();planraw=(root/'plan.json').read_bytes();plan=json.loads(planraw)
if hashlib.sha256(planraw).hexdigest()!='7b47337f9d93714ac2c4b2b5ed19f7dde21bd008a52257484bb6b19909468dcf' or hashlib.sha256(raw).hexdigest()!=plan['source_sha256']:raise ValueError('plan or source mismatch')
o=json.loads(raw);items=o['candidates'];assert len(items)==188 and all(x['status']=='HASH_METADATA_FOUND' for x in items)
(root/'indexes').mkdir(mode=0o700);(root/'metadata').mkdir(mode=0o700);(root/'intent.json').write_text(json.dumps({'status':'STARTED_NO_RETRY','count':len(items),'plan_sha256':hashlib.sha256(planraw).hexdigest()})+'\n');lock=threading.Lock();budget=[0];requests=[0]
def get(url,limit,accept=None):
 with lock:
  requests[0]+=1
  if requests[0]>plan['request_cap']:raise ValueError('request cap')
 headers={'User-Agent':'VoltaFabric-dependency-admission'}
 if accept:headers['Accept']=accept
 req=urllib.request.Request(url,headers=headers)
 with urllib.request.urlopen(req,timeout=plan['timeout']) as f:
  chunks=[];n=0
  while True:
   part=f.read(min(65536,limit+1-n))
   if not part:break
   n+=len(part)
   with lock:
    budget[0]+=len(part)
    if budget[0]>plan['aggregate_cap']:raise ValueError('aggregate cap')
   if n>limit:raise ValueError('per response cap')
   chunks.append(part)
 return b''.join(chunks)
def save(path,data):
 with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def task(x):
 name=x['name'];version=x['version'];result={'name':name,'version':version,'filename':x['filename'],'wheel_sha256':x['sha256']}
 try:
  project=canonicalize_name(name);idx=get('https://pypi.org/simple/'+urllib.parse.quote(project,safe='')+'/',plan['index_cap'],'application/vnd.pypi.simple.v1+json');j=json.loads(idx);matches=[f for f in j['files'] if f.get('filename')==x['filename']]
  if len(matches)!=1:raise ValueError('index file missing or ambiguous')
  f=matches[0];u=urllib.parse.urlsplit(f['url']);selected=urllib.parse.urlsplit(x['selected_link'])
  if u.scheme!='https' or u.hostname!='files.pythonhosted.org' or u.path!=selected.path or f.get('hashes',{}).get('sha256')!=x['sha256']:raise ValueError('index artifact mismatch')
  mh=f.get('core-metadata') or f.get('dist-info-metadata')
  if not isinstance(mh,dict) or not isinstance(mh.get('sha256'),str):raise ValueError('missing hashed metadata')
  expected=mh['sha256'];meta=get(urllib.parse.urlunsplit((u.scheme,u.netloc,u.path+'.metadata','','')),plan['metadata_cap']);actual=hashlib.sha256(meta).hexdigest()
  if actual!=expected:raise ValueError('metadata hash mismatch')
  m=email.message_from_bytes(meta)
  if canonicalize_name(m.get('Name',''))!=project or Version(m.get('Version',''))!=Version(version):raise ValueError('metadata identity mismatch')
  key=hashlib.sha256((name+'=='+version).encode()).hexdigest();save(root/'indexes'/(key+'.json'),idx);save(root/'metadata'/(key+'.metadata'),meta)
  result.update({'status':'VERIFIED','index_sha256':hashlib.sha256(idx).hexdigest(),'metadata_sha256':actual,'metadata_bytes':len(meta),'requires_dist':m.get_all('Requires-Dist',[]),'requires_python':m.get('Requires-Python')})
 except Exception as e:result.update({'status':'FAILED','error_class':type(e).__name__,'reason':str(e)[:180]})
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=plan['workers']) as pool:results=list(pool.map(task,items))
results.sort(key=lambda x:x['name']);out={'status':'PEP658_METADATA_SURVEY','plan_sha256':hashlib.sha256(planraw).hexdigest(),'total':len(results),'verified':sum(x['status']=='VERIFIED' for x in results),'failed':sum(x['status']!='VERIFIED' for x in results),'requests':requests[0],'network_bytes':budget[0],'candidates':results,'limits':'No wheels downloaded, no dependency graph admission, local 1cat excluded, no override or GPU.'};save(root/'terminal.json',(json.dumps(out,sort_keys=True,indent=2)+'\n').encode());print(json.dumps({k:out[k] for k in ('status','total','verified','failed','requests','network_bytes')}));print('failures',[(x['name'],x.get('reason')) for x in results if x['status']!='VERIFIED'])
