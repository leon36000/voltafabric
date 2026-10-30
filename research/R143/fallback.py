import json,hashlib,urllib.request,zipfile,email.parser,os,pathlib
root=pathlib.Path(__file__).parent
repo=root.parent/'gitlab-import-r113-20261028/repository/research/R142/terminal.json'
c=next(x for x in json.loads(repo.read_bytes())['candidates'] if x['name']=='pydantic-core')
assert c['size'] < 3_000_000 and c['status']=='HASH_METADATA_FOUND'
plan={'operation':'one exact wheel; no install, one request, no retry','url':c['selected_link'],'size':c['size'],'sha256':c['sha256'],'metadata_limit':1048576}
planbytes=(json.dumps(plan,sort_keys=True,indent=2)+'\n').encode()
with open(root/'fallback-plan.json','xb') as f:f.write(planbytes);f.flush();os.fsync(f.fileno())
result={'plan_sha256':hashlib.sha256(planbytes).hexdigest(),'status':'UNKNOWN'}
try:
 req=urllib.request.Request(plan['url'],headers={'Accept-Encoding':'identity'})
 with urllib.request.urlopen(req,timeout=15) as resp:
  data=resp.read(plan['size']+1)
 assert len(data)==plan['size'] and hashlib.sha256(data).hexdigest()==plan['sha256'], 'wheel length or hash mismatch'
 wp=root/c['filename']
 with open(wp,'xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 with zipfile.ZipFile(wp) as z:
  names=[i for i in z.infolist() if i.filename.endswith('.dist-info/METADATA')]
  assert len(names)==1 and names[0].file_size<=plan['metadata_limit']
  raw=z.read(names[0]);m=email.parser.BytesParser().parsebytes(raw)
  assert m['Name'].lower().replace('_','-')==c['name'] and m['Version']==c['version']
 mp=root/'pydantic-core.metadata'
 with open(mp,'xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 result.update(status='VERIFIED_WHEEL_AND_METADATA',wheel_sha256=plan['sha256'],metadata_sha256=hashlib.sha256(raw).hexdigest(),metadata_size=len(raw),name=c['name'],version=c['version'],requires_dist=m.get_all('Requires-Dist',[]),requires_python=m['Requires-Python'])
except Exception as e:
 result.update(status='FAILED_NO_RETRY',error_class=type(e).__name__,reason=str(e)[:300])
with open(root/'fallback-receipt.json','x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps({k:v for k,v in result.items() if k!='requires_dist'}))
