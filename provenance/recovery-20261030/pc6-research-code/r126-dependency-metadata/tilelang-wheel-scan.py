import json,urllib.request
out=[]
for v in ['0.1.9','0.1.10']:
 req=urllib.request.Request('https://pypi.org/pypi/tilelang/'+v+'/json',headers={'User-Agent':'VoltaFabric-dependency-admission'})
 with urllib.request.urlopen(req,timeout=12) as f:o=json.load(f)
 a=[]
 for x in o.get('urls',[]):
  n=x.get('filename','')
  if n.endswith('.whl') and ('x86_64' in n or n.endswith('-any.whl')):
   a.append({'filename':n,'bytes':x['size'],'sha256':x['digests']['sha256'],'yanked':x['yanked']})
 out.append({'version':v,'x86_or_any_wheels':a})
print(json.dumps(out))
