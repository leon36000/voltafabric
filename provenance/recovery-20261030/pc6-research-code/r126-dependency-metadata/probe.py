import json, urllib.request, urllib.error, sys
packages=[('tilelang','0.1.9'),('tilelang','0.1.10'),('apache-tvm-ffi','0.1.9'),('apache-tvm-ffi','0.1.10'),('torch','2.10.0')]
output={'scope':'PyPI release metadata only; no wheel download, install, import or GPU','results':[]}
for name,version in packages:
 item={'name':name,'version':version}
 try:
  u='https://pypi.org/pypi/'+name+'/'+version+'/json'; req=urllib.request.Request(u,headers={'User-Agent':'VoltaFabric-dependency-admission'});
  with urllib.request.urlopen(req,timeout=12) as f:
   raw=f.read(12000001)
   if len(raw)>12000000: raise ValueError('metadata exceeds cap')
  o=json.loads(raw); matches=[]
  for x in o.get('urls',[]):
   n=x.get('filename','')
   if n.endswith('.whl') and (('cp312' in n and ('x86_64' in n or 'any.whl' in n)) or n.endswith('py3-none-any.whl')):
    matches.append({'filename':n,'size':x.get('size'),'sha256':x.get('digests',{}).get('sha256'),'yanked':x.get('yanked')})
  item.update({'status':'READ','requires_python':o.get('info',{}).get('requires_python'),'requires_dist':o.get('info',{}).get('requires_dist'),'matching_wheels':matches})
 except Exception as e:item.update({'status':'ERROR','error_class':type(e).__name__})
 output['results'].append(item)
print(json.dumps(output,sort_keys=True))
