import json,urllib.request
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.utils import parse_wheel_filename
from pip._vendor.packaging.tags import sys_tags

def fetch(n,v):
 u='https://pypi.org/pypi/'+n+'/'+v+'/json'
 with urllib.request.urlopen(urllib.request.Request(u,headers={'User-Agent':'VoltaFabric-dependency-admission'}),timeout=12) as f:
  raw=f.read(12000001)
 if len(raw)>12000000:raise ValueError('metadata cap')
 return json.loads(raw)
tags=list(sys_tags());order={tag:i for i,tag in enumerate(tags)}; base=fetch('torch','2.10.0'); out={'scope':'PyPI metadata only, exact Linux x86 pinned torch CUDA dependencies, no install/download/GPU','candidates':[],'errors':[]}
for raw in base['info']['requires_dist']:
 r=Requirement(raw);name=r.name.lower().replace('_','-')
 if not (name.startswith('nvidia-') or name in ('cuda-bindings','triton')):continue
 if r.marker and not r.marker.evaluate():continue
 specs=list(r.specifier)
 if len(specs)!=1 or specs[0].operator!='==':out['errors'].append({'name':name,'error':'not_exact'});continue
 ver=specs[0].version
 try:
  o=fetch(name,ver); candidates=[]
  for x in o.get('urls',[]):
   if x.get('yanked') or not x['filename'].endswith('.whl'):continue
   try: wt=parse_wheel_filename(x['filename'])[3]
   except Exception:continue
   ranks=[order[t] for t in wt if t in order]
   if ranks:candidates.append((min(ranks),x))
  if not candidates:out['errors'].append({'name':name,'version':ver,'error':'no_compatible_wheel'});continue
  x=min(candidates,key=lambda t:(t[0],t[1]['filename']))[1];out['candidates'].append({'name':name,'version':ver,'filename':x['filename'],'size':x['size'],'sha256':x['digests']['sha256'],'url':x['url']})
 except Exception as e:out['errors'].append({'name':name,'version':ver,'error':type(e).__name__})
out['total_bytes']=sum(x['size'] for x in out['candidates']);print(json.dumps(out,sort_keys=True))
