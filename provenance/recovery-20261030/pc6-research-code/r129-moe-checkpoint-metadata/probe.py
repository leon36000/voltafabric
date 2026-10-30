import json,urllib.request,urllib.parse
ids=['nvidia/Qwen3.6-35B-A3B-NVFP4','ornith-ai/Ornith-1.5-35B-A3B-NVFP4'];out=[]
for repo in ids:
 r={'repo':repo}
 try:
  req=urllib.request.Request('https://huggingface.co/api/models/'+repo+'?blobs=true',headers={'User-Agent':'VoltaFabric-admission'})
  with urllib.request.urlopen(req,timeout=15) as f:data=f.read(6000001)
  if len(data)>6000000:raise ValueError('metadata cap')
  j=json.loads(data);weights=[x for x in j.get('siblings',[]) if x.get('rfilename','').endswith('.safetensors')];r.update({'status':'OK','sha':j.get('sha'),'private':j.get('private'),'gated':j.get('gated'),'tags_license':[t for t in j.get('tags',[]) if t.startswith('license:')],'weights':[{'name':x.get('rfilename'),'size':x.get('size'),'lfs_sha256':x.get('lfs',{}).get('sha256')} for x in weights],'weight_bytes_sum_known':sum(x.get('size',0) for x in weights if isinstance(x.get('size'),int)),'weight_sizes_complete':bool(weights) and all(isinstance(x.get('size'),int) for x in weights)})
 except Exception as e:r.update({'status':'ERROR','error_class':type(e).__name__})
 out.append(r)
print(json.dumps({'scope':'public repository metadata only; no weights read or downloaded','results':out},sort_keys=True))
