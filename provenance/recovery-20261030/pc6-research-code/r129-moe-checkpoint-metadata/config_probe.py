import hashlib,json,urllib.request
items=[('nvidia/Qwen3.6-35B-A3B-NVFP4','1355db6a052410cfd62085d94b58866fd0f2c3c5'),('ornith-ai/Ornith-1.5-35B-A3B-NVFP4','94e431d9cc47fa1986a7a1a4e9a80f7f118b03aa')];out=[]
for repo,rev in items:
 x={'repo':repo,'revision':rev}
 try:
  req=urllib.request.Request('https://huggingface.co/'+repo+'/resolve/'+rev+'/config.json',headers={'User-Agent':'VoltaFabric-admission'})
  with urllib.request.urlopen(req,timeout=15) as f: b=f.read(2000001)
  if len(b)>2000000:raise ValueError('config cap')
  j=json.loads(b);keys=['architectures','model_type','num_experts','num_local_experts','num_experts_per_tok','hidden_size','quantization_config','text_config'];x.update({'status':'OK','config_sha256':hashlib.sha256(b).hexdigest(),'fields':{k:j[k] for k in keys if k in j}})
 except Exception as e:x.update({'status':'ERROR','error_class':type(e).__name__})
 out.append(x)
print(json.dumps({'scope':'small pinned config JSON only; no model weights','results':out},sort_keys=True))
