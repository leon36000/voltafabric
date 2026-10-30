import json,urllib.request,hashlib,collections
repo='nvidia/Qwen3.6-35B-A3B-NVFP4';rev='1355db6a052410cfd62085d94b58866fd0f2c3c5';url='https://huggingface.co/'+repo+'/resolve/'+rev+'/model.safetensors.index.json';req=urllib.request.Request(url,headers={'User-Agent':'VoltaFabric-index-admission'})
with urllib.request.urlopen(req,timeout=15) as f:b=f.read(12000001)
if len(b)>12000000:raise ValueError('index too large')
j=json.loads(b);m=j['weight_map']; keys=sorted(k for k in m if 'layers.0.' in k and ('expert' in k.lower() or 'mlp.gate' in k.lower()));counts=collections.Counter(m[k] for k in keys);print(json.dumps({'repo':repo,'revision':rev,'index_sha256':hashlib.sha256(b).hexdigest(),'index_bytes':len(b),'total_tensors':len(m),'layer0_expert_or_gate_tensors':len(keys),'layer0_names':keys[:40],'layer0_shard_counts':dict(counts),'scope':'pinned index JSON only; no weight bytes, no GPU'},sort_keys=True))
