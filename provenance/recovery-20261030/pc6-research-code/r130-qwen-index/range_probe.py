import urllib.request,hashlib,json,re
u='https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/1355db6a052410cfd62085d94b58866fd0f2c3c5/model.safetensors.index.json'; req=urllib.request.Request(u,headers={'User-Agent':'VoltaFabric-index-admission','Range':'bytes=0-262143','Accept-Encoding':'identity'})
try:
 with urllib.request.urlopen(req,timeout=15) as f:
  status=f.status;cr=f.headers.get('Content-Range','')
  if status!=206 or not cr.startswith('bytes 0-262143/'):raise ValueError('range not honored')
  b=f.read(262145)
 if len(b)!=262144:raise ValueError('partial size mismatch')
 lines=b.decode('utf-8','strict').splitlines(); matches=[x.strip() for x in lines if 'layers.0.' in x and ('experts.' in x or 'mlp.gate' in x)];print(json.dumps({'status':'PARTIAL_RANGE_ONLY','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content_range':cr,'matched_count':len(matches),'first_matches':matches[:12],'partial_not_complete_index':True}))
except Exception as e:print(json.dumps({'status':'FAILED_NO_RETRY','error_class':type(e).__name__}));raise SystemExit(1)
