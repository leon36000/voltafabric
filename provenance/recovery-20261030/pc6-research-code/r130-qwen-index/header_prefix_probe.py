import urllib.request,hashlib,json,re
u='https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/1355db6a052410cfd62085d94b58866fd0f2c3c5/model-00001-of-00003.safetensors?download=true';req=urllib.request.Request(u,headers={'User-Agent':'VoltaFabric-header-admission','Range':'bytes=8-262151','Accept-Encoding':'identity'})
try:
 with urllib.request.urlopen(req,timeout=15) as f:
  cr=f.headers.get('Content-Range','')
  if f.status!=206 or not cr.startswith('bytes 8-262151/'):raise ValueError('range not honored')
  b=f.read(262145)
 if len(b)!=262144:raise ValueError('length mismatch')
 s=b.decode('utf-8','strict');r={}
 for m in re.finditer(r'"(model\.language_model\.layers\.0\.mlp\.experts\.0\.[^"]+)"\s*:\s*(\{[^{}]*\})',s):
  try:r[m.group(1)]=json.loads(m.group(2))
  except json.JSONDecodeError:pass
 print(json.dumps({'status':'PARTIAL_HEADER_ONLY','bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'content_range':cr,'expert0_fields':r,'tensor_payload_read':False,'header_complete':False}))
except Exception as e:print(json.dumps({'status':'FAILED_NO_RETRY','error_class':type(e).__name__}));raise SystemExit(1)
