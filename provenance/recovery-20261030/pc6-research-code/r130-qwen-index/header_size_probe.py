import urllib.request,struct,json,hashlib
u='https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/1355db6a052410cfd62085d94b58866fd0f2c3c5/model-00001-of-00003.safetensors'; req=urllib.request.Request(u,headers={'User-Agent':'VoltaFabric-header-admission','Range':'bytes=0-7','Accept-Encoding':'identity'})
try:
 with urllib.request.urlopen(req,timeout=15) as f:
  status=f.status; cr=f.headers.get('Content-Range','')
  if status!=206 or not cr.startswith('bytes 0-7/'):raise ValueError('range not honored')
  b=f.read(9)
 if len(b)!=8:raise ValueError('unexpected length')
 n=struct.unpack('<Q',b)[0];print(json.dumps({'status':'HEADER_LENGTH_ONLY','length':n,'sha256_prefix':hashlib.sha256(b).hexdigest(),'content_range':cr,'tensor_payload_read':False}))
except Exception as e:print(json.dumps({'status':'FAILED_NO_RETRY','error_class':type(e).__name__}));raise SystemExit(1)
