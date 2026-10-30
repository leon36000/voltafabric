import urllib.request,hashlib,json
url='https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/resolve/1355db6a052410cfd62085d94b58866fd0f2c3c5/model-00001-of-00003.safetensors?download=true'; length=6937600; start=8;end=start+length-1;total=10006877608
req=urllib.request.Request(url,headers={'User-Agent':'VoltaFabric-header-admission','Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
def strict(pairs):
 d={}
 for k,v in pairs:
  if k in d:raise ValueError('duplicate JSON key')
  d[k]=v
 return d
with urllib.request.urlopen(req,timeout=25) as f:
 cr=f.headers.get('Content-Range','')
 if f.status!=206 or cr!=f'bytes {start}-{end}/{total}':raise ValueError('range not honored')
 b=f.read(length+1)
if len(b)!=length:raise ValueError('header length mismatch')
j=json.loads(b,object_pairs_hook=strict);prefix='model.language_model.layers.0.mlp.experts.0.'; fields=['input_scale','weight','weight_scale','weight_scale_2'];names=[prefix+proj+'.'+field for proj in ('gate_proj','up_proj','down_proj') for field in fields];missing=[n for n in names if n not in j]
if missing:raise ValueError('missing '+repr(missing))
selected={n:j[n] for n in names}
for n,v in selected.items():
 if not isinstance(v,dict) or not isinstance(v.get('data_offsets'),list) or len(v['data_offsets'])!=2:raise ValueError('invalid offset')
 a,z=v['data_offsets']
 if type(a)!=int or type(z)!=int or not (0<=a<=z<=total-8-length):raise ValueError('out-of-bounds offset')
print(json.dumps({'status':'FULL_HEADER_EXPERT0_ONLY','header_sha256':hashlib.sha256(b).hexdigest(),'header_bytes':len(b),'content_range':cr,'shard_bytes':total,'selected':selected,'scope':'header metadata only; no tensor payload, no model/GPU, full shard SHA not verified'},sort_keys=True))
