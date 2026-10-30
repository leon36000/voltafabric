import collections,hashlib,json,math,struct
from pathlib import Path
p=Path('/home/pc6/NAOS/60_travaux/VoltaFabric-A53-20261012/.runs/r132-qwen-expert-sample');r=json.loads((p/'terminal.json').read_text());assert r['status']=='RANGES_FETCHED_NO_NUMERICAL_ORACLE'; out={'status':'CPU_BYTE_FORMAT_SCREEN','gpu':False,'full_shard_sha_verified':False,'projection_order':['down','gate','up']}
for n in ('scalars','block_scales','codes'):
 b=(p/(n+'.bin')).read_bytes();assert len(b)==r['samples'][n]['bytes'] and hashlib.sha256(b).hexdigest()==r['samples'][n]['sha256']
 if n=='scalars':
  q=struct.unpack('<6f',b);out['scalar_bits']=[b[i:i+4].hex() for i in range(0,24,4)];out['scalar_finite_positive']=all(math.isfinite(x) and x>0 for x in q)
 elif n=='block_scales':
  out['scale_bytes']=len(b);out['negative_sign_bytes']=sum(bool(x&128) for x in b);out['e4m3_nan_bytes']=sum((x&127)==127 for x in b);out['zero_bytes']=sum((x&127)==0 for x in b)
 else:
  lo=collections.Counter(x&15 for x in b);hi=collections.Counter(x>>4 for x in b);out['codes_bytes']=len(b);out['nibble_histogram']=[lo[i]+hi[i] for i in range(16)]
assert out['scalar_finite_positive'];out['scope']='Independent byte-format checks only; no dequant oracle, output, source LFS verification or GPU';print(json.dumps(out,sort_keys=True))
