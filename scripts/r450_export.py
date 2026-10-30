"""Extract bounded vectors from an identified isolated layer observation."""
import json, pathlib, struct, sys
import numpy as np
p=pathlib.Path(sys.argv[1]); r=json.loads((p/'result.json').read_text())
def vec(key,length):
    a=np.asarray(r[key],dtype='<f4')
    if a.shape!=(length,) or not np.isfinite(a).all(): raise ValueError(key)
    return a
for key,length in [('ple_residual',10240),('gdn',19044),('activation',2560)]:
    with open(p/(key+'.f32'),'xb') as f: f.write(vec(key,length).tobytes())
with open(p/'post.bin','xb') as f:
    for key,n in [('z',6144),('normalized',6144),('attention',2560),('post_attention',10240),('activation',2560),('injection',4)]: f.write(vec(key,n).tobytes())
    if not all(len(r[k])==10 for k in ('experts','weights','logits')): raise ValueError('route length')
    for a,b,c in zip(r['experts'],r['weights'],r['logits']): f.write(struct.pack('<Iff',a,b,c))
print('exported',r['run_id'],r['experts'])
