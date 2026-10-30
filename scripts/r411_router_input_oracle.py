"""Independent NumPy reference for token-42 residual write and FFN read."""
import os
import json
import sys
import numpy as np
if len(sys.argv) != 6:
    raise SystemExit('usage: oracle DENSE EMBEDDING GDN_OUTPUT RESULT ROUTE_JSON')
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)
before = os.fstat(fd)
def raw(offset, size):
    value = os.pread(fd, size, offset)
    if len(value) != size: raise ValueError('short read')
    return value
def f32(offset, size): return np.frombuffer(raw(offset, size), dtype='<f4')
def bf16(offset, size, shape): return (np.frombuffer(raw(offset,size),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
def round_bf16(x):
    bits = np.asarray(x,dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7fff) + ((bits >> 16) & 1)) & np.uint32(0xffff0000)).view(np.float32)
def calc(res, offsets):
    norm,down,up,inj=offsets
    g=f32(norm,40960).reshape(4,2560).astype(np.float64)
    xn=(res.astype(np.float64) / np.sqrt(np.mean(res.astype(np.float64)**2,axis=1,keepdims=True)+1e-6)*g).astype(np.float32).reshape(-1)
    xq=round_bf16(xn)
    d=bf16(down,6553600,(320,10240)).astype(np.float64)
    u=bf16(up,6553600,(10240,320)).astype(np.float64)
    i=bf16(inj,81920,(4,10240)).astype(np.float64)
    lo=(d @ xq.astype(np.float64)).astype(np.float32)/np.float32(4)
    lo=round_bf16((lo/(1+np.exp(-lo))).astype(np.float32))
    gate=(u @ lo.astype(np.float64)).astype(np.float32)
    v=(xn/(1+np.exp(-gate))).astype(np.float32).reshape(4,2560)
    mixed=np.zeros(2560,dtype=np.float32)
    for row in v: mixed=(mixed+row).astype(np.float32)
    mixed=(mixed/np.float32(4)).astype(np.float32)
    inject=(i @ xq.astype(np.float64)).astype(np.float32)
    return mixed,inject
emb=np.fromfile(sys.argv[2],dtype='<f4'); gdn=np.fromfile(sys.argv[3],dtype='<f4'); result=np.fromfile(sys.argv[4],dtype='<f4')
if len(emb)!=2560 or len(gdn)!=14848 or len(result)!=12804: raise ValueError('input shape')
res=np.tile(emb,(4,1)); _,inj=calc(res,(22410240,15774720,22451200,22328320))
w=(2/(1+np.exp(-(inj.astype(np.float64)/4)))).astype(np.float32)
post=(res+(gdn[12288:][None,:]*w[:,None])).astype(np.float32)
mixed,ffn_inj=calc(post,(35640320,29004800,35681280,35558400))
for name,actual,expected in [('router',result[:2560],mixed),('inject',result[2560:2564],ffn_inj),('residual',result[2564:],post.ravel())]:
    diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64)); print(name,'count',diff.size,'max_abs',diff.max())
    if not np.isfinite(expected).all() or not np.isfinite(actual).all() or diff.max()>1e-5: raise ValueError(name+' mismatch')
weights=bf16(13148160,2621440,(512,2560)).astype(np.float64)
logits=(weights @ round_bf16(result[:2560]).astype(np.float64)).astype(np.float32)
exps=np.exp(logits.astype(np.float64)-float(logits.max()))
probs=exps*np.float32(1.0/exps.sum())
ids=np.lexsort((np.arange(512),-probs))[:10]
coeff=probs[ids]/max(probs[ids].sum(),2**-14)
with open(sys.argv[5],encoding='utf-8') as handle: actual=json.load(handle)
if [x['expert'] for x in actual] != ids.tolist(): raise ValueError('route ids mismatch')
err=np.abs(np.array([x['weight'] for x in actual])-coeff)
print('route ids',ids.tolist(),'max_weight_error',err.max())
if not np.isfinite(coeff).all() or err.max()>1e-6: raise ValueError('route weight mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise ValueError('dense changed')
