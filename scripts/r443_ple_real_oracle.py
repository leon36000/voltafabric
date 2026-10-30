"""Independent numpy oracle over observed BF16/F32/F16 checkpoint spans."""
import json, sys, hashlib, os
import numpy as np
m=json.load(open(sys.argv[1])); observed=json.load(open(sys.argv[3])); emb=np.asarray(json.load(open(sys.argv[4]))['values'],dtype=np.float32)
with open(sys.argv[2],'rb') as f:
    def data(name,dtype):
        s=m[name]; f.seek(s['offset']); b=f.read(s['bytes'])
        if len(b)!=s['bytes'] or hashlib.sha256(b).hexdigest()!=s['sha256']: raise ValueError(name)
        return np.frombuffer(b,dtype=dtype)
    key=(data('key','<u2').astype(np.uint32)<<16).view(np.float32).reshape(10240,2560)
    value=(data('value','<u2').astype(np.uint32)<<16).view(np.float32).reshape(2560,2560)
    nk=data('norm_key','<f4').reshape(4,2560); nq=data('norm_query','<f4').reshape(4,2560); nc=data('norm_conv','<f4').reshape(4,2560)
    kernel=data('kernel','<f2').astype(np.float32).reshape(10240,4)
    projected_key=(key.astype(np.float64) @ emb.astype(np.float64)).astype(np.float32).reshape(4,2560)
    projected_value=(value.astype(np.float64) @ emb.astype(np.float64)).astype(np.float32)
idx=np.arange(10240,dtype=np.int32)
hidden=((idx%101-50).astype(np.float32)/np.float32(100)).reshape(4,2560)
hi=np.arange(10240*9,dtype=np.int32); history=((hi%37-18).astype(np.float32)/np.float32(50)).reshape(10240,9)
def norm(x,w):
    mean=np.mean((x*x).astype(np.float64),axis=1).astype(np.float32)
    scale=np.float32(1)/np.sqrt(mean+np.float32(1e-6))
    return (x*scale[:,None])*w
k=norm(projected_key,nk); q=norm(hidden,nq)
s=(np.sum(k.astype(np.float64)*q.astype(np.float64),axis=1).astype(np.float32)/np.sqrt(np.float32(2560)))
gate=np.float32(1)/(np.float32(1)+np.exp(-np.sign(s)*np.sqrt(np.maximum(np.abs(s),np.float32(1e-6)))))
gated=gate[:,None]*projected_value[None,:]; normalized=norm(gated,nc).ravel()
acc=np.zeros(10240,dtype=np.float32)
for i, v in enumerate([history[:,0],history[:,3],history[:,6],normalized]): acc=acc+kernel[:,i]*v
res=hidden.ravel()+gated.ravel()+acc/(np.float32(1)+np.exp(-acc))
next_hist=np.concatenate((history[:,1:],normalized[:,None]),axis=1).ravel()
for name,expected in [('residual',res),('normalized',normalized),('next_history',next_hist)]:
    got=np.asarray(observed[name],dtype=np.float32)
    if got.shape!=expected.shape or not np.isfinite(got).all(): raise ValueError(name+' shape/finite')
    delta=np.abs(got-expected)
    if not np.allclose(got,expected,rtol=2e-5,atol=2e-5): raise ValueError(name+' mismatch '+str(delta.max()))
    print(name,'count',len(got),'max_abs',float(delta.max()))
