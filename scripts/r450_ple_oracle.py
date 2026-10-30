"""Independent sequential PLE math over observed weights and two actual embeddings."""
import hashlib, json, sys
import numpy as np
manifest=json.load(open(sys.argv[1])); observed=json.load(open(sys.argv[3])); embed_paths=sys.argv[4:]
if len(embed_paths)!=1: raise ValueError('expected one embedding')
with open(sys.argv[2],'rb') as f:
    def data(name,dtype):
        spec=manifest[name]; f.seek(spec['offset']); b=f.read(spec['bytes'])
        if len(b)!=spec['bytes'] or hashlib.sha256(b).hexdigest()!=spec['sha256']: raise ValueError(name)
        return np.frombuffer(b,dtype=dtype)
    key=(data('key','<u2').astype(np.uint32)<<16).view(np.float32).reshape(10240,2560).astype(np.float64)
    value=(data('value','<u2').astype(np.uint32)<<16).view(np.float32).reshape(2560,2560).astype(np.float64)
    nk=data('norm_key','<f4').reshape(4,2560); nq=data('norm_query','<f4').reshape(4,2560); nc=data('norm_conv','<f4').reshape(4,2560)
    kernel=data('kernel','<f2').astype(np.float32).reshape(10240,4)
def norm(x,w):
    mean=np.mean((x*x).astype(np.float64),axis=1).astype(np.float32)
    return (x*(np.float32(1)/np.sqrt(mean+np.float32(1e-6)))[:,None])*w
history=np.zeros((10240,9),dtype=np.float32)
for pos,p in enumerate(embed_paths):
    row=json.load(open(p)); emb=np.asarray(row['values'],dtype=np.float32)
    if row['token']!=42 or emb.shape!=(2560,): raise ValueError('embedding identity')
    bits=emb.view(np.uint32); emb=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    rawkey=(key@emb.astype(np.float64)).astype(np.float32).reshape(4,2560)
    rawvalue=(value@emb.astype(np.float64)).astype(np.float32)
    idx=np.arange(10240,dtype=np.int32)
    hidden=((idx%137).astype(np.float32)-np.float32(68))*np.float32(0.0001)
    hidden=hidden.reshape(4,2560); k=norm(rawkey,nk); q=norm(hidden,nq)
    score=np.sum((k*q).astype(np.float64),axis=1).astype(np.float32)/np.sqrt(np.float32(2560))
    gate=np.float32(1)/(np.float32(1)+np.exp(-np.sign(score)*np.sqrt(np.maximum(np.abs(score),np.float32(1e-6)))))
    gated=gate[:,None]*rawvalue[None,:]; normalized=norm(gated,nc).ravel()
    acc=np.zeros(10240,dtype=np.float32)
    for i,v in enumerate([history[:,0],history[:,3],history[:,6],normalized]): acc=acc+kernel[:,i]*v
    result=hidden.ravel()+(gated.ravel()+acc/(np.float32(1)+np.exp(-acc)))
    got=np.asarray(observed['ple_residual'],dtype=np.float32)
    if got.shape!=(10240,): raise ValueError('residual hash')
    if not np.allclose(got,result,rtol=2e-5,atol=2e-5): raise ValueError('PLE residual mismatch')
    print('position',pos,'values',len(got),'max_abs',float(np.max(np.abs(got-result))))
    history=np.concatenate((history[:,1:],normalized[:,None]),axis=1)
