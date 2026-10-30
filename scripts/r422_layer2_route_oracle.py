"""CPU oracle for SC117 layer-2 norm, residual, FFN read, and routing."""
import os,struct,sys
import numpy as np
if len(sys.argv)!=5:raise SystemExit('usage: oracle DENSE RESIDUAL GDN RESULT')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
def read(o,n):
 b=os.pread(fd,n,o)
 if len(b)!=n:raise ValueError('short span')
 return b
def f32(o,n):return np.frombuffer(read(o,n),dtype='<f4')
def bf(o,n,shape):return (np.frombuffer(read(o,n),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
def rb(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
r=np.fromfile(sys.argv[2],dtype='<f4');gdn=np.fromfile(sys.argv[3],dtype='<f4')
with open(sys.argv[4],'rb') as f: raw=f.read()
if len(r)!=10240 or len(gdn)!=19044 or len(raw)!=110728:raise ValueError('shapes')
out=np.frombuffer(raw,dtype='<f4',count=27652); route=[struct.unpack_from('<Iff',raw,27652*4+i*12) for i in range(10)]
z=out[:6144].reshape(48,128).astype(np.float64);o=gdn[12804:18948].reshape(48,128).astype(np.float64);gamma=f32(168117376,512).astype(np.float64)
y=o/np.sqrt((o*o).mean(axis=1,keepdims=True)+1e-6)*gamma/(1+np.exp(-z))
inj=gdn[2560:2564].astype(np.float64);weight=(2/(1+np.exp(-inj/4))).astype(np.float32);post=(r.reshape(4,2560)+out[12288:14848][None,:]*weight[:,None]).astype(np.float32)
g=f32(160867072,40960).reshape(4,2560).astype(np.float64);p=post.astype(np.float64);xn=(p/np.sqrt(np.mean(p*p,axis=1,keepdims=True)+1e-6)*g).astype(np.float32).reshape(-1);xq=rb(xn)
down=bf(154231552,6553600,(320,10240)).astype(np.float64);up=bf(160908032,6553600,(10240,320)).astype(np.float64);iw=bf(160785152,81920,(4,10240)).astype(np.float64)
lo=(down@xq.astype(np.float64)).astype(np.float32)/np.float32(4);lo=rb((lo/(1+np.exp(-lo))).astype(np.float32));zff=(up@lo.astype(np.float64)).astype(np.float32);v=(xn/(1+np.exp(-zff))).astype(np.float32).reshape(4,2560);mixed=np.zeros(2560,np.float32)
for row in v:mixed=(mixed+row).astype(np.float32)
mixed=(mixed/np.float32(4)).astype(np.float32);inject=(iw@xq.astype(np.float64)).astype(np.float32)
for name,actual,expected in [('norm',out[6144:12288],y.ravel()),('post',out[14848:25088],post.ravel()),('mixed',out[25088:27648],mixed),('inject',out[27648:27652],inject)]:
 err=np.abs(actual.astype(np.float64)-expected);print(name,'count',len(err),'max_abs',err.max())
 if not np.isfinite(actual).all() or not np.isfinite(expected).all() or err.max()>1e-5:raise ValueError(name+' mismatch')
w=bf(138374912,2621440,(512,2560)).astype(np.float64);logits=(w@rb(out[25088:27648]).astype(np.float64)).astype(np.float32);exps=np.exp(logits.astype(np.float64)-float(logits.max()));probs=exps*np.float32(1/exps.sum());ids=np.lexsort((np.arange(512),-probs))[:10];coeff=probs[ids]/max(probs[ids].sum(),2**-14)
if [x[0] for x in route]!=ids.tolist():raise ValueError('route ids mismatch')
err=np.abs(np.array([x[1] for x in route])-coeff);print('ids',ids.tolist(),'weights_max_abs',err.max())
if not np.isfinite(coeff).all() or err.max()>1e-6:raise ValueError('route weights mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('dense changed')
print('attention projections are consumed as input and sampled separately')
