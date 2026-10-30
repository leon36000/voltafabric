"""Independent NumPy check of layer-2 HC, gates and zero-state recurrence."""
import os,sys
import numpy as np
if len(sys.argv)!=4:raise SystemExit('usage: oracle DENSE RESIDUAL RESULT')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
def read(o,n):
 b=os.pread(fd,n,o)
 if len(b)!=n:raise ValueError('short span')
 return b
def f32(o,n):return np.frombuffer(read(o,n),dtype='<f4')
def bf(o,n,shape):return (np.frombuffer(read(o,n),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
def rb(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
r=np.fromfile(sys.argv[2],dtype='<f4');out=np.fromfile(sys.argv[3],dtype='<f4')
if len(r)!=10240 or len(out)!=19044:raise ValueError('shapes')
r=r.reshape(4,2560).astype(np.float64);g=f32(147636992,40960).reshape(4,2560).astype(np.float64)
xn=(r/np.sqrt(np.mean(r*r,axis=1,keepdims=True)+1e-6)*g).astype(np.float32).reshape(-1);xq=rb(xn)
down=bf(141001472,6553600,(320,10240)).astype(np.float64);up=bf(147677952,6553600,(10240,320)).astype(np.float64);wi=bf(147555072,81920,(4,10240)).astype(np.float64)
lo=(down@xq.astype(np.float64)).astype(np.float32)/np.float32(4);lo=rb((lo/(1+np.exp(-lo))).astype(np.float32));z=(up@lo.astype(np.float64)).astype(np.float32)
v=(xn/(1+np.exp(-z))).astype(np.float32).reshape(4,2560);mixed=np.zeros(2560,dtype=np.float32)
for row in v:mixed=(mixed+row).astype(np.float32)
mixed=(mixed/np.float32(4)).astype(np.float32);inj=(wi@xq.astype(np.float64)).astype(np.float32)
alpha=bf(167461824,245760,(48,2560)).astype(np.float64);beta=bf(167707584,245760,(48,2560)).astype(np.float64)
a=f32(167461632,192);dt=f32(168117184,192);conv=f32(167953344,163840).reshape(10240,4)
mx=out[:2560].astype(np.float64);aa=(alpha@mx).astype(np.float32);bb=(beta@mx).astype(np.float32);z=aa+dt;gate=(np.where(z>20,z,np.logaddexp(np.float32(0),z))*a).astype(np.float32);bt=(1/(1+np.exp(-bb.astype(np.float64)))).astype(np.float32)
qkv=out[2564:12804];t=(qkv*conv[:,3]).astype(np.float32);h=(t/(1+np.exp(-t.astype(np.float64)))).astype(np.float32);q=h[:2048].reshape(16,128);k=h[2048:4096].reshape(16,128);v=h[4096:].reshape(48,128)
q=(q/np.sqrt((q.astype(np.float64)**2).sum(axis=1,keepdims=True)+1e-6)).astype(np.float32);k=(k/np.sqrt((k.astype(np.float64)**2).sum(axis=1,keepdims=True)+1e-6)).astype(np.float32)
y=np.empty((48,128),np.float64)
for j in range(48):y[j]=np.dot(q[j%16].astype(np.float64),k[j%16].astype(np.float64))*float(bt[j])*v[j].astype(np.float64)/np.sqrt(128)
for name,actual,expected in [('mixed',out[:2560],mixed),('inject',out[2560:2564],inj),('recurrence',out[12804:18948],y.ravel()),('gate',out[18948:18996],gate),('beta',out[18996:],bt)]:
 err=np.abs(actual.astype(np.float64)-expected);print(name,'count',len(err),'max_abs',err.max())
 if not np.isfinite(actual).all() or not np.isfinite(expected).all() or err.max()>1e-5:raise ValueError(name+' mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('dense changed')
print('QKV is input to oracle; zero state does not exercise gate decay')
