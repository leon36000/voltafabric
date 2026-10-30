"""Independent NumPy check for SC117 output head hyperconnection mix."""
import os,sys
import numpy as np
if len(sys.argv)!=4:raise SystemExit('usage: oracle DENSE RESIDUAL HEAD_OUTPUT')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
def read(o,n):
 b=os.pread(fd,n,o)
 if len(b)!=n:raise ValueError('short')
 return b
def bf(o,n,shape):return (np.frombuffer(read(o,n),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
def rb(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
r=np.fromfile(sys.argv[2],dtype='<f4').reshape(4,2560).astype(np.float64);gamma=np.frombuffer(read(6553600,40960),dtype='<f4').reshape(4,2560).astype(np.float64)
x=(r/np.sqrt(np.mean(r*r,axis=1,keepdims=True)+1e-6)*gamma).astype(np.float32).reshape(-1);xq=rb(x)
down=bf(0,6553600,(320,10240)).astype(np.float64);up=bf(6594560,6553600,(10240,320)).astype(np.float64)
lo=(down@xq.astype(np.float64)).astype(np.float32)/np.float32(4);lo=rb((lo/(1+np.exp(-lo))).astype(np.float32));z=(up@lo.astype(np.float64)).astype(np.float32);v=(x/(1+np.exp(-z))).astype(np.float32).reshape(4,2560);mixed=np.zeros(2560,np.float32)
for row in v:mixed=(mixed+row).astype(np.float32)
mixed=(mixed/np.float32(4)).astype(np.float32)
a=np.fromfile(sys.argv[3],dtype='<f4',count=2560);err=np.max(np.abs(a.astype(np.float64)-mixed.astype(np.float64)));print('mixed_max_abs',err)
if len(a)!=2560 or not np.isfinite(a).all() or err>1e-5:raise ValueError('head mix mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('dense changed')
