"""Independent first-position SC117 QSA mix, norm and one-cell attention check."""
import os,sys
import numpy as np
if len(sys.argv)!=4:raise SystemExit('usage: oracle DENSE RESIDUAL OUTPUT')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
def read(o,n):
 b=os.pread(fd,n,o)
 if len(b)!=n:raise ValueError('short')
 return b
def f(o,n):return np.frombuffer(read(o,n),dtype='<f4')
def bf(o,n,shape):return (np.frombuffer(read(o,n),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape)
def rb(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32);return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
r=np.fromfile(sys.argv[2],dtype='<f4').reshape(4,2560).astype(np.float64);o=np.fromfile(sys.argv[3],dtype='<f4')
if len(o)!=37380 or not np.isfinite(o).all():raise ValueError('output shape')
g=f(177382016,40960).reshape(4,2560).astype(np.float64);xn=(r/np.sqrt(np.mean(r*r,axis=1,keepdims=True)+1e-6)*g).astype(np.float32).reshape(-1);xq=rb(xn)
down=bf(170746496,6553600,(320,10240)).astype(np.float64);up=bf(177422976,6553600,(10240,320)).astype(np.float64);iw=bf(177300096,81920,(4,10240)).astype(np.float64)
lo=(down@xq.astype(np.float64)).astype(np.float32)/np.float32(4);lo=rb((lo/(1+np.exp(-lo))).astype(np.float32));z=(up@lo.astype(np.float64)).astype(np.float32);v=(xn/(1+np.exp(-z))).astype(np.float32).reshape(4,2560);mixed=np.zeros(2560,np.float32)
for row in v:mixed=(mixed+row).astype(np.float32)
mixed=(mixed/np.float32(4)).astype(np.float32);inject=(iw@xq.astype(np.float64)).astype(np.float32)
qfull=o[2564:14852].reshape(24,512);q=qfull[:,:256].astype(np.float64);qgamma=f(168118912,1024).astype(np.float64);qn=q/np.sqrt(np.mean(q*q,axis=1,keepdims=True)+1e-6)*qgamma
gate=qfull[:,256:];k=o[27140:27652].reshape(2,256).astype(np.float64);kgamma=f(168117888,1024).astype(np.float64);kn=k/np.sqrt(np.mean(k*k,axis=1,keepdims=True)+1e-6)*kgamma
value=o[28164:28676].reshape(2,256).astype(np.float16).astype(np.float64);gated=(np.repeat(value,12,axis=0)/(1+np.exp(-gate.astype(np.float64)))).astype(np.float32)
for name,got,want in [('mixed',o[:2560],mixed),('inject',o[2560:2564],inject),('qnorm',o[14852:20996],qn.ravel()),('gate',o[20996:27140],gate.ravel()),('knorm',o[27652:28164],kn.ravel()),('one_cell',o[28676:34820],gated.ravel())]:
 err=np.max(np.abs(got.astype(np.float64)-want));print(name,'max_abs',err)
 if not np.isfinite(want).all() or err>1e-5:raise ValueError(name+' mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed')
