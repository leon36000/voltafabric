"""Independent f64 NumPy two-token GDN recurrence using observed QKV projections."""
import os,sys
import numpy as np
if len(sys.argv)!=4:raise SystemExit('usage: oracle DENSE GDN0 GDN1')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
k=np.frombuffer(os.pread(fd,163840,167953344),dtype='<f4').reshape(10240,4).astype(np.float64)
g=[np.fromfile(p,dtype='<f4') for p in sys.argv[2:]]
if any(len(x)!=19044 or not np.isfinite(x).all() for x in g):raise ValueError('GDN shape')
state=np.zeros((48,128,128),dtype=np.float64);past=np.zeros(10240,dtype=np.float64)
for i,a in enumerate(g):
 qkv=a[2564:12804].astype(np.float64); conv=past*k[:,2]+qkv*k[:,3]; conv=conv/(1+np.exp(-conv))
 q=conv[:2048].reshape(16,128); kk=conv[2048:4096].reshape(16,128); v=conv[4096:].reshape(48,128)
 q=q/np.sqrt(np.sum(q*q,axis=1,keepdims=True)+1e-6);kk=kk/np.sqrt(np.sum(kk*kk,axis=1,keepdims=True)+1e-6)
 beta=a[18996:].astype(np.float64);decay=np.exp(a[18948:18996].astype(np.float64));out=np.empty((48,128)); zero=np.empty_like(out)
 for h in range(48):
  qh=q[h%16];kh=kk[h%16];s=state[h]*decay[h];delta=(v[h]-kh@s)*beta[h];state[h]=s+np.outer(kh,delta);out[h]=(qh@state[h])/np.sqrt(128)
  zero[h]=np.dot(qh,kh)*beta[h]*v[h]/np.sqrt(128)
 expected=out.ravel();actual=a[12804:18948].astype(np.float64);err=np.max(np.abs(actual-expected));difference=np.max(np.abs(expected-zero.ravel()))
 print('token',i,'max_abs',err,'nonzero_state_effect',difference)
 if not np.isfinite(expected).all() or err>1e-5 or (i==1 and difference<1e-7):raise ValueError('GDN mismatch or no second-state effect')
 past=qkv
end=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(end.st_ino,end.st_size,end.st_mtime_ns,end.st_ctime_ns):raise ValueError('dense changed')
