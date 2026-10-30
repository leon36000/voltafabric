"""Check layer 3 hyperconnection from actual residuals against separately computed NumPy values."""
import hashlib,json,os,sys
from pathlib import Path
import numpy as np
if len(sys.argv)!=4: raise SystemExit('usage: hyper_oracle DENSE MANIFEST TRACE')
m=json.loads(Path(sys.argv[2]).read_text()); fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW); before=os.fstat(fd)
if before.st_size!=m['dense_bytes']: raise ValueError('dense size')
def span(name):
 v=m[name]; b=os.pread(fd,v['bytes'],v['offset'])
 if len(b)!=v['bytes'] or hashlib.sha256(b).hexdigest()!=v['sha256']: raise ValueError('source span '+name)
 return b
def bf(name,shape): return (np.frombuffer(span(name),dtype='<u2').astype(np.uint32)<<16).view(np.float32).reshape(shape).astype(np.float64)
def rb(x):
 u=np.asarray(x,dtype=np.float32).view(np.uint32)
 return ((u+np.uint32(0x7fff)+((u>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
g=np.frombuffer(span('norm'),dtype='<f4').reshape(4,2560).astype(np.float64)
down=bf('down',(320,10240)); up=bf('up',(10240,320)); iw=bf('inject',(4,10240))
a=np.fromfile(sys.argv[3],dtype='<f4')
if len(a)!=92*47620 or not np.isfinite(a).all(): raise ValueError('trace shape or finite')
a=a.reshape(92,47620); maxima={'mixed':0.0,'inject':0.0}; worst={}; negative_rejected=False
for pos,rec in enumerate(a):
 r=rec[:10240].reshape(4,2560).astype(np.float64); o=rec[10240:]
 xn=(r/np.sqrt(np.mean(r*r,axis=1,keepdims=True)+1e-6)*g).astype(np.float32).reshape(-1); xq=rb(xn)
 lo=(down@xq.astype(np.float64)).astype(np.float32)/np.float32(4)
 lo=rb((lo/(1+np.exp(-lo))).astype(np.float32)); z=(up@lo.astype(np.float64)).astype(np.float32)
 v=(xn/(1+np.exp(-z))).astype(np.float32).reshape(4,2560); mixed=np.zeros(2560,np.float32)
 for row in v: mixed=(mixed+row).astype(np.float32)
 mixed=(mixed/np.float32(4)).astype(np.float32); inject=(iw@xq.astype(np.float64)).astype(np.float32)
 for name,got,want in [('mixed',o[:2560],mixed),('inject',o[2560:2564],inject)]:
  err=float(np.max(np.abs(got.astype(np.float64)-want)))
  if not np.isfinite(err) or err>1e-5: raise ValueError(f'{pos} {name} {err}')
  if err>maxima[name]: maxima[name]=err; worst[name]=pos
  if pos==91 and name=='mixed':
   mutated=got.copy(); mutated[0]+=np.float32(0.1)
   negative_rejected=float(np.max(np.abs(mutated.astype(np.float64)-want)))>1e-5
if not negative_rejected: raise ValueError('negative control accepted')
after=os.fstat(fd)
if (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('dense source changed')
print(json.dumps({'status':'PASS','positions':92,'values':92*(2560+4),'max_abs':maxima,'worst_position':worst,'negative_control':'rejected','dense_spans':['norm','down','up','inject']},sort_keys=True))
