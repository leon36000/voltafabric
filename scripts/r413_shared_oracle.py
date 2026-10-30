"""NumPy checks shared-expert arithmetic after the three quantized projections."""
import json
import os
import sys
import numpy as np
if len(sys.argv)!=5: raise SystemExit('usage: oracle SHARD ROUTER_INPUT COLLECTOR_JSON OUTPUT')
fd=os.open(sys.argv[1],os.O_RDONLY|os.O_NOFOLLOW);before=os.fstat(fd)
w=np.frombuffer(os.pread(fd,5120,1711058944),dtype='<u2').astype(np.uint32)
w=(w<<16).view(np.float32)
x=np.fromfile(sys.argv[2],dtype='<f4');o=np.fromfile(sys.argv[4],dtype='<f4')
with open(sys.argv[3],encoding='utf-8') as f: routed=np.array(json.load(f)['result']['output'][0],dtype=np.float64)
if len(w)!=2560 or len(x)!=12804 or len(o)!=19841 or len(routed)!=2560:raise ValueError('shape')
bits=x[:2560].copy().view(np.uint32);xq=((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
dot=np.dot(xq.astype(np.float64),w.astype(np.float64));scalar=np.float32(1/(1+np.exp(-dot)))
gate=o[:640].astype(np.float64);up=o[640:1280].astype(np.float64);mid=(gate/(1+np.exp(-gate))*up).astype(np.float32)
shared=(o[1920:4480]*scalar).astype(np.float32);combined=(routed+shared.astype(np.float64)).astype(np.float32)
inject=x[2560:2564].astype(np.float64);weight=(2/(1+np.exp(-inject/4))).astype(np.float32)
residual=(x[2564:].reshape(4,2560)+combined[None,:]*weight[:,None]).astype(np.float32)
for name,actual,expected in [('mid',o[1280:1920],mid),('shared',o[4480:7040],shared),('combined',o[7040:9600],combined),('residual',o[9600:19840],residual.ravel()),('scalar',o[19840:],np.array([scalar]))]:
 diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64));print(name,'count',len(diff),'max_abs',diff.max())
 if not np.isfinite(actual).all() or not np.isfinite(expected).all() or diff.max()>1e-5:raise ValueError(name+' mismatch')
after=os.fstat(fd)
if (before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('shard changed')
