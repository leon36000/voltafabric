"""Extract legacy QSA records from a verified layer-input trace for comparison."""
import hashlib,os,sys
import numpy as np
if len(sys.argv)!=3:raise SystemExit('usage: split_trace TRACE OUTPUT')
x=np.fromfile(sys.argv[1],dtype='<f4')
if len(x)!=92*47620 or not np.isfinite(x).all():raise ValueError('trace shape')
b=x.reshape(92,47620)[:,10240:].copy().tobytes()
fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print(len(b),hashlib.sha256(b).hexdigest())
