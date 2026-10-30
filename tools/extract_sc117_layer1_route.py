#!/usr/bin/env python3
"""Stage the exact R415 layer-1 activation and route from a pinned result."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

if len(sys.argv)!=3: raise SystemExit('usage: extract RESULT OUTPUT_DIRECTORY')
path=Path(sys.argv[1]); dest=Path(sys.argv[2])
if not path.is_absolute() or not dest.is_absolute(): raise SystemExit('absolute paths required')
fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
try:
 st=os.fstat(fd)
 if not stat.S_ISREG(st.st_mode) or st.st_size!=110728: raise ValueError('wrong source type or size')
 data=os.read(fd,110729);after=os.fstat(fd)
 if len(data)!=110728 or (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns):raise ValueError('source changed')
 if hashlib.sha256(data).hexdigest()!='f252ac7c7cf54d959b195db3e704c0de62e17a74417342e81f505a4887492bcf':raise ValueError('source digest mismatch')
finally: os.close(fd)
activation=data[25088*4:27648*4]
if any(not math.isfinite(v) for v in struct.unpack('<2560f',activation)):raise ValueError('nonfinite activation')
records=[]
for i in range(10):
 expert,weight,logit=struct.unpack_from('<Iff',data,27652*4+i*12)
 if expert>=512 or not math.isfinite(weight) or not 0<=weight<=1 or not math.isfinite(logit):raise ValueError('invalid route')
 records.append({'expert':expert,'weight':weight,'logit':logit})
if len({x['expert'] for x in records})!=10:raise ValueError('duplicate expert')
route=(json.dumps(records,separators=(',',':'),allow_nan=False)+'\n').encode()
with open(dest/'activation.f32','xb') as f:f.write(activation)
with open(dest/'route.json','xb') as f:f.write(route)
print(json.dumps({'activation_sha256':hashlib.sha256(activation).hexdigest(),'route_sha256':hashlib.sha256(route).hexdigest(),'ids':[x['expert'] for x in records]}))
