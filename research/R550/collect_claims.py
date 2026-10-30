"""Collect exact local claim texts after the one-shot run has finished."""
import json,os,sys
from pathlib import Path
if len(sys.argv)!=3:raise SystemExit('usage: collect_claims LEDGER OUTPUT')
p=Path(sys.argv[1]); paths=sorted(p.glob('*.claim.json'))
if not 1<=len(paths)<=104448: raise ValueError('claim count')
a={f.name:f.read_text() for f in paths}
b=json.dumps(a,sort_keys=True,separators=(',',':')).encode()
fd=os.open(sys.argv[2],os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
with os.fdopen(fd,'wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
print('claims',len(a),'bytes',len(b))
