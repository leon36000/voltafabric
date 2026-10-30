import base64,gzip,hashlib,json
from pathlib import Path
p=Path(__file__).parent
meta=json.loads((p/'METADATA.json').read_text())
for name in ('selection','observation'):
    raw=(p/(name+'.json')).read_bytes()
    meta[name+'_sha256']=hashlib.sha256(raw).hexdigest()
    encoded=base64.b64encode(gzip.compress(raw,mtime=0)).decode('ascii')
    out=p/(name+'-parts'); out.mkdir(exist_ok=False)
    parts=[encoded[i:i+32000] for i in range(0,len(encoded),32000)]
    for n,part in enumerate(parts): (out/f'part-{n:03}.txt').write_text(part+'\n')
    meta[name+'_parts']=len(parts)
(p/'METADATA.json').write_text(json.dumps(meta,sort_keys=True,indent=2)+'\n')
print(json.dumps(meta))
