"""Retain exact local claims and receipt summary in bounded text chunks."""
import base64,gzip,hashlib,json
from pathlib import Path
p=Path(__file__).parent
result=(p/'result.json').read_bytes(); obj=json.loads(result)
trace=obj['receipt']['qsa_trace']; trace.pop('base64'); trace['base64_omitted']=True
summary=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
claims=(p/'claims.json').read_bytes()
meta={'full_result_sha256':hashlib.sha256(result).hexdigest(),'trace_sha256':trace['sha256'],'trace_bytes':trace['positions']*trace['words_per_position']*4}
for name,data in [('summary',summary),('claims',claims)]:
 s=base64.b64encode(gzip.compress(data,compresslevel=9,mtime=0)).decode(); q=p/(name+'-parts');q.mkdir(exist_ok=False)
 for i in range(0,len(s),32000): (q/('part-%03d.txt'%(i//32000))).write_text(s[i:i+32000]+'\n')
 meta[name+'_parts']=(len(s)+31999)//32000;meta[name+'_sha256']=hashlib.sha256(data).hexdigest()
(p/'METADATA.json').write_text(json.dumps(meta,indent=2,sort_keys=True)+'\n')
print(json.dumps(meta,sort_keys=True))
