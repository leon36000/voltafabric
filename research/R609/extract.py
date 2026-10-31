"""Extract validated trace into fresh local files, never a worker or claim."""
import base64,hashlib,json,sys
from pathlib import Path
mode=sys.argv[1]
assert mode in ('scalar','vector')
w=Path('/home/pc1/NAOS')/('vf-r609-'+mode)
assert (w/'EXIT_STATUS').read_text().strip()=='0'
x=json.loads((w/'RESULT.json').read_bytes()); t=x['receipt']['qsa_trace']
assert t['version']=='vf-qsa-trace-v2-layer3-input-f32le' and t['positions']==13 and t['words_per_position']==47620
raw=base64.b64decode(t['base64'],validate=True)
assert len(raw)==13*47620*4 and hashlib.sha256(raw).hexdigest()==t['sha256']
with (w/'trace-v2.f32').open('xb') as f:f.write(raw)
with (w/'qsa-v1.f32').open('xb') as f:
 for i in range(13):f.write(raw[(i*47620+10240)*4:(i+1)*47620*4])
print(mode,t['sha256'])
