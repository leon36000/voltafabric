"""Compare two complete, independently plan-bound layer-3 trace receipts."""
import base64,hashlib,json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[2]
arrays={}; outputs={}
for mode in ('scalar','vector'):
 p=root/'.runs'/('vf-r609-'+mode)/'RESULT.json'; x=json.loads(p.read_bytes()); r=x['receipt']; t=r['qsa_trace']
 expected=json.loads((Path(__file__).parent/('PREPARED-'+mode+'.json')).read_text())
 assert r['plan_id']==expected['plan_id'] and r['prompt_tokens']==expected['prompt_token_ids']
 assert r['native_math_backend']==('scalar-reference-ggml-v1' if mode=='scalar' else 'vector-ggml-v1')
 assert t['version']=='vf-qsa-trace-v2-layer3-input-f32le' and t['positions']==13 and t['words_per_position']==47620
 raw=base64.b64decode(t['base64'],validate=True);assert hashlib.sha256(raw).hexdigest()==t['sha256'] and len(raw)==13*47620*4
 arrays[mode]=np.frombuffer(raw,dtype='<f4').reshape(13,47620)
 assert np.isfinite(arrays[mode]).all()
 outputs[mode]={'receipt_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'trace_sha256':t['sha256'],
                'generated_token_ids':r['generated_token_ids'],'run_id':r['run_id'],'plan_id':r['plan_id']}
segments={'input_residual':(0,10240),'mixed':(10240,12800),'injection':(12800,12804),
          'raw_query_gate':(12804,25092),'query':(25092,31236),'gate':(31236,37380),
          'raw_key':(37380,37892),'key':(37892,38404),'value':(38404,38916),
          'gated_attention':(38916,45060),'attention':(45060,47620)}
comparison={}
for name,(a,b) in segments.items():
 x=arrays['scalar'][:,a:b];y=arrays['vector'][:,a:b];d=np.abs(x.astype(np.float64)-y.astype(np.float64))
 bypos=np.max(d,axis=1); comparison[name]={'different_values':int(np.count_nonzero(x!=y)),
   'max_abs':float(d.max()),'first_different_position':next((i for i,v in enumerate(bypos) if v>0),None)}
print(json.dumps({'positions':13,'outputs':outputs,'segments':comparison,
                  'comparison_is_numerical_not_performance':True},indent=2,sort_keys=True))
