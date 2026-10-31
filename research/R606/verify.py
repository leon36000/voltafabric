"""Verify the single fresh layer receipt and reconstruct its residual independently."""
import ctypes
import hashlib
import json
import math
import struct
from pathlib import Path

root = Path(__file__).resolve().parents[2]
w = root / '.runs/vf-r606-work'
h = lambda x: hashlib.sha256(x).hexdigest()
f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
raw = (w/'PLAN.json').read_bytes(); plan = json.loads(raw); plan_hash = h(raw)
assert (w/'SUBMITTED').read_text().strip() == plan_hash
assert (w/'EXIT_STATUS').read_text().strip() == '0'
assert plan['gpu'] is False and plan['network_worker'] is False
assert plan['ggml_base_sha256'] == '54558b38a3f14789ce7b8cdab493788c2dd6ff495a3953970da66b026c47fc23'
assert plan['ggml_cpu_sha256'] == '42e39eea5894bed527fef2feb9bf8a23e970733fa9ea892fd060cafe5505412a'
assert plan['runner_sha256'] == h((root/'research/R606/run.py').read_bytes())
assert plan['layer_source_sha256'] == h((root/'research/R606/layer.rs').read_bytes())
claims = list((w/'ledger').glob('*.claim.json')); assert len(claims) == 1
claim_bytes = claims[0].read_bytes(); claim=json.loads(claim_bytes); stamp=claim['stamp']
assert claim['status'] == 'RESERVED_UNKNOWN_NO_REPLAY'
assert stamp == {'run_id':plan['run_id'], 'model_revision':'sc117-local-observation', 'plan_id':'r606-'+plan_hash, 'layer':2, 'batch':0, 'epoch':1}
assert isinstance(claim['attempt'], int) and claim['attempt'] > 0
key=b'VF_NATIVE_BATCH_V1\0'
for s in (stamp['run_id'],stamp['model_revision'],stamp['plan_id']):
    b=s.encode();key+=struct.pack('<Q',len(b))+b
key+=struct.pack('<IQQ',stamp['layer'],stamp['batch'],stamp['epoch'])
assert claims[0].name == 'native-'+h(key)+'.claim.json'
data=json.loads((w/'stdout').read_bytes()); assert data['run_id']==plan['run_id'] and data['next_token']==1 and data['attempt']==claim['attempt']
assert len(data['closed'])==2 and all(not x['killed'] and x['exit_code']==0 and x['bye_execs']==1 for x in data['closed'])
t=data['tokens'][0]; old=json.loads((root/'.runs/vf-r605-work/r428.json').read_bytes())['tokens'][0]
assert t['ids']==old['ids'] and t['weights']==old['weights']
assert t['activation_sha']==old['activation_sha']=='c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a'
assert t['gdn_sha']==old['gdn_sha']
assert all(t[k]==old[k] for k in ('post','shared','injection'))
row_bytes=(root/'.runs/vf-r604-work/collector.f64').read_bytes()
assert h(row_bytes)==t['collector_sha']=='ef0a699feb2e5d21cea882c1f7c8cbd517883a2de419e0c511aa91e784514ac5'
row=struct.unpack('<2560d',row_bytes); post=list(map(f32,t['post'])); shared=list(map(f32,t['shared'])); inject=list(map(f32,t['injection']))
assert (len(post),len(shared),len(inject))==(10240,2560,4)
lib=ctypes.CDLL('libm.so.6'); expf=lib.expf; expf.argtypes=[ctypes.c_float]; expf.restype=ctypes.c_float
weights=[]
for z in inject:
    x=f32(z/4.0)
    if x>=0: sig=f32(1.0/f32(1.0+expf(f32(-x))))
    else:
        e=expf(x); sig=f32(e/f32(1.0+e))
    weights.append(f32(2.0*sig))
combined=[f32(r+s) for r,s in zip(row,shared)]
residual=struct.pack('<10240f',*(f32(post[j*2560+i]+f32(combined[i]*weights[j])) for j in range(4) for i in range(2560)))
assert h(residual)==t['residual_sha']=='0350630913cfe2b369d8bc7043e2cbeae352910229468516a6d55aa960173e90'
assert all(math.isfinite(x) for x in struct.unpack('<10240f',residual))
result={'plan_sha256':plan_hash,'run_id':plan['run_id'],'claim_sha256':h(claim_bytes),'claim_status':claim['status'],
        'receipt_sha256':h((w/'stdout').read_bytes()),'layer_binary_sha256':plan['layer_binary_sha256'],'worker_sha256':plan['worker_sha256'],
        'activation_sha256':t['activation_sha'],'gdn_sha256':t['gdn_sha'],'collector_sha256':t['collector_sha'],
        'residual_sha256':t['residual_sha'],'expert_ids':t['ids'],'oracle_values':10240,
        'actual_layer_transaction':True,'gdn_next_token':data['next_token'], 'gpu':False,'network_worker':False,'generation':False}
print(json.dumps(result,indent=2,sort_keys=True))
