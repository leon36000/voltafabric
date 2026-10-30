#!/usr/bin/env python3
"""Independent encoding and plan-vector check for R520."""
import hashlib,json,struct

def prompt_hash(tokens):
    return hashlib.sha256(b''.join(struct.pack('<I', token) for token in tokens)).hexdigest()
plan={'version':'vf-sc117-worker-plan-v10','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1',
      'model_revision':'sc117-local-observation',
      'index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24',
      'epoch':1,'layers':48,'prompt_length':2,'prompt_tokens_sha256':prompt_hash([7,7]),
      'max_new_tokens':16,'eos_token_id':248046,'prompt_mode':'plain','payload_budget':300,
      'rank_limits':{'2':100,'3':200},
      'endpoints':{'2':{'rank':2,'argv':['/worker']},'3':{'rank':3,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','pc1-lan','/worker','--guarded-remote','--expected-exe-sha256','a'*64]}}}
def digest():return 'r520-'+hashlib.sha256(json.dumps(plan,separators=(',',':')).encode()).hexdigest()
assert plan['prompt_tokens_sha256']=='28625f27a2f3d7f965ec406e885f2f1050684660b9941c56e95d96c46bcb42f9'
assert digest()=='r520-a7dd5a61211985b8cca9753209422805e3adb0e0aa3afb14052a50b731c83f06'
plan['prompt_tokens_sha256']=prompt_hash([1,2]); a=digest()
plan['prompt_tokens_sha256']=prompt_hash([2,1]); assert digest()!=a
print('PASS: little-endian token digest and plan')
