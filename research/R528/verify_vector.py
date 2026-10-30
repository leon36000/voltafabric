"""Independent fixed vector: one guarded local and one SSH rank."""
import hashlib, json, struct
h=lambda x:hashlib.sha256(x).hexdigest()
d='a'*64
plan={'version':'vf-sc117-worker-plan-v14','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','ple_manifest_sha256':d,'epoch':1,'layers':48,'prompt_length':2,'prompt_tokens_sha256':h(struct.pack('<II',7,7)),'max_new_tokens':16,'eos_token_id':248046,'prompt_mode':'plain','payload_budget':300,'rank_limits':{'2':100,'3':200},'endpoints':{'2':{'rank':2,'argv':['/worker','--guarded-local','--expected-exe-sha256',d]},'3':{'rank':3,'argv':['/usr/bin/ssh','-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','pc1-lan','/worker','--guarded-remote','--expected-exe-sha256',d]}}}
plan['known_hosts_sha256']={'3':d}
plan['shared_lease']={'path':'/lease','dev':1,'ino':2,'uid':3}
id='r528-'+h(json.dumps(plan,separators=(',',':')).encode());print(id)
