"""Independent v13 plans for the synthetic pinned SSH host keys; no network."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent
h=lambda b:hashlib.sha256(b).hexdigest()
tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
ph=h(b''.join(struct.pack('<I',x) for x in tokens))
worker='/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'
wd='49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409'
path='/home/pc1/NAOS/vf-r529-work/known-hosts';host='pc1-lan';digest='a'*64
opts=['UserKnownHostsFile='+path,'GlobalKnownHostsFile=/dev/null','HostKeyAlias='+host,'VerifyHostKeyDNS=no','UpdateHostKeys=no','KnownHostsCommand=none','CheckHostIP=no','ProxyCommand=none','ProxyJump=none','PermitLocalCommand=no','ClearAllForwardings=yes']
remote=['/usr/bin/ssh']+[x for v in opts for x in ('-o',v)]+['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',host,'/opt/vf-worker','--guarded-remote','--expected-exe-sha256',digest]
plan={'version':'vf-sc117-worker-plan-v15','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','ple_manifest_sha256':'436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824','epoch':1,'layers':48,'prompt_length':13,'prompt_tokens_sha256':ph,'max_new_tokens':1,'eos_token_id':248046,'prompt_mode':'chat','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker,'--guarded-local','--expected-exe-sha256',wd]},'3':{'rank':3,'argv':remote}},'known_hosts_sha256':{}}
base=json.loads((p/'R53-BASE.json').read_text())
suffix=json.loads((p/'R53-SUFFIX.json').read_text())
marker=json.loads((p/'R53-STARTED.json').read_text())
plan['known_hosts_sha256']={'3':'b8887e50af0cdd3da028df44a11edadb352ad80f74a86542a3d32beef9755176'}
lease=base['context']['shared_lease'];plan['shared_lease']={k:lease[k] for k in ('path','dev','ino','uid')}
plan['r53_namespace']={'root':'/home/pc1/NAOS/vf-r529-work/synthetic-r53','dev':66312,'ino':32953844,'uid':1000,'base_id':'r53-r529-integration-a'}
pid='r529-'+h(json.dumps(plan,separators=(',',':')).encode())
for record,effective in ((base,'r53-r529-integration-a'),(suffix,'r53-r529-integration-a-duplicate-slot')):
 assert record['status']=='RESERVED_UNKNOWN_NO_REPLAY' and record['base_id']=='r53-r529-integration-a' and record['effective_run_id']==effective
 assert record['context']=={'plan_id':pid,'shared_lease':lease}
assert marker=={'status':'RESERVED_UNKNOWN_NO_REPLAY','base_id':'r53-r529-integration-a','plan_id':pid}
assert pid=='r529-ae4237e41848abc43f843c9d46d1eb443fd75c690fa53e2fb7f3b64279815b7e'
print('PASS: independent v15 plan and R53 identities',pid)
