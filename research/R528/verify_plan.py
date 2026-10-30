"""Independent v13 plans for the synthetic pinned SSH host keys; no network."""
import hashlib,json,struct
from pathlib import Path
p=Path(__file__).parent
h=lambda b:hashlib.sha256(b).hexdigest()
tokens=[248045,846,198,12675,248046,198,248045,74455,198,248068,271,248069,271]
ph=h(b''.join(struct.pack('<I',x) for x in tokens))
worker='/home/pc1/NAOS/vf-r525-work/target/release/vf-rank-session'
wd='49216f6ce76a637be323dd0aa88c69e7788b4fa8ff13f5663ee103c5e4a22409'
path='/home/pc1/NAOS/vf-r528-work/known-hosts';host='pc1-lan';digest='a'*64
opts=['UserKnownHostsFile='+path,'GlobalKnownHostsFile=/dev/null','HostKeyAlias='+host,'VerifyHostKeyDNS=no','UpdateHostKeys=no','KnownHostsCommand=none','CheckHostIP=no','ProxyCommand=none','ProxyJump=none','PermitLocalCommand=no','ClearAllForwardings=yes']
remote=['/usr/bin/ssh']+[x for v in opts for x in ('-o',v)]+['-T','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10',host,'/opt/vf-worker','--guarded-remote','--expected-exe-sha256',digest]
plan={'version':'vf-sc117-worker-plan-v14','wire_policy':'native-f32-raw-if-advertised-otherwise-json-v1','eviction_policy':'rank-lru-then-size-v2','origin_policy':'stable-source-digest-cache-v1','model_revision':'sc117-local-observation','index_sha256':'adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24','ple_manifest_sha256':'436066427aaca7f163943b3f0f00c98cb59ddb5df7b2fa410b90d474ab1fe824','epoch':1,'layers':48,'prompt_length':13,'prompt_tokens_sha256':ph,'max_new_tokens':1,'eos_token_id':248046,'prompt_mode':'chat','payload_budget':3221225472,'rank_limits':{'2':1073741824,'3':2147483648},'endpoints':{'2':{'rank':2,'argv':[worker,'--guarded-local','--expected-exe-sha256',wd]},'3':{'rank':3,'argv':remote}},'known_hosts_sha256':{}}
proof=json.loads((p/'GUARD.json').read_text())
plan['known_hosts_sha256']={'3':proof['hostkeys_sha256']}
lease=proof['synthetic_lease']
plan['shared_lease']={k:lease[k] for k in ('path','dev','ino','uid')}
pid='r528-'+h(json.dumps(plan,separators=(',',':')).encode())
assert proof['status']=='PASS' and proof['ledger_claims']==0 and proof['plan_id']==pid
print('PASS: independent v14 plan',pid)
