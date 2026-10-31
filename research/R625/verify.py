"""Independent count and checksum check of the recovered PC6 session archive."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).parent
raw=(root/'provenance/recovery-20261030/pc6-ledgers/research-ledgers.json').read_bytes()
a=json.loads((p/'AUDIT.json').read_bytes());data=json.loads(raw)['files'];r=a['pc6']
assert r['part_sha256']==[hashlib.sha256(raw).hexdigest()]
sessions={k.rsplit('/',1)[0] for k in data if k.endswith('/identity.json')}
claims={k for k in data if '/batch-' in k and k.endswith('.claim.json')}
done={k for k in data if '/batch-' in k and k.endswith('.done.json')}
unknown=sorted(k for k in claims if k.replace('.claim.json','.done.json') not in done)
orphan=sorted(k for k in done if k.replace('.done.json','.claim.json') not in claims)
assert (len(data),len(sessions),len(claims),len(done))==(121,20,33,32)
assert unknown==r['claims_without_done'] and orphan==r['done_without_claim']==[]
assert sum(parent+'/closed.json' not in data for parent in sessions)==r['sessions_without_closed_marker']==4
assert a['shared_session_identity_count']==a['shared_run_id_count']==0
assert not a['ready_for_live_use'] and not r['ready_for_live_use'] and not r['done_payloads_verified']
print('PASS: archive hash, counts, missing markers and no live qualification')
