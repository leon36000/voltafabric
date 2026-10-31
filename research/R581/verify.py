"""Independent count and pairing check of the retained PC1 archive parts."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).parents[2]
parts = sorted((root / 'provenance/recovery-20261030/pc1-ledgers').glob('part-*.json'))
result = json.loads((Path(__file__).parent / 'AUDIT.json').read_bytes())
assert len(parts) == 5
records = {}
for p in parts:
    part = json.loads(p.read_bytes())
    assert part['original_root'] == result['original_root']
    assert not set(records).intersection(part['files'])
    records.update(part['files'])
assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in parts] == result['part_sha256']
claims = {k for k in records if '/batch-' in k and k.endswith('.claim.json')}
done = {k for k in records if '/batch-' in k and k.endswith('.done.json')}
unmatched_claims = sorted(k for k in claims if k.replace('.claim.json','.done.json') not in done)
unmatched_done = sorted(k for k in done if k.replace('.done.json','.claim.json') not in claims)
sessions = {k.rsplit('/',1)[0] for k in records if k.endswith('/identity.json')}
assert (len(records), len(sessions), len(claims), len(done)) == (4398, 653, 1212, 1210)
assert unmatched_claims == result['claims_without_done'] and unmatched_done == result['done_without_claim']
assert len(unmatched_claims) == 2 and not unmatched_done
assert sum(parent+'/closed.json' not in records for parent in sessions) == result['sessions_without_closed_marker'] == 24
assert sum(parent+'/process.json' not in records for parent in sessions) == result['sessions_without_process_marker'] == 1
assert result['ready_for_live_use'] is False and result['done_payloads_verified'] is False
print('PASS: archived counts, original part hashes and unmatched claim paths; no live admission')
