"""Link two unfinished snapshot claims to frozen handoff entries and source bytes.

The handoff is an unauthenticated snapshot; matching a recorded report hash is
not verification of the absent report. This script never replays any identity.
"""
if not __debug__:
    raise RuntimeError('verification requires assertions enabled')

import hashlib
import json
from pathlib import Path

root = Path(__file__).parents[2]
archive = {}
part_hashes = []
for path in sorted((root / 'provenance/recovery-20261030/pc1-ledgers').glob('part-*.json')):
    raw = path.read_bytes(); part_hashes.append(hashlib.sha256(raw).hexdigest())
    part = json.loads(raw)
    assert not set(archive).intersection(part['files'])
    archive.update(part['files'])
audit = json.loads((root / 'research/R581/AUDIT.json').read_bytes())
assert audit['part_sha256'] == part_hashes
handoff_path = root / 'provenance/recovery-20261030/handoff/PRODUCTIVITY_CURRENT.json'
handoff_bytes = handoff_path.read_bytes(); handoff = json.loads(handoff_bytes)
targets = {
    'R278': {'path_prefix':'r278-owner-failure-r1/', 'run_id':'r278-8d2f00d5fe3544a9b64ecbc57f02d495',
             'files': {'remote.py':'remote_code_sha256', 'failing_worker.py':'worker_code_sha256'}},
    'R287': {'path_prefix':'r287-failure-isolation-r1/', 'run_id':'r287-90fae92c0bd84c21b36e70f5ed399c2d',
             'files': {'multiplex.py':'multiplex.py', 'owner_service.py':'owner_service.py', 'proxy.py':'proxy.py', 'test_real.py':'test_real.py'}},
}
results = []
for round_id, config in targets.items():
    matches = [n for n in audit['claims_without_done'] if n.startswith(config['path_prefix'])]
    assert len(matches) == 1
    name = matches[0]; claim = json.loads(archive[name]); entry = handoff[round_id]
    assert claim['run_id'] == config['run_id'] == entry['run_id']
    assert claim['attempt'] == 'assigned-by-rust' and claim['batch'] == 0
    parent = name.rsplit('/',1)[0]
    assert json.loads(archive[parent+'/identity.json']) == {k:claim[k] for k in ('run_id','model_revision','plan_id','layer','epoch')}
    assert name.replace('.claim.json','.done.json') not in archive and parent+'/closed.json' not in archive
    verified = {}
    for filename, key in config['files'].items():
        digest = hashlib.sha256((root / 'research' / round_id / filename).read_bytes()).hexdigest()
        expected = entry[key] if round_id == 'R278' else entry['source_sha256'][key]
        assert digest == expected
        verified[filename] = digest
    assert isinstance(entry['report_sha256'], str) and len(entry['report_sha256']) == 64
    results.append({'round':round_id,'run_id':claim['run_id'],'claim':name,
                    'handoff_status':entry['status'],'handoff_limits':entry['limits'],
                    'report_hash_recorded':entry['report_sha256'],
                    'report_content_verified':False,'source_hashes_verified':verified,
                    'classification':'CLAIM_WITHOUT_DONE_NO_REPLAY'})
assert len(audit['claims_without_done']) == len(results) == 2
print(json.dumps({'handoff_sha256':hashlib.sha256(handoff_bytes).hexdigest(),
                  'archive_parts_sha256':part_hashes,'links':results,
                  'ready_for_live_use':False}, sort_keys=True, separators=(',', ':')))
