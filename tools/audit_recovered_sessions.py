#!/usr/bin/env python3
"""Read-only conservative identity lookup across the two recovered archives.

Observed identities are consumed. Missing identities are NOT authorized: the
archives may be incomplete and neither is a shared R53 ledger or a lease.
"""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('snapshot_audit', ROOT / 'tools/audit_pc1_session_snapshots.py')
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
PC1 = ['0e6f77c94f3512ad3f537600492e98c2860272c7aac898522842a5766e8cf86f',
       '84e0aff9e063ae16cdb5a222d1715a792e3a48bd4619c529fe39648febb22a51',
       '6df3c6993720b52dbd318545a5f470c3bea3d258b6a77a5442edc6f7e0997754',
       '35d99851c20e53bf204283266d504be5bec49edc331a95c0dbf968f93d6319d0',
       '15389e25576f170be175699985c5cd6dc78d4876b714647650fee0f1d07388d0']
PC6 = ['258ff2f6e887c1bd11f2aa245fe9695c184e5306c9675c9bd11b830b65c7e1a3']


def audit_pair(pc1_paths, pc6_paths, expected_pc1, expected_pc6, candidate=None):
    first, first_sessions = AUDIT.audit(pc1_paths)
    second, second_sessions = AUDIT.audit(pc6_paths)
    if first['part_sha256'] != expected_pc1 or second['part_sha256'] != expected_pc6:
        raise ValueError('archive hash mismatch')
    result = {'pc1': first, 'pc6': second,
              'shared_session_identity_count': len(first_sessions.keys() & second_sessions.keys()),
              'shared_run_id_count': len({s['identity']['run_id'] for s in first_sessions.values()} &
                                         {s['identity']['run_id'] for s in second_sessions.values()}),
              'ready_for_live_use': False,
              'scope': 'Frozen possibly incomplete archives only; neither a live R53 ledger nor a shared lease.'}
    for host in ('pc1', 'pc6'):
        result[host]['limitation'] = 'Snapshot of ' + host.upper() + ' records; absent identities are not authorized.'
    if candidate is not None:
        digest = AUDIT.identity(candidate)
        hosts = [name for name, sessions in (('pc1', first_sessions), ('pc6', second_sessions)) if digest in sessions]
        result['candidate'] = {'sha256': digest, 'observed_hosts': hosts,
                               'status': 'CONSUMED_NO_REPLAY' if hosts else 'NOT_OBSERVED_NOT_AUTHORIZED'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', help='JSON file containing the exact five session identity fields')
    args = parser.parse_args()
    candidate = AUDIT.read_part(args.candidate)[0] if args.candidate else None
    paths1 = [ROOT / f'provenance/recovery-20261030/pc1-ledgers/part-{i}.json' for i in range(1, 6)]
    paths6 = [ROOT / 'provenance/recovery-20261030/pc6-ledgers/research-ledgers.json']
    result = audit_pair(paths1, paths6, PC1, PC6, candidate)
    print(json.dumps(result, sort_keys=True, separators=(',', ':')))
    return 1 if any(result[h]['claims_without_done'] or result[h]['done_without_claim'] or
                    result[h]['sessions_without_closed_marker'] or result[h]['nonzero_close_markers'] or
                    result[h]['sessions_without_process_marker'] for h in ('pc1', 'pc6')) else 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
        raise SystemExit('Archive inspection refused: ' + str(error))
