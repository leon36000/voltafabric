#!/usr/bin/env python3
"""Build a bounded batch spec and check route admissions against R552."""
import hashlib, json, sys
from pathlib import Path
if len(sys.argv) != 4: raise SystemExit('usage: extract_spec.py ROUTE_JSON R552_RESULT OUTPUT')
route_path, result_path, output = map(Path, sys.argv[1:])
r = json.loads(route_path.read_bytes()); receipt = json.loads(result_path.read_bytes())['receipt']
assert r['trace_sha256'] == 'a22a989c0b7b27a88f2f63ba558091cd510115be2295014c401c7c2c293346f2'
assert r['source_plan'] == receipt['plan_id'] and receipt['workers_closed'] == 96
a = [x for x in receipt['dynamic_admissions'] if x[0] == 3]
routes = [[c['expert'] for c in x['choices']] for x in r['records']]
assert len(routes) == 2 and [x[1] for x in a[:10]] == routes[0]
new = [x for x in routes[1] if x not in routes[0]]
assert [x[1] for x in a[10:10 + len(new)]] == new
ids = list(dict.fromkeys(routes[0] + routes[1])); latest = {x[1]: x for x in a}
meta = {str(e): {'rank': latest[e][2], 'local': latest[e][3], 'sha256': latest[e][4]} for e in ids}
spec = {'route_trace_sha256': hashlib.sha256(route_path.read_bytes()).hexdigest(), 'inputs': r['records'],
        'experts': meta, 'source_run_id': receipt['run_id'], 'source_plan_id': receipt['plan_id']}
payload = (json.dumps(spec, sort_keys=True, separators=(',', ':')) + '\n').encode()
with output.open('xb') as f: f.write(payload)
print(hashlib.sha256(payload).hexdigest())
