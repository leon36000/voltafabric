"""Verify the real-format receipt and its pinned input identities, without rerunning it."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).parent;w=root/'.runs/vf-r626-real'
h=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
raw=(p/'PLAN.json').read_bytes();plan=json.loads(raw)
assert (w/'SUBMITTED').read_text().strip()==hashlib.sha256(raw).hexdigest()
assert h(root/plan['binary'])==plan['binary_sha256'] and h(p/'run.py')==plan['runner_sha256']
for name,expected in plan['files'].items():assert h(root/name)==expected,name
assert json.loads((w/'RESULT.json').read_text())==json.loads((p/'RESULT.json').read_text())=={'atomic_failure_verified':True,'cases':28,'exact_f64_values':71680,'format_pairs':7}
assert len(plan['cases'])==28 and len({(x['gu'],x['down']) for x in plan['cases']})==7
print('PASS: receipt, plan and input hashes; no new worker')
