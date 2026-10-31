"""Check the stored PC1 native comparison receipt, without rerunning it."""
import hashlib,json
from pathlib import Path
p=Path(__file__).parent;root=p.parents[1];w=root/'.runs/vf-r627-work';raw=(p/'NATIVE-PLAN.json').read_bytes()
assert (w/'NATIVE-SUBMITTED').read_text().strip()==hashlib.sha256(raw).hexdigest()
r=json.loads((w/'NATIVE-RESULT.json').read_text());assert r=={'experts':10,'exact_f64_values':25600,'atomic_failure_verified':True}
print(json.dumps(r,sort_keys=True))
