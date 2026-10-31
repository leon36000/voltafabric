import hashlib, json
from pathlib import Path
p = Path(__file__).parent
m = json.loads((p / "CLAIMS-MANIFEST.json").read_text())
records = []
for part in m["parts"]:
    raw = (p / part["path"]).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == part["sha256"]
    rows = json.loads(raw)
    assert len(rows) == part["count"]
    records.extend(rows)
raw = (json.dumps(records, sort_keys=True, separators=(",", ":")) + "\n").encode()
assert len(records) == m["count"]
assert hashlib.sha256(raw).hexdigest() == m["original_sha256"]
print("verified", len(records), m["original_sha256"])
