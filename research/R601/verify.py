"""Verify retained R601 claim bytes and independently derive reservation names."""
import hashlib
import json
import struct
from pathlib import Path

p = Path(__file__).parent
plan_bytes = (p / "PLAN.json").read_bytes()
plan_hash = hashlib.sha256(plan_bytes).hexdigest()
plan = json.loads(plan_bytes)
records = json.loads((p / "CLAIMS.json").read_bytes())
assert len(records) == 3
seen = set()
kinds = []
for record in records:
    raw = record["raw"].encode()
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    c = json.loads(raw)
    s = c["stamp"]
    assert c["status"] == "RESERVED_UNKNOWN_NO_REPLAY"
    assert s == {"run_id": plan["run_id"], "model_revision": "synthetic-r601", "plan_id": "r601-" + plan_hash, "layer": 0, "batch": s["batch"], "epoch": 1}
    eviction = "binding" in c
    key = b"VF_NATIVE_EVICTION_V1\0" if eviction else b"VF_NATIVE_BATCH_V1\0"
    for text in (s["run_id"], s["model_revision"], s["plan_id"]):
        value = text.encode(); key += struct.pack("<Q", len(value)) + value
    key += struct.pack("<I", s["layer"])
    if eviction:
        binding = c["binding"]
        assert (binding["expert"], binding["rank"], binding["local"]) == (2, 2, 1)
        key += struct.pack("<I", binding["expert"])
    key += struct.pack("<QQ", s["batch"], s["epoch"])
    digest = hashlib.sha256(key).hexdigest()
    name = ("native-evict-" if eviction else "native-") + digest + ".claim.json"
    assert name == record["file"] and name not in seen
    assert not eviction or c["operation_id"] == digest
    seen.add(name)
    kinds.append(("evict" if eviction else "batch", s["batch"], c["attempt"]))
assert sorted(kinds) == [("batch", 0, 1), ("batch", 1, 2), ("evict", 0, 1)]
assert "1 passed; 0 failed" in (p / "RESULT.txt").read_text()
print("PASS: exact retained bytes and three distinct non-replayable claim identities")
