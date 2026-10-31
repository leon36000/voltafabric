"""Check run-bound receipts, sticky admissions, and retained non-replayable claims."""
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

p = Path(__file__).parent
plan = json.loads((p / "PLAN.json").read_bytes())
prepared = json.loads((p / "PREPARED.json").read_bytes())
result = json.loads((p / "RESULT.json").read_bytes())
claims = json.loads((p / "CLAIMS.json").read_bytes())
r = result["receipt"]
assert r["plan_id"] == plan["engine_plan_id"] == prepared["plan_id"]
assert r["rank_affinity"] is True and r["layer_count"] == 48 and r["workers_closed"] == 96
assert set(r["worker_transports"].values()) == {"local"}
assert r["prompt_tokens"] == prepared["prompt_token_ids"] == result["prompt_token_ids"]
assert r["prompt_tokens_sha256"] == prepared["prompt_tokens_sha256"]
seen_rank = {(layer, expert): 2 + expert for layer in range(48) for expert in (0, 1)}
readmissions = 0
for layer, expert, rank, local, digest in r["dynamic_admissions"]:
    key = (layer, expert)
    if key in seen_rank:
        readmissions += 1
        assert seen_rank[key] == rank
    else:
        seen_rank[key] = rank
    assert local >= 0 and len(digest) == 64
for e in r["evictions"]:
    b = e["binding"]
    assert seen_rank[(b["layer"], b["expert"])] == b["rank"]
assert readmissions > 0
assert all(x[0] in (2, 3) and x[1] == x[2] for x in r["worker_exec_counts"])
assert len(r["worker_exec_counts"]) == 96
assert result["output"]["token_ids"] == r["generated_token_ids"]
assert result["output"]["run_id"] == r["run_id"]
previous = json.loads((p.parent / "R571" / "RESULT.json").read_bytes())
for field in ("prompt_tokens", "input_tokens", "generated_token_ids", "output_sha256", "embedding_sha256", "logits_sha256"):
    assert r[field] == previous["receipt"][field], field
assert result["output"]["text"] == previous["output"]["text"]
sticky = json.loads((p.parent / "R578" / "RESULT.json").read_bytes())["receipt"]
for field in ("input_tokens", "generated_token_ids", "embedding_sha256", "output_sha256", "logits_sha256", "dynamic_admissions"):
    assert r[field] == sticky[field], field
kinds = Counter()
filenames = set()
for record in claims:
    raw = record["raw"].encode()
    assert hashlib.sha256(raw).hexdigest() == record["sha256"]
    claim = json.loads(raw)
    stamp = claim["stamp"]
    assert claim["status"] == "RESERVED_UNKNOWN_NO_REPLAY"
    assert stamp["run_id"] == r["run_id"] and stamp["plan_id"] == r["plan_id"]
    assert stamp["model_revision"] == "sc117-local-observation" and stamp["epoch"] == 1
    ev = "binding" in claim
    key = b"VF_NATIVE_EVICTION_V1\0" if ev else b"VF_NATIVE_BATCH_V1\0"
    for s in (stamp["run_id"], stamp["model_revision"], stamp["plan_id"]):
        b = s.encode(); key += struct.pack("<Q", len(b)) + b
    key += struct.pack("<I", stamp["layer"])
    if ev:
        key += struct.pack("<I", claim["binding"]["expert"])
    key += struct.pack("<QQ", stamp["batch"], stamp["epoch"])
    digest = hashlib.sha256(key).hexdigest()
    name = ("native-evict-" if ev else "native-") + digest + ".claim.json"
    assert name == record["file"] and name not in filenames
    assert not ev or claim["operation_id"] == digest
    filenames.add(name)
    kinds["evictions" if ev else "batches"] += 1
assert kinds["evictions"] == len(r["evictions"])
assert kinds["batches"] == 48 * ((len(r["prompt_tokens"]) // 2) + (len(r["prompt_tokens"]) % 2) + max(0, len(r["input_tokens"]) - len(r["prompt_tokens"])))
print("PASS", json.dumps({"claims": dict(kinds), "readmissions": readmissions, "generated": r["generated_token_ids"], "output": result["output"]}, sort_keys=True))
