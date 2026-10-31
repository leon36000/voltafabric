"""Check run-bound receipts, sticky admissions, and retained non-replayable claims."""
import hashlib
import json
import struct
from collections import Counter
from pathlib import Path

p = Path(__file__).parent
plan = json.loads((p / "PLAN.json").read_bytes())
prepared = json.loads((p / "PREPARED.json").read_bytes())
work = p.parents[1] / ".runs/vf-r629-work"
result = json.loads((work / "RESULT.json").read_bytes())
claims = json.loads((work / "CLAIMS.json").read_bytes())
assert (work / "EXIT_STATUS").read_text().strip() == "0"
assert (work / "SUBMITTED").read_text().strip() == hashlib.sha256((p / "PLAN.json").read_bytes()).hexdigest()
assert not (work / "error.log").read_bytes()
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
assert r["native_math_backend"] == "scalar-reference-ggml-v1"
previous = json.loads((p.parents[1] / ".runs/vf-r616-work/RESULT.json").read_bytes())
assert r["prompt_tokens"] == previous["receipt"]["prompt_tokens"]
comparison = {field: r[field][:len(previous["receipt"][field])] == previous["receipt"][field] for field in ("input_tokens", "generated_token_ids", "output_sha256", "embedding_sha256", "logits_sha256")}
assert all(comparison.values()), comparison
assert r["dynamic_admissions"][:len(previous["receipt"]["dynamic_admissions"])] == previous["receipt"]["dynamic_admissions"]
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
summary = {"claims":dict(kinds), "readmissions":readmissions, "generated_token_ids":r["generated_token_ids"],
           "generated_text":result["output"]["text"], "stop_reason":result["output"]["stop_reason"],
           "comparison_with_r616_scalar_prefix":comparison, "run_id":r["run_id"],"plan_id":r["plan_id"],
           "workers_closed":r["workers_closed"],"backend":r["native_math_backend"],
           "receipt_sha256":hashlib.sha256((work / "RESULT.json").read_bytes()).hexdigest(),
           "gpu":False,"network_worker":False}
print(json.dumps(summary,indent=2,sort_keys=True))
