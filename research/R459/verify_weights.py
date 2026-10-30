#!/usr/bin/env python3
"""Cross-check the locally observed seed and admitted blobs against GGUF slices."""
import json
import sys
from observe_sc117_experts import run

result_path, index_path, pack_path, shard_path = sys.argv[1:]
r = json.load(open(result_path, encoding="utf-8"))
expected = {(layer, expert): digest
            for layer, hashes in enumerate(r["seed_hashes"])
            for expert, digest in zip(r["seed_experts"], hashes, strict=True)}
for layer, expert, rank, local, digest in r["dynamic_admissions"]:
    if (layer, expert) in expected:
        raise ValueError("duplicate expert admission")
    expected[(layer, expert)] = digest
selections = sorted(expected)
records = []
for i in range(0, len(selections), 16):
    records.extend(run(index_path, pack_path, selections[i:i + 16], shard_path)["experts"])
if not all(expected[(x["layer"], x["expert"])] == x["sha256"]
           and x["matches_source_shard_slices"] for x in records):
    raise ValueError("observed source differs from admitted digest")
print(json.dumps({"scope": "local observation; not publisher authentication", "experts": records}, sort_keys=True))
