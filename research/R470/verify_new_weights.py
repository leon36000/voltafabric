#!/usr/bin/env python3
"""Compare new admitted expert blobs with independently observed GGUF slices."""
import json
import sys
from observe_sc117_experts import run

result_path, prior_path, index_path, pack_path, shard_path = sys.argv[1:]
r = json.load(open(result_path, encoding='utf-8'))['receipt']
prior = json.load(open(prior_path, encoding='utf-8'))['experts']
known = {(x['layer'], x['expert']): x['sha256'] for x in prior if x['matches_source_shard_slices']}
if len(known) != len(prior):
    raise ValueError('invalid prior observation')
expected = {(layer, expert): digest for layer, hashes in enumerate(r['seed_hashes'])
            for expert, digest in zip(r['seed_experts'], hashes, strict=True)}
for layer, expert, rank, local, digest in r['dynamic_admissions']:
    if (layer, expert) in expected:
        raise ValueError('duplicate expert admission')
    expected[(layer, expert)] = digest
if any(expected.get(key) != digest for key, digest in known.items()):
    raise ValueError('prior observed expert changed')
new = sorted(expected.keys() - known.keys())
records = []
for i in range(0, len(new), 16):
    records.extend(run(index_path, pack_path, new[i:i+16], shard_path)['experts'])
if len(records) != len(new) or any(expected[(x['layer'],x['expert'])] != x['sha256'] or not x['matches_source_shard_slices'] for x in records):
    raise ValueError('new expert observation differs')
print(json.dumps({'scope':'local consistency; not publisher authentication','prior_count':len(known),'new_count':len(records),'total_count':len(expected),'experts':records},sort_keys=True))
