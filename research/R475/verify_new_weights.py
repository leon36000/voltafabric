#!/usr/bin/env python3
"""Compare newly seen expert blobs with independent local GGUF observations."""
import json
import sys
from observe_sc117_experts import run

result_path, prior_a_path, prior_b_path, prior_c_path, index_path, pack_path, shard_path = sys.argv[1:]
r = json.load(open(result_path, encoding='utf-8'))['receipt']
a = json.load(open(prior_a_path, encoding='utf-8'))['experts']
b = json.load(open(prior_b_path, encoding='utf-8'))['experts']
c = json.load(open(prior_c_path, encoding='utf-8'))['experts']
prior = {}
for v in a+b+c:
    key=(v['layer'],v['expert'])
    if not v['matches_source_shard_slices'] or (key in prior and prior[key]!=v['sha256']):
        raise ValueError('inconsistent previous observation')
    prior[key]=v['sha256']
expected={(layer,expert):digest for layer,hashes in enumerate(r['seed_hashes'])
          for expert,digest in zip(r['seed_experts'],hashes,strict=True)}
repeated=0
for layer,expert,rank,local,digest in r['dynamic_admissions']:
    key=(layer,expert)
    if key in expected:
        if expected[key]!=digest: raise ValueError('re-admitted digest differs')
        repeated+=1
    expected[key]=digest
for key in expected.keys() & prior.keys():
    if expected[key]!=prior[key]: raise ValueError('previously observed digest differs')
new=sorted(expected.keys()-prior.keys()); records=[]
for i in range(0,len(new),16):
    records.extend(run(index_path,pack_path,new[i:i+16],shard_path)['experts'])
if len(records)!=len(new) or any(expected[(x['layer'],x['expert'])]!=x['sha256'] or not x['matches_source_shard_slices'] for x in records):
    raise ValueError('new source observation differs')
print(json.dumps({'scope':'local consistency; not publisher authentication',
                  'prior_matching':len(expected.keys() & prior.keys()), 'new_count':len(new),
                  'unique_total':len(expected),'repeat_admissions':repeated,'experts':records},sort_keys=True))
