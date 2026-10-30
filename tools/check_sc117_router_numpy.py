#!/usr/bin/env python3
"""Optional independent NumPy check of the SC117 CPU reference router."""
import argparse
import json
import subprocess
import numpy as np

ap = argparse.ArgumentParser(description=__doc__)
for item in ("dense", "input", "router", "digest"):
    ap.add_argument("--"+item, required=True)
a = ap.parse_args()
with open(a.dense,"rb") as f:
    f.seek(13_148_160)
    raw=f.read(2_621_440)
if len(raw)!=2_621_440: raise SystemExit("short router read")
w=np.frombuffer(raw,dtype="<u2").astype(np.uint32).reshape(512,2560)
w=(w<<16).view(np.float32).astype(np.float64)
with open(a.input,"rb") as f: x=np.frombuffer(f.read(10240),dtype="<f4").copy()
if x.size!=2560: raise SystemExit("short input")
bits=x.view(np.uint32)
rounded=((bits+(np.uint32(0x7fff)+((bits>>16)&1))) & np.uint32(0xffff0000)).view(np.float32).astype(np.float64)
logits=(w@rounded).astype(np.float32)
ex=np.exp(logits.astype(np.float64)-float(logits.max()))
inv=np.float32(1/ex.sum())
prob=(ex*float(inv)).astype(np.float32)
ids=np.argsort(-prob,kind="stable")[:10]
weights=(prob[ids].astype(np.float64)/max(sum(map(float,prob[ids])),2**-14)).astype(np.float32)
route=json.loads(subprocess.check_output([a.router,a.dense,a.input,a.digest]))
if [r["expert"] for r in route]!=ids.tolist(): raise SystemExit("router IDs differ")
weight_error=max(abs(float(r["weight"])-float(wt)) for r,wt in zip(route,weights))
logit_error=max(abs(float(r["logit"])-float(logits[int(i)])) for r,i in zip(route,ids))
if weight_error>1e-7: raise SystemExit("router weights differ")
print(json.dumps({"status":"ok","ids":ids.tolist(),"max_weight_difference":weight_error,"max_logit_difference":logit_error},sort_keys=True))
