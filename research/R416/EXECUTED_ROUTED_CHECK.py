#!/usr/bin/env python3
"""One-shot local CPU route and expert check using operator-reviewed observations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import tempfile
import uuid

INDEX_SHA = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24"

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("pack", "index", "dense", "input", "observation", "bridge", "worker", "oracle", "router", "work-root", "router-digest"):
        ap.add_argument("--" + name, required=True)
    ap.add_argument("--collector", help="optional local vf executable for explicit FP16/f64 collection")
    ap.add_argument("--model-input-sha256", help="expected SHA-256 of bounded model-derived input file")
    ap.add_argument("--layer", type=int, choices=(0,1), default=0)
    ap.add_argument("--route-json", help="explicit observed route JSON; required for layer 1")
    ap.add_argument("--route-sha256", help="SHA-256 of the explicit route JSON")
    a = ap.parse_args()
    if a.layer == 1 and (not a.model_input_sha256 or not a.route_json or not a.route_sha256):
        ap.error("layer 1 requires model input and route digests")
    if bool(a.route_json) != bool(a.route_sha256): ap.error("explicit route requires its digest")
    for path in (a.pack,a.index,a.dense,a.input,a.observation,a.bridge,a.worker,a.oracle,a.router,a.work_root):
        if not os.path.isabs(path): ap.error("all paths must be absolute")
    work = Path(tempfile.mkdtemp(prefix="routed-session-", dir=a.work_root)); ledger = work / "ledger"; ledger.mkdir()
    input_path = a.input
    if a.model_input_sha256:
        fd = os.open(a.input, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            st = os.fstat(fd)
            import stat
            if not stat.S_ISREG(st.st_mode) or not 10240 <= st.st_size <= 1048576: raise ValueError("invalid model input file")
            raw = os.read(fd, st.st_size + 1)
            after = os.fstat(fd)
            if len(raw)!=st.st_size or (st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns): raise ValueError("input changed")
            if hashlib.sha256(raw).hexdigest()!=a.model_input_sha256: raise ValueError("model input digest mismatch")
        finally: os.close(fd)
        input_path = str(work / "activation.f32")
        with open(input_path,"xb") as f: f.write(raw[:10240])
    if a.route_json:
        if not os.path.isabs(a.route_json): ap.error("route path must be absolute")
        route_data = Path(a.route_json).read_bytes()
        if len(route_data)>16384 or hashlib.sha256(route_data).hexdigest()!=a.route_sha256: raise ValueError("explicit route digest mismatch")
        route = json.loads(route_data)
    else:
        route = json.loads(subprocess.run([a.router,a.dense,input_path,a.router_digest], check=True, timeout=30, capture_output=True, text=True).stdout)
    # JSON decimal text must round-trip to the original binary32 coefficients.
    for entry in route:
        entry["weight"]=struct.unpack("<f",struct.pack("<f",entry["weight"]))[0]
    ids = [x["expert"] for x in route]
    if len(ids)!=10 or len(set(ids))!=10 or any(not 0<=i<512 for i in ids): raise ValueError("invalid CPU route")
    if any(not math.isfinite(x["weight"]) or x["weight"]<0 for x in route): raise ValueError("invalid route weights")
    observations = json.loads(Path(a.observation).read_text())
    if observations.get("index_sha256")!=INDEX_SHA or observations.get("kind")!="sc117-observation-not-authentication": raise ValueError("invalid observation")
    records=observations.get("experts",[])
    if [r.get("expert") for r in records]!=ids or any(r.get("layer")!=a.layer or r.get("matches_source_shard_slices") is not True for r in records):
        raise ValueError("observation does not cover route in slot order")
    refs=[]
    for expert in ids:
        path=work / f"reference-{expert}.bin"
        command=[a.oracle,a.pack,str(expert),str(path)]
        if a.model_input_sha256: command.append(input_path)
        if a.layer == 1: command.append("1")
        subprocess.run(command,check=True,timeout=30,stdout=subprocess.DEVNULL)
        raw=path.read_bytes()
        if len(raw)!=2560*12: raise ValueError("invalid oracle output")
        refs.append((struct.unpack("<2560f",raw[:10240]),struct.unpack("<2560d",raw[10240:])))
    input_bytes=Path(input_path).read_bytes()[:10240]
    if len(input_bytes)!=10240 or struct.unpack("<2560f",input_bytes)!=refs[0][0] or any(r[0]!=refs[0][0] for r in refs):
        raise ValueError("route and oracle inputs differ")
    run_id="sc117-gate-cpu-"+uuid.uuid4().hex
    stamp={"run_id":run_id,"model_revision":"sc117-local-observation-r399","plan_id":"cpu-gate-10-local","layer":a.layer,"batch":0,"epoch":1}
    copies=[]; bindings=[]; orders={"2":[],"3":[]}
    for slot,(expert,record) in enumerate(zip(ids,records)):
        rank=2+slot%2; local=slot//2; orders[str(rank)].append(local)
        copies.append({"expert":expert,"rank":rank,"local":local})
        bindings.append({"model_revision":stamp["model_revision"],"rank":rank,"layer":a.layer,"expert":expert,"local":local,"epoch":1,
                         "index_sha256":INDEX_SHA,"blob_sha256":record["sha256"]})
    init={"template":{"hidden":2560,"experts":ids,"routes":[[{"expert":e,"weight":float(r["weight"])} for e,r in zip(ids,route)]],
                      "registry":{"epoch":1,"copies":copies},"stamp":stamp,"preferred_owners":{},"resident_order":orders},
          "model_experts":list(range(512)),"index":Path(a.index).read_text(),"pack_path":a.pack,"ledger_path":str(ledger),
          "endpoints":[{"rank":rank,"argv":[a.worker]} for rank in (2,3)],"bindings":bindings,
          "resident_payload_limit":10*(2_176_000 if a.layer==0 else 1_510_400),"timeout_ms":30_000}
    batch={"op":"batch","batch":1,"tokens":1,"top_k":10,"inputs":refs[0][0],"ids":ids,"kinds":[-1]*10}
    payload="".join(json.dumps(x,allow_nan=False)+"\n" for x in (init,batch,{"op":"shutdown"}))
    child=subprocess.Popen([a.bridge],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    try: out,err=child.communicate(payload,timeout=90)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGKILL); child.communicate(); raise RuntimeError("unknown outcome; never replay")
    if child.returncode!=0: raise RuntimeError("bridge failed; never replay: "+err[-1000:])
    replies=[json.loads(line) for line in out.splitlines()]
    if len(replies)!=3 or replies[0].get("state")!="ready" or replies[2].get("state")!="closed": raise ValueError("incomplete bridge conversation")
    result=replies[1]; rows=result.get("cpu_rows",[])
    if result.get("state")!="completed" or result.get("batch")!=1 or [r.get("position") for r in rows]!=list(range(10)):
        raise ValueError("invalid result coverage")
    # The bridge transmits f32 values as JSON decimals; restore the precise
    # binary32 values before half conversion or downstream arithmetic.
    for row in rows:
        row["value"]=[struct.unpack("<f",struct.pack("<f",v))[0] for v in row["value"]]
    max_error=0.0; max_combined=0.0
    for r,ref in zip(rows,refs):
        if len(r["value"])!=2560: raise ValueError("invalid output width")
        for got,want in zip(r["value"],ref[1]):
            if not math.isfinite(got) or not math.isfinite(want): raise ValueError("nonfinite result")
            error=abs(got-want); max_error=max(max_error,error)
            if error>1e-7+1e-3*abs(want): raise ValueError("expert output differs from scalar reference")
    for col in range(2560):
        observed=sum(float(route[i]["weight"])*rows[i]["value"][col] for i in range(10))
        expected=sum(float(route[i]["weight"])*refs[i][1][col] for i in range(10))
        max_combined=max(max_combined,abs(observed-expected))
        if abs(observed-expected)>1e-7+1e-3*abs(expected): raise ValueError("weighted output differs")
    result_summary={"status":"ok","run_id":run_id,"ids":ids,"weights":[r["weight"] for r in route],
                    "max_row_error":max_error,"max_weighted_error":max_combined,"work_dir":str(work),
                    "scope":"CPU reference route, local real expert sessions; " + ("model-derived activation" if a.model_input_sha256 else "synthetic activation")}
    if a.collector:
        if not os.path.isabs(a.collector): raise ValueError("collector path must be absolute")
        claims=list(ledger.glob("*.claim.json"))
        if len(claims)!=1: raise ValueError("unexpected durable reservation count")
        claim=json.loads(claims[0].read_text())
        if claim.get("status")!="RESERVED_UNKNOWN_NO_REPLAY" or claim.get("stamp",{}).get("run_id")!=run_id:
            raise ValueError("unexpected durable reservation identity")
        attempt=claim["attempt"]
        problem=json.loads(json.dumps(init["template"]))
        problem["stamp"]["batch"]=1
        outputs=[]; half_rows=[]
        for row in rows:
            bits=[int.from_bytes(struct.pack("<e",v),"little") for v in row["value"]]
            half_rows.append(bits)
        for rank in (3,2):  # Deliberately reverse rank arrival; Collector restores slot order.
            slots=[]; bits=[]
            for slot,record in enumerate(copies):
                if record["rank"]==rank:
                    slots.append({"token":0,"slot":slot,"expert":record["expert"],"local":record["local"]})
                    bits.extend(half_rows[slot])
            outputs.append({"stamp":problem["stamp"],"rank":rank,"attempt":attempt,"rows":len(slots),
                            "columns":2560,"slots":slots,"fp16_bits":bits})
        request={"problem":problem,"expected_attempt":attempt,"outputs":outputs}
        proc=subprocess.run([a.collector,"collect-fp16","-"],input=json.dumps(request,allow_nan=False),
                            capture_output=True,text=True,check=True,timeout=30)
        collected=json.loads(proc.stdout)
        if not collected.get("ok") or collected["result"].get("accepted_rows")!=10: raise ValueError("collector refused")
        vector=collected["result"]["output"]
        if len(vector)!=1 or len(vector[0])!=2560: raise ValueError("collector output shape")
        max_collector_diff=0.0; max_fp16_loss=0.0
        for col,got in enumerate(vector[0]):
            want=sum(route[slot]["weight"]*struct.unpack("<e",half_rows[slot][col].to_bytes(2,"little"))[0] for slot in range(10))
            full=sum(route[slot]["weight"]*rows[slot]["value"][col] for slot in range(10))
            max_collector_diff=max(max_collector_diff,abs(got-want)); max_fp16_loss=max(max_fp16_loss,abs(got-full))
            if not math.isfinite(got) or abs(got-want)>1e-15: raise ValueError("collector differs from ordered FP16 reference")
        result_summary.update({"collector":"FP16 decoded, f64 accumulation", "collector_max_diff":max_collector_diff,
                               "fp16_max_loss_vs_unweighted_fp32":max_fp16_loss,"attempt":attempt})
        with open(work / "collector-output.json","x",encoding="utf-8") as f: json.dump(collected,f,allow_nan=False)
    print(json.dumps(result_summary,sort_keys=True))

if __name__=="__main__":
    try: main()
    except (OSError,ValueError,KeyError,RuntimeError,subprocess.SubprocessError) as error:
        print("diagnostic failed: "+str(error),file=sys.stderr); sys.exit(1)
