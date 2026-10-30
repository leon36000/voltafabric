import json,pathlib,sys
request=json.loads(sys.stdin.readline())
assert request=={'op':'hello','protocol':1,'rank':2}
print(json.dumps({'reply':'hello','rank':2,'capabilities':['cpu_qpn2_resident_raw_v1'],'max_chunk_bytes':4194304}),flush=True)
remaining=sys.stdin.buffer.read()
pathlib.Path(sys.argv[1],'legacy-evidence.json').write_text(json.dumps({'post_hello_bytes':len(remaining)}))
