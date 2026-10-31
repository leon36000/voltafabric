import hashlib, json
from pathlib import Path
p=Path(__file__).parent
m=json.loads((p/"RESULT-MANIFEST.json").read_text())
chunks=[]
for part in m["parts"]:
    data=(p/part["path"]).read_bytes()
    assert len(data)==part["bytes"] and hashlib.sha256(data).hexdigest()==part["sha256"]
    chunks.append(data)
raw=b"".join(chunks)
assert len(raw)==m["bytes"] and hashlib.sha256(raw).hexdigest()==m["original_sha256"]
json.loads(raw)
print("verified result",m["bytes"],m["original_sha256"])
