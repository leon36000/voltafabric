"""Bind the current generation trace to the previous full-vocabulary oracle."""
import hashlib,json,struct
from pathlib import Path
root=Path(__file__).resolve().parents[2];p=Path(__file__).parent
r=json.loads((p/'RESULT.json').read_text());prior=json.loads((root/'research/R623/RESULT.json').read_text())
head=(root/'.runs/vf-r627-work/head.bin').read_bytes();digest=hashlib.sha256(head).hexdigest()
assert len(head)==1003524 and digest==r['head_sha256']==prior['head_sha256']
assert struct.unpack('<I',head[-4:])[0]==r['generated_token_ids'][0]==prior['reference_argmax']==9419
assert prior['rows']==248320 and prior['violations']==0
print(json.dumps({'head_sha256':digest,'inherited_oracle':'R623 identical bytes','rows':prior['rows'],'violations':prior['violations'],'argmax':9419},indent=2,sort_keys=True))
