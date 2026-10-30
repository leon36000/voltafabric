"""Check the retained R520 read-only probe against R473's prior local observation."""
import hashlib,json
from pathlib import Path
raw=(Path(__file__).parent/'ORIGIN.json').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='06a59f6e6641ed18b5b199e2b5e07ff8200cd95ba661479a0b4a9ad9c261006f'
r=json.loads(raw)
assert (r['layer'],r['expert'],r['hits'],r['misses'],r['entries'])==(2,255,1,1,1)
assert r['sha256']=='8a17e1a2a5f9a21e11d2bae64fc99a656c810e8ed16d15bed9a09800fccd9cb5'
print('PASS: prior observed source digest; one cache miss and hit')
