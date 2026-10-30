"""Check preserved diagnostic outputs, not correctness of all model executions."""
import hashlib,json
from pathlib import Path
p=Path(__file__).parent
hashes={'explainability_probe.cpp':'47e7012cb14e065d63e195ae67b75bdd7066a6a6e5af07dc7ae2a404f8680f87',
'final-expert75.out':'465a80c8c6f94067705682f78e111a918a5a0ee0b9dbe95c8fcaaaa5983c53b1',
'final-mid.out':'b891c4ebfb671de267a8b31ebad4a9a7be5808e7bd128550b2338aa087220a8e',
'final-none.out':'a93c53c4159575614a9cbd1a82606dbdccf3b4d2250d01c70122ffebe5b08806',
'final-output.out':'6764f82826d43c4c75af5640a2115ccb47bcea22f1773aa57e0e2d9a43b26915'}
for file,digest in hashes.items(): assert hashlib.sha256((p/file).read_bytes()).hexdigest()==digest
s=json.loads((p/'STATUS.json').read_text())
assert s['expert255']=={'classification':'STRICT_FAIL_BOUND_EXPLAINED','exit':19,'strict_violations':1169,'envelope_violations':0}
assert s['fault_mid']=={'exit':17,'rejected':True} and s['fault_output']=={'exit':18,'rejected':True}
assert s['expert75']=={'classification':'STRICT_PASS','exit':0,'strict_violations':0}
assert 'original_scalar_violations=1169' in (p/'final-none.out').read_text()
assert 'classification=STRICT_FAIL_BOUND_EXPLAINED' in (p/'final-none.out').read_text()
assert 'mid_violations=1' in (p/'final-mid.out').read_text()
assert 'explainability_envelope_violations=1' in (p/'final-output.out').read_text()
assert 'classification=STRICT_PASS' in (p/'final-expert75.out').read_text()
print('PASS_SUMMARY: strict failure preserved; corrupted intermediate and output refused')
