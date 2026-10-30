"""Verify the retained quantization result; trace and captured bytes are needed to rerun."""
import hashlib,json
from pathlib import Path
p=Path(__file__).parent
hashes={'quant_trace.cpp':'98bca0dbc8fb6ca3a5e0056220f3fee61c4f36c50b142370b08c22b81363e9fb',
'verify_quant.py':'8915ec28c38143d03158bab604a558198d439bbc398f2aebd6e63db87952f2f1',
'RESULT.json':'2ede7922fa07c6f3aa2b6a24630821a899de17b09ebcc6e3b5d21366d46599ce',
'negative-q8k.txt':'bedf9a5fc25214ecf8efd1cd2db454ed3c72b40a44b3caf48c5ef01d2aeac3c6',
'negative-q8.txt':'ad2f4aeeadd3710945c8becf4b596fb97b4fe963fe9fcc076bf12bf7b8fe47b7'}
for name,digest in hashes.items(): assert hashlib.sha256((p/name).read_bytes()).hexdigest()==digest
r=json.loads((p/'RESULT.json').read_text())
assert r=={'positions':92,'q8_0_blocks':17664,'q8k_blocks':920,'quant_bytes':869216,'quant_sha256':'a12b7cd214277039e94458a44c824645823505fd1057ae9a311534ccd6e887c5','status':'PASS','trace_sha256':'8787cdb8133d0962c23e273507ea471dc4d45014ac7c4f2d323705764da2c54e'}
assert 'Q8_K mismatch position=91 byte=4' in (p/'negative-q8k.txt').read_text()
assert 'Q8_0 mismatch position=91 byte=2' in (p/'negative-q8.txt').read_text()
print('PASS_SUMMARY: 92 real positions; both mutation controls rejected')
