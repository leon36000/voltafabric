#!/usr/bin/env python3
"""Check the ordinary Collector's order and coefficients independently."""
import hashlib, json, struct, sys
from pathlib import Path

def f32(x): return struct.unpack('<f', struct.pack('<f', x))[0]
def u64(x): return struct.unpack('<Q', struct.pack('<d', x))[0]
def main():
    if len(sys.argv) != 3: raise SystemExit('usage: verify.py SPEC RESULT')
    spec_bytes = Path(sys.argv[1]).read_bytes()
    assert hashlib.sha256(spec_bytes).hexdigest() == '3c20888b55920f5b108115ca4d24208de578755bc286c7c8ae8301959ecfea20'
    spec = json.loads(spec_bytes); result = json.loads(Path(sys.argv[2]).read_bytes())
    assert result['spec_sha256'] == hashlib.sha256(spec_bytes).hexdigest()
    assert {r['rank'] for r in result['receipts']} == {2, 3}
    for t in range(2):
        weights = [f32(x['weight']) for x in spec['inputs'][t]['choices']]
        rows = result['raw_f32_bits'][t]
        assert len(rows) == 10 and all(len(row) == 2560 for row in rows)
        for col in range(2560):
            value = 0.0
            for slot in range(10):
                number = struct.unpack('<f', struct.pack('<I', rows[slot][col]))[0]
                half = struct.unpack('<e', struct.pack('<e', number))[0]
                value += weights[slot] * half
            assert u64(value) == result['collected_f64_bits'][t][col], (t, col)
    print(json.dumps({'status':'pass','tokens':2,'slots':20,'collector_elements':5120,
                      'run_id':result['run_id'],'plan_sha256':result['plan_sha256'],
                      'receipts':result['receipts']}))
if __name__ == '__main__': main()
