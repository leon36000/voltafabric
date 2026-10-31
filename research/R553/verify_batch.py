#!/usr/bin/env python3
"""Independent scalar audit against the pinned pre-grouping FP32 outputs."""
import hashlib
import json
import struct
import sys
from pathlib import Path

OLD_HASHES = ("a988a43e1182724453111498f3dceef54d442354e4c486be6ea0c53131fc974b",
              "9e7d39ebb18c3838593f302d35e12c2e32f42192c9e9fbb6f3abb81cb64b93ae")
IDS = (12, 64, 75, 159, 181, 209, 255, 257, 272, 302, 480, 492)
ROUTES = ((255, 12, 75, 181), (181, 75, 12, 255))
WEIGHTS = ((0.125, 0.25, 0.375, 0.25), (0.5, 0.125, 0.125, 0.25))

def bits64(x):
    return struct.unpack('<Q', struct.pack('<d', x))[0]

def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: verify_batch.py RESULT OLD0 OLD1')
    result = json.loads(Path(sys.argv[1]).read_bytes())
    assert len(result['receipts']) == 2 and {r['rank'] for r in result['receipts']} == {2, 3}
    for token in range(2):
        old = Path(sys.argv[2 + token]).read_bytes()
        assert len(old) == 12 * 2560 * 4 and hashlib.sha256(old).hexdigest() == OLD_HASHES[token]
        words = struct.unpack('<30720I', old)
        for slot, expert in enumerate(ROUTES[token]):
            pos = IDS.index(expert) * 2560
            assert result['raw_f32_bits'][token][slot] == list(words[pos:pos + 2560]), (token, slot)
        assert len(result['collected_f64_bits'][token]) == 2560
        for col in range(2560):
            value = 0.0
            for slot, weight in enumerate(WEIGHTS[token]):
                bits = result['raw_f32_bits'][token][slot][col]
                f32 = struct.unpack('<f', struct.pack('<I', bits))[0]
                f16 = struct.unpack('<e', struct.pack('<e', f32))[0]
                value += weight * f16
            assert bits64(value) == result['collected_f64_bits'][token][col], (token, col)
    print(json.dumps({'status':'pass','raw_elements':20480,'collector_elements':5120,
                      'old_baseline_sha256':OLD_HASHES,'run_id':result['run_id'],
                      'plan_sha256':result['plan_sha256'],'receipts':result['receipts']}))

if __name__ == '__main__':
    main()
