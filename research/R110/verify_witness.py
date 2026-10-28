#!/usr/bin/env python3
"""One synthetic IQ3_S block, actual archived codebook and actual Rust decoder.

This is a type-level counterexample. It never opens real model payloads.
"""
from fractions import Fraction
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    receipts = {}
    for name in ('iq3_witness', 'mxfp4_witness'):
        binary = ROOT / name
        result = subprocess.run([str(binary)], cwd=ROOT, text=True, capture_output=True, timeout=15, check=False)
        receipts[name] = dict(argv=[str(binary)], binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                              stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode)
        if result.returncode != 0:
            raise AssertionError(f'{name} exited {result.returncode}')
    iq = json.loads(receipts['iq3_witness']['stdout'])
    fp = json.loads(receipts['mxfp4_witness']['stdout'])
    assert iq == dict(block_bytes=110, grid_index=2, grid_word=0x01010105, d_binary16_bits=0x2c00,
                      scale_multiplier=1, first_weight_numerator=5, denominator=16,
                      control_grid_index=0, control_weight_numerator=1)
    target = Fraction(iq['first_weight_numerator'], iq['denominator'])
    control = Fraction(iq['control_weight_numerator'], iq['denominator'])
    codes = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2), Fraction(2), Fraction(3), Fraction(4), Fraction(6))
    # Exact rational oracle independent of Rust float arithmetic; all scale bytes 0..254.
    values = [sign * code * (Fraction(2) ** (scale - 127)) for sign in (1, -1) for code in codes for scale in range(255)]
    assert len(values) == fp['pairs_tested'] == 4080
    assert sum(x == target for x in values) == fp['matches_5_over_16'] == 0
    assert sum(x == control for x in values) == fp['matches_1_over_16'] > 0
    assert min(abs(x-target) for x in values) == Fraction(1,16)
    assert fp['best_absolute_error_numerator'] == 1 and fp['denominator'] == 16
    assert fp['reserved_scale_rejected'] is True
    # IQ3_S groups 256 elements in 110 bytes; eight MXFP4 blocks use 8*(16+1)=136 bytes.
    assert 8 * (16+1) == 136 != iq['block_bytes']
    receipt = dict(status='UNIVERSAL_EXACT_IQ3S_TO_E2M1_E8M0_REJECTED_BY_SYNTHETIC_WITNESS',
                   synthetic=True, actual_model_payload_read=False, actual_kernel_executed=False,
                   target='5/16', nearest_mxfp4='4/16 or 6/16', absolute_error='1/16',
                   relative_error='20%', finite_scale_bytes=255, nibble_codes=16,
                   actual_rust_functions=['voltafabric::quant::decode_fp4','voltafabric::deepseek_v41::decode_e8m0'],
                   only_one_synthetic_lane=True, IQ3S_256_bytes=110, MXFP4_256_bytes=136,
                   scope='Historical R46 archived GGML grid and documented IQ3_S lane decode; no claim on actual SC117 payload or installed Strata binary.',
                   commands=receipts)
    (ROOT/'RESULT.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, sort_keys=True))

if __name__ == '__main__':
    main()
