#!/usr/bin/env python3
"""Validate recorded independent comparisons; no model execution occurs."""
import json
from pathlib import Path
p = Path(__file__).parent
for name, expected_sha, mismatch in (
    ('REAL', 'c8a0744a87fade2652acfb34c802a49354379f558791bdf63e53f1f124bcc88a', 0),
    ('BOUNDARY', '06f1665640dd1c45abedc210c281e3ac3d5d5d9c6d18c0682c7b40f57c7a4c11', 480),
):
    a = json.loads((p / (name + '_PC1.json')).read_text())
    b = json.loads((p / (name + '_PC6.json')).read_text())
    if a != b or a['input_sha256'] != expected_sha:
        raise ValueError('CPU observations differ')
    for kind in ('q80','q8k'):
        if a[kind]['reference_mismatch_bytes'] != 0 or a[kind]['nearest_even_mismatch_bytes'] != 0:
            raise ValueError('scalar oracle mismatch')
    if a['q80']['optimized_mismatch_bytes'] != mismatch or a['q8k']['optimized_mismatch_bytes'] != 0:
        raise ValueError('unexpected rounding result')
print('PASS: two CPU observations match; Q8_0 tie modes differ on the constructed boundary')
