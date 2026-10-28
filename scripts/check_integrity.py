#!/usr/bin/env python3
"""Validate delivered files before modification; this is integrity, not a signature."""
import hashlib
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
manifest = root/'SHA256SUMS'
failures = []
for line in manifest.read_text().splitlines():
    expected, name = line.split('  ', 1)
    p = (root/name).resolve()
    if not p.is_relative_to(root) or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
        failures.append(name)
print('Integrity PASS' if not failures else 'Integrity FAIL: '+', '.join(failures))
sys.exit(bool(failures))
