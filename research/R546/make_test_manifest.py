"""Derive a test-only Cargo manifest without auto-discovered/explicit binaries."""
from pathlib import Path
import sys
if len(sys.argv) != 2:
    raise SystemExit('usage: make_test_manifest INPUT_CARGO_TOML > OUTPUT_CARGO_TOML')
lines=Path(sys.argv[1]).read_text().splitlines()
out=[];skip=False; packages=0
for line in lines:
    if line.startswith('[[bin]]'):
        skip=True
        continue
    if skip and line.startswith('[') and not line.startswith('[[bin]]'):
        skip=False
    if not skip:
        out.append(line)
    if line == '[package]':
        packages+=1
        out.append('autobins = false')
if packages != 1:
    raise ValueError('expected one package')
sys.stdout.write('\n'.join(out)+'\n')
