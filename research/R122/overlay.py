#!/usr/bin/env python3
"""Prepare, but do not install, the pinned Skinny patch set and upstream originals."""
import argparse
import ast
import hashlib
import json
import os
import stat
import zipfile
from pathlib import Path

PATHS = {
    'modelopt.py': 'vllm/model_executor/layers/quantization/modelopt.py',
    'marlin.py': 'vllm/model_executor/kernels/linear/nvfp4/marlin.py',
}
PINNED = {
    'wheel': '8a628983ad9d675559910372643220c418b307ddc7fd52ac65a7f5fbcb104bc6',
    'originals': {'modelopt.py': '1c58627e479e13aea2ad26c5da122a3a65a7335c3a06a812d4a7f5a20e94ef11',
                  'marlin.py': '6dd9892f215049f24bd6479cca32463a6256baba5957b3ed549e35f210502e06'},
    'patches': {'modelopt.py': '7d8288560318c37c340c6aff44c7d2c32ff80d7c22bd9b2eb82cb82e6bd1fb8f',
                'marlin.py': 'b2e1f7c402d83f133244337fc2352a640e5cb5b67e8279938e2da18f4a7cf147',
                'skinny_moe.py': 'fa6a5459d3b913594104618076973390c28c8c252cc2edf35998e7b047339abd'},
}


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def _open_regular(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    if not stat.S_ISREG(os.fstat(fd).st_mode):
        os.close(fd)
        raise ValueError('not a regular file')
    return os.fdopen(fd, 'rb')


def _write(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def prepare_overlay(wheel, sources, destination, expected):
    """Read and validate bytes before creating a new private destination.

    Output is review material only, not an importable installation or GPU proof.
    A partial destination after an I/O error is left for reconciliation.
    """
    destination = Path(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if set(sources) != {'modelopt.py', 'marlin.py', 'skinny_moe.py'}:
        raise ValueError('unexpected patch set')
    patches = {}
    for name, path in sources.items():
        with _open_regular(path) as f:
            data = f.read(2_000_001)
        if len(data) > 2_000_000 or _digest(data) != expected['patches'][name]:
            raise ValueError('patch mismatch: ' + name)
        ast.parse(data, filename=name)
        patches[name] = data
    originals = {}
    with _open_regular(wheel) as f:
        digest = hashlib.sha256()
        while chunk := f.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != expected['wheel']:
            raise ValueError('wheel mismatch')
        f.seek(0)
        with zipfile.ZipFile(f) as z:
            names = z.namelist()
            if any('skinny_moe' in name.lower() for name in names):
                raise ValueError('wheel already contains skinny module')
            for name, member in PATHS.items():
                if names.count(member) != 1:
                    raise ValueError('ambiguous or missing wheel member')
                item = z.getinfo(member)
                if item.file_size > 2_000_000:
                    raise ValueError('wheel member too large')
                data = z.read(item)
                if _digest(data) != expected['originals'][name]:
                    raise ValueError('wheel member mismatch: ' + name)
                originals[name] = data
    destination.mkdir(mode=0o700)
    (destination / 'originals').mkdir(mode=0o700)
    (destination / 'overlay').mkdir(mode=0o700)
    for name, data in originals.items():
        _write(destination / 'originals' / name, data)
    for name, data in patches.items():
        _write(destination / 'overlay' / name, data)
    result = {'status': 'PREPARED_NOT_INSTALLED', 'wheel_sha256': expected['wheel'],
              'originals': {k: _digest(v) for k, v in originals.items()},
              'patches': {k: _digest(v) for k, v in patches.items()},
              'note': 'Two donor patches and VoltaFabric adapter only; remaining bootstrap patches, dependencies and runtime are not admitted.'}
    _write(destination / 'manifest.json', (json.dumps(result, sort_keys=True, indent=2) + '\n').encode())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('wheel', 'modelopt', 'marlin', 'skinny', 'out'):
        parser.add_argument('--' + name, required=True, type=Path)
    a = parser.parse_args()
    print(json.dumps(prepare_overlay(a.wheel, {'modelopt.py': a.modelopt, 'marlin.py': a.marlin,
                                               'skinny_moe.py': a.skinny}, a.out, PINNED), sort_keys=True))


if __name__ == '__main__':
    main()
