"""Independent NumPy RoPE and causal attention oracle for 13 standalone QSA prefixes.

Inputs to this check are observed projections, not a trace from the live Segment.
"""
import json
import os
import sys
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit('usage: oracle DENSE PREFIXES')
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)
before = os.fstat(fd)
def read(offset, count):
    data = os.pread(fd, count, offset)
    if len(data) != count:
        raise ValueError('short dense span')
    return np.frombuffer(data, dtype='<f4').astype(np.float64)
qgamma, kgamma = read(168118912, 1024), read(168117888, 1024)
values = np.fromfile(sys.argv[2], dtype='<f4')
if len(values) != 13 * 37380 or not np.isfinite(values).all():
    raise ValueError('prefix dimensions or nonfinite values')
prefixes = values.reshape(13, 37380)
keys, vals = [], []
maxima = {'query': 0.0, 'key': 0.0, 'attention': 0.0}
worst = {}
def norm_rope(raw, gamma, position):
    x = raw.astype(np.float64)
    norm = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-6) * gamma
    norm = norm.astype(np.float32).astype(np.float64)
    theta = position * np.power(10_000_000.0, -np.arange(32, dtype=np.float64) / 32.0)
    a, b = norm[:, :32].copy(), norm[:, 32:64].copy()
    norm[:, :32] = a*np.cos(theta) - b*np.sin(theta)
    norm[:, 32:64] = a*np.sin(theta) + b*np.cos(theta)
    return norm.astype(np.float32)
for position, p in enumerate(prefixes):
    qfull = p[2564:14852].reshape(24, 512)
    raw_q, gate = qfull[:, :256], qfull[:, 256:].astype(np.float64)
    if not np.array_equal(gate, p[20996:27140].reshape(24, 256)):
        raise ValueError(f'position {position}: gate layout mismatch')
    raw_k = p[27140:27652].reshape(2, 256)
    q = norm_rope(raw_q, qgamma, position)
    k = norm_rope(raw_k, kgamma, position)
    errors = {'query': float(np.max(np.abs(q.astype(np.float64)-p[14852:20996].reshape(24, 256)))),
              'key': float(np.max(np.abs(k.astype(np.float64)-p[27652:28164].reshape(2, 256))))}
    keys.append(k.astype(np.float16).astype(np.float64))
    vals.append(p[28164:28676].reshape(2, 256).astype(np.float16).astype(np.float64))
    keys_a, vals_a = np.asarray(keys), np.asarray(vals)
    expected = np.empty((24, 256), dtype=np.float64)
    for h in range(24):
        scores = keys_a[:, h//12, :] @ q[h].astype(np.float64) / 16.0
        weights = np.exp(scores - scores.max()); weights /= weights.sum()
        expected[h] = (weights @ vals_a[:, h//12, :]) / (1.0 + np.exp(-gate[h]))
    errors['attention'] = float(np.max(np.abs(expected-p[28676:34820].reshape(24, 256))))
    for name, error in errors.items():
        if not np.isfinite(error) or error > 1e-5:
            raise ValueError(f'position {position}: {name} error {error}')
        if error > maxima[name]:
            maxima[name], worst[name] = error, position
    if position in (0, 1, 12):
        print(json.dumps({'position':position, 'errors':errors}, sort_keys=True))
after = os.fstat(fd)
if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
    raise ValueError('dense source changed')
print(json.dumps({'status':'PASS', 'positions':13, 'maxima':maxima, 'worst_positions':worst}, sort_keys=True))
