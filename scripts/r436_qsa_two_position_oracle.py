"""Independent NumPy check of SC117 position-0/1 QSA RoPE and FP16 KV attention.

Consumes two standalone prefix observations, not outputs captured from the live
Segment. Projection inputs are observed; this does not check projection parity.
"""
import os
import sys
import numpy as np

if len(sys.argv) != 3:
    raise SystemExit("usage: oracle DENSE PREFIXES")
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)
before = os.fstat(fd)
def read(offset, count):
    data = os.pread(fd, count, offset)
    if len(data) != count:
        raise ValueError("short dense span")
    return np.frombuffer(data, dtype="<f4").astype(np.float64)
qgamma, kgamma = read(168118912, 1024), read(168117888, 1024)
values = np.fromfile(sys.argv[2], dtype="<f4")
if len(values) != 2 * 37380 or not np.isfinite(values).all():
    raise ValueError("prefix dimensions or finite values")
prefixes = values.reshape(2, 37380)
cache_keys, cache_values = [], []
def norm_rope(raw, gamma, position):
    x = raw.astype(np.float64)
    norm = x / np.sqrt(np.mean(x*x, axis=1, keepdims=True) + 1e-6) * gamma
    norm = norm.astype(np.float32).astype(np.float64)
    theta = position * np.power(10_000_000.0, -np.arange(32, dtype=np.float64) / 32.0)
    a, b = norm[:, :32].copy(), norm[:, 32:64].copy()
    norm[:, :32] = a * np.cos(theta) - b * np.sin(theta)
    norm[:, 32:64] = a * np.sin(theta) + b * np.cos(theta)
    return norm.astype(np.float32)

for position, p in enumerate(prefixes):
    qfull = p[2564:14852].reshape(24, 512)
    raw_q, gate = qfull[:, :256], qfull[:, 256:]
    raw_k = p[27140:27652].reshape(2, 256)
    q = norm_rope(raw_q, qgamma, position)
    k = norm_rope(raw_k, kgamma, position)
    errors = {"query": float(np.max(np.abs(q.astype(np.float64) - p[14852:20996].reshape(24, 256)))),
              "key": float(np.max(np.abs(k.astype(np.float64) - p[27652:28164].reshape(2, 256))))}
    cache_keys.append(k.astype(np.float16).astype(np.float64))
    cache_values.append(p[28164:28676].reshape(2, 256).astype(np.float16).astype(np.float64))
    expected = np.empty((24, 256), dtype=np.float64)
    for head in range(24):
        kh = head // 12
        scores = np.array([np.dot(q[head].astype(np.float64), k0[kh]) / 16.0 for k0 in cache_keys])
        weights = np.exp(scores - scores.max()); weights /= weights.sum()
        combined = sum(w * v[kh] for w, v in zip(weights, cache_values))
        expected[head] = combined / (1.0 + np.exp(-gate[head].astype(np.float64)))
    errors["attention"] = float(np.max(np.abs(expected - p[28676:34820].reshape(24, 256))))
    print("position", position, errors)
    if max(errors.values()) > 1e-5 or not all(np.isfinite(v) for v in errors.values()):
        raise ValueError("QSA two-position oracle mismatch")
after = os.fstat(fd)
if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
    raise ValueError("dense source changed")
