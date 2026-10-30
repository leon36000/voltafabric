"""Compare the R409 CPU diagnostic with NumPy; inputs are local observed bytes."""
import os
import sys
import numpy as np

if len(sys.argv) != 5:
    raise SystemExit('usage: oracle DENSE MIXED QKV RESULT')
dense, mixed, qkv_path, result = sys.argv[1:]
fd = os.open(dense, os.O_RDONLY | os.O_NOFOLLOW)
before = os.fstat(fd)
def read(offset, size):
    data = os.pread(fd, size, offset)
    if len(data) != size:
        raise ValueError('short dense read')
    return data
def f32(data):
    return np.frombuffer(data, dtype='<f4')
def bf16(offset):
    words = np.frombuffer(read(offset, 245760), dtype='<u2').astype(np.uint32)
    return (words << 16).view(np.float32).reshape(48, 2560)
a = f32(read(42234880, 192))
alpha, beta = bf16(42235072), bf16(42480832)
conv = f32(read(42726592, 163840)).reshape(10240, 4)
dt = f32(read(42890432, 192))
after = os.fstat(fd)
if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
    raise ValueError('dense changed during read')
x = np.fromfile(mixed, dtype='<f4')
qkv = np.fromfile(qkv_path, dtype='<f4')
r = np.fromfile(result, dtype='<f4')
if len(x) != 2560 or len(qkv) != 10240 or len(r) != 6240:
    raise ValueError('unexpected input shape')
av = (alpha.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
bv = (beta.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)
z = av + dt
gate = (np.where(z > 20, z, np.logaddexp(np.float32(0), z)) * a).astype(np.float32)
b = (1 / (1 + np.exp(-bv.astype(np.float64)))).astype(np.float32)
t = (qkv * conv[:, 3]).astype(np.float32)
h = (t / (1 + np.exp(-t.astype(np.float64)))).astype(np.float32)
q, k, v = h[:2048].reshape(16, 128), h[2048:4096].reshape(16, 128), h[4096:].reshape(48, 128)
q = (q / np.sqrt((q.astype(np.float64) ** 2).sum(axis=1, keepdims=True) + 1e-6)).astype(np.float32)
k = (k / np.sqrt((k.astype(np.float64) ** 2).sum(axis=1, keepdims=True) + 1e-6)).astype(np.float32)
out = np.empty((48, 128), dtype=np.float64)
for head in range(48):
    out[head] = np.dot(q[head % 16].astype(np.float64), k[head % 16].astype(np.float64)) * float(b[head]) * v[head].astype(np.float64) / np.sqrt(128)
for name, actual, expected, tolerance in [('output', r[:6144], out.ravel(), 1e-5), ('gate', r[6144:6192], gate, 1e-5), ('beta', r[6192:], b, 1e-6)]:
    error = np.abs(actual.astype(np.float64) - expected)
    print(name, 'max_abs', error.max(), 'count', len(error))
    if not np.isfinite(actual).all() or not np.isfinite(expected).all() or error.max() >= tolerance:
        raise ValueError(name + ' mismatch')
print('passed; zero state means decay gate does not affect first output')
