"""Check every SC117 GDN output-normalization element independently with NumPy."""
import os
import sys
import numpy as np
if len(sys.argv) != 4:
    raise SystemExit('usage: oracle DENSE R409_RESULT R410_RESULT')
fd = os.open(sys.argv[1], os.O_RDONLY | os.O_NOFOLLOW)
a = os.fstat(fd)
gamma = np.frombuffer(os.pread(fd, 512, 42890624), dtype='<f4')
b = os.fstat(fd)
if len(gamma) != 128 or (a.st_ino, a.st_size, a.st_mtime_ns, a.st_ctime_ns) != (b.st_ino, b.st_size, b.st_mtime_ns, b.st_ctime_ns):
    raise ValueError('dense span unavailable or changed')
r = np.fromfile(sys.argv[2], dtype='<f4')
s = np.fromfile(sys.argv[3], dtype='<f4')
if len(r) != 6240 or len(s) != 14848:
    raise ValueError('wrong diagnostic shapes')
o = r[:6144].reshape(48, 128).astype(np.float64)
z = s[:6144].reshape(48, 128).astype(np.float64)
y = o / np.sqrt(np.mean(o * o, axis=1, keepdims=True) + 1e-6) * gamma.astype(np.float64) / (1 + np.exp(-z))
error = np.abs(y.ravel() - s[6144:12288])
print('output_norm max_abs', error.max(), 'count', error.size)
if not np.isfinite(y).all() or not np.isfinite(s).all() or error.max() > 1e-5:
    raise ValueError('nonfinite or mismatch')
