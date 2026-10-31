"""Independent reconstruction of the sticky policy's plan domain."""
import hashlib

base = "r531-" + "a" * 64
actual = "r576-" + hashlib.sha256(b"vf-expert-rank-affinity-v1\0" + base.encode("ascii")).hexdigest()
expected = "r576-14d8a594dfddda0d9fb3f4ec30d5c53b0c5b1672eb6f88903ebfd05d5354f557"
assert actual == expected
assert actual != base
print(actual)
