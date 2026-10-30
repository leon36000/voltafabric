"""Generate bounded local-observation SC117 QSA manifests using GGUF metadata.

Requires the pinned Strata gguf-py; observed hashes do not authenticate the model.
Writes one new small manifest per QSA layer, never checkpoint data.
"""
import hashlib
import json
import os
import pathlib
import sys
from gguf import GGUFReader

if len(sys.argv) != 5:
    raise SystemExit("usage: r437_qsa_manifests DENSE INDEX SHARD NEW_OUTPUT_DIR")
dense_path, index_path, shard_path, output_path = sys.argv[1:]
out = pathlib.Path(output_path)
out.mkdir(exist_ok=False)
index = {}
with open(index_path, encoding="utf-8") as f:
    for line in f:
        parts = line.split()
        if parts and not parts[0].startswith("#"):
            if parts[0] in index:
                raise ValueError("duplicate dense tensor")
            index[parts[0]] = (int(parts[3]), int(parts[4]))
tensors = {t.name: t for t in GGUFReader(shard_path).tensors}
names = {"norm": "hc_attn_norm.weight", "down": "hc_attn_down.weight", "up": "hc_attn_up.weight",
         "inject": "hc_attn_inject.weight", "qnorm": "attn_q_norm.weight", "knorm": "attn_k_norm.weight"}
def span(f, offset, size):
    if offset < 0 or not 0 < size <= 32 * 1024 * 1024:
        raise ValueError("span out of bounds")
    f.seek(offset)
    b = f.read(size)
    if len(b) != size:
        raise ValueError("short source")
    return {"offset": offset, "bytes": size, "sha256": hashlib.sha256(b).hexdigest()}
def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns

with open(dense_path, "rb") as dense, open(shard_path, "rb") as shard:
    db, sb = os.fstat(dense.fileno()), os.fstat(shard.fileno())
    if db.st_size != 1538035200 or sb.st_size != 55063448064:
        raise ValueError("unexpected SC117 source lengths")
    for layer in range(3, 48, 4):
        m = {"model_revision": "sc117-local-observation", "layer": layer,
             "dense_bytes": db.st_size, "shard_bytes": sb.st_size}
        for key, name in names.items():
            m[key] = span(dense, *index[f"blk.{layer}.{name}"])
        q = tensors[f"blk.{layer}.attn_q.weight"]
        if list(q.shape) != [2560, 12288] or int(q.tensor_type) not in (12, 13, 14, 23) or int(q.n_bytes) % 2:
            raise ValueError("unexpected query metadata")
        half = int(q.n_bytes) // 2
        m["q_type"] = int(q.tensor_type)
        m["q0"] = span(shard, int(q.data_offset), half)
        m["q1"] = span(shard, int(q.data_offset) + half, half)
        for key in ("k", "v", "output"):
            tensor = tensors[f"blk.{layer}.attn_{key}.weight"]
            kind = int(tensor.tensor_type)
            expected_shape = [6144, 2560] if key == "output" else [2560, 512]
            if list(tensor.shape) != expected_shape or kind not in ((8,) if key == "output" else (12, 13, 14, 23)):
                raise ValueError("unexpected projection metadata")
            m["out" if key == "output" else key] = span(shard, int(tensor.data_offset), int(tensor.n_bytes))
            if key != "output":
                m[key + "_type"] = kind
        if identity(db) != identity(os.fstat(dense.fileno())) or identity(sb) != identity(os.fstat(shard.fileno())):
            raise ValueError("source changed while hashing")
        payload = (json.dumps(m, indent=2) + "\n").encode()
        with open(out / f"layer{layer}.json", "xb") as f:
            f.write(payload)
        print(layer, m["q_type"], m["k_type"], m["v_type"], hashlib.sha256(payload).hexdigest())
