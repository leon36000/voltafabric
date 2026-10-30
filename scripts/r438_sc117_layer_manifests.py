"""Generate local SC117 GDN, FFN and shared-stage manifests, not model provenance.

Uses the pinned Strata gguf-py and hashes bounded source spans. Writes only small
new JSON files; no checkpoint or pack data is copied.
"""
import hashlib
import json
import os
import pathlib
import sys
from gguf import GGUFReader

if len(sys.argv) != 5:
    raise SystemExit("usage: r438_sc117_layer_manifests DENSE INDEX SHARD NEW_OUTPUT_DIR")
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
            index[parts[0]] = int(parts[3]), int(parts[4])
tensors = {t.name: t for t in GGUFReader(shard_path).tensors}
def identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns
def span(f, offset, size):
    if offset < 0 or not 0 < size <= 32 * 1024 * 1024:
        raise ValueError("span out of bounds")
    f.seek(offset)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("short source")
    return {"offset": offset, "bytes": size, "sha256": hashlib.sha256(data).hexdigest()}
def save(name, value):
    b = (json.dumps(value, indent=2) + "\n").encode()
    with open(out / name, "xb") as f:
        f.write(b)
    return hashlib.sha256(b).hexdigest()

with open(dense_path, "rb") as dense, open(shard_path, "rb") as shard:
    db, sb = os.fstat(dense.fileno()), os.fstat(shard.fileno())
    if db.st_size != 1538035200 or sb.st_size != 55063448064:
        raise ValueError("unexpected SC117 source lengths")
    for layer in range(48):
        prefix = f"blk.{layer}."
        def ds(name, expected):
            offset, size = index[prefix + name]
            if size != expected:
                raise ValueError(f"unexpected dense size {layer} {name}")
            return span(dense, offset, size)
        def ss(name, types, shape):
            tensor = tensors[prefix + name]
            kind = int(tensor.tensor_type)
            if kind not in types or list(tensor.shape) != shape:
                raise ValueError(f"unexpected GGUF metadata {layer} {name}")
            return span(shard, int(tensor.data_offset), int(tensor.n_bytes)), kind
        common = {"model_revision": "sc117-local-observation", "layer": layer,
                  "dense_bytes": db.st_size, "shard_bytes": sb.st_size}
        shared = {"model_revision": common["model_revision"], "layer": layer, "shard_bytes": sb.st_size}
        shared["scalar"], _ = ss("ffn_gate_inp_shexp.weight", (30,), [2560])
        shared["gate"], shared["gate_type"] = ss("ffn_gate_shexp.weight", (12, 13, 14, 23), [2560, 640])
        shared["up"], shared["up_type"] = ss("ffn_up_shexp.weight", (12, 13, 14, 23), [2560, 640])
        shared["down"], _ = ss("ffn_down_shexp.weight", (8,), [640, 2560])
        ffn = {"model_revision": common["model_revision"], "layer": layer, "dense_bytes": db.st_size}
        for key, name, size in (("norm", "hc_ffn_norm.weight", 40960), ("down", "hc_ffn_down.weight", 6553600),
                                ("up", "hc_ffn_up.weight", 6553600), ("inject", "hc_ffn_inject.weight", 81920),
                                ("router", "ffn_gate_inp.weight", 2621440)):
            ffn[key] = ds(name, size)
        hashes = {"shared": save(f"layer{layer}-shared.json", shared)}
        if layer % 4 == 3:
            hashes["ffn"] = save(f"layer{layer}-ffn.json", ffn)
        else:
            gdn = dict(common)
            for key, name, size in (("norm", "hc_attn_norm.weight", 40960), ("down", "hc_attn_down.weight", 6553600),
                                    ("up", "hc_attn_up.weight", 6553600), ("inject", "hc_attn_inject.weight", 81920),
                                    ("a", "ssm_a", 192), ("alpha", "ssm_alpha.weight", 245760),
                                    ("beta", "ssm_beta.weight", 245760), ("conv", "ssm_conv1d.weight", 163840),
                                    ("dt", "ssm_dt.bias", 192)):
                gdn[key] = ds(name, size)
            gdn["qkv"], gdn["qkv_type"] = ss("attn_qkv.weight", (12, 13, 14, 23), [2560, 10240])
            post = dict(common)
            post.update({key: ffn[key] for key in ("norm", "down", "up", "inject", "router")})
            post["gamma"] = ds("ssm_norm.weight", 512)
            post["gate"], post["gate_type"] = ss("attn_gate.weight", (12, 13, 14, 23), [2560, 6144])
            post["out"], _ = ss("ssm_out.weight", (8,), [6144, 2560])
            hashes["gdn"] = save(f"layer{layer}-gdn.json", gdn)
            hashes["post"] = save(f"layer{layer}-post.json", post)
        if identity(db) != identity(os.fstat(dense.fileno())) or identity(sb) != identity(os.fstat(shard.fileno())):
            raise ValueError("source changed during hashing")
        print(json.dumps({"layer": layer, "manifests": hashes}), flush=True)
