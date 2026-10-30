"""Check local PLE gather against GGUF metadata, Python integer hashing and gguf-py.

The hash calculation is independent; IQ4_NL decoding uses llama.cpp gguf-py,
so it shares the documented quantization format and codebook with the Rust path.
"""
import hashlib
import json
import os
import sys
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

if len(sys.argv) < 4:
    raise SystemExit("usage: oracle SHARD1 SHARD2 RESULT [RESULT ...]")
shard1, shard2 = GGUFReader(sys.argv[1]), GGUFReader(sys.argv[2])
field = lambda name: shard1.fields["qwen4exp.ple." + name].contents()
mult, vocab, offsets, eos = field("layer_multipliers"), field("head_vocab_sizes"), field("head_offsets"), field("eos_token_id")
if field("layers") != [1] or field("ngram_size") != 3 or field("heads_per_ngram") != 8 or field("conv_kernel") != 4:
    raise ValueError("unexpected PLE geometry")
table = next(t for t in shard2.tensors if t.name == "per_layer_token_embd.weight")
if int(table.tensor_type) != 20 or table.shape.tolist() != [160, 320001536] or int(table.data_offset) != 192:
    raise ValueError("unexpected PLE table metadata")
fd = os.open(sys.argv[2], os.O_RDONLY | os.O_NOFOLLOW)
before = os.fstat(fd)
for path in sys.argv[3:]:
    observed = json.load(open(path, encoding="utf-8"))
    token, previous = observed["token"], observed["previous"]
    context = [token]; cut = False
    for prior in reversed(previous):
        cut = cut or prior < 0 or prior == eos
        context.append(eos if cut else prior)
    mask = (1 << 64) - 1
    bigram = ((context[0] * mult[0]) & mask) ^ ((context[1] * mult[1]) & mask)
    trigram = bigram ^ ((context[2] * mult[2]) & mask)
    expected_rows = [offsets[h] + (bigram if h < 8 else trigram) % vocab[h] for h in range(16)]
    if observed["rows"] != expected_rows:
        raise ValueError("row selection differs")
    decoded = []
    for row in expected_rows:
        raw = os.pread(fd, 90, 192 + row * 90)
        if len(raw) != 90:
            raise ValueError("short table row")
        decoded.extend(dequantize(np.frombuffer(raw, dtype=np.uint8).reshape(1, 90), GGMLQuantizationType.IQ4_NL).ravel())
    expected = np.asarray(decoded, dtype="<f4")
    actual = np.asarray(observed["values"], dtype="<f4")
    if actual.shape != (2560,) or not np.isfinite(actual).all() or not np.array_equal(actual, expected):
        raise ValueError("decoded PLE values differ")
    digest = hashlib.sha256(actual.tobytes()).hexdigest()
    if digest != observed["values_sha256"]:
        raise ValueError("result digest differs")
    print("token", token, "rows", len(expected_rows), "values", len(actual), "bit_equal", True, "sha256", digest)
after = os.fstat(fd)
if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
    raise ValueError("table source changed")
