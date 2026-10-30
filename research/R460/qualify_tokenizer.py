#!/usr/bin/env python3
"""Compare observed tokenizer pack with GGUF metadata; no model execution."""
import hashlib
import json
import pathlib
import sys

from adapters.strata_tokenizer.adapter import GreedyDecoder, load_sc117, decode_greedy_receipt
from adapters.strata_tokenizer.vendor.gguf_reader import GGUFFile

pack, shard, manifest_path, result_path = map(pathlib.Path, sys.argv[1:])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
hashes = {k: manifest[k] for k in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}
identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
before = shard.stat()
if before.st_size != 55_063_448_064:
    raise ValueError("unexpected source size")
tokenizer, eos = load_sc117(pack, hashes)
metadata = GGUFFile(shard).metadata
if identity(before) != identity(shard.stat()):
    raise ValueError("checkpoint changed during metadata read")
if (metadata.get("tokenizer.ggml.model") != "gpt2"
        or metadata.get("tokenizer.ggml.pre") != "qwen35"
        or metadata.get("tokenizer.ggml.tokens") != tokenizer.tokens
        or metadata.get("tokenizer.ggml.token_type") != tokenizer.token_types
        or metadata.get("tokenizer.ggml.eos_token_id") != eos):
    raise ValueError("tokenizer pack differs from checkpoint metadata")
special_ids = {key: int(value) for key, value in metadata.items()
               if key.startswith("tokenizer.ggml.") and key.endswith("_token_id")}
if tokenizer.special_ids != special_ids or metadata.get("tokenizer.ggml.add_bos_token", False) is not False:
    raise ValueError("tokenizer special IDs or BOS policy differ from checkpoint")
merges = [" ".join(pair) for pair in tokenizer.ranks]
if metadata.get("tokenizer.ggml.merges") != merges:
    raise ValueError("tokenizer merge order differs from checkpoint")
run = json.loads(result_path.read_text(encoding="utf-8"))
if run["layer_count"] != 48 or len(run["generated_token_ids"]) != 2 or run["input_tokens"][1] != run["generated_token_ids"][0]:
    raise ValueError("not a two-step 48-layer autoregressive receipt")
ids = [run["input_tokens"][0], *run["generated_token_ids"]]
decoded_receipt = decode_greedy_receipt(tokenizer, eos, run)
decoder = GreedyDecoder(tokenizer, eos)
chunks = [decoder.push(i)[0] for i in ids]
chunks.append(decoder.finish())
text = "".join(chunks)
# Independent byte-alphabet reconstruction for the specific observed pieces.
visible = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
alphabet = {chr(b): b for b in visible}
for offset, b in enumerate(x for x in range(256) if x not in visible):
    alphabet[chr(256 + offset)] = b
raw_reference = bytes(alphabet[ch] for i in ids for ch in tokenizer.tokens[i])
if raw_reference.decode("utf-8", errors="strict") != text:
    raise ValueError("independent byte-alphabet reconstruction differs")
if tokenizer.encode(text) != ids or tokenizer.decode(ids, errors="strict") != text:
    raise ValueError("observed token sequence did not round-trip")
stop = GreedyDecoder(tokenizer, eos)
if stop.push(eos) != ("", True):
    raise ValueError("EOS policy failed")
examples = ["Hello, world!", "été 😀", "a\n\n b", "e\u0301 café", "中文 42"]
for value in examples:
    if tokenizer.decode(tokenizer.encode(value), errors="strict") != value:
        raise ValueError("UTF-8 round-trip failed")
print(json.dumps({"scope":"local observed metadata; not publisher authentication", "ids":ids,
                  "input_token":ids[0],"generated_ids":ids[1:],"decoded_all":text,
                  "decoded_generated":tokenizer.decode(ids[1:],errors="strict"),
                  "receipt_text":decoded_receipt["text"],"stop_reason":decoded_receipt["stop_reason"],
                  "eos":eos,"vocabulary_count":len(tokenizer.tokens),"merge_count":len(merges),
                  "metadata_exact":True,"independent_byte_decode":True,"roundtrip_examples":len(examples),
                  "source_identity_sha256":hashlib.sha256(repr(identity(before)).encode()).hexdigest()},
                 sort_keys=True,ensure_ascii=False))
