"""Bounded SC117 one- or two-token prompt runner, with no automatic retry.

Run under an externally admitted resource scope. The caller authenticates the
model, tokenizer manifest, worker, engine binary, paths and ledger. A failure
may leave consumed claims; inspect the ledger rather than replaying it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import stat
import subprocess
import sys

from .adapter import _unique_object, decode_greedy_receipt, load_sc117

MAX_PROMPT_BYTES = 4096
MAX_RECEIPT_BYTES = 4 * 1024 * 1024


def _binary_sha(path: str) -> str:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 128 * 1024 * 1024:
            raise ValueError("invalid engine binary")
        with os.fdopen(fd, "rb", closefd=False) as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        after = os.fstat(fd)
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if fields(before) != fields(after):
            raise ValueError("engine binary changed during hash")
        return digest
    finally:
        os.close(fd)


def run_prompt(prompt: str, tokenizer, eos: int, binary: str, engine_paths: list[str], *, max_new_tokens: int = 2, runner=subprocess.run) -> dict:
    """Run once, and publish decoded text only after the full receipt validates."""
    if not isinstance(prompt, str) or not 0 < len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES:
        raise ValueError("prompt must be nonempty and bounded UTF-8")
    if len(engine_paths) != 8 or any(not pathlib.Path(p).is_absolute() for p in [binary, *engine_paths]):
        raise ValueError("absolute engine and eight input paths required")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 8:
        raise ValueError("generation limit must be between one and eight")
    prompt_ids = tokenizer.encode(prompt, parse_special=False)
    if (not 1 <= len(prompt_ids) <= 8 or any(type(t) is not int or not 0 <= t < len(tokenizer.tokens)
            or t == eos or tokenizer.token_types[t] != 1 for t in prompt_ids)):
        raise ValueError("this qualification accepts one to eight ordinary prompt tokens")
    command = [binary, *engine_paths, "48", ",".join(str(t) for t in prompt_ids)]
    if max_new_tokens != 2:
        command.append(str(max_new_tokens))
    completed = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError("engine failed; reconcile its ledger and do not replay UNKNOWN claims")
    if len(completed.stdout) > MAX_RECEIPT_BYTES or len(completed.stderr) > MAX_RECEIPT_BYTES:
        raise ValueError("engine response exceeds limit")
    receipt = json.loads(completed.stdout, object_pairs_hook=_unique_object)
    input_tokens = receipt.get("input_tokens") if isinstance(receipt, dict) else None
    if not isinstance(input_tokens, list) or input_tokens[:len(prompt_ids)] != prompt_ids:
        raise ValueError("engine receipt differs from prompt token")
    if receipt.get("requested_max_new_tokens", 2) != max_new_tokens:
        raise ValueError("engine receipt differs from generation limit")
    decoded = decode_greedy_receipt(tokenizer, eos, receipt)
    return {"prompt_token_ids": prompt_ids, "output": decoded, "receipt": receipt}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--tokenizer-manifest", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2)
    for name in ("dense", "shard", "pack", "index", "session", "ledger", "ple-shard", "ple-manifest"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    raw = sys.stdin.buffer.read(MAX_PROMPT_BYTES + 1)
    if len(raw) > MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds byte limit")
    prompt = raw.decode("utf-8", errors="strict")
    if _binary_sha(args.binary) != args.binary_sha256:
        raise ValueError("engine binary digest mismatch")
    manifest = json.loads(pathlib.Path(args.tokenizer_manifest).read_text(encoding="utf-8"))
    hashes = {k: manifest[k] for k in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}
    tokenizer, eos = load_sc117(args.tokenizer_dir, hashes)
    paths = [args.dense, args.shard, args.pack, args.index, args.session,
             args.ledger, args.ple_shard, args.ple_manifest]
    result = run_prompt(prompt, tokenizer, eos, args.binary, paths, max_new_tokens=args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("prompt run refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
