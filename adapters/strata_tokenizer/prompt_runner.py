"""Bounded SC117 prompt runner, with no automatic retry.

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
import selectors
import stat
import subprocess
import sys

from .adapter import (MAX_GENERATED_TOKENS, MAX_INPUT_POSITIONS, MAX_PROMPT_TOKENS, _unique_object,
                      decode_greedy_receipt, load_sc117)

MAX_PROMPT_BYTES = 4096
MAX_RECEIPT_BYTES = 32 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
_ENGINE_FAILURES = {
    b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity\n": "rank_payload_capacity",
    b"prefill-greedy qualification refused: INVALID_INPUT: expert payload budget exceeded\n": "global_payload_capacity",
    b"prefill-greedy qualification refused: INVALID_INPUT: eviction target cannot be met safely\n": "eviction_target",
    b"prefill-greedy qualification refused: INVALID_INPUT: eviction limit\n": "eviction_limit",
    b"prefill-greedy qualification refused: worker shutdown receipt\n": "worker_shutdown",
}

def _run_bounded(command, *, stdout_limit=MAX_RECEIPT_BYTES, stderr_limit=MAX_STDERR_BYTES,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False):
    """Drain both child pipes, retaining only bounded prefixes; never retry or kill."""
    if stdout != subprocess.PIPE or stderr != subprocess.PIPE or check or stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("invalid bounded capture configuration")
    output = [bytearray(), bytearray()]
    truncated = [False, False]
    limits = (stdout_limit, stderr_limit)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ, 0)
            selector.register(child.stderr, selectors.EVENT_READ, 1)
            while selector.get_map():
                for key, _ in selector.select():
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    idx = key.data
                    available = limits[idx] - len(output[idx])
                    output[idx].extend(chunk[:available])
                    if len(chunk) > available:
                        truncated[idx] = True
        returncode = child.wait()
    result = subprocess.CompletedProcess(command, returncode, bytes(output[0]), bytes(output[1]))
    result.stdout_truncated, result.stderr_truncated = truncated
    return result


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


def _chat_prompt_ids(prompt, tokenizer, eos):
    """One text-only user message; template controls are distinct from user text."""
    text = prompt.strip()
    if not text:
        raise ValueError("empty chat message")
    names = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    ids = [tokenizer.ids.get(name) for name in names]
    if (any(type(i) is not int or not 0 <= i < len(tokenizer.tokens) for i in ids)
            or [tokenizer.token_types[i] for i in ids] != [3, 3, 4, 4]
            or ids[1] != eos):
        raise ValueError("unsupported chat control tokens")
    prefix = "<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    rendered = prefix + text + suffix
    result = tokenizer.encode(rendered, parse_special=True, plain=((len(prefix), len(prefix) + len(text)),))
    if any(type(i) is not int or not 0 <= i < len(tokenizer.tokens) for i in result):
        raise ValueError("invalid chat token")
    controls = [i for i in result if tokenizer.token_types[i] != 1]
    if controls != [ids[0], ids[1], ids[0], ids[2], ids[3]]:
        raise ValueError("unexpected chat control sequence")
    return result


def run_prompt(prompt: str, tokenizer, eos: int, binary: str, engine_paths: list[str], *, max_new_tokens: int = 2, chat: bool = False, runner=_run_bounded) -> dict:
    """Run once, and publish decoded text only after the full receipt validates."""
    if not isinstance(prompt, str) or not 0 < len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES:
        raise ValueError("prompt must be nonempty and bounded UTF-8")
    if len(engine_paths) != 8 or any(not pathlib.Path(p).is_absolute() for p in [binary, *engine_paths]):
        raise ValueError("absolute engine and eight input paths required")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_GENERATED_TOKENS:
        raise ValueError("generation limit must be between one and 32")
    if (type(eos) is not int or not 0 <= eos < len(tokenizer.tokens)
            or tokenizer.token_types[eos] != 3):
        raise ValueError("invalid tokenizer EOS token")
    if type(chat) is not bool:
        raise ValueError("invalid prompt mode")
    prompt_ids = _chat_prompt_ids(prompt, tokenizer, eos) if chat else tokenizer.encode(prompt, parse_special=False)
    if (not 1 <= len(prompt_ids) <= MAX_PROMPT_TOKENS or any(type(t) is not int or not 0 <= t < len(tokenizer.tokens)
            or (not chat and (t == eos or tokenizer.token_types[t] != 1)) for t in prompt_ids)):
        raise ValueError("invalid prompt tokens or token bound")
    if len(prompt_ids) + max_new_tokens - 1 > MAX_INPUT_POSITIONS:
        raise ValueError("prompt and generation exceed 39 input positions")
    command = [binary, *engine_paths, "48", ",".join(str(t) for t in prompt_ids)]
    command.extend((str(max_new_tokens), str(eos)))
    if chat:
        command.append("chat")
    completed = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        category = "unclassified" if getattr(completed, "stderr_truncated", False) else _ENGINE_FAILURES.get(completed.stderr, "unclassified")
        raise RuntimeError(f"engine failed ({category}, exit={completed.returncode}); reconcile its ledger and do not replay UNKNOWN claims")
    if (getattr(completed, "stdout_truncated", False) or getattr(completed, "stderr_truncated", False)
            or len(completed.stdout) > MAX_RECEIPT_BYTES or len(completed.stderr) > MAX_STDERR_BYTES):
        raise ValueError("engine response exceeds limit")
    receipt = json.loads(completed.stdout, object_pairs_hook=_unique_object)
    input_tokens = receipt.get("input_tokens") if isinstance(receipt, dict) else None
    if not isinstance(input_tokens, list) or input_tokens[:len(prompt_ids)] != prompt_ids:
        raise ValueError("engine receipt differs from prompt token")
    if receipt.get("requested_max_new_tokens", 2) != max_new_tokens:
        raise ValueError("engine receipt differs from generation limit")
    if type(receipt.get("eos_token_id")) is not int or receipt["eos_token_id"] != eos:
        raise ValueError("engine receipt differs from tokenizer EOS")
    if chat and receipt.get("prompt_mode") != "chat":
        raise ValueError("engine receipt differs from prompt mode")
    decoded = decode_greedy_receipt(tokenizer, eos, receipt)
    return {"prompt_token_ids": prompt_ids, "output": decoded, "receipt": receipt}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--tokenizer-manifest", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--chat", action="store_true", help="format one text-only user turn with the SC117 template")
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
    result = run_prompt(prompt, tokenizer, eos, args.binary, paths, max_new_tokens=args.max_new_tokens, chat=args.chat)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("prompt run refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
