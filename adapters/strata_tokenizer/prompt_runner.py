"""Bounded SC117 prompt runner, with no automatic retry.

Run under an externally admitted resource scope. The caller authenticates the
model, tokenizer manifest, worker, engine binary, paths and ledger. A failure
may leave consumed claims; inspect the ledger rather than replaying it.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pathlib
import selectors
import stat
import struct
import subprocess
import sys

from .adapter import (MAX_GENERATED_TOKENS, MAX_INPUT_POSITIONS, MAX_PROMPT_TOKENS, _unique_object,
                      decode_greedy_receipt, load_sc117)

MAX_PROMPT_BYTES = 4096
MAX_RECEIPT_BYTES = 128 * 1024 * 1024
MAX_STDERR_BYTES = 64 * 1024
TRACE_WORDS = 37380
TRACE_MAX_POSITIONS = 96


def _verify_qsa_trace(receipt: dict) -> None:
    trace = receipt.get("qsa_trace")
    inputs = receipt.get("input_tokens")
    is_v2 = isinstance(receipt.get("plan_id"), str) and receipt["plan_id"].startswith("r544-")
    version = "vf-qsa-trace-v2-layer3-input-f32le" if is_v2 else "vf-qsa-trace-v1-layer3-f32le"
    words = TRACE_WORDS + 10240 if is_v2 else TRACE_WORDS
    if (not isinstance(trace, dict) or set(trace) != {"version", "layer", "positions", "words_per_position", "sha256", "base64"}
            or trace["version"] != version or trace["layer"] != 3
            or type(trace["positions"]) is not int or not isinstance(inputs, list)
            or trace["positions"] != len(inputs) or not 1 <= len(inputs) <= TRACE_MAX_POSITIONS
            or trace["words_per_position"] != words or not isinstance(trace["base64"], str)
            or not isinstance(trace["sha256"], str)):
        raise ValueError("invalid QSA trace metadata")
    encoded = trace["base64"].encode("ascii")
    expected_bytes = len(inputs) * words * 4
    if len(encoded) != ((expected_bytes + 2) // 3) * 4:
        raise ValueError("invalid QSA trace size")
    raw = base64.b64decode(encoded, validate=True)
    if (len(raw) != expected_bytes or base64.b64encode(raw) != encoded
            or hashlib.sha256(raw).hexdigest() != trace["sha256"]):
        raise ValueError("invalid QSA trace digest or encoding")
_ENGINE_FAILURES = {
    b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity\n": "rank_payload_capacity",
    b"prefill-greedy qualification refused: INVALID_INPUT: expert payload budget exceeded\n": "global_payload_capacity",
    b"prefill-greedy qualification refused: INVALID_INPUT: eviction target cannot be met safely\n": "eviction_target",
    b"prefill-greedy qualification refused: INVALID_INPUT: eviction limit\n": "eviction_limit",
    b"prefill-greedy qualification refused: worker shutdown receipt\n": "worker_shutdown",
}

def _run_bounded(command, *, stdout_limit=MAX_RECEIPT_BYTES, stderr_limit=MAX_STDERR_BYTES,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, pass_fds=()):
    """Drain both child pipes, retaining only bounded prefixes; never retry or kill."""
    if stdout != subprocess.PIPE or stderr != subprocess.PIPE or check or stdout_limit < 0 or stderr_limit < 0:
        raise ValueError("invalid bounded capture configuration")
    output = [bytearray(), bytearray()]
    truncated = [False, False]
    limits = (stdout_limit, stderr_limit)
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, pass_fds=pass_fds) as child:
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


def _chat_messages(prompt):
    """Accept a bounded alternating text history, optionally starting with system."""
    messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    if not isinstance(messages, list) or not 1 <= len(messages) <= 8:
        raise ValueError("invalid chat history")
    offset = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    if len(messages) <= offset or (len(messages) - offset) % 2 != 1:
        raise ValueError("chat history must end with a user message")
    total = 0
    for i, message in enumerate(messages):
        expected = "system" if i < offset else ("user" if (i - offset) % 2 == 0 else "assistant")
        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                or message["role"] != expected or not isinstance(message["content"], str)
                or not message["content"].strip()):
            raise ValueError("invalid chat message")
        total += len(message["content"].encode("utf-8"))
        if total > MAX_PROMPT_BYTES:
            raise ValueError("chat history exceeds byte limit")
    return messages


def _chat_prompt_ids(prompt, tokenizer, eos):
    """Render a text-only history while keeping its content literal."""
    messages = _chat_messages(prompt)
    names = ("<|im_start|>", "<|im_end|>", "<think>", "</think>")
    ids = [tokenizer.ids.get(name) for name in names]
    if (any(type(i) is not int or not 0 <= i < len(tokenizer.tokens) for i in ids)
            or [tokenizer.token_types[i] for i in ids] != [3, 3, 4, 4]
            or ids[1] != eos):
        raise ValueError("unsupported chat control tokens")
    rendered, spans, expected = "", [], []
    for message in messages:
        role, text = message["role"], message["content"].strip()
        rendered += "<|im_start|>" + role + "\n"
        expected.append(ids[0])
        if role == "assistant":
            rendered += "<think>\n\n</think>\n\n"
            expected.extend(ids[2:])
        start = len(rendered)
        rendered += text
        spans.append((start, len(rendered)))
        rendered += "<|im_end|>\n"
        expected.append(ids[1])
    rendered += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    expected.extend((ids[0], ids[2], ids[3]))
    result = tokenizer.encode(rendered, parse_special=True, plain=spans)
    if any(type(i) is not int or not 0 <= i < len(tokenizer.tokens) for i in result):
        raise ValueError("invalid chat token")
    controls = [i for i in result if tokenizer.token_types[i] != 1]
    if controls != expected:
        raise ValueError("unexpected chat control sequence")
    return result


def run_prompt(prompt, tokenizer, eos: int, binary: str, engine_paths: list[str], *, max_new_tokens: int = 2, chat: bool = False, require_bound_plan: bool = True, prepare_plan: bool = False, runner=_run_bounded) -> dict:
    """Run once; strict mode requires a current plan and prompt digest, not attestation.

    Legacy receipt compatibility requires an explicit opt-out. A caller still
    authenticates the engine and plan inputs.
    """
    if type(chat) is not bool:
        raise ValueError("invalid prompt mode")
    if type(require_bound_plan) is not bool:
        raise ValueError("invalid bound-plan mode")
    if type(prepare_plan) is not bool:
        raise ValueError("invalid prepare-plan mode")
    if chat:
        _chat_messages(prompt)
    elif not isinstance(prompt, str) or not 0 < len(prompt.encode("utf-8")) <= MAX_PROMPT_BYTES:
        raise ValueError("prompt must be nonempty and bounded UTF-8")
    if len(engine_paths) != 8 or any(not pathlib.Path(p).is_absolute() for p in [binary, *engine_paths]):
        raise ValueError("absolute engine and eight input paths required")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= MAX_GENERATED_TOKENS:
        raise ValueError("generation limit must be between one and 128")
    if (type(eos) is not int or not 0 <= eos < len(tokenizer.tokens)
            or tokenizer.token_types[eos] != 3):
        raise ValueError("invalid tokenizer EOS token")
    prompt_ids = _chat_prompt_ids(prompt, tokenizer, eos) if chat else tokenizer.encode(prompt, parse_special=False)
    if (not 1 <= len(prompt_ids) <= MAX_PROMPT_TOKENS or any(type(t) is not int or not 0 <= t < len(tokenizer.tokens)
            or (not chat and (t == eos or tokenizer.token_types[t] != 1)) for t in prompt_ids)):
        raise ValueError("invalid prompt tokens or token bound")
    if len(prompt_ids) + max_new_tokens - 1 > MAX_INPUT_POSITIONS:
        raise ValueError("prompt and generation exceed 128 input positions")
    expected_plan = os.environ.get("VF_EXPECTED_PLAN_ID")
    if expected_plan is not None and (not expected_plan.startswith(("r512-", "r518-", "r520-", "r523-", "r525-", "r527-", "r528-", "r529-", "r531-", "r537-", "r544-")) or len(expected_plan) != 69
                                      or any(c not in "0123456789abcdef" for c in expected_plan[5:])):
        raise ValueError("invalid expected plan identity")
    command = [binary, *(["--plan-only"] if prepare_plan else []), *engine_paths, "48", ",".join(str(t) for t in prompt_ids)]
    command.extend((str(max_new_tokens), str(eos)))
    if chat:
        command.append("chat")
    # The coordinator must inherit the same open file description holding the
    # externally selected shared lease; it validates the identity and lock.
    options = {}
    if not prepare_plan and any(os.environ.get(f"VF_RANK{rank}_SSH_HOST") for rank in (2, 3)):
        raw_fd = os.environ.get("VF_SHARED_LEASE_FD", "")
        if not raw_fd.isascii() or not raw_fd.isdecimal() or int(raw_fd) < 3:
            raise ValueError("inherited shared lease fd required")
        fd = int(raw_fd)
        os.fstat(fd)
        options["pass_fds"] = (fd,)
    completed = runner(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, **options)
    if completed.returncode != 0:
        category = "unclassified" if getattr(completed, "stderr_truncated", False) else _ENGINE_FAILURES.get(completed.stderr, "unclassified")
        raise RuntimeError(f"engine failed ({category}, exit={completed.returncode}); reconcile its ledger and do not replay UNKNOWN claims")
    if (getattr(completed, "stdout_truncated", False) or getattr(completed, "stderr_truncated", False)
            or len(completed.stdout) > MAX_RECEIPT_BYTES or len(completed.stderr) > MAX_STDERR_BYTES):
        raise ValueError("engine response exceeds limit")
    if prepare_plan and len(completed.stdout) > 1024:
        raise ValueError("invalid prepared plan receipt")
    receipt = json.loads(completed.stdout, object_pairs_hook=_unique_object)
    if prepare_plan:
        if not isinstance(receipt, dict) or set(receipt) != {"plan_id", "prompt_tokens_sha256"}:
            raise ValueError("invalid prepared plan receipt")
        plan = receipt["plan_id"]
        if (not isinstance(plan, str) or not plan.startswith(("r512-", "r518-", "r520-", "r523-", "r525-", "r527-", "r528-", "r529-", "r531-", "r537-", "r544-")) or len(plan) != 69
                or any(c not in "0123456789abcdef" for c in plan[5:])):
            raise ValueError("current bound plan identity required")
        prompt_hash = hashlib.sha256(b"".join(struct.pack("<I", token) for token in prompt_ids)).hexdigest()
        if receipt["prompt_tokens_sha256"] != prompt_hash:
            raise ValueError("engine receipt differs from prompt digest")
        if expected_plan is not None and plan != expected_plan:
            raise ValueError("engine receipt differs from expected plan")
        return {"plan_id": plan, "prompt_tokens_sha256": prompt_hash, "prompt_token_ids": prompt_ids}
    input_tokens = receipt.get("input_tokens") if isinstance(receipt, dict) else None
    if not isinstance(input_tokens, list) or input_tokens[:len(prompt_ids)] != prompt_ids:
        raise ValueError("engine receipt differs from prompt token")
    if require_bound_plan and (receipt.get("prompt_tokens") != prompt_ids
                               or any(type(t) is not int for t in receipt.get("prompt_tokens", []))):
        raise ValueError("engine receipt differs from prompt tokens")
    plan_id = receipt.get("plan_id")
    if expected_plan is not None and plan_id != expected_plan:
        raise ValueError("engine receipt differs from expected plan")
    if require_bound_plan:
        if (not isinstance(plan_id, str) or not plan_id.startswith(("r512-", "r518-", "r520-", "r523-", "r525-", "r527-", "r528-", "r529-", "r531-", "r537-", "r544-")) or len(plan_id) != 69
                or any(c not in "0123456789abcdef" for c in plan_id[5:])):
            raise ValueError("current bound plan identity required")
    if require_bound_plan or "prompt_tokens_sha256" in receipt or (isinstance(plan_id, str) and plan_id.startswith(("r506-", "r508-", "r512-", "r518-", "r520-", "r523-", "r525-", "r527-", "r528-", "r529-", "r531-", "r537-", "r544-"))):
        prompt_hash = hashlib.sha256(b"".join(struct.pack("<I", token) for token in prompt_ids)).hexdigest()
        if receipt.get("prompt_tokens_sha256") != prompt_hash:
            raise ValueError("engine receipt differs from prompt digest")
    if receipt.get("requested_max_new_tokens", 2) != max_new_tokens:
        raise ValueError("engine receipt differs from generation limit")
    if type(receipt.get("eos_token_id")) is not int or receipt["eos_token_id"] != eos:
        raise ValueError("engine receipt differs from tokenizer EOS")
    if chat and receipt.get("prompt_mode") != "chat":
        raise ValueError("engine receipt differs from prompt mode")
    if isinstance(plan_id, str) and plan_id.startswith(("r537-", "r544-")):
        _verify_qsa_trace(receipt)
    elif "qsa_trace" in receipt:
        raise ValueError("QSA trace requires trace-bound plan")
    decoded = decode_greedy_receipt(tokenizer, eos, receipt)
    return {"prompt_token_ids": prompt_ids, "output": decoded, "receipt": receipt}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokenizer-dir", required=True)
    p.add_argument("--tokenizer-manifest", required=True)
    p.add_argument("--binary", required=True)
    p.add_argument("--binary-sha256", required=True)
    p.add_argument("--max-new-tokens", type=int, default=2)
    p.add_argument("--prepare-plan", action="store_true", help="prepare a plan without starting workers or generating text")
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--chat", action="store_true", help="format one text-only user turn")
    modes.add_argument("--messages-json", action="store_true", help="read a bounded text-only chat history as JSON")
    for name in ("dense", "shard", "pack", "index", "session", "ledger", "ple-shard", "ple-manifest"):
        p.add_argument("--" + name, required=True)
    args = p.parse_args()
    raw = sys.stdin.buffer.read(MAX_PROMPT_BYTES + 1)
    if len(raw) > MAX_PROMPT_BYTES:
        raise ValueError("prompt exceeds byte limit")
    prompt = raw.decode("utf-8", errors="strict")
    if args.messages_json:
        prompt = json.loads(prompt, object_pairs_hook=_unique_object)
    if _binary_sha(args.binary) != args.binary_sha256:
        raise ValueError("engine binary digest mismatch")
    manifest = json.loads(pathlib.Path(args.tokenizer_manifest).read_text(encoding="utf-8"))
    hashes = {k: manifest[k] for k in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}
    tokenizer, eos = load_sc117(args.tokenizer_dir, hashes)
    paths = [args.dense, args.shard, args.pack, args.index, args.session,
             args.ledger, args.ple_shard, args.ple_manifest]
    result = run_prompt(prompt, tokenizer, eos, args.binary, paths, max_new_tokens=args.max_new_tokens, chat=args.chat or args.messages_json, require_bound_plan=True, prepare_plan=args.prepare_plan)
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        print("prompt run refused: " + str(error), file=sys.stderr)
        raise SystemExit(1)
