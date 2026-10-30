"""Pinned SC117 tokenizer pack and explicit streaming termination policy.

Hashes identify the caller-selected local files; callers must authenticate model
provenance and tokenizer semantics independently. This is not a chat template.
"""
from __future__ import annotations

import codecs
import hashlib
import json
import os
import pathlib
import stat

from .vendor.strata_tokenizer import Tokenizer

_LIMITS = {"vocab.json": 16_000_000, "merges.txt": 16_000_000,
           "token_type.json": 4_000_000, "tokenizer.json": 16_384}
MAX_PROMPT_TOKENS = 8
MAX_GENERATED_TOKENS = 32


def _identity(st):
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _load(root: pathlib.Path, name: str, digest: str) -> bytes:
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("invalid expected tokenizer digest")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(root / name, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= _LIMITS[name]:
            raise ValueError("invalid tokenizer file size or type")
        with os.fdopen(fd, "rb", closefd=False) as f:
            data = f.read(_LIMITS[name] + 1)
        if len(data) != before.st_size or _identity(before) != _identity(os.fstat(fd)):
            raise ValueError("tokenizer file changed during read")
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError("tokenizer digest mismatch")
        return data
    finally:
        os.close(fd)


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate JSON key")
        out[key] = value
    return out


def _json(raw):
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)


def load_sc117(directory, expected_hashes: dict[str, str]) -> tuple[Tokenizer, int]:
    """Open caller-pinned locally observed tokenizer files; return tokenizer and EOS ID."""
    if set(expected_hashes) != set(_LIMITS):
        raise ValueError("exactly four tokenizer digests required")
    root = pathlib.Path(directory)
    raw = {name: _load(root, name, expected_hashes[name]) for name in _LIMITS}
    cfg = _json(raw["tokenizer.json"])
    vocab = _json(raw["vocab.json"])
    types = _json(raw["token_type.json"])
    merges = raw["merges.txt"].decode("utf-8").splitlines()
    if (not isinstance(cfg, dict) or cfg.get("model") != "gpt2" or cfg.get("pre") != "qwen35"
            or cfg.get("vocab_size") != 248_320 or cfg.get("n_merges") != 247_587
            or cfg.get("add_bos_token") is not False):
        raise ValueError("unsupported tokenizer configuration")
    if not isinstance(vocab, dict) or len(vocab) != 248_320:
        raise ValueError("invalid tokenizer vocabulary")
    if not isinstance(types, list) or len(types) != len(vocab) or any(type(t) is not int for t in types):
        raise ValueError("invalid tokenizer token types")
    if len(merges) != 247_587 or len(set(merges)) != len(merges):
        raise ValueError("invalid or duplicate tokenizer merges")
    tokens = [None] * len(vocab)
    for token, i in vocab.items():
        if not isinstance(token, str) or type(i) is not int or not 0 <= i < len(tokens) or tokens[i] is not None:
            raise ValueError("invalid or duplicate tokenizer ID")
        tokens[i] = token
    if any(token is None for token in tokens):
        raise ValueError("incomplete tokenizer IDs")
    special = cfg.get("special_ids")
    if not isinstance(special, dict):
        raise ValueError("missing tokenizer special IDs")
    eos = special.get("tokenizer.ggml.eos_token_id")
    if type(eos) is not int or not 0 <= eos < len(tokens) or types[eos] != 3:
        raise ValueError("invalid EOS token")
    return Tokenizer(tokens, merges, types, pre="qwen35", special_ids=special), eos


class GreedyDecoder:
    """Strict UTF-8 incremental decoder; EOS stops, other control tokens refuse."""
    def __init__(self, tokenizer: Tokenizer, eos: int):
        if type(eos) is not int or not 0 <= eos < len(tokenizer.tokens):
            raise ValueError("invalid EOS token")
        self.tokenizer = tokenizer
        self.eos = eos
        self.stopped = False
        self._utf8 = codecs.getincrementaldecoder("utf-8")("strict")

    def push(self, token: int) -> tuple[str, bool]:
        if self.stopped:
            raise ValueError("decoder stopped")
        try:
            if type(token) is not int or not 0 <= token < len(self.tokenizer.tokens):
                raise ValueError("invalid token")
            if token == self.eos:
                text = self._utf8.decode(b"", final=True)
                self.stopped = True
                return text, True
            if self.tokenizer.token_types and self.tokenizer.token_types[token] not in (1, 4):
                raise ValueError("unexpected control or unsupported token type")
            return self._utf8.decode(self.tokenizer.token_bytes(token), final=False), False
        except (ValueError, UnicodeError, KeyError, IndexError):
            self.stopped = True
            raise

    def finish(self) -> str:
        if self.stopped:
            raise ValueError("decoder already stopped")
        self.stopped = True
        return self._utf8.decode(b"", final=True)


def decode_greedy_receipt(tokenizer: Tokenizer, eos: int, receipt: dict) -> dict:
    """Decode a bounded R459/R462-style receipt atomically; caller authenticates it."""
    if not isinstance(receipt, dict) or receipt.get("layer_count") != 48:
        raise ValueError("full layer receipt required")
    inputs, generated = receipt.get("input_tokens"), receipt.get("generated_token_ids")
    requested = receipt.get("requested_max_new_tokens", 2)
    if (type(requested) is not int or not 1 <= requested <= MAX_GENERATED_TOKENS
            or not isinstance(inputs, list) or not inputs or not isinstance(generated, list)
            or not 1 <= len(generated) <= requested):
        raise ValueError("invalid autoregressive receipt")
    prompt = receipt.get("prompt_tokens", inputs[:1])
    if (not isinstance(prompt, list) or not 1 <= len(prompt) <= MAX_PROMPT_TOKENS
            or inputs != prompt + generated[:-1] or len(inputs) > MAX_PROMPT_TOKENS + MAX_GENERATED_TOKENS - 1
            or any(type(token) is not int or not 0 <= token < len(tokenizer.tokens) for token in inputs)
            or not isinstance(receipt.get("run_id"), str) or not receipt["run_id"].strip()
            or receipt.get("workers_closed") != 96):
        raise ValueError("invalid autoregressive receipt")
    for key, count in (("output_sha256", len(inputs)), ("logits_sha256", len(generated))):
        hashes = receipt.get(key)
        if (not isinstance(hashes, list) or len(hashes) != count
                or any(not isinstance(h, str) or len(h) != 64
                       or any(c not in "0123456789abcdef" for c in h) for h in hashes)):
            raise ValueError("invalid receipt output hashes")
    decoder = GreedyDecoder(tokenizer, eos)
    chunks = []
    for i, token in enumerate(generated):
        piece, stopped = decoder.push(token)
        chunks.append(piece)
        if stopped and i != len(generated) - 1:
            raise ValueError("tokens supplied after EOS")
    reason = "eos" if decoder.stopped else "token_limit"
    if not decoder.stopped:
        if len(generated) != requested:
            raise ValueError("incomplete generation receipt")
        chunks.append(decoder.finish())
    return {"run_id": receipt["run_id"], "text": "".join(chunks),
            "token_ids": list(generated), "stop_reason": reason}
