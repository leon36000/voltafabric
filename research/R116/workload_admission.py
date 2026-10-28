"""Private R100 workload byte admission; see CONTRACT.md. No benchmark execution."""
import hashlib
import json
import os
import re
import stat

LIMIT = 256 * 1024
MAX_INT = 2**63 - 1


class Invalid(ValueError):
    """Malformed, inconsistent, unavailable, or unbound workload evidence."""


def _invalid(message):
    raise Invalid(message)


def _path_bytes(path):
    if (type(path) is not str or not path or len(path) > 4096 or
            "\x00" in path or not os.path.isabs(path)):
        _invalid("path must be a nonempty absolute string")
    # O_NONBLOCK prevents a path replaced by a FIFO from hanging before fstat.
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                _invalid("path is not a regular file")
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                raw = stream.read(LIMIT + 1)
        finally:
            if fd >= 0:
                os.close(fd)
    except Invalid:
        raise
    except (OSError, ValueError) as exc:
        raise Invalid("file cannot be read") from exc
    if len(raw) > LIMIT:
        _invalid("file exceeds byte limit")
    return raw, hashlib.sha256(raw).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _invalid("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_value):
    _invalid("nonfinite JSON constant")


def _document(raw, schema):
    try:
        value = json.loads(raw.decode("utf-8", errors="strict"),
                           object_pairs_hook=_unique_object,
                           parse_constant=_reject_constant)
    except Invalid:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise Invalid("invalid UTF-8 JSON") from exc
    if type(value) is not dict or set(value) != {"schema", "requests"}:
        _invalid("document fields are invalid")
    if type(value["schema"]) is not str or value["schema"] != schema:
        _invalid("document schema is invalid")
    if type(value["requests"]) is not list or not value["requests"]:
        _invalid("requests must be a nonempty list")
    return value["requests"]


def _identifier(value):
    if (type(value) is not str or not value or len(value) > 4096 or
            not any(not char.isspace() for char in value)):
        _invalid("request id is invalid")
    return value


def _corpus(raw):
    entries = _document(raw, "r100-prompt-corpus-v1")
    ids = []
    seen = set()
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"id", "prompt"}:
            _invalid("corpus request fields are invalid")
        ident = _identifier(entry["id"])
        prompt = entry["prompt"]
        if type(prompt) is not str or not prompt:
            _invalid("prompt is invalid")
        if ident in seen:
            _invalid("duplicate request id")
        seen.add(ident)
        ids.append(ident)
    return ids


def _counts(raw, maximum_per_request=None):
    entries = _document(raw, "r100-token-counts-v1")
    ids = []
    total = 0
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"id", "count"}:
            _invalid("count request fields are invalid")
        ident = _identifier(entry["id"])
        count = entry["count"]
        if type(count) is not int or not 1 <= count <= MAX_INT:
            _invalid("token count is invalid")
        if maximum_per_request is not None and count > maximum_per_request:
            _invalid("output token count exceeds declared maximum")
        total += count
        if total > MAX_INT:
            _invalid("token count sum exceeds limit")
        ids.append(ident)
    return ids, total


def admit_workload(corpus_path, input_counts_path, output_counts_path, expected):
    """Return the exact summary defined in CONTRACT.md, or raise Invalid."""
    corpus_raw, corpus_hash = _path_bytes(corpus_path)
    input_raw, input_hash = _path_bytes(input_counts_path)
    output_raw, output_hash = _path_bytes(output_counts_path)

    if type(expected) is not dict or set(expected) != {
            "prompt_corpus_sha256", "token_counts_sha256",
            "output_token_counts_sha256", "requests", "output_tokens",
            "max_new_tokens"}:
        _invalid("expected fields are invalid")
    for key in ("prompt_corpus_sha256", "token_counts_sha256",
                "output_token_counts_sha256"):
        digest = expected[key]
        if type(digest) is not str or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            _invalid("expected digest is invalid")
    for key in ("requests", "output_tokens", "max_new_tokens"):
        number = expected[key]
        if type(number) is not int or not 1 <= number <= MAX_INT:
            _invalid("expected number is invalid")
    if (expected["prompt_corpus_sha256"] != corpus_hash or
            expected["token_counts_sha256"] != input_hash or
            expected["output_token_counts_sha256"] != output_hash):
        _invalid("artifact hash mismatch")

    corpus_ids = _corpus(corpus_raw)
    input_ids, input_total = _counts(input_raw)
    output_ids, output_total = _counts(
        output_raw, maximum_per_request=expected["max_new_tokens"])
    if input_ids != corpus_ids or output_ids != corpus_ids:
        _invalid("request identities or order mismatch")
    if len(corpus_ids) != expected["requests"]:
        _invalid("request count mismatch")
    if output_total != expected["output_tokens"]:
        _invalid("output token count mismatch")

    return {
        "schema": "r100-workload-admission-v1",
        "requests": len(corpus_ids),
        "input_tokens": input_total,
        "output_tokens": output_total,
        "prompt_corpus_sha256": corpus_hash,
        "token_counts_sha256": input_hash,
        "output_token_counts_sha256": output_hash,
    }
