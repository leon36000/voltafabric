"""Private CPU spike: bounded framing helpers for a future Session/owner seam.

These are stricter than the Session protocol and are not a production worker.
"""
import json
import math
import struct

MAX_HEADER = 64 * 1024
MAX_BODY = 16 * 1024 * 1024 - 64 * 1024
MAX_TOTAL = 16 * 1024 * 1024

class WireError(ValueError):
    pass

def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise WireError('duplicate JSON key')
        out[key] = value
    return out

def _constant(_):
    raise WireError('nonfinite JSON number')

def read_header(stream):
    line = stream.readline(MAX_HEADER + 1)
    if not line or len(line) > MAX_HEADER or not line.endswith(b'\n'):
        raise WireError('missing, unterminated or oversized header')
    try:
        obj = json.loads(line, object_pairs_hook=_unique, parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise WireError('invalid JSON header') from exc
    if type(obj) is not dict:
        raise WireError('header must be an object')
    return obj

def read_exact(stream, size):
    if type(size) is not int or not 0 <= size <= MAX_BODY:
        raise WireError('body length outside bounds')
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk or len(chunk) > remaining:
            raise WireError('truncated or invalid body')
        chunks.append(chunk)
        remaining -= len(chunk)
    return b''.join(chunks)

def decode_fp16_inputs(header, body):
    if type(body) is not bytes or type(header) is not dict:
        raise WireError('invalid input representation')
    declared = header.get('activation_bytes')
    rows = header.get('inputs')
    if type(declared) is not int or not 0 <= declared <= MAX_BODY or declared != len(body) or type(rows) is not list:
        raise WireError('invalid input length or rows')
    tokens, values, offset = [], [], 0
    for row in rows:
        if type(row) is not dict or set(row) != {'token', 'values'}:
            raise WireError('invalid input row metadata')
        token, n = row['token'], row['values']
        if type(token) is not int or not 0 <= token < 2**64 or type(n) is not int or not 0 < n <= 65536:
            raise WireError('invalid token or row length')
        if tokens and token <= tokens[-1]:
            raise WireError('tokens must be distinct and ordered')
        end = offset + n * 2
        if end > len(body):
            raise WireError('row exceeds body')
        row_values = list(struct.unpack_from(f'<{n}e', body, offset))
        if any(not math.isfinite(value) for value in row_values):
            raise WireError('nonfinite FP16 input')
        tokens.append(token); values.append(row_values); offset = end
    if offset != len(body):
        raise WireError('row lengths do not consume body')
    return tokens, values

def write_frame(stream, header, body=b''):
    if type(header) is not dict or type(body) is not bytes or len(body) > MAX_BODY:
        raise WireError('invalid reply frame')
    try:
        line = json.dumps(header, separators=(',', ':'), allow_nan=False).encode('utf-8') + b'\n'
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise WireError('invalid reply JSON') from exc
    if len(line) > MAX_HEADER or len(line) + len(body) > MAX_TOTAL:
        raise WireError('reply exceeds bounds')
    data = memoryview(line + body)
    while data:
        written = stream.write(data)
        if type(written) is not int or written <= 0 or written > len(data):
            raise WireError('short or invalid frame write')
        data = data[written:]
    stream.flush()
