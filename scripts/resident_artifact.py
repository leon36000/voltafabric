"""Strict CPU-only reader for the QPN2_RESIDENT_V1 Safetensors artifact.

The writer is a separate Rust CLI boundary. This module accepts only the six
compressed resident planes and bounded identity metadata; it never builds dense
weights, moves tensors to a GPU, or caches artifacts. The opened file must remain
quiescent while read, as required by the artifact producer contract.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import struct
from typing import Any

import torch

SCHEMA = "QPN2_RESIDENT_V1"
MAX_HEADER_BYTES = 8 * 1024 * 1024
MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
MAX_EXPERTS = 1024
MAX_DIM = 65536
# A compressed resident projection can be larger than the historical 1M-element
# bound. The loader owns compressed planes only; the complete six-plane payload
# remains capped at 128 MiB below. This admits DeepSeek's 11,796,480-element
# gate/up and down planes without opening unbounded metadata shapes or dense
# allocations.
MAX_PLANE_ELEMENTS = 1 << 24
_METADATA_KEYS = {"vf_schema", "vf_info"}
_INFO_KEYS = {
    "model_revision",
    "layer",
    "rank",
    "epoch",
    "payload_bytes",
    "physical_experts",
    "resident_order",
    "shape",
}
_TENSOR_KEYS = {"codes13", "scales13", "gscales13", "codes2", "scales2", "gscales2"}
_U8_KEYS = {"codes13", "scales13", "codes2", "scales2"}


@dataclass(frozen=True)
class ResidentArtifact:
    """Validated CPU tensors; frozen fields do not make tensor values immutable."""

    metadata: dict[str, Any]
    tensors: dict[str, torch.Tensor]


def _fail(message: str) -> ValueError:
    return ValueError("invalid QPN2 resident artifact: " + message)


def _is_uint(value: Any, bits: int) -> bool:
    return type(value) is int and 0 <= value < (1 << bits)


def _is_utf8_text(value: Any) -> bool:
    """A str with a UTF-8 form; a lone surrogate (JSON "\\ud800") has none."""
    if type(value) is not str:
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _pairs_without_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise _fail("duplicate JSON key")
        result[key] = value
    return result


def _decode_json(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_pairs_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise _fail("invalid Safetensors header JSON") from exc
    if type(value) is not dict:
        raise _fail("header must be an object")
    return value


def _validate_info(
    info: Any,
) -> tuple[dict[str, Any], dict[str, tuple[str, list[int], int]]]:
    if type(info) is not dict or set(info) != _INFO_KEYS:
        raise _fail("metadata fields differ")
    revision = info["model_revision"]
    if (
        not _is_utf8_text(revision)
        or not revision.strip()
        or len(revision.encode("utf-8")) > 256
    ):
        raise _fail("model_revision is invalid")
    for field in ("layer", "rank"):
        if not _is_uint(info[field], 32):
            raise _fail(field + " must be u32")
    for field in ("epoch", "payload_bytes"):
        if not _is_uint(info[field], 64):
            raise _fail(field + " must be u64")
    if info["payload_bytes"] > MAX_PAYLOAD_BYTES:
        raise _fail("payload exceeds 128 MiB")

    shape = info["shape"]
    if (
        type(shape) is not list
        or len(shape) != 5
        or any(not _is_uint(v, 64) for v in shape)
    ):
        raise _fail("shape must be five unsigned integers")
    e, n13, k13, n2, k2 = shape
    physical = info["physical_experts"]
    order = info["resident_order"]
    if (
        type(physical) is not list
        or type(order) is not list
        or len(physical) > MAX_EXPERTS
    ):
        raise _fail("expert identity arrays are invalid")
    if e == 0:
        if shape != [0, 0, 0, 0, 0] or physical or order or info["payload_bytes"] != 0:
            raise _fail("empty rank must use the all-zero sentinel")
        expected_shapes = {
            "codes13": ("U8", [0], 0),
            "scales13": ("U8", [0], 0),
            "gscales13": ("F32", [0, 0], 0),
            "codes2": ("U8", [0], 0),
            "scales2": ("U8", [0], 0),
            "gscales2": ("F32", [0, 0], 0),
        }
    else:
        if not 1 <= e <= MAX_EXPERTS or len(physical) != e or len(order) != e:
            raise _fail("expert count differs from physical identity")
        if any(not 0 < v <= MAX_DIM for v in (n13, k13, n2, k2)):
            raise _fail("nonempty dimensions outside bounds")
        if n13 % 64 or k13 % 64 or n2 % 32 or k2 % 64:
            raise _fail("dimensions are not QPN2 tile aligned")
        if n13 // 2 != k2 or n2 != k13:
            raise _fail("dimensions do not form a uniform SwiGLU")
        gate_elements = (n13 // 2) * k13
        up_elements = gate_elements
        down_elements = n2 * k2
        if any(
            elements > MAX_PLANE_ELEMENTS
            for elements in (gate_elements, up_elements, down_elements)
        ):
            raise _fail("QPN2 expert plane exceeds 16,777,216 elements")
        ids, locals_ = [], []
        for item in physical:
            if type(item) is not dict or set(item) != {"id", "local"}:
                raise _fail("physical expert entry fields differ")
            if not _is_uint(item["id"], 32) or not _is_uint(item["local"], 32):
                raise _fail("physical expert IDs must be u32")
            ids.append(item["id"])
            locals_.append(item["local"])
        if len(set(ids)) != e or len(set(locals_)) != e:
            raise _fail("physical expert IDs or locals are duplicated")
        if any(not _is_uint(local, 32) for local in order):
            raise _fail("resident_order entries must be u32")
        if len(set(order)) != e or order != locals_:
            raise _fail("resident_order must exactly match physical local order")
        c13, s13, g13 = e * n13 * k13 // 2, e * n13 * k13 // 16, e * (n13 // 32)
        c2, s2, g2 = e * n2 * k2 // 2, e * n2 * k2 // 16, e * (n2 // 32)
        expected_payload = c13 + s13 + 4 * g13 + c2 + s2 + 4 * g2
        if (
            expected_payload != info["payload_bytes"]
            or expected_payload > MAX_PAYLOAD_BYTES
        ):
            raise _fail("payload byte count differs from plane layout")
        expected_shapes = {
            "codes13": ("U8", [c13], c13),
            "scales13": ("U8", [s13], s13),
            "gscales13": ("F32", [e, n13 // 32], 4 * g13),
            "codes2": ("U8", [c2], c2),
            "scales2": ("U8", [s2], s2),
            "gscales2": ("F32", [e, n2 // 32], 4 * g2),
        }
    if type(info["payload_bytes"]) is not int:
        raise _fail("payload_bytes must be an integer")
    return info, expected_shapes


def _stable_stat(st: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        st.st_dev,
        st.st_ino,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
        st.st_mode,
    )


def _sha256_fd(fd: int, file_size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while offset < file_size:
        count = min(1024 * 1024, file_size - offset)
        chunk = os.pread(fd, count, offset)
        if len(chunk) != count:
            raise _fail("artifact changed or truncated while hashing")
        digest.update(chunk)
        offset += count
    return digest.hexdigest()


def _open_header(fd: int, file_size: int) -> tuple[dict[str, Any], dict[str, Any], int]:
    if file_size < 8 or file_size > MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES + 8:
        raise _fail("file size outside bounded range")
    prefix = os.pread(fd, 8, 0)
    if len(prefix) != 8:
        raise _fail("truncated header length")
    header_bytes = struct.unpack("<Q", prefix)[0]
    if not 0 < header_bytes <= MAX_HEADER_BYTES:
        raise _fail("header exceeds 8 MiB or is empty")
    data_start = 8 + header_bytes
    if data_start > file_size:
        raise _fail("truncated header")
    raw = os.pread(fd, header_bytes, 8)
    if len(raw) != header_bytes:
        raise _fail("truncated header body")
    header = _decode_json(raw)
    if set(header) != _TENSOR_KEYS | {"__metadata__"}:
        raise _fail("tensor names differ")
    metadata = header["__metadata__"]
    if type(metadata) is not dict or set(metadata) != _METADATA_KEYS:
        raise _fail("Safetensors metadata fields differ")
    if metadata["vf_schema"] != SCHEMA or type(metadata["vf_schema"]) is not str:
        raise _fail("unsupported schema")
    if type(metadata["vf_info"]) is not str:
        raise _fail("vf_info must be a JSON string")
    # surrogatepass keeps a lone surrogate as bytes that _decode_json then refuses.
    info = _decode_json(metadata["vf_info"].encode("utf-8", "surrogatepass"))
    info, expected = _validate_info(info)

    intervals = []
    for name in _TENSOR_KEYS:
        desc = header[name]
        if type(desc) is not dict or set(desc) != {"dtype", "shape", "data_offsets"}:
            raise _fail("tensor descriptor fields differ")
        dtype, shape, byte_count = expected[name]
        if desc["dtype"] != dtype or type(desc["dtype"]) is not str:
            raise _fail(name + " dtype differs")
        descriptor_shape = desc["shape"]
        if (
            type(descriptor_shape) is not list
            or any(not _is_uint(value, 64) for value in descriptor_shape)
            or descriptor_shape != shape
        ):
            raise _fail(name + " shape differs")
        offsets = desc["data_offsets"]
        if (
            type(offsets) is not list
            or len(offsets) != 2
            or any(not _is_uint(v, 64) for v in offsets)
        ):
            raise _fail(name + " offsets are invalid")
        start, end = offsets
        if end < start or end - start != byte_count:
            raise _fail(name + " byte span differs from shape")
        intervals.append((start, end, name))
    cursor = 0
    for start, end, _ in sorted(intervals):
        if start != cursor:
            raise _fail("tensor spans overlap or leave gaps")
        cursor = end
    if data_start + cursor != file_size:
        raise _fail("file has truncated or trailing payload bytes")
    return {"vf_schema": SCHEMA, "vf_info": metadata["vf_info"]}, info, data_start


def load_stack(
    path: str | os.PathLike[str],
    expected: dict[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> ResidentArtifact:
    """Load and validate one immutable-on-disk artifact into owned CPU tensors.

    `expected`, when provided, must be the complete vf_info object. Matching local
    metadata is an identity consistency check, not authenticity or attestation.
    `expected_sha256`, when provided, binds this load to the exact artifact bytes
    produced by the Rust session. The hash and tensor reads use the same open file
    descriptor; the existing before/after stat check still requires quiescence.
    No loaded tensor is cached; returned tensor contents remain mutable by callers.
    """
    if expected_sha256 is not None and (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(char not in "0123456789abcdef" for char in expected_sha256)
    ):
        raise _fail("expected SHA-256 must be 64 lowercase hexadecimal characters")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        fd = os.open(Path(path), flags)
    except OSError as exc:
        raise _fail(
            "cannot open regular artifact without following final symlink"
        ) from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise _fail("artifact is not a regular file")
        if (
            before.st_size < 8
            or before.st_size > MAX_PAYLOAD_BYTES + MAX_HEADER_BYTES + 8
        ):
            raise _fail("file size outside bounded range")
        if expected_sha256 is not None:
            if _sha256_fd(fd, before.st_size) != expected_sha256:
                raise _fail("artifact SHA-256 differs from expected bytes")
        custom_metadata, info, _ = _open_header(fd, before.st_size)
        if expected is not None:
            _validate_info(expected)
            if info != expected:
                raise _fail("artifact identity differs from expected metadata")
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise _fail("safetensors.safe_open is unavailable") from exc
        proc_path = Path("/proc/self/fd") / str(fd)
        tensors: dict[str, torch.Tensor] = {}
        try:
            with safe_open(str(proc_path), framework="pt", device="cpu") as file:
                metadata = file.metadata()
                if metadata != custom_metadata:
                    raise _fail("parsed metadata differs from Safetensors metadata")
                for name in (
                    "codes13",
                    "scales13",
                    "gscales13",
                    "codes2",
                    "scales2",
                    "gscales2",
                ):
                    tensor = file.get_tensor(name)
                    if tensor.device.type != "cpu":
                        raise _fail("tensor unexpectedly left CPU")
                    tensor = tensor.contiguous().clone()
                    tensors[name] = tensor
        except ValueError:
            raise
        except Exception as exc:
            raise _fail("Safetensors reader rejected file") from exc
        after = os.fstat(fd)
        if _stable_stat(before) != _stable_stat(after):
            raise _fail("source file changed while reading; quiescence required")
        _, expected_shapes = _validate_info(info)
        for name, tensor in tensors.items():
            dtype = torch.uint8 if name in _U8_KEYS else torch.float32
            _, shape, _ = expected_shapes[name]
            if (
                tensor.dtype != dtype
                or list(tensor.shape) != shape
                or not tensor.is_contiguous()
            ):
                raise _fail(name + " loaded tensor dtype/shape/layout differs")
        if info["shape"][0] > 0:
            from adapters.skinny_moe import _validate_sm70_scale_domain

            scales13, scales2 = tensors["scales13"], tensors["scales2"]
            if bool((scales13 >= 0x7F).any()) or bool((scales2 >= 0x7F).any()):
                raise _fail("scales must be unsigned finite E4M3")
            _validate_sm70_scale_domain(
                scales13, tensors["gscales13"], info["shape"][2]
            )
            _validate_sm70_scale_domain(scales2, tensors["gscales2"], info["shape"][4])
        return ResidentArtifact(metadata=deepcopy(info), tensors=tensors)
    finally:
        os.close(fd)
