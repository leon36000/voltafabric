#!/usr/bin/env python3
"""Fetch six pinned FP4/UE8M0 ranges per selected backbone layer-0 expert.

The default selection is expert 383 (A106 paths and existing receipt fields retained).
A120 adds an allowlist for experts 96 and 192 from the same pinned shard, with
at most two experts and 37,601,280 payload bytes per invocation. Payload bytes
are preserved verbatim; nothing is decoded.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import urllib.error
import urllib.parse
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
SHARD = "model-00003-of-00048.safetensors"
SHARD_BYTES = 7_389_759_032
HEADER_PATH = (
    ROOT / "evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
)
HEADER_SHA256 = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449"
HEADER_BUDGET_BYTES = 16 * 1024 * 1024
INDEX_PATH = (
    ROOT
    / "evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
)
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
INDEX_BUDGET_BYTES = 16 * 1024 * 1024
EXPERT_ID = 383
LEGACY_SELECTION = (EXPERT_ID,)
ROUTED_EXPERT_COUNT = 384
SUPPORTED_EXPERT_IDS = frozenset({96, 192, EXPERT_ID})
MAX_EXPERTS_PER_INVOCATION = 2
TENSOR_ORDER = (
    ("w1.weight", "I8", [2304, 2560]),
    ("w1.scale", "F8_E8M0", [2304, 160]),
    ("w2.weight", "I8", [5120, 1152]),
    ("w2.scale", "F8_E8M0", [5120, 72]),
    ("w3.weight", "I8", [2304, 2560]),
    ("w3.scale", "F8_E8M0", [2304, 160]),
)
EXPECTED_PAYLOAD_BYTES = 18_800_640
PAYLOAD_CAP_BYTES = 37_601_280
MAX_HF_REDIRECTS = 3
# Hugging Face documents download redirects to storage/CDN hosts under these
# two official domains. Keep the suffix check label-boundary aware.
# https://huggingface.co/docs/hub/models-downloading#downloading-behind-a-proxy-or-firewall
HF_REDIRECT_ROOTS = ("huggingface.co", "hf.co")
OUTPUT = ROOT / "fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors"
EVIDENCE = ROOT / "evidence/continuation_a106_real_second_expert"
RECEIPT = EVIDENCE / "A106_EXPERT383_RANGE_RECEIPT.json"
WORK = ROOT / ".work/a106-expert383-range-payload"
# Root for non-default (A120) selections; tests point it at a scratch dir.
ARTIFACT_ROOT = ROOT
A120_EVIDENCE_DIR = "evidence/continuation_a120_selected_experts"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RangeReadError(RuntimeError):
    """A bounded range response failed validation after a known amount was read."""

    def __init__(
        self,
        message: str,
        *,
        status: int | None,
        content_range: str,
        body_bytes_read: int,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.content_range = content_range
        self.body_bytes_read = body_bytes_read


def trusted_hf_https_url(url: str) -> tuple[str, int]:
    """Validate an HTTPS URL within Hugging Face's documented service domains."""
    parsed = urllib.parse.urlsplit(url)
    try:
        port = parsed.port
    except ValueError:
        raise ValueError("invalid redirect port") from None
    host = parsed.hostname
    if (
        parsed.scheme.lower() != "https"
        or not host
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
        or port not in (None, 443)
        or not any(
            host == root or host.endswith(f".{root}") for root in HF_REDIRECT_ROOTS
        )
    ):
        raise ValueError(
            "redirect target is outside the trusted HTTPS Hugging Face domains"
        )
    return host, 443


class PinnedHfRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Follow only HTTPS redirects to Hugging Face's documented service domains."""

    def __init__(self) -> None:
        super().__init__()
        self.max_redirections = MAX_HF_REDIRECTS
        self.redirects: list[dict[str, object]] = []

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        target = urllib.parse.urljoin(req.full_url, newurl)
        try:
            host, port = trusted_hf_https_url(target)
        except ValueError as error:
            raise urllib.error.HTTPError(
                target, code, str(error), headers, fp
            ) from None
        if len(self.redirects) >= self.max_redirections:
            raise urllib.error.HTTPError(
                target, code, "too many Hugging Face redirects", headers, fp
            )
        redirected = super().redirect_request(req, fp, code, msg, headers, target)
        if redirected is not None:
            # Record only scheme/host/port and status; signed query strings are
            # intentionally excluded from evidence.
            self.redirects.append(
                {"status": code, "scheme": "https", "host": host, "port": port}
            )
        return redirected


def open_range_request(
    request: urllib.request.Request, timeout: float = 30
) -> tuple[object, list[dict[str, object]]]:
    """Open one range request with per-call redirect restrictions and evidence."""
    try:
        trusted_hf_https_url(request.full_url)
    except ValueError as error:
        raise urllib.error.URLError(str(error)) from None
    redirect_handler = PinnedHfRedirectHandler()
    opener = urllib.request.build_opener(redirect_handler)
    response = opener.open(request, timeout=timeout)
    try:
        trusted_hf_https_url(response.geturl())
    except (AttributeError, ValueError) as error:
        response.close()
        raise urllib.error.URLError(f"untrusted final response URL: {error}") from None
    return response, redirect_handler.redirects


def load_pinned_header() -> tuple[bytes, dict[str, object]]:
    header_bytes = HEADER_PATH.read_bytes()
    if sha256(header_bytes) != HEADER_SHA256:
        raise ValueError("pinned layer-zero header SHA-256 mismatch")
    if len(header_bytes) > HEADER_BUDGET_BYTES:
        raise ValueError("pinned expert header exceeds the configured header budget")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError("pinned header is not a JSON object")
    return header_bytes, header


def load_pinned_weight_map() -> dict[str, object]:
    if INDEX_PATH.stat().st_size > INDEX_BUDGET_BYTES:
        raise ValueError("pinned index exceeds the configured index budget")
    index_bytes = INDEX_PATH.read_bytes()
    if sha256(index_bytes) != INDEX_SHA256:
        raise ValueError("pinned checkpoint index SHA-256 mismatch")
    index = json.loads(index_bytes)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict):
        raise ValueError("pinned checkpoint index has no weight_map")
    return weight_map


def validate_expert_ids(expert_ids: object) -> tuple[int, ...]:
    """Return the canonical (sorted) selection or raise ValueError."""
    if expert_ids is None:
        return LEGACY_SELECTION
    if isinstance(expert_ids, (str, bytes)) or not hasattr(expert_ids, "__iter__"):
        raise ValueError("expert selection must be a sequence of integer IDs")
    ids = list(expert_ids)
    if not ids:
        raise ValueError("expert selection is empty")
    if len(ids) > MAX_EXPERTS_PER_INVOCATION:
        raise ValueError("too many experts requested for one invocation")
    for value in ids:
        if type(value) is not int:
            raise ValueError(f"expert ID {value!r} is not an integer")
        if not 0 <= value < ROUTED_EXPERT_COUNT:
            raise ValueError(f"expert ID {value} is outside the routed expert domain")
        if value not in SUPPORTED_EXPERT_IDS:
            raise ValueError(f"expert ID {value} is not in the pinned allowlist")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate expert ID in selection")
    if len(ids) * EXPECTED_PAYLOAD_BYTES > PAYLOAD_CAP_BYTES:
        raise ValueError("expert selection exceeds the payload byte cap")
    return tuple(sorted(ids))


def expert_ranges(
    expert_id: int,
    header_bytes: bytes,
    header: dict[str, object],
    weight_map: dict[str, object],
) -> list[dict[str, object]]:
    data_start = 8 + len(header_bytes)
    payload_boundary = SHARD_BYTES - data_start
    prefix = f"layers.0.ffn.experts.{expert_id}."
    expected_names = {prefix + suffix for suffix, _, _ in TENSOR_ORDER}
    if {name for name in header if name.startswith(prefix)} != expected_names:
        raise ValueError(
            f"pinned header tensors for expert {expert_id} are out of domain"
        )
    if {name for name in weight_map if name.startswith(prefix)} != expected_names:
        raise ValueError(
            f"pinned index tensors for expert {expert_id} are out of domain"
        )
    for name in sorted(expected_names):
        if weight_map[name] != SHARD:
            raise ValueError(f"pinned index maps {name} outside {SHARD}")
    specs = []
    total = 0
    for suffix, expected_dtype, expected_shape in TENSOR_ORDER:
        name = prefix + suffix
        entry = header.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"pinned header is missing {name}")
        if entry.get("dtype") != expected_dtype or entry.get("shape") != expected_shape:
            raise ValueError(f"pinned header dtype/shape mismatch for {name}")
        offsets = entry.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(value) is not int for value in offsets)
        ):
            raise ValueError(f"pinned header offsets are malformed for {name}")
        start, end = offsets
        expected_bytes = 1
        for dimension in expected_shape:
            expected_bytes *= dimension
        if start < 0 or end <= start or end - start != expected_bytes:
            raise ValueError(f"pinned header extent mismatch for {name}")
        if end > payload_boundary:
            raise ValueError(f"pinned header range escapes source shard for {name}")
        absolute_start = data_start + start
        absolute_end = data_start + end - 1
        specs.append(
            {
                "name": name,
                "dtype": expected_dtype,
                "shape": expected_shape,
                "data_offsets": [start, end],
                "range_start": absolute_start,
                "range_end": absolute_end,
                "range": f"bytes={absolute_start}-{absolute_end}",
                "expected_bytes": expected_bytes,
            }
        )
        total += expected_bytes
    if total != EXPECTED_PAYLOAD_BYTES:
        raise ValueError(f"expected {EXPECTED_PAYLOAD_BYTES} expert bytes, got {total}")
    return specs


def selected_ranges(
    expert_ids: object = None,
) -> tuple[bytes, tuple[int, ...], list[dict[str, object]]]:
    ids = validate_expert_ids(expert_ids)
    header_bytes, header = load_pinned_header()
    weight_map = load_pinned_weight_map()
    specs = []
    for expert_id in ids:
        specs.extend(expert_ranges(expert_id, header_bytes, header, weight_map))
    ordered = sorted(specs, key=lambda spec: spec["range_start"])
    for previous, current in zip(ordered, ordered[1:]):
        if current["range_start"] <= previous["range_end"]:
            raise ValueError("selected tensor byte ranges overlap")
    total = sum(spec["expected_bytes"] for spec in specs)
    if total != EXPECTED_PAYLOAD_BYTES * len(ids) or total > PAYLOAD_CAP_BYTES:
        raise ValueError("selected payload bytes miss or exceed the byte budget")
    return header_bytes, ids, specs


def pinned_ranges(expert_id: int = EXPERT_ID) -> tuple[bytes, list[dict[str, object]]]:
    """A106-compatible single-expert entry point (now also index-checked)."""
    header_bytes, _, specs = selected_ranges([expert_id])
    return header_bytes, specs


def selection_label(expert_ids: tuple[int, ...]) -> str:
    if len(expert_ids) == 1:
        return f"expert{expert_ids[0]}"
    return "experts" + "-".join(str(value) for value in expert_ids)


def artifact_paths(expert_ids: tuple[int, ...]) -> dict[str, Path]:
    """One output/receipt/work identity per canonical selection."""
    if tuple(expert_ids) == LEGACY_SELECTION:
        return {
            "root": ROOT,
            "output": OUTPUT,
            "evidence": EVIDENCE,
            "receipt": RECEIPT,
            "work": WORK,
        }
    label = selection_label(tuple(expert_ids))
    evidence = ARTIFACT_ROOT / A120_EVIDENCE_DIR
    return {
        "root": ARTIFACT_ROOT,
        "output": ARTIFACT_ROOT
        / f"fixtures/deepseek_v41_layer0_{label}_314113f8.safetensors",
        "evidence": evidence,
        "receipt": evidence / f"A120_{label.upper()}_RANGE_RECEIPT.json",
        "work": ARTIFACT_ROOT / f".work/a120-{label}-range-payload",
    }


def payload_filename(name: str, expert_ids: tuple[int, ...]) -> str:
    expert, weight, kind = name.rsplit(".", 3)[-3:]
    base = f"{weight}_{kind}.bin"
    if tuple(expert_ids) == LEGACY_SELECTION:
        return base
    return f"expert{expert}_{base}"


def validate_range_spec(spec: dict[str, object]) -> None:
    start, end, expected = (
        spec["range_start"],
        spec["range_end"],
        spec["expected_bytes"],
    )
    if any(type(value) is not int for value in (start, end, expected)):
        raise ValueError("range bounds must be integers")
    if not 0 <= start <= end < SHARD_BYTES or end - start + 1 != expected:
        raise ValueError("range bounds escape the shard or mismatch the tensor size")
    if expected > PAYLOAD_CAP_BYTES:
        raise ValueError("single range exceeds the payload byte cap")


def read_range(
    spec: dict[str, object],
) -> tuple[bytes, str, list[dict[str, object]]]:
    validate_range_spec(spec)
    start, end = spec["range_start"], spec["range_end"]
    request = urllib.request.Request(
        f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{SHARD}",
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "VoltaFabric-pinned-expert-range/1.0",
        },
    )
    response, redirects = open_range_request(request, timeout=30)
    with response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")
        expected_content_range = f"bytes {start}-{end}/{SHARD_BYTES}"
        if status != 206 or content_range != expected_content_range:
            raise RangeReadError(
                "server did not honor the exact pinned tensor byte range",
                status=status,
                content_range=content_range,
                body_bytes_read=0,
            )
        expected_bytes = spec["expected_bytes"]
        content_length = response.headers.get("Content-Length")
        if content_length is not None and content_length != str(expected_bytes):
            raise RangeReadError(
                "response Content-Length differs from the exact tensor range",
                status=status,
                content_range=content_range,
                body_bytes_read=0,
            )
        # Read at most one byte beyond the requested range. This detects an
        # overlong response without buffering an accidentally un-ranged shard.
        body = response.read(expected_bytes + 1)
        if len(body) != expected_bytes:
            raise RangeReadError(
                "response body length differs from the exact tensor range",
                status=status,
                content_range=content_range,
                body_bytes_read=len(body),
            )
        return body, content_range, redirects


def make_safetensors_fixture(
    specs: list[dict[str, object]],
    payloads: dict[str, bytes],
    expected_total: int | None = None,
) -> bytes:
    header: dict[str, dict[str, object]] = {}
    ordered_payloads = []
    data_offset = 0
    if expected_total is None:
        expected_total = EXPECTED_PAYLOAD_BYTES
    if expected_total > PAYLOAD_CAP_BYTES:
        raise ValueError("fixture payload total exceeds the payload byte cap")
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("fixture tensor names are not unique")
    if set(payloads) != {spec["name"] for spec in specs}:
        raise ValueError(
            "fixture payload names do not match the six pinned tensor names"
        )
    for spec in specs:
        name = spec["name"]
        payload = payloads[name]
        if len(payload) != spec["expected_bytes"]:
            raise ValueError(f"payload byte count mismatch for {name}")
        end = data_offset + len(payload)
        header[name] = {
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": [data_offset, end],
        }
        ordered_payloads.append(payload)
        data_offset = end
    if data_offset != expected_total:
        raise ValueError("assembled payload byte count changed")
    encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded_header += b" " * ((-len(encoded_header)) % 8)
    return (
        struct.pack("<Q", len(encoded_header))
        + encoded_header
        + b"".join(ordered_payloads)
    )


def atomic_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_new_file(path: Path, data: bytes) -> None:
    """Write via an exclusive temp file and hard-link; never replaces `path`."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "xb") as handle:
        handle.write(data)
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def validate_resume_receipt(
    receipt: dict[str, object],
    specs: list[dict[str, object]],
    expert_ids: tuple[int, ...] = LEGACY_SELECTION,
) -> dict[str, bytes]:
    paths = artifact_paths(expert_ids)
    root, work = paths["root"], paths["work"]
    expected_ranges = [
        {
            "name": spec["name"],
            "dtype": spec["dtype"],
            "shape": spec["shape"],
            "data_offsets": spec["data_offsets"],
            "range_start": spec["range_start"],
            "range_end": spec["range_end"],
            "expected_bytes": spec["expected_bytes"],
        }
        for spec in specs
    ]
    if (
        receipt.get("status")
        not in {"IN_PROGRESS", "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"}
        or receipt.get("repository") != REPOSITORY
        or receipt.get("revision") != REVISION
        or receipt.get("shard") != SHARD
        or receipt.get("header_sha256") != HEADER_SHA256
        or receipt.get("requested_tensors") != expected_ranges
        or (
            tuple(expert_ids) != LEGACY_SELECTION
            and (
                receipt.get("expert_ids") != list(expert_ids)
                or receipt.get("index_sha256") != INDEX_SHA256
            )
        )
    ):
        raise ValueError("partial receipt does not match the pinned expert extraction")
    entries = receipt.get("completed_tensors")
    if not isinstance(entries, list) or any(not isinstance(x, dict) for x in entries):
        raise ValueError("partial tensor receipt is malformed")
    names = [entry.get("name") for entry in entries]
    if len(names) != len(set(names)):
        raise ValueError("partial tensor receipt contains duplicate range identities")
    expected_by_name = {spec["name"]: spec for spec in specs}
    if not set(names).issubset(expected_by_name):
        raise ValueError("partial tensor receipt contains an unrequested range")
    payloads = {}
    total = 0
    for entry in entries:
        name = entry["name"]
        spec = expected_by_name[name]
        path = work / payload_filename(name, expert_ids)
        relative = str(path.relative_to(root))
        if entry.get("payload_path") != relative:
            raise ValueError(f"{name}: temporary payload path differs from the receipt")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root.resolve()):
            raise ValueError(f"{name}: temporary payload path escapes the project")
        payload = resolved.read_bytes()
        if (
            entry.get("http_status") != 206
            or entry.get("range")
            != f"bytes {spec['range_start']}-{spec['range_end']}/{SHARD_BYTES}"
            or len(payload) != spec["expected_bytes"]
            or entry.get("bytes") != len(payload)
            or entry.get("sha256") != sha256(payload)
        ):
            raise ValueError(f"{name}: saved range payload fails receipt validation")
        payloads[name] = payload
        total += len(payload)
    if receipt.get("completed_response_body_bytes") != total:
        raise ValueError("partial response body byte total is inconsistent")
    expected_temp_names = {payload_filename(name, expert_ids) for name in names}
    actual_temp_names = (
        {path.name for path in work.iterdir()} if work.exists() else set()
    )
    if actual_temp_names != expected_temp_names:
        raise ValueError(
            "unreceipted temporary payload exists; inspect before resuming"
        )
    return payloads


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only after inspecting the partial receipt and local ranges.",
    )
    parser.add_argument(
        "--expert-id",
        dest="expert_ids",
        action="append",
        type=int,
        help="Layer-0 expert to extract (repeatable, max 2; default 383).",
    )
    args = parser.parse_args(argv)
    try:
        header_bytes, expert_ids, specs = selected_ranges(args.expert_ids)
    except (OSError, ValueError) as error:
        raise SystemExit(f"expert selection rejected: {error}") from None
    paths = artifact_paths(expert_ids)
    root, output, receipt_path, work = (
        paths["root"],
        paths["output"],
        paths["receipt"],
        paths["work"],
    )
    label = selection_label(expert_ids)
    legacy = expert_ids == LEGACY_SELECTION
    expected_total = EXPECTED_PAYLOAD_BYTES * len(expert_ids)
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    if receipt_temporary.exists():
        raise SystemExit(
            f"{label} receipt temp exists; inspect before starting or resuming"
        )
    paths["evidence"].mkdir(parents=True, exist_ok=True)
    if output.exists() or output.with_suffix(output.suffix + ".tmp").exists():
        raise SystemExit(f"{label} fixture already exists; refusing to overwrite it")
    if receipt_path.exists():
        if not args.resume:
            raise SystemExit(
                "range receipt exists; inspect it and pass --resume explicitly"
            )
        receipt = json.loads(receipt_path.read_bytes())
        try:
            payloads = validate_resume_receipt(receipt, specs, expert_ids)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SystemExit(
                f"partial extraction receipt is invalid: {error}"
            ) from None
    else:
        if args.resume:
            raise SystemExit("--resume requested but no partial range receipt exists")
        if work.exists() and any(work.iterdir()):
            raise SystemExit(
                "unreceipted payload files exist; inspect before any fetch"
            )
        work.mkdir(parents=True, exist_ok=True)
        receipt = {
            "schema_version": 1,
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "IN_PROGRESS",
            "repository": REPOSITORY,
            "revision": REVISION,
            "shard": SHARD,
            "shard_bytes": SHARD_BYTES,
        }
        if legacy:
            receipt["expert_id"] = EXPERT_ID
        else:
            receipt["expert_ids"] = list(expert_ids)
            receipt["selection_label"] = label
            receipt["index_path"] = str(INDEX_PATH.relative_to(ROOT))
            receipt["index_sha256"] = INDEX_SHA256
            receipt["payload_byte_cap"] = PAYLOAD_CAP_BYTES
            receipt["max_experts_per_invocation"] = MAX_EXPERTS_PER_INVOCATION
        receipt.update(
            {
                "header_path": str(HEADER_PATH.relative_to(ROOT)),
                "header_sha256": sha256(header_bytes),
                "full_checkpoint_downloaded": False,
                "other_expert_payloads_downloaded": 0,
                "requested_payload_bytes": expected_total,
                "completed_response_body_bytes": 0,
                "requested_tensors": [
                    {
                        "name": spec["name"],
                        "dtype": spec["dtype"],
                        "shape": spec["shape"],
                        "data_offsets": spec["data_offsets"],
                        "range_start": spec["range_start"],
                        "range_end": spec["range_end"],
                        "expected_bytes": spec["expected_bytes"],
                    }
                    for spec in specs
                ],
                "completed_tensors": [],
            }
        )
        atomic_json(receipt_path, receipt)
        payloads = {}

    completed = {entry["name"] for entry in receipt["completed_tensors"]}
    for spec in specs:
        name = spec["name"]
        if name in completed:
            continue
        stage = "range_request"
        returned_body_bytes = 0
        try:
            if (
                receipt["completed_response_body_bytes"] + spec["expected_bytes"]
                > PAYLOAD_CAP_BYTES
            ):
                stage = "budget_check"
                raise ValueError("next range would exceed the payload byte cap")
            payload, content_range, redirects = read_range(spec)
            returned_body_bytes = len(payload)
            stage = "persist_range"
            local_path = work / payload_filename(name, expert_ids)
            temporary = local_path.with_suffix(local_path.suffix + ".tmp")
            with temporary.open("xb") as stream:
                stream.write(payload)
            temporary.replace(local_path)
            receipt["completed_tensors"].append(
                {
                    "name": name,
                    "http_status": 206,
                    "range": content_range,
                    "bytes": len(payload),
                    "sha256": sha256(payload),
                    "payload_path": str(local_path.relative_to(root)),
                    "redirect_chain": redirects,
                }
            )
            receipt["completed_response_body_bytes"] += len(payload)
            receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(receipt_path, receipt)
            payloads[name] = payload
        except Exception as error:
            receipt["status"] = "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
            receipt["last_failure"] = {
                "tensor": name,
                "stage": stage,
                "exception_type": type(error).__name__,
                "http_status": (
                    error.code
                    if isinstance(error, urllib.error.HTTPError)
                    else getattr(error, "status", None)
                ),
                "content_range": getattr(error, "content_range", ""),
                "response_body_bytes_read_before_failure": getattr(
                    error, "body_bytes_read", returned_body_bytes
                ),
                "automatic_retry": False,
            }
            receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
            atomic_json(receipt_path, receipt)
            raise SystemExit(
                f"{label} range extraction stopped at {name}/{stage}; receipt saved"
            ) from None

    if len(receipt["completed_tensors"]) != len(specs):
        raise SystemExit("expected exactly six tensors per expert; inspect the receipt")
    fixture = make_safetensors_fixture(specs, payloads, expected_total)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_new_file(output, fixture)
    except OSError as error:
        receipt["status"] = "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
        receipt["last_failure"] = {
            "tensor": None,
            "stage": "write_fixture",
            "exception_type": type(error).__name__,
            "automatic_retry": False,
        }
        atomic_json(receipt_path, receipt)
        raise SystemExit(f"{label} fixture write failed; receipt saved") from None
    receipt["status"] = f"PASS_PINNED_{label.upper().replace('-', '_')}_RANGES_ONLY"
    receipt["fixture_path"] = str(output.relative_to(root))
    receipt["fixture_bytes"] = len(fixture)
    receipt["fixture_sha256"] = sha256(fixture)
    receipt["payload_bytes_preserved_verbatim"] = True
    receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt.pop("last_failure", None)
    atomic_json(receipt_path, receipt)
    summary = {"status": receipt["status"]}
    if legacy:
        summary["expert_id"] = EXPERT_ID
    else:
        summary["expert_ids"] = list(expert_ids)
    summary.update(
        {
            "tensor_count": len(receipt["completed_tensors"]),
            "payload_bytes": expected_total,
            "fixture_bytes": len(fixture),
            "fixture_sha256": receipt["fixture_sha256"],
            "full_checkpoint_downloaded": False,
        }
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
