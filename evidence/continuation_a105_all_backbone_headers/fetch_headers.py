#!/usr/bin/env python3
"""Fetch only pinned Safetensors JSON headers using explicit bounded HTTP Range reads."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import prod
from pathlib import Path
import re
import struct
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INDEX_PATH = (
    ROOT
    / "evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
)
RECEIPT_PATH = OUT / "A105_FETCH_RECEIPT.json"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
REPOSITORY = "deepseek-ai/DeepSeek-V4.1-Flash"
MAX_HEADER_BYTES = 16 * 1024 * 1024
EXPERT_KEY = re.compile(
    r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w1|w2|w3)\.(weight|scale)\Z"
)
EXPECTED_PLANES = {
    (projection, plane)
    for projection in ("w1", "w2", "w3")
    for plane in ("weight", "scale")
}
EXPECTED_SHAPES = {
    "w1": {"weight": [2304, 2560], "scale": [2304, 160]},
    "w2": {"weight": [5120, 1152], "scale": [5120, 72]},
    "w3": {"weight": [2304, 2560], "scale": [2304, 160]},
}
EXPECTED_DTYPES = {"weight": "I8", "scale": "F8_E8M0"}
DTYPE_BYTES = {"I8": 1, "F8_E8M0": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}


def write_receipt(receipt: dict[str, object]) -> None:
    temporary = RECEIPT_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(RECEIPT_PATH)


def index_shards() -> tuple[bytes, dict[str, set[str]]]:
    index_bytes = INDEX_PATH.read_bytes()
    if hashlib.sha256(index_bytes).hexdigest() != INDEX_SHA256:
        raise ValueError("pinned index SHA-256 mismatch")
    weight_map = json.loads(index_bytes)["weight_map"]
    shard_names = sorted(
        {
            source
            for name, source in weight_map.items()
            if name.startswith("layers.") and ".ffn.experts." in name
        }
    )
    by_shard = {
        shard: {name for name, source in weight_map.items() if source == shard}
        for shard in shard_names
    }
    groups: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    layer_source: dict[int, str] = {}
    unparsed_expert_keys = []
    for shard, names in by_shard.items():
        for name in names:
            match = EXPERT_KEY.fullmatch(name)
            if not match:
                if ".ffn.experts." in name:
                    unparsed_expert_keys.append(name)
                continue
            layer_text, expert_text, projection, plane = match.groups()
            layer, expert = int(layer_text), int(expert_text)
            groups[(layer, expert)].add((projection, plane))
            if layer in layer_source and layer_source[layer] != shard:
                raise ValueError(f"backbone layer {layer} is split across shards")
            layer_source[layer] = shard
    if len(shard_names) != 40 or len(groups) != 40 * 384:
        raise ValueError(
            "pinned index is not the expected 40-layer, 384-expert backbone"
        )
    if set(layer_source) != set(range(40)) or any(
        v != EXPECTED_PLANES for v in groups.values()
    ):
        raise ValueError(
            "pinned index contains missing or incomplete backbone expert groups"
        )
    if unparsed_expert_keys:
        raise ValueError(
            f"unrecognized backbone expert index key: {unparsed_expert_keys[0]}"
        )
    if len(set(layer_source.values())) != 40:
        raise ValueError("pinned backbone layers do not map one-to-one to shards")
    # Layer zero already has a separately pinned A103 header fixture. Fetching it
    # again would add an unnecessary remote request and contradict the 39-header
    # acceptance contract for this operation.
    preexisting_shard = layer_source[0]
    if preexisting_shard != "model-00003-of-00048.safetensors":
        raise ValueError("layer-zero shard differs from the pinned A103 header")
    return index_bytes, {
        shard: names for shard, names in by_shard.items() if shard != preexisting_shard
    }


def get_range(
    shard: str, start: int, end: int, expected_total: int | None = None
) -> tuple[bytes, int, str]:
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{shard}"
    request = urllib.request.Request(
        url,
        headers={
            "Range": f"bytes={start}-{end}",
            "Accept-Encoding": "identity",
            "User-Agent": "VoltaFabric-pinned-header-crosscheck/1.0",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        status = getattr(response, "status", response.getcode())
        content_range = response.headers.get("Content-Range", "")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
        if status != 206 or match is None:
            raise RuntimeError("server did not honor the bounded HTTP Range request")
        response_start, response_end, total = map(int, match.groups())
        if (response_start, response_end) != (start, end):
            raise RuntimeError("server returned a different byte range than requested")
        if expected_total is not None and total != expected_total:
            raise RuntimeError("remote file size changed between header Range requests")
        expected_bytes = end - start + 1
        body = response.read(expected_bytes)
        if len(body) != expected_bytes or response.read(1):
            raise RuntimeError(
                "response body length does not match the exact requested range"
            )
        return body, total, content_range


def validate_header(
    shard: str, header_bytes: bytes, file_bytes: int, indexed_names: set[str]
) -> tuple[int, int, int]:
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"{shard}: header is not a JSON object")
    names = set(header) - {"__metadata__"}
    if names != indexed_names:
        raise ValueError(f"{shard}: header/index name sets do not match")
    groups: dict[int, set[tuple[str, str]]] = defaultdict(set)
    expert_count = 0
    for name in names:
        entry = header[name]
        match = EXPERT_KEY.fullmatch(name)
        if match is None:
            if ".ffn.experts." in name:
                raise ValueError(f"{shard}: unrecognized expert key")
        else:
            layer_text, expert_text, projection, plane = match.groups()
            layer, expert = int(layer_text), int(expert_text)
            if str(layer) != layer_text or str(expert) != expert_text:
                raise ValueError(f"{shard}: noncanonical expert ID")
            if (
                entry["dtype"] != EXPECTED_DTYPES[plane]
                or entry["shape"] != EXPECTED_SHAPES[projection][plane]
            ):
                raise ValueError(f"{shard}: expert dtype or shape mismatch")
            identity = (projection, plane)
            if identity in groups[expert]:
                raise ValueError(f"{shard}: duplicate plane in expert {expert}")
            groups[expert].add(identity)
            expert_count += 1
        dtype = entry["dtype"]
        shape = entry["shape"]
        offsets = entry["data_offsets"]
        if dtype not in DTYPE_BYTES or not isinstance(shape, list) or len(offsets) != 2:
            raise ValueError(f"{shard}: invalid Safetensors dtype/shape/offsets")
        start, end = offsets
        if start < 0 or end < start or end - start != prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError(f"{shard}: tensor size does not match shape/dtype")
        payload_bytes = file_bytes - (8 + len(header_bytes))
        if end > payload_bytes:
            raise ValueError(f"{shard}: tensor exceeds the source payload boundary")
    if set(groups) != set(range(384)) or any(
        planes != EXPECTED_PLANES for planes in groups.values()
    ):
        raise ValueError(f"{shard}: incomplete 384-expert six-plane coverage")
    if expert_count != 2304:
        raise ValueError(f"{shard}: unexpected expert tensor count")
    return len(names), len(names) - expert_count, expert_count


def _layer_for_shard(names: set[str]) -> int:
    layers = {
        int(match.group(1))
        for name in names
        if (match := EXPERT_KEY.fullmatch(name)) is not None
    }
    if len(layers) != 1:
        raise ValueError(
            "a fetched backbone shard must contain exactly one backbone layer"
        )
    return next(iter(layers))


def _validate_resume_receipt(
    receipt: dict[str, object],
    index_bytes: bytes,
    by_shard: dict[str, set[str]],
) -> dict[str, object]:
    """Validate every persisted fetch result before an explicit resume can skip it."""
    if receipt.get("status") not in {
        "IN_PROGRESS",
        "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY",
    }:
        raise ValueError("fetch receipt is not resumable")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("repository") != REPOSITORY
        or receipt.get("revision") != REVISION
        or receipt.get("index_sha256") != hashlib.sha256(index_bytes).hexdigest()
        or receipt.get("index_sha256") != INDEX_SHA256
        or receipt.get("max_header_bytes") != MAX_HEADER_BYTES
        or receipt.get("full_checkpoint_downloaded") is not False
        or receipt.get("payload_downloaded") is not False
        or receipt.get("requested_shards") != sorted(by_shard)
    ):
        raise ValueError("partial receipt does not match the pinned A105 request")
    entries = receipt.get("shards")
    if not isinstance(entries, list):
        raise ValueError("partial receipt shard records are not a list")
    files = [entry.get("file") for entry in entries if isinstance(entry, dict)]
    if len(files) != len(entries):
        raise ValueError("partial receipt contains a malformed shard record")
    if len(files) != len(set(files)):
        raise ValueError("partial receipt contains a duplicate shard")
    if not set(files).issubset(by_shard):
        raise ValueError("partial receipt contains a shard outside this request")

    total_body_bytes = 0
    for entry in entries:
        shard = entry["file"]
        header_path = OUT / f"{shard}.header.{REVISION[:8]}.json"
        expected_relative_path = str(header_path.relative_to(ROOT))
        if entry.get("header_path") != expected_relative_path:
            raise ValueError(f"{shard}: partial receipt header path is unexpected")
        resolved_header = header_path.resolve(strict=True)
        if not resolved_header.is_relative_to(ROOT):
            raise ValueError(f"{shard}: partial receipt header path escapes project")
        header_bytes = resolved_header.read_bytes()
        header_size = len(header_bytes)
        file_size = entry.get("file_bytes")
        if (
            entry.get("http_status") != 206
            or not isinstance(file_size, int)
            or file_size <= 8 + header_size
            or entry.get("header_bytes") != header_size
            or header_size > MAX_HEADER_BYTES
            or entry.get("payload_bytes_read") != 8 + header_size
            or entry.get("range_0_7") != f"bytes 0-7/{file_size}"
            or entry.get("range_header") != f"bytes 8-{7 + header_size}/{file_size}"
            or hashlib.sha256(header_bytes).hexdigest() != entry.get("header_sha256")
        ):
            raise ValueError(f"{shard}: saved header does not match its range receipt")
        tensor_count, non_expert_count, expert_tensor_count = validate_header(
            shard, header_bytes, file_size, by_shard[shard]
        )
        expected_metadata = {
            "header_tensor_count": tensor_count,
            "layer": _layer_for_shard(by_shard[shard]),
            "backbone_expert_count": 384,
            "expert_tensor_count": expert_tensor_count,
            "non_expert_tensor_count": non_expert_count,
            "header_index_names_match": True,
            "all_expert_groups_complete": True,
        }
        if any(
            entry.get(name) != expected for name, expected in expected_metadata.items()
        ):
            raise ValueError(f"{shard}: saved header coverage metadata is inconsistent")
        total_body_bytes += 8 + header_size
    if receipt.get("network_response_body_bytes") != total_body_bytes:
        raise ValueError(
            "partial receipt aggregate response byte count is inconsistent"
        )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue only after inspecting the saved partial receipt.",
    )
    args = parser.parse_args()
    index_bytes, by_shard = index_shards()
    previous = None
    if RECEIPT_PATH.exists():
        if not args.resume:
            raise SystemExit(
                "fetch receipt exists; inspect it and pass --resume explicitly to continue"
            )
        previous = json.loads(RECEIPT_PATH.read_bytes())
        try:
            receipt = _validate_resume_receipt(previous, index_bytes, by_shard)
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise SystemExit(f"partial fetch receipt is invalid: {error}") from None
    else:
        if args.resume:
            raise SystemExit("--resume requested but no partial fetch receipt exists")
        receipt = {
            "schema_version": 1,
            "observed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "IN_PROGRESS",
            "repository": REPOSITORY,
            "revision": REVISION,
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
            "max_header_bytes": MAX_HEADER_BYTES,
            "full_checkpoint_downloaded": False,
            "payload_downloaded": False,
            "network_response_body_bytes": 0,
            "requested_shards": sorted(by_shard),
            "shards": [],
        }
        write_receipt(receipt)

    completed = {item["file"] for item in receipt["shards"]}
    for shard in sorted(set(by_shard) - completed):
        orphan = OUT / f"{shard}.header.{REVISION[:8]}.json"
        if orphan.exists():
            raise SystemExit(
                f"unreceipted header exists for {shard}; inspect it before any explicit resume"
            )
    for shard in sorted(by_shard):
        if shard in completed:
            continue
        stage = "prefix_range"
        try:
            prefix, file_bytes, range_prefix = get_range(shard, 0, 7)
            header_bytes_length = struct.unpack("<Q", prefix)[0]
            if header_bytes_length == 0 or header_bytes_length > MAX_HEADER_BYTES:
                raise RuntimeError(
                    "Safetensors header length is outside the configured limit"
                )
            stage = "header_range"
            header_bytes, _, range_header = get_range(
                shard, 8, 7 + header_bytes_length, expected_total=file_bytes
            )
            tensor_count, non_expert_count, expert_tensor_count = validate_header(
                shard, header_bytes, file_bytes, by_shard[shard]
            )
            layer = _layer_for_shard(by_shard[shard])
            header_path = OUT / f"{shard}.header.{REVISION[:8]}.json"
            temporary_header_path = header_path.with_suffix(header_path.suffix + ".tmp")
            temporary_header_path.write_bytes(header_bytes)
            temporary_header_path.replace(header_path)
            entry = {
                "file": shard,
                "http_status": 206,
                "range_0_7": range_prefix,
                "range_header": range_header,
                "file_bytes": file_bytes,
                "header_bytes": len(header_bytes),
                "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
                "header_path": str(header_path.relative_to(ROOT)),
                "payload_bytes_read": 8 + len(header_bytes),
                "header_tensor_count": tensor_count,
                "layer": layer,
                "backbone_expert_count": 384,
                "expert_tensor_count": expert_tensor_count,
                "non_expert_tensor_count": non_expert_count,
                "header_index_names_match": True,
                "all_expert_groups_complete": True,
            }
            receipt["shards"].append(entry)
            receipt["network_response_body_bytes"] += entry["payload_bytes_read"]
            receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
            write_receipt(receipt)
        except Exception as error:
            status = error.code if isinstance(error, urllib.error.HTTPError) else None
            receipt["status"] = "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
            receipt["last_failure"] = {
                "shard": shard,
                "stage": stage,
                "exception_type": type(error).__name__,
                "http_status": status,
                "replay_policy": "Inspect this receipt and the local header before any explicit --resume operation.",
            }
            receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
            write_receipt(receipt)
            raise SystemExit(
                f"bounded header fetch stopped at {shard}/{stage}; receipt saved"
            ) from None

    if len(receipt["shards"]) != 39:
        receipt["status"] = "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
        write_receipt(receipt)
        raise SystemExit(
            "expected 39 missing backbone headers; inspect the saved receipt"
        )
    receipt["status"] = "PASS_BOUNDED_HTTP_RANGE_HEADERS_ONLY"
    receipt["payload_downloaded"] = False
    receipt["observed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt.pop("last_failure", None)
    write_receipt(receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "completed_headers": len(receipt["shards"]),
                "response_body_bytes": receipt["network_response_body_bytes"],
                "payload_downloaded": receipt["payload_downloaded"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
