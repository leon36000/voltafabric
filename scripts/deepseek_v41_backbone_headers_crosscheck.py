#!/usr/bin/env python3
"""Cross-check every pinned DeepSeek backbone expert shard header and index entry."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import prod
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = (
    ROOT
    / "evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
)
BACKBONE_HEADER_DIR = ROOT / "evidence/continuation_a105_all_backbone_headers"
PREEXISTING_HEADER_PATH = (
    ROOT / "evidence/continuation_a97_model_metadata/model-00003.header.314113f8.json"
)
PREEXISTING_HEADER_RECEIPT = (
    ROOT
    / "evidence/continuation_a103_official_shard_header/A103_OFFICIAL_SHARD_HEADER_CROSSCHECK.json"
)
FETCH_RECEIPT_PATH = BACKBONE_HEADER_DIR / "A105_FETCH_RECEIPT.json"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
BACKBONE_KEY = re.compile(
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
MAX_HEADER_BYTES = 16 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def discover() -> tuple[dict[str, object], dict[str, object], bytes]:
    index_bytes = INDEX_PATH.read_bytes()
    digest = sha256(index_bytes)
    if digest != INDEX_SHA256:
        raise ValueError("pinned DeepSeek official index SHA-256 mismatch")
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    groups: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    source_by_layer: dict[int, str] = {}
    malformed = []
    for name, source in weight_map.items():
        if ".ffn.experts." not in name:
            continue
        match = BACKBONE_KEY.fullmatch(name)
        if match is None:
            if name.startswith("mtp."):
                continue
            malformed.append(name)
            continue
        layer_text, expert_text, projection, plane = match.groups()
        layer, expert = int(layer_text), int(expert_text)
        groups[(layer, expert)].add((projection, plane))
        old_source = source_by_layer.setdefault(layer, source)
        if old_source != source:
            raise ValueError(f"backbone layer {layer} is split across index shards")
    if malformed:
        raise ValueError(f"unrecognized expert key in pinned index: {malformed[0]}")
    if set(groups) != {(layer, expert) for layer in range(40) for expert in range(384)}:
        raise ValueError(
            "pinned backbone expert identity coverage is not layers 0..39 by experts 0..383"
        )
    if any(planes != EXPECTED_PLANES for planes in groups.values()):
        raise ValueError(
            "pinned backbone index contains an incomplete six-plane expert"
        )
    if len(source_by_layer) != 40 or len(set(source_by_layer.values())) != 40:
        raise ValueError("pinned backbone layers do not map one-to-one to 40 shards")

    prior = json.loads(PREEXISTING_HEADER_RECEIPT.read_bytes())
    prior_header = prior["header"]
    shards = []
    for layer in range(40):
        shard = source_by_layer[layer]
        if layer == 0:
            header_path = PREEXISTING_HEADER_PATH
            header_sha = prior_header["sha256"]
            header_bytes = prior_header["bytes"]
            file_bytes = prior["shard_bytes"]
            origin = "preexisting_pinned_a103_header"
        else:
            header_path = BACKBONE_HEADER_DIR / f"{shard}.header.{REVISION[:8]}.json"
            header_sha = None
            header_bytes = None
            file_bytes = None
            origin = "bounded_http_range_pending"
        shards.append(
            {
                "file": shard,
                "layer": layer,
                "expert_count": 384,
                "expert_tensor_count": 2304,
                "index_name_count": sum(
                    1 for name, source in weight_map.items() if source == shard
                ),
                "header_path": str(header_path.relative_to(ROOT)),
                "header_sha256": header_sha,
                "header_bytes": header_bytes,
                "file_bytes": file_bytes,
                "origin": origin,
            }
        )
    summary = {
        "schema_version": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PINNED_INDEX_BACKBONE_DISCOVERY",
        "repository": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": REVISION,
        "index_sha256": digest,
        "total_index_keys": len(weight_map),
        "backbone_layers": 40,
        "backbone_experts": 15_360,
        "backbone_expert_tensor_entries": 92_160,
        "backbone_shard_count": 40,
        "one_layer_per_shard": True,
        "all_groups_have_six_index_planes": True,
        "unparsed_expert_keys": 0,
        "shards": shards,
    }
    return summary, index, index_bytes


def crosscheck() -> dict[str, object]:
    discovery, index, _ = discover()
    fetch = json.loads(FETCH_RECEIPT_PATH.read_bytes())
    if fetch["status"] != "PASS_BOUNDED_HTTP_RANGE_HEADERS_ONLY":
        raise ValueError("bounded header fetch receipt is not a completed pass")
    if fetch["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash":
        raise ValueError("unexpected repository in the bounded header fetch receipt")
    if fetch["revision"] != REVISION or fetch["index_sha256"] != INDEX_SHA256:
        raise ValueError(
            "header fetch receipt does not match the pinned model revision/index"
        )
    if fetch["full_checkpoint_downloaded"] or fetch["payload_downloaded"]:
        raise ValueError("fetch receipt claims a Safetensors payload was downloaded")
    fetched_by_name = {entry["file"]: entry for entry in fetch["shards"]}
    expected_missing = {
        entry["file"] for entry in discovery["shards"] if entry["layer"] != 0
    }
    if (
        fetch["requested_shards"] != sorted(expected_missing)
        or fetch["max_header_bytes"] != MAX_HEADER_BYTES
        or len(fetched_by_name) != len(fetch["shards"])
        or set(fetched_by_name) != expected_missing
    ):
        raise ValueError(
            "range receipt does not uniquely cover exactly the 39 missing backbone headers"
        )

    weight_map = index["weight_map"]
    shard_results = []
    all_groups: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
    total_network_body_bytes = 0
    total_expert_tensors = 0
    total_non_expert_tensors = 0
    for item in discovery["shards"]:
        shard = item["file"]
        layer = item["layer"]
        if layer == 0:
            fetch_info = {
                "file_bytes": item["file_bytes"],
                "header_bytes": item["header_bytes"],
                "header_sha256": item["header_sha256"],
                "header_path": item["header_path"],
                "origin": item["origin"],
            }
        else:
            fetch_info = fetched_by_name[shard]
            if fetch_info["http_status"] != 206:
                raise ValueError(
                    f"{shard}: expected HTTP 206 for both bounded Range responses"
                )
            if fetch_info["range_0_7"] != f"bytes 0-7/{fetch_info['file_bytes']}":
                raise ValueError(f"{shard}: invalid 8-byte Safetensors prefix Range")
            if fetch_info["payload_bytes_read"] != 8 + fetch_info["header_bytes"]:
                raise ValueError(
                    f"{shard}: response body exceeded or missed the requested header bytes"
                )
            if fetch_info["range_header"] != (
                f"bytes 8-{7 + fetch_info['header_bytes']}/{fetch_info['file_bytes']}"
            ):
                raise ValueError(f"{shard}: invalid bounded Safetensors header Range")
            if fetch_info["header_bytes"] > fetch["max_header_bytes"]:
                raise ValueError(
                    f"{shard}: Safetensors header exceeded the configured maximum"
                )
            if fetch_info["payload_bytes_read"] != 8 + fetch_info["header_bytes"]:
                raise ValueError(
                    f"{shard}: response body exceeds the prefix plus JSON header"
                )
            if (
                fetch_info["layer"] != layer
                or fetch_info["backbone_expert_count"] != 384
                or fetch_info["expert_tensor_count"] != 2304
                or fetch_info["header_index_names_match"] is not True
                or fetch_info["all_expert_groups_complete"] is not True
            ):
                raise ValueError(f"{shard}: fetch coverage receipt is inconsistent")
            total_network_body_bytes += fetch_info["payload_bytes_read"]

        expected_path = (
            PREEXISTING_HEADER_PATH
            if layer == 0
            else BACKBONE_HEADER_DIR / f"{shard}.header.{REVISION[:8]}.json"
        ).resolve(strict=True)
        if not expected_path.is_relative_to(ROOT):
            raise ValueError(f"{shard}: header fixture path escapes the project")
        expected_relative = str(expected_path.relative_to(ROOT))
        if fetch_info["header_path"] != expected_relative:
            raise ValueError(f"{shard}: unexpected local header fixture path")
        header_bytes = expected_path.read_bytes()
        header_sha = sha256(header_bytes)
        if (
            len(header_bytes) != fetch_info["header_bytes"]
            or header_sha != fetch_info["header_sha256"]
        ):
            raise ValueError(f"{shard}: header size or SHA-256 mismatch")
        header = json.loads(header_bytes)
        names = set(header) - {"__metadata__"}
        indexed_names = {name for name, source in weight_map.items() if source == shard}
        if names != indexed_names:
            raise ValueError(
                f"{shard}: header/index name mismatch: header-only={len(names - indexed_names)}, "
                f"index-only={len(indexed_names - names)}"
            )
        payload_bytes = fetch_info["file_bytes"] - (8 + len(header_bytes))
        max_data_end = 0
        groups: dict[int, set[tuple[str, str]]] = defaultdict(set)
        expert_tensors = 0
        non_expert_tensors = 0
        for name in sorted(names):
            entry = header[name]
            shape = entry["shape"]
            offsets = entry["data_offsets"]
            if (
                not isinstance(shape, list)
                or not shape
                or any(not isinstance(dim, int) or dim <= 0 for dim in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
            ):
                raise ValueError(f"{shard}: malformed shape/offsets for {name}")
            start, end = offsets
            dtype = entry["dtype"]
            if start < 0 or end < start or dtype not in DTYPE_BYTES:
                raise ValueError(f"{shard}: invalid dtype/offsets for {name}")
            if end - start != prod(shape) * DTYPE_BYTES[dtype]:
                raise ValueError(
                    f"{shard}: shape/dtype byte extent mismatch for {name}"
                )
            if end > payload_bytes:
                raise ValueError(
                    f"{shard}: tensor extends beyond the shard payload for {name}"
                )
            max_data_end = max(max_data_end, end)
            match = BACKBONE_KEY.fullmatch(name)
            if match is None:
                if ".ffn.experts." in name:
                    raise ValueError(f"{shard}: unrecognized expert key {name}")
                non_expert_tensors += 1
                continue
            layer_text, expert_text, projection, plane = match.groups()
            parsed_layer, expert = int(layer_text), int(expert_text)
            if (
                parsed_layer != layer
                or str(parsed_layer) != layer_text
                or str(expert) != expert_text
            ):
                raise ValueError(
                    f"{shard}: noncanonical or unexpected expert identity {name}"
                )
            if (
                dtype != EXPECTED_DTYPES[plane]
                or shape != EXPECTED_SHAPES[projection][plane]
            ):
                raise ValueError(
                    f"{shard}: official MXFP4 dtype/shape mismatch for {name}"
                )
            identity = (projection, plane)
            if identity in groups[expert]:
                raise ValueError(
                    f"{shard}: duplicate plane for expert {expert}: {identity}"
                )
            groups[expert].add(identity)
            all_groups[(layer, expert)].add(identity)
            expert_tensors += 1
        if (
            len(groups) != 384
            or set(groups) != set(range(384))
            or any(planes != EXPECTED_PLANES for planes in groups.values())
            or expert_tensors != 2304
        ):
            raise ValueError(
                f"{shard}: backbone layer {layer} is not 384 complete six-plane experts"
            )
        total_expert_tensors += expert_tensors
        total_non_expert_tensors += non_expert_tensors
        shard_results.append(
            {
                "file": shard,
                "layer": layer,
                "file_bytes": fetch_info["file_bytes"],
                "header_bytes": len(header_bytes),
                "header_sha256": header_sha,
                "header_tensor_count": len(names),
                "index_name_set_matches": True,
                "backbone_experts": len(groups),
                "expert_tensors": expert_tensors,
                "non_expert_tensors": non_expert_tensors,
                "max_tensor_data_end": max_data_end,
                "payload_bytes": payload_bytes,
                "all_experts_complete": True,
            }
        )
    if len(all_groups) != 15_360 or total_expert_tensors != 92_160:
        raise ValueError("aggregate backbone expert coverage changed")
    if total_network_body_bytes != fetch["network_response_body_bytes"]:
        raise ValueError("aggregate bounded HTTP response body byte count mismatch")
    return {
        "schema_version": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PINNED_OFFICIAL_ALL_BACKBONE_HEADERS_AND_INDEX_SCHEMA_ONLY",
        "repository": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": REVISION,
        "index_sha256": INDEX_SHA256,
        "full_checkpoint_downloaded": False,
        "shard_payload_downloaded": False,
        "backbone_layers": 40,
        "backbone_experts": len(all_groups),
        "expert_tensor_entries": total_expert_tensors,
        "non_expert_tensor_entries": total_non_expert_tensors,
        "bounded_network_response_body_bytes": total_network_body_bytes,
        "shards": shard_results,
        "limits": [
            "Only the pinned index and Safetensors headers were validated; shard payload bytes were not requested or read.",
            "No Rust Safetensors loading of 2.6–7.4 GB shards, GPU/CUDA/V100, inference or performance claim.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = discover()[0] if args.discover_only else crosscheck()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
