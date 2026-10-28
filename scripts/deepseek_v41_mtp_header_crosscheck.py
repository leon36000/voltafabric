#!/usr/bin/env python3
"""Independently compare the pinned MTP shard headers with the official index."""

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
EVIDENCE = ROOT / "evidence/continuation_a104_mtp_namespace"
INDEX_PATH = (
    ROOT
    / "evidence/continuation_a97_model_metadata/model.safetensors.index.314113f8.json"
)
FETCH_PATH = EVIDENCE / "A104_MTP_HEADER_FETCH_RECEIPT.json"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
EXPERT_KEY = re.compile(
    r"mtp\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w1|w2|w3)\.(weight|scale)\Z"
)
SHARDS = {
    "model-00044-of-00048.safetensors": {
        "sha256": "bcfa221d7a325e0a910e262d782c0e97bd5e5b7cbde6c9b480bb88ab7c23a36e",
        "layer": 0,
        "tensor_count": 801,
        "non_expert_count": 33,
        "file_bytes": 2_652_728_736,
    },
    "model-00045-of-00048.safetensors": {
        "sha256": "c8f2cad66c6387793b335cbc10231661b61f2ec7a8c814bccb8d6ae418200675",
        "layer": 1,
        "tensor_count": 798,
        "non_expert_count": 30,
        "file_bytes": 2_573_998_176,
    },
    "model-00046-of-00048.safetensors": {
        "sha256": "1baad4e59ff44a15727f0c294d015da70a90502fd5d1e844c1c0939a8194793e",
        "layer": 2,
        "tensor_count": 802,
        "non_expert_count": 34,
        "file_bytes": 2_706_402_896,
    },
}
DTYPE_BYTES = {"I8": 1, "F8_E8M0": 1, "F8_E4M3": 1, "BF16": 2, "F32": 4}
EXPECTED_SHAPES = {
    "w1": {"weight": [2304, 2560], "scale": [2304, 160]},
    "w2": {"weight": [5120, 1152], "scale": [5120, 72]},
    "w3": {"weight": [2304, 2560], "scale": [2304, 160]},
}
EXPECTED_DTYPES = {"weight": "I8", "scale": "F8_E8M0"}
EXPECTED_PLANES = {
    (projection, plane)
    for projection in ("w1", "w2", "w3")
    for plane in ("weight", "scale")
}


def crosscheck() -> dict[str, object]:
    index_bytes = INDEX_PATH.read_bytes()
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if index_sha256 != INDEX_SHA256:
        raise ValueError("pinned official Safetensors index digest mismatch")
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    fetch = json.loads(FETCH_PATH.read_bytes())
    if fetch["repository"] != "deepseek-ai/DeepSeek-V4.1-Flash":
        raise ValueError("unexpected checkpoint repository in bounded fetch receipt")
    if fetch["revision"] != REVISION or fetch["index_sha256"] != INDEX_SHA256:
        raise ValueError(
            "bounded MTP header fetch is not tied to the pinned revision/index"
        )
    if fetch["payload_downloaded"] or fetch["full_checkpoint_downloaded"]:
        raise ValueError("fetch receipt claims checkpoint payload was downloaded")
    if [item["file"] for item in fetch["shards"]] != list(SHARDS):
        raise ValueError(
            "bounded fetch receipt does not contain the three pinned MTP shards"
        )

    all_groups: dict[tuple[int, int], dict[tuple[str, str], dict[str, object]]] = (
        defaultdict(dict)
    )
    shard_receipts = []
    total_body_bytes = 0
    total_expert_tensors = 0
    total_non_expert_tensors = 0
    for fetched in fetch["shards"]:
        shard = fetched["file"]
        expected = SHARDS[shard]
        if (
            fetched["http_status"] != 206
            or fetched["all_experts_exactly_six_planes"] is not True
            or fetched["file_bytes"] != expected["file_bytes"]
            or fetched["range_0_7"] != f"bytes 0-7/{expected['file_bytes']}"
        ):
            raise ValueError(
                f"{shard}: bounded Range receipt is not a complete HTTP 206 header read"
            )
        expected_path = EVIDENCE / f"{shard}.header.{REVISION[:8]}.json"
        if fetched["header_path"] != str(expected_path.relative_to(ROOT)):
            raise ValueError(f"{shard}: unexpected bounded header path")
        header_path = expected_path.resolve(strict=True)
        if not header_path.is_relative_to(ROOT):
            raise ValueError(f"{shard}: header fixture path escapes the project root")
        header_bytes = header_path.read_bytes()
        header_sha256 = hashlib.sha256(header_bytes).hexdigest()
        if (
            header_sha256 != expected["sha256"]
            or header_sha256 != fetched["header_sha256"]
            or len(header_bytes) != fetched["header_bytes"]
            or len(header_bytes) > fetch["max_header_bytes"]
            or fetched["range_header"]
            != f"bytes 8-{7 + len(header_bytes)}/{expected['file_bytes']}"
        ):
            raise ValueError(f"{shard}: official header digest mismatch")
        header = json.loads(header_bytes)
        names = set(header) - {"__metadata__"}
        indexed_names = {name for name, source in weight_map.items() if source == shard}
        if names != indexed_names:
            raise ValueError(
                f"{shard}: header/index name mismatch: header-only={len(names - indexed_names)}, "
                f"index-only={len(indexed_names - names)}"
            )
        if (
            len(names) != expected["tensor_count"]
            or len(names) != fetched["header_tensor_count"]
        ):
            raise ValueError(f"{shard}: unexpected header tensor count")

        data_start = 8 + len(header_bytes)
        payload_bytes = fetched["file_bytes"] - data_start
        max_data_end = 0
        expert_count = 0
        non_expert_count = 0
        groups: dict[tuple[int, int], set[tuple[str, str]]] = defaultdict(set)
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
                raise ValueError(f"{shard}: malformed shape/offset metadata for {name}")
            start, end = offsets
            if start < 0 or end < start:
                raise ValueError(f"{shard}: invalid byte offsets for {name}")
            dtype = entry["dtype"]
            if (
                dtype not in DTYPE_BYTES
                or end - start != prod(shape) * DTYPE_BYTES[dtype]
            ):
                raise ValueError(
                    f"{shard}: dtype/shape byte extent mismatch for {name}"
                )
            if end > payload_bytes:
                raise ValueError(
                    f"{shard}: tensor extends beyond shard payload for {name}"
                )
            max_data_end = max(max_data_end, end)

            match = EXPERT_KEY.fullmatch(name)
            if match is None:
                if ".ffn.experts." in name:
                    raise ValueError(f"{shard}: unrecognized expert identity {name}")
                non_expert_count += 1
                continue
            layer_text, expert_text, projection, plane = match.groups()
            layer, expert = int(layer_text), int(expert_text)
            if (
                layer != expected["layer"]
                or str(layer) != layer_text
                or str(expert) != expert_text
            ):
                raise ValueError(
                    f"{shard}: noncanonical or unexpected MTP identity {name}"
                )
            if entry["dtype"] != EXPECTED_DTYPES[plane]:
                raise ValueError(f"{shard}: unexpected MXFP4 dtype for {name}")
            if shape != EXPECTED_SHAPES[projection][plane]:
                raise ValueError(f"{shard}: unexpected MXFP4 shape for {name}")
            group = groups[(layer, expert)]
            identity = (projection, plane)
            if identity in group:
                raise ValueError(
                    f"{shard}: duplicate expert plane {identity} for {layer}/{expert}"
                )
            group.add(identity)
            all_groups[(layer, expert)].setdefault(identity, entry)
            expert_count += 1

        if not groups or any(planes != EXPECTED_PLANES for planes in groups.values()):
            raise ValueError(f"{shard}: incomplete MTP expert group")
        if (
            len(groups) != 128
            or {expert for _, expert in groups} != set(range(128))
            or expert_count != 768
            or non_expert_count != expected["non_expert_count"]
        ):
            raise ValueError(f"{shard}: expert or non-expert coverage changed")
        total_body_bytes += 8 + len(header_bytes)
        total_expert_tensors += expert_count
        total_non_expert_tensors += non_expert_count
        shard_receipts.append(
            {
                "file": shard,
                "file_bytes": fetched["file_bytes"],
                "header_bytes": len(header_bytes),
                "header_sha256": header_sha256,
                "header_tensor_count": len(names),
                "index_name_set_matches": True,
                "mtp_layer": layer,
                "mtp_expert_ids": [0, 127],
                "mtp_experts": len(groups),
                "expert_tensors": expert_count,
                "non_expert_tensors": non_expert_count,
                "max_tensor_data_end": max_data_end,
                "payload_bytes": payload_bytes,
                "all_expert_groups_complete": True,
            }
        )

    if (
        total_body_bytes != fetch["network_payload_bytes_read_from_response_bodies"]
        or total_body_bytes != 255_176
        or len(all_groups) != 384
        or total_expert_tensors != 2_304
        or total_non_expert_tensors != 97
    ):
        raise ValueError("aggregate pinned MTP coverage or bounded read bytes changed")
    indexed_mtp = {
        name
        for name in weight_map
        if name.startswith("mtp.") and ".ffn.experts." in name
    }
    if len(indexed_mtp) != 2_304:
        raise ValueError(
            "official index no longer contains exactly 2,304 MTP expert planes"
        )

    return {
        "schema_version": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PINNED_OFFICIAL_MTP_HEADERS_AND_INDEX_SCHEMA_ONLY",
        "repository": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": REVISION,
        "index_sha256": index_sha256,
        "full_checkpoint_downloaded": False,
        "shard_payload_downloaded": False,
        "network_range_response_body_bytes": total_body_bytes,
        "mtp_expert_count": len(all_groups),
        "mtp_expert_tensor_count": total_expert_tensors,
        "non_expert_tensor_count": total_non_expert_tensors,
        "mtp_layers": [0, 1, 2],
        "experts_per_layer": 128,
        "shards": shard_receipts,
        "limits": [
            "Validated only the pinned Safetensors index and JSON headers; no expert weight payload was fetched or parsed.",
            "No GPU, CUDA, V100, inference, performance, or model-quality claim.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = crosscheck()
    encoded = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
