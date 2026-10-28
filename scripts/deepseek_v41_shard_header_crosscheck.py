#!/usr/bin/env python3
"""Cross-check the pinned DeepSeek Safetensors shard header against its index."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from math import prod
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence/continuation_a97_model_metadata"
HEADER_PATH = EVIDENCE / "model-00003.header.314113f8.json"
INDEX_PATH = EVIDENCE / "model.safetensors.index.314113f8.json"
RECEIPT_PATH = EVIDENCE / "A97_EXPERT0_EXTRACTION_RECEIPT.json"
REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
SHARD = "model-00003-of-00048.safetensors"
SHARD_BYTES = 7_389_759_032
HEADER_SHA256 = "71213f444f9c15a4af6dbcbddc36f7611542884ebb3aadef54de650071357449"
INDEX_SHA256 = "74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8"
EXPERT_NAME = re.compile(
    r"layers\.(0|[1-9][0-9]*)\.ffn\.experts\.(0|[1-9][0-9]*)\.(w1|w2|w3)\.(weight|scale)\Z"
)
EXPECTED_SHAPES = {
    "w1": {"weight": [2304, 2560], "scale": [2304, 160]},
    "w2": {"weight": [5120, 1152], "scale": [5120, 72]},
    "w3": {"weight": [2304, 2560], "scale": [2304, 160]},
}
EXPECTED_DTYPES = {"weight": "I8", "scale": "F8_E8M0"}
DTYPE_BYTES = {
    "I8": 1,
    "F8_E8M0": 1,
    "F8_E4M3": 1,
    "BF16": 2,
    "F32": 4,
}
EXPECTED_PLANES = {
    (projection, plane)
    for projection in ("w1", "w2", "w3")
    for plane in ("weight", "scale")
}


def crosscheck() -> dict[str, object]:
    header_bytes = HEADER_PATH.read_bytes()
    index_bytes = INDEX_PATH.read_bytes()
    header_sha256 = hashlib.sha256(header_bytes).hexdigest()
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    if header_sha256 != HEADER_SHA256 or index_sha256 != INDEX_SHA256:
        raise ValueError("pinned shard header or index digest mismatch")

    extraction = json.loads(RECEIPT_PATH.read_bytes())
    if extraction["source"]["repo"] != "deepseek-ai/DeepSeek-V4.1-Flash":
        raise ValueError("unexpected checkpoint repository")
    if extraction["source"]["revision"] != REVISION:
        raise ValueError("unexpected checkpoint revision")
    if extraction["source"]["shard"] != SHARD:
        raise ValueError("unexpected source shard")
    if extraction["source"]["shard_bytes"] != SHARD_BYTES:
        raise ValueError("unexpected source shard size")

    header = json.loads(header_bytes)
    index = json.loads(index_bytes)
    weight_map = index["weight_map"]
    tensor_names = set(header) - {"__metadata__"}
    indexed_names = {name for name, shard in weight_map.items() if shard == SHARD}
    if tensor_names != indexed_names:
        raise ValueError(
            f"header/index tensor mismatch: {len(tensor_names - indexed_names)} "
            f"unindexed header names, {len(indexed_names - tensor_names)} missing header names"
        )
    if len(tensor_names) != 2334:
        raise ValueError(f"unexpected shard tensor count: {len(tensor_names)}")

    data_start = 8 + len(header_bytes)
    shard_payload_bytes = SHARD_BYTES - data_start
    groups: dict[tuple[int, int], dict[tuple[str, str], dict[str, object]]] = (
        defaultdict(dict)
    )
    max_data_end = 0
    expert_tensor_count = 0
    non_expert_tensor_count = 0
    for name in sorted(tensor_names):
        entry = header[name]
        offsets = entry["data_offsets"]
        shape = entry["shape"]
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ValueError(f"invalid data offsets for {name}")
        start, end = offsets
        dtype = entry["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"unsupported Safetensors dtype in pinned shard: {dtype}")
        expected_bytes = prod(shape) * DTYPE_BYTES[dtype]
        if end - start != expected_bytes:
            raise ValueError(f"shape and byte extent disagree for {name}")
        if end > shard_payload_bytes:
            raise ValueError(f"tensor extends beyond the pinned shard for {name}")
        max_data_end = max(max_data_end, end)
        match = EXPERT_NAME.fullmatch(name)
        if not match:
            if ".ffn.experts." in name:
                raise ValueError(f"unsupported expert tensor key: {name}")
            non_expert_tensor_count += 1
            continue

        layer, expert, projection, plane = match.groups()
        layer_id, expert_id = int(layer), int(expert)
        if str(layer_id) != layer or str(expert_id) != expert:
            raise ValueError(f"noncanonical numeric identity in {name}")
        if entry["dtype"] != EXPECTED_DTYPES[plane]:
            raise ValueError(f"unexpected dtype for {name}: {entry['dtype']}")
        if shape != EXPECTED_SHAPES[projection][plane]:
            raise ValueError(f"unexpected shape for {name}: {shape}")
        group = groups[(layer_id, expert_id)]
        identity = (projection, plane)
        if identity in group:
            raise ValueError(
                f"duplicate expert plane for {layer_id}/{expert_id}: {identity}"
            )
        group[identity] = entry
        expert_tensor_count += 1

    if not groups:
        raise ValueError("pinned shard header contains no recognized experts")
    for identity, planes in groups.items():
        if set(planes) != EXPECTED_PLANES:
            missing = sorted(EXPECTED_PLANES - set(planes))
            extra = sorted(set(planes) - EXPECTED_PLANES)
            raise ValueError(
                f"incomplete expert {identity}: missing={missing}, extra={extra}"
            )
    if (
        len(groups) != 384
        or expert_tensor_count != 2304
        or non_expert_tensor_count != 30
    ):
        raise ValueError(
            "pinned shard expert/non-expert counts changed: "
            f"groups={len(groups)}, expert_tensors={expert_tensor_count}, "
            f"non_expert_tensors={non_expert_tensor_count}"
        )
    if {layer for layer, _ in groups} != {0}:
        raise ValueError("unexpected layer set for the pinned model-00003 shard")
    if {expert for _, expert in groups} != set(range(384)):
        raise ValueError("expert identities are not exactly the canonical range 0..383")

    return {
        "schema_version": 1,
        "observed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_PINNED_OFFICIAL_FULL_HEADER_INDEX_SCHEMA_ONLY",
        "repository": "deepseek-ai/DeepSeek-V4.1-Flash",
        "revision": REVISION,
        "license": "MIT",
        "shard": SHARD,
        "shard_bytes": SHARD_BYTES,
        "full_shard_payload_downloaded": False,
        "full_checkpoint_downloaded": False,
        "header": {
            "path": str(HEADER_PATH.relative_to(ROOT)),
            "sha256": header_sha256,
            "bytes": len(header_bytes),
            "data_start_bytes": data_start,
            "payload_bytes": shard_payload_bytes,
            "max_tensor_data_end": max_data_end,
            "tensor_count": len(tensor_names),
        },
        "index": {
            "path": str(INDEX_PATH.relative_to(ROOT)),
            "sha256": index_sha256,
            "all_header_names_map_to_source_shard": True,
        },
        "expert_contract": {
            "layers": [0],
            "expert_ids": [0, 383],
            "expert_count": len(groups),
            "expert_tensor_count": expert_tensor_count,
            "non_expert_tensor_count": non_expert_tensor_count,
            "planes_per_expert": 6,
            "projection_shapes": EXPECTED_SHAPES,
            "dtypes": EXPECTED_DTYPES,
            "all_experts_have_exactly_six_expected_planes": True,
            "all_ids_use_canonical_decimal_spelling": True,
        },
        "limits": [
            "This verifies only a pinned Safetensors index and JSON header, not the 7.39 GB shard payload.",
            "No Rust Safetensors deserialization of the complete shard, GPU, CUDA, V100, or inference was run.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = crosscheck()
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        if not args.out.is_absolute():
            args.out = (ROOT / args.out).resolve()
        if not args.out.is_relative_to(ROOT):
            raise SystemExit("output path escapes the project root")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output)
        print(args.out.relative_to(ROOT))
    else:
        sys.stdout.write(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
