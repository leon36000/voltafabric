#!/usr/bin/env python3
"""Observe selected SC117 blob hashes; output is evidence, not authentication."""
import argparse
import hashlib
import json
import os
import re
import stat
import sys

SIZE = 50_292_326_400
INDEX_SHA = "adba4968661f3d59f749a676a87948e29236c81997f649f5526daf71148edc24"
SOURCE_NAME = "Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf"
SOURCE_SIZE = 55_063_448_064
BLOCKS = {18: (256, 98), 21: (256, 110), 22: (256, 82),
          23: (256, 136), 20: (32, 18), 42: (64, 18)}


def run(index_path, pack_path, selections, source_path=None):
    if not 1 <= len(selections) <= 16 or len(set(selections)) != len(selections):
        raise ValueError("select 1 to 16 unique experts")
    with open(index_path, "rb") as f:
        index = f.read(65537)
    if len(index) > 65536 or hashlib.sha256(index).hexdigest() != INDEX_SHA:
        raise ValueError("index digest differs from pinned SC117 index")
    rows = []
    for line in index.decode("utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) not in (8, 9) or not all(re.fullmatch(r"[0-9]+", x) for x in fields[:8]):
            raise ValueError("invalid index row")
        values = [int(x) for x in fields[:8]]
        if values[0] != len(rows):
            raise ValueError("invalid index ordering")
        rows.append(values)
    if len(rows) != 48:
        raise ValueError("invalid layer count")
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(pack_path, flags)
    source_fd = None
    try:
        source_before = None
        if source_path is not None:
            if os.path.basename(source_path) != SOURCE_NAME:
                raise ValueError("unexpected source shard name")
            source_fd = os.open(source_path, flags)
            source_before = os.fstat(source_fd)
            if not stat.S_ISREG(source_before.st_mode) or source_before.st_size != SOURCE_SIZE:
                raise ValueError("unexpected source shard type or size")
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size != SIZE:
            raise ValueError("pack is not an SC117-sized regular file")
        records = []
        for layer, expert in selections:
            if not 0 <= layer < 48 or not 0 <= expert < 512:
                raise ValueError("layer or expert out of range")
            row = rows[layer]
            offset, length = row[3] + expert * row[4], row[4]
            if length < 1 or length > 4 * 1024 * 1024 or offset + length > SIZE:
                raise ValueError("expert span out of bounds")
            digest = hashlib.sha256()
            done = 0
            while done < length:
                chunk = os.pread(fd, min(65536, length - done), offset + done)
                if not chunk:
                    raise ValueError("short expert read")
                digest.update(chunk)
                done += len(chunk)
            record = {"layer": layer, "expert": expert, "offset": offset,
                      "bytes": length, "sha256": digest.hexdigest()}
            if source_fd is not None:
                if len(line_for_layer(index, layer).split()) != 8:
                    raise ValueError("selected layer uses a different source shard")
                try:
                    ge, gb = BLOCKS[row[1]]
                    de, db = BLOCKS[row[2]]
                except KeyError as error:
                    raise ValueError("unsupported native format") from error
                if 2560 % ge or 640 % de:
                    raise ValueError("unsupported block alignment")
                gu = (2560 // ge) * gb * 640
                down = (640 // de) * db * 2560
                if 2 * gu + down != length:
                    raise ValueError("expert geometry mismatch")
                source_digest = hashlib.sha256()
                for start, count in ((row[5], gu), (row[6], gu), (row[7], down)):
                    at = start + expert * count
                    if at + count > SOURCE_SIZE:
                        raise ValueError("source span out of bounds")
                    done = 0
                    while done < count:
                        chunk = os.pread(source_fd, min(65536, count - done), at + done)
                        if not chunk:
                            raise ValueError("short source read")
                        source_digest.update(chunk)
                        done += len(chunk)
                if source_digest.hexdigest() != record["sha256"]:
                    raise ValueError("expert pack and source slices differ")
                record["matches_source_shard_slices"] = True
            records.append(record)
        after = os.fstat(fd)
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if identity(before) != identity(after):
            raise ValueError("pack changed while observing")
        if source_fd is not None and identity(source_before) != identity(os.fstat(source_fd)):
            raise ValueError("source shard changed while observing")
        result = {"kind": "sc117-observation-not-authentication", "index_sha256": INDEX_SHA,
                  "pack": {"device": before.st_dev, "inode": before.st_ino,
                           "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                           "ctime_ns": before.st_ctime_ns}, "experts": records}
        if source_before is not None:
            result["source"] = {"name": SOURCE_NAME, "device": source_before.st_dev,
                                "inode": source_before.st_ino, "bytes": source_before.st_size,
                                "mtime_ns": source_before.st_mtime_ns, "ctime_ns": source_before.st_ctime_ns}
        return result
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(fd)


def line_for_layer(index, layer):
    lines = [line for line in index.decode("utf-8").splitlines() if line and not line.startswith("#")]
    return lines[layer]


def selection(value):
    if not re.fullmatch(r"[0-9]+:[0-9]+", value):
        raise argparse.ArgumentTypeError("use LAYER:EXPERT")
    return tuple(map(int, value.split(":")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--pack", required=True)
    parser.add_argument("--source-shard", help="optional first GGUF shard for exact slice comparison")
    parser.add_argument("--expert", type=selection, action="append", required=True)
    args = parser.parse_args()
    try:
        result = run(args.index, args.pack, args.expert, args.source_shard)
    except (OSError, ValueError, UnicodeError) as error:
        print("observation refused: " + str(error), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
