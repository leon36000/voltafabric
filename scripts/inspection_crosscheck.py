#!/usr/bin/env python3
"""Exact legacy snapshot and compact inspection oracle, CPU only."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_SHA = "2599d26d52eaf1ca6758492fb35154c441a643dac49b1256c97b9a8f52382bb4"
GOLDEN_BYTES = 22308553


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    args = parser.parse_args()
    request = ROOT / "fixtures/resident_import_qwen_expert0.json"
    with tempfile.TemporaryFile() as output:
        result = subprocess.run(
            [str(args.binary), "resident-import", str(request)],
            cwd=ROOT,
            stdout=output,
            stderr=subprocess.PIPE,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        length = output.tell()
        output.seek(0)
        digest = hashlib.sha256()
        while block := output.read(65536):
            digest.update(block)
        assert length == GOLDEN_BYTES, length
        assert digest.hexdigest() == GOLDEN_SHA, digest.hexdigest()
    result = subprocess.run(
        [str(args.binary), "resident-inspect", str(request)],
        cwd=ROOT,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout) < 4096
    actual = json.loads(result.stdout)
    assert actual["ok"] and actual["schema_version"] == 1
    summary = actual["result"]
    assert summary["scope"] == "CPU_COMPRESSED_INSPECTION_ONLY"
    assert summary["cuda_executed"] is False
    assert summary["payload_bytes"] == 1769856
    assert summary["shape"] == [1, 1024, 2048, 2048, 512]
    expected = dict(
        codes13=1048576,
        codes2=524288,
        scales13=131072,
        scales2=65536,
        gscales13=32,
        gscales2=64,
    )
    assert summary["plane_elements"] == expected
    assert not set(expected).intersection(summary), summary
    # Counts come from compressed U8 planes and exact direct F32 factors.
    assert (
        sum(expected[n] * (4 if n.startswith("gscales") else 1) for n in expected)
        == summary["payload_bytes"]
    )
    assert (
        summary["file_bytes"]
        == (ROOT / json.loads(request.read_text())["path"]).stat().st_size
    )
    print(
        json.dumps(
            dict(
                status="PASS",
                legacy_snapshot_bytes=length,
                legacy_snapshot_sha256=GOLDEN_SHA,
                summary_bytes=len(result.stdout),
                cuda_executed=False,
                scope="CPU_JSON_COMPATIBILITY_ONLY",
            )
        )
    )


if __name__ == "__main__":
    main()
