from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import struct
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/deepseek_v41_extract_expert_fixture.py"
SPEC = importlib.util.spec_from_file_location("a106_expert383_extractor", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(extractor)


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, *, status: int, content_range: str) -> None:
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Range": content_range}
        self.read_sizes: list[int] = []

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def getcode(self) -> int:
        return self.status

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return super().read(size)


class A106Expert383ExtractorTests(unittest.TestCase):
    def test_pinned_ranges_are_exactly_six_expert383_tensors(self) -> None:
        header_bytes, specs = extractor.pinned_ranges()

        self.assertEqual(len(header_bytes), 258_136)
        self.assertEqual(len(specs), 6)
        self.assertEqual(
            [spec["name"].rsplit(".", 1)[-1] for spec in specs],
            ["weight", "scale", "weight", "scale", "weight", "scale"],
        )
        self.assertEqual(
            [spec["range_start"] for spec in specs],
            [
                6_186_518_072,
                357_744_440,
                6_192_416_312,
                358_113_080,
                6_198_314_552,
                358_481_720,
            ],
        )
        self.assertEqual(sum(spec["expected_bytes"] for spec in specs), 18_800_640)
        for spec in specs:
            self.assertEqual(
                spec["range_end"] - spec["range_start"] + 1,
                spec["expected_bytes"],
            )
            self.assertIn(".experts.383.", spec["name"])

    def test_range_reader_accepts_only_exact_206_response(self) -> None:
        spec = {
            "range_start": 10,
            "range_end": 12,
            "expected_bytes": 3,
        }
        response = FakeResponse(
            b"abc", status=206, content_range=f"bytes 10-12/{extractor.SHARD_BYTES}"
        )
        with patch.object(
            extractor,
            "open_range_request",
            return_value=(response, []),
        ) as open_range:
            body, content_range, redirects = extractor.read_range(spec)

        self.assertEqual(body, b"abc")
        self.assertEqual(content_range, f"bytes 10-12/{extractor.SHARD_BYTES}")
        self.assertEqual(redirects, [])
        self.assertEqual(open_range.call_count, 1)
        request = open_range.call_args.args[0]
        self.assertEqual(request.get_header("Range"), "bytes=10-12")
        self.assertEqual(response.read_sizes, [4])

    def test_invalid_or_overlong_range_is_bounded_and_never_retried(self) -> None:
        spec = {"range_start": 10, "range_end": 12, "expected_bytes": 3}
        bad_response = FakeResponse(
            b"a" * 100,
            status=200,
            content_range=f"bytes 10-12/{extractor.SHARD_BYTES}",
        )
        with patch.object(
            extractor,
            "open_range_request",
            return_value=(bad_response, []),
        ) as open_range:
            with self.assertRaises(extractor.RangeReadError) as caught:
                extractor.read_range(spec)
        self.assertEqual(open_range.call_count, 1)
        self.assertEqual(caught.exception.body_bytes_read, 0)
        self.assertEqual(bad_response.read_sizes, [])

        overlong = FakeResponse(
            b"abcd" + b"z" * 100,
            status=206,
            content_range=f"bytes 10-12/{extractor.SHARD_BYTES}",
        )
        with patch.object(
            extractor,
            "open_range_request",
            return_value=(overlong, []),
        ) as open_range:
            with self.assertRaises(extractor.RangeReadError) as caught:
                extractor.read_range(spec)
        self.assertEqual(open_range.call_count, 1)
        self.assertEqual(caught.exception.body_bytes_read, 4)
        self.assertEqual(overlong.read_sizes, [4])

    def test_fixture_repackages_bytes_without_changing_payloads(self) -> None:
        specs = [
            {
                "name": "layers.0.ffn.experts.383.w1.weight",
                "dtype": "I8",
                "shape": [1, 2],
                "expected_bytes": 2,
            },
            {
                "name": "layers.0.ffn.experts.383.w1.scale",
                "dtype": "F8_E8M0",
                "shape": [1, 2],
                "expected_bytes": 2,
            },
        ]
        payloads = {specs[0]["name"]: b"\x12\x34", specs[1]["name"]: b"\x7f\x80"}
        with patch.object(extractor, "EXPECTED_PAYLOAD_BYTES", 4):
            fixture = extractor.make_safetensors_fixture(specs, payloads)

        header_size = struct.unpack("<Q", fixture[:8])[0]
        header_end = 8 + header_size
        header = json.loads(fixture[8:header_end])
        self.assertEqual(header[specs[0]["name"]]["data_offsets"], [0, 2])
        self.assertEqual(header[specs[1]["name"]]["data_offsets"], [2, 4])
        self.assertEqual(fixture[header_end:], b"\x12\x34\x7f\x80")


if __name__ == "__main__":
    unittest.main()
