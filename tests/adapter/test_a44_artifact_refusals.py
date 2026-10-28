"""Each refusal of scripts/resident_artifact.py, driven by an input that violates it alone.

Every test starts from an artifact (or vf_info object) that the reader accepts,
changes exactly one fact so that a single operand of one `if ..: raise` guard is
true while all earlier guards pass, and asserts the exact exception type and the
full anchored message. Disabling that operand therefore either admits the input
or surfaces a different exception/message. Artifacts are written by a tiny local
Safetensors encoder into a private temporary directory, so headers can be broken
field by field; reader-misbehaviour guards use a fake `safe_open`. CPU only.
"""

from __future__ import annotations

import contextlib
import copy
import json
import os
from pathlib import Path
import re
import struct
import sys
import tempfile
import unittest
from unittest import mock

import torch

SCRIPT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SCRIPT_ROOT))
from scripts import resident_artifact as ra  # noqa: E402

PREFIX = "invalid QPN2 resident artifact: "
BASE_SHAPE = [1, 128, 64, 64, 64]
# F32 planes first so that every float tensor starts 4-byte aligned.
LAYOUT = ("gscales13", "gscales2", "codes13", "scales13", "codes2", "scales2")


def exact(message: str) -> str:
    return "^" + re.escape(PREFIX + message) + "$"


def payload(shape):
    e, n13, k13, n2, k2 = shape
    return (
        e * n13 * k13 // 2
        + e * n13 * k13 // 16
        + 4 * e * (n13 // 32)
        + e * n2 * k2 // 2
        + e * n2 * k2 // 16
        + 4 * e * (n2 // 32)
    )


def make_info(shape=BASE_SHAPE, **updates):
    shape = list(shape)
    physical = [{"id": 90 + i, "local": 7 + i} for i in range(shape[0])]
    info = {
        "model_revision": "test-rev",
        "layer": 3,
        "rank": 2,
        "epoch": 9,
        "payload_bytes": payload(shape),
        "physical_experts": physical,
        "resident_order": [p["local"] for p in physical],
        "shape": shape,
    }
    info.update(updates)
    return info


def make_tensors():
    e, n13, k13, n2, k2 = BASE_SHAPE
    return {
        "gscales13": torch.full((e, n13 // 32), 0.25, dtype=torch.float32),
        "gscales2": torch.full((e, n2 // 32), 0.25, dtype=torch.float32),
        "codes13": torch.full((e * n13 * k13 // 2,), 0x22, dtype=torch.uint8),
        "scales13": torch.full((e * n13 * k13 // 16,), 0x38, dtype=torch.uint8),
        "codes2": torch.full((e * n2 * k2 // 2,), 0x22, dtype=torch.uint8),
        "scales2": torch.full((e * n2 * k2 // 16,), 0x38, dtype=torch.uint8),
    }


def tensor_bytes(tensor):
    flat = tensor.contiguous().reshape(-1)
    if tensor.dtype == torch.float32:
        return struct.pack("<%df" % flat.numel(), *flat.tolist())
    return bytes(flat.tolist())


def encode(
    info=None,
    tensors=None,
    *,
    header_mut=None,
    raw_mut=None,
    data_prefix=b"",
    trailing=b"",
):
    """Serialize a Safetensors file; hooks break one field before encoding."""
    info = make_info() if info is None else info
    tensors = make_tensors() if tensors is None else tensors
    header = {
        "__metadata__": {
            "vf_schema": ra.SCHEMA,
            "vf_info": json.dumps(info, separators=(",", ":")),
        }
    }
    data, cursor = [], 0
    for name in LAYOUT:
        blob = tensor_bytes(tensors[name])
        dtype = "F32" if tensors[name].dtype == torch.float32 else "U8"
        header[name] = {
            "dtype": dtype,
            "shape": list(tensors[name].shape),
            "data_offsets": [cursor, cursor + len(blob)],
        }
        data.append(blob)
        cursor += len(blob)
    if header_mut is not None:
        header_mut(header)
    raw = json.dumps(header, separators=(",", ":"))
    if raw_mut is not None:
        raw = raw_mut(raw)
    encoded = raw.encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)
    return (
        struct.pack("<Q", len(encoded))
        + encoded
        + data_prefix
        + b"".join(data)
        + trailing
    )


class FakeReader:
    """Stand-in for safetensors.safe_open returning chosen metadata/tensors."""

    def __init__(self, metadata, tensors):
        self._metadata = metadata
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def metadata(self):
        return self._metadata

    def get_tensor(self, name):
        return self._tensors[name]


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a44-artifact-")
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.path = self.dir / "resident.safetensors"

    def write(self, blob: bytes, name="resident.safetensors") -> Path:
        path = self.dir / name
        path.write_bytes(blob)
        return path

    def refuse(self, message, fn, *args, **kwargs):
        with self.assertRaisesRegex(ValueError, exact(message)):
            fn(*args, **kwargs)

    def refuse_file(self, message, blob, **kwargs):
        self.refuse(message, ra.load_stack, self.write(blob), **kwargs)

    def refuse_info(self, message, info):
        self.refuse(message, ra._validate_info, info)


# ------------------------------------------------------------------ controls


class Controls(Base):
    def test_baseline_file_loads_with_expected_identity(self):
        info = make_info()
        loaded = ra.load_stack(self.write(encode(info)), expected=info)
        self.assertEqual(loaded.metadata, info)
        for name, tensor in make_tensors().items():
            self.assertTrue(torch.equal(loaded.tensors[name], tensor))

    def test_near_miss_info_controls_are_admitted(self):
        for shape in (
            BASE_SHAPE,
            [2, 128, 64, 64, 64],
            [3, 2048, 512, 512, 1024],
            [1, 1024, 2048, 2048, 512],
        ):
            with self.subTest(shape=shape):
                ra._validate_info(make_info(shape))
        ra._validate_info(make_info([0, 0, 0, 0, 0]))
        ra._validate_info(make_info(model_revision="a" * 256))
        ra._validate_info(make_info(layer=(1 << 32) - 1, epoch=(1 << 64) - 1))
        reversed_ = make_info([2, 128, 64, 64, 64])
        reversed_["physical_experts"].reverse()
        reversed_["resident_order"].reverse()
        ra._validate_info(reversed_)

    def test_deepseek_compressed_plane_geometry_is_admitted(self):
        shape = [1, 4608, 5120, 5120, 2304]
        info = make_info(shape)
        self.assertLess(info["payload_bytes"], ra.MAX_PAYLOAD_BYTES)
        validated, planes = ra._validate_info(info)
        self.assertEqual(validated, info)
        self.assertEqual(ra.MAX_PLANE_ELEMENTS, 16_777_216)
        self.assertEqual((4608 // 2) * 5120, 11_796_480)
        self.assertEqual(5120 * 2304, 11_796_480)
        self.assertEqual(
            planes["codes13"], ("U8", [4608 * 5120 // 2], 4608 * 5120 // 2)
        )
        self.assertEqual(planes["codes2"], ("U8", [5120 * 2304 // 2], 5120 * 2304 // 2))

    def test_fake_reader_baseline_is_admitted(self):
        path = self.write(encode())
        meta = {
            "vf_schema": ra.SCHEMA,
            "vf_info": json.dumps(make_info(), separators=(",", ":")),
        }
        with mock.patch(
            "safetensors.safe_open",
            lambda *a, **k: FakeReader(meta, make_tensors()),
        ):
            ra.load_stack(path)


# --------------------------------------------------------------- JSON decode


class JsonRefusals(Base):
    def test_m452_header_and_vf_info_must_be_objects(self):
        def as_list(raw):
            return "[]"

        self.refuse_file("header must be an object", encode(raw_mut=as_list))

        def info_list(header):
            header["__metadata__"]["vf_info"] = "[]"

        self.refuse_file("header must be an object", encode(header_mut=info_list))

    def test_m479_duplicate_json_key(self):
        def duplicate(raw):
            desc = json.loads(raw)["codes13"]
            return raw[:-1] + ',"codes13":' + json.dumps(desc) + "}"

        self.refuse_file("duplicate JSON key", encode(raw_mut=duplicate))


# --------------------------------------------------------------- vf_info rules


class InfoRefusals(Base):
    def test_m453_info_not_dict_with_matching_key_set(self):
        self.refuse_info("metadata fields differ", sorted(ra._INFO_KEYS))

    def test_m454_info_extra_key(self):
        self.refuse_info("metadata fields differ", make_info(extra=1))

    def test_m455_revision_not_string(self):
        self.refuse_info("model_revision is invalid", make_info(model_revision=5))

    def test_m456_revision_blank(self):
        self.refuse_info("model_revision is invalid", make_info(model_revision="  "))

    def test_m457_revision_over_256_bytes(self):
        self.refuse_info(
            "model_revision is invalid", make_info(model_revision="a" * 257)
        )

    def test_m480_layer_and_rank_must_be_u32(self):
        for field in ("layer", "rank"):
            with self.subTest(field=field):
                self.refuse_info(field + " must be u32", make_info(**{field: 1 << 32}))

    def test_m481_epoch_must_be_u64(self):
        self.refuse_info("epoch must be u64", make_info(epoch=-1))
        self.refuse_info("epoch must be u64", make_info(epoch=1 << 64))

    def test_m458_payload_over_128_mib_with_consistent_layout(self):
        # 152 experts of a valid plane layout need exactly this many bytes, so
        # the declared payload is self-consistent and only the 128 MiB cap fails.
        shape = [152, 2048, 512, 512, 1024]
        self.assertGreater(payload(shape), ra.MAX_PAYLOAD_BYTES)
        self.refuse_info("payload exceeds 128 MiB", make_info(shape))

    def test_m459_shape_not_list(self):
        info = make_info()
        info["shape"] = tuple(info["shape"])
        self.refuse_info("shape must be five unsigned integers", info)

    def test_m460_shape_length(self):
        for shape in ([1, 128, 64, 64, 64, 0], [1, 128, 64, 64]):
            with self.subTest(shape=shape):
                info = make_info()
                info["shape"] = shape
                self.refuse_info("shape must be five unsigned integers", info)

    def test_m461_shape_entry_not_uint(self):
        info = make_info()
        info["shape"] = [True, 128, 64, 64, 64]
        self.refuse_info("shape must be five unsigned integers", info)

    def test_m462_physical_not_list(self):
        info = make_info()
        info["physical_experts"] = tuple(info["physical_experts"])
        self.refuse_info("expert identity arrays are invalid", info)

    def test_m463_order_not_list(self):
        info = make_info()
        info["resident_order"] = tuple(info["resident_order"])
        self.refuse_info("expert identity arrays are invalid", info)

    def test_m464_physical_longer_than_max_experts(self):
        info = make_info()
        info["physical_experts"] = [
            {"id": i, "local": i} for i in range(ra.MAX_EXPERTS + 1)
        ]
        self.refuse_info("expert identity arrays are invalid", info)

    def test_m482_to_m485_empty_rank_sentinel(self):
        message = "empty rank must use the all-zero sentinel"
        empty = make_info([0, 0, 0, 0, 0])
        cases = {
            "shape": {"shape": [0, 128, 64, 64, 64]},
            "physical": {"physical_experts": [{"id": 1, "local": 1}]},
            "order": {"resident_order": [1]},
            "payload": {"payload_bytes": 5},
        }
        for label, update in cases.items():
            with self.subTest(label=label):
                info = copy.deepcopy(empty)
                info.update(update)
                self.refuse_info(message, info)

    def test_m487_physical_count_differs(self):
        info = make_info()
        info["physical_experts"] = []
        self.refuse_info("expert count differs from physical identity", info)

    def test_m488_order_count_differs(self):
        info = make_info()
        info["resident_order"] = [7, 7]
        self.refuse_info("expert count differs from physical identity", info)

    def test_m489_dimension_zero(self):
        info = make_info()
        info["shape"] = [1, 0, 64, 64, 64]
        self.refuse_info("nonempty dimensions outside bounds", info)

    def test_m490_to_m493_tile_alignment(self):
        message = "dimensions are not QPN2 tile aligned"
        for label, shape in (
            ("n13", [1, 129, 64, 64, 64]),
            ("k13", [1, 128, 96, 96, 64]),
            ("n2", [1, 128, 64, 48, 64]),
            ("k2", [1, 64, 64, 64, 32]),
        ):
            with self.subTest(label=label):
                self.refuse_info(message, make_info(shape))

    def test_m494_m495_swiglu_shape(self):
        message = "dimensions do not form a uniform SwiGLU"
        for label, shape in (
            ("n13/2 != k2", [1, 256, 64, 64, 64]),
            ("n2 != k13", [1, 128, 128, 64, 64]),
        ):
            with self.subTest(label=label):
                self.refuse_info(message, make_info(shape))

    def test_m496_plane_element_limit(self):
        shape = [1, 8192, 4160, 4160, 4096]
        self.assertLess(payload(shape), ra.MAX_PAYLOAD_BYTES)
        self.refuse_info(
            "QPN2 expert plane exceeds 16,777,216 elements", make_info(shape)
        )

    def test_m519_physical_entry_not_dict(self):
        info = make_info()
        info["physical_experts"] = [["id", "local"]]
        self.refuse_info("physical expert entry fields differ", info)

    def test_m520_physical_entry_extra_field(self):
        info = make_info()
        info["physical_experts"][0]["x"] = 1
        self.refuse_info("physical expert entry fields differ", info)

    def test_m521_physical_id_not_u32(self):
        info = make_info()
        info["physical_experts"][0]["id"] = 1 << 32
        self.refuse_info("physical expert IDs must be u32", info)

    def test_m522_physical_local_not_u32(self):
        info = make_info()
        info["physical_experts"][0]["local"] = 1 << 32
        info["resident_order"] = [1 << 32]
        self.refuse_info("physical expert IDs must be u32", info)

    def test_m497_duplicate_physical_ids(self):
        info = make_info([2, 128, 64, 64, 64])
        info["physical_experts"][1]["id"] = info["physical_experts"][0]["id"]
        self.refuse_info("physical expert IDs or locals are duplicated", info)

    def test_m498_duplicate_physical_locals(self):
        info = make_info([2, 128, 64, 64, 64])
        info["physical_experts"][1]["local"] = 7
        info["resident_order"] = [7, 7]
        self.refuse_info("physical expert IDs or locals are duplicated", info)

    def test_m499_order_entry_not_u32(self):
        info = make_info()
        info["physical_experts"][0]["local"] = 1
        info["resident_order"] = [True]
        self.refuse_info("resident_order entries must be u32", info)

    def test_m501_order_differs_from_physical_locals(self):
        info = make_info([2, 128, 64, 64, 64])
        info["resident_order"].reverse()
        self.refuse_info("resident_order must exactly match physical local order", info)

    def test_m502_payload_differs_from_layout(self):
        info = make_info()
        info["payload_bytes"] += 1
        self.refuse_info("payload byte count differs from plane layout", info)


# ------------------------------------------------------------ file framing


class FramingRefusals(Base):
    def test_m466_file_shorter_than_length_prefix(self):
        for blob in (b"", b"\x01\x00\x00\x00"):
            with self.subTest(size=len(blob)):
                self.refuse_file("file size outside bounded range", blob)

    def test_m467_file_larger_than_bound(self):
        path = self.dir / "huge.safetensors"
        with open(path, "wb") as handle:
            handle.truncate(ra.MAX_PAYLOAD_BYTES + ra.MAX_HEADER_BYTES + 9)
        self.refuse("file size outside bounded range", ra.load_stack, path)

    def test_m468_short_prefix_read(self):
        path = self.write(b"\x01\x00\x00\x00")
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        self.refuse("truncated header length", ra._open_header, fd, 100)

    def test_m469_header_length_zero_or_over_8_mib(self):
        for length in (0, ra.MAX_HEADER_BYTES + 1):
            with self.subTest(length=length):
                self.refuse_file(
                    "header exceeds 8 MiB or is empty",
                    struct.pack("<Q", length) + b"{}      ",
                )

    def test_m470_header_longer_than_file(self):
        blob = struct.pack("<Q", 100) + b"{" + b" " * 41
        self.refuse_file("truncated header", blob)

    def test_m471_short_header_body_read(self):
        path = self.write(struct.pack("<Q", 100) + b"{" * 10)
        fd = os.open(path, os.O_RDONLY)
        self.addCleanup(os.close, fd)
        self.refuse("truncated header body", ra._open_header, fd, 200)

    def test_m478_trailing_and_truncated_payload(self):
        self.refuse_file(
            "file has truncated or trailing payload bytes", encode(trailing=b"\0")
        )
        self.refuse_file("file has truncated or trailing payload bytes", encode()[:-1])

    def test_m517_directory_is_not_regular(self):
        self.refuse("artifact is not a regular file", ra.load_stack, self.dir)


# ---------------------------------------------------------- header structure


class HeaderRefusals(Base):
    def test_m472_tensor_names_differ(self):
        def extra(header):
            header["extra"] = {"dtype": "U8", "shape": [0], "data_offsets": [0, 0]}

        def missing(header):
            del header["codes2"]

        for label, mut in (("extra", extra), ("missing", missing)):
            with self.subTest(label=label):
                self.refuse_file("tensor names differ", encode(header_mut=mut))

    def test_m473_metadata_not_dict(self):
        def as_list(header):
            header["__metadata__"] = ["vf_schema", "vf_info"]

        self.refuse_file(
            "Safetensors metadata fields differ", encode(header_mut=as_list)
        )

    def test_m474_metadata_extra_key(self):
        def extra(header):
            header["__metadata__"]["x"] = "y"

        self.refuse_file("Safetensors metadata fields differ", encode(header_mut=extra))

    def test_m475_unsupported_schema(self):
        def other(header):
            header["__metadata__"]["vf_schema"] = "QPN2_RESIDENT_V2"

        self.refuse_file("unsupported schema", encode(header_mut=other))

    def test_m477_vf_info_not_string(self):
        def as_object(header):
            header["__metadata__"]["vf_info"] = make_info()

        self.refuse_file("vf_info must be a JSON string", encode(header_mut=as_object))

    def test_m504_descriptor_not_dict(self):
        def as_list(header):
            header["codes13"] = ["dtype", "shape", "data_offsets"]

        self.refuse_file("tensor descriptor fields differ", encode(header_mut=as_list))

    def test_m505_descriptor_extra_field(self):
        def extra(header):
            header["codes13"]["x"] = 1

        self.refuse_file("tensor descriptor fields differ", encode(header_mut=extra))

    def test_m506_dtype_differs(self):
        def int8(header):
            header["codes13"]["dtype"] = "I8"

        self.refuse_file("codes13 dtype differs", encode(header_mut=int8))

    def test_m508_descriptor_shape_not_list(self):
        def scalar(header):
            header["codes13"]["shape"] = header["codes13"]["shape"][0]

        self.refuse_file("codes13 shape differs", encode(header_mut=scalar))

    def test_m509_descriptor_shape_entry_not_uint(self):
        for value in (True, 1.0):
            with self.subTest(value=value):

                def change(header, value=value):
                    header["gscales13"]["shape"][0] = value

                self.refuse_file("gscales13 shape differs", encode(header_mut=change))

    def test_m510_descriptor_shape_differs(self):
        def reshape(header):
            header["codes13"]["shape"] = [2048, 2]

        self.refuse_file("codes13 shape differs", encode(header_mut=reshape))

    def test_m511_offsets_not_list(self):
        def scalar(header):
            header["codes13"]["data_offsets"] = 5

        self.refuse_file("codes13 offsets are invalid", encode(header_mut=scalar))

    def test_m512_offsets_length(self):
        def three(header):
            start, end = header["codes13"]["data_offsets"]
            header["codes13"]["data_offsets"] = [start, end, end]

        self.refuse_file("codes13 offsets are invalid", encode(header_mut=three))

    def test_m513_offset_entry_not_uint(self):
        def floats(header):
            start, end = header["codes13"]["data_offsets"]
            header["codes13"]["data_offsets"] = [float(start), float(end)]

        self.refuse_file("codes13 offsets are invalid", encode(header_mut=floats))

    def test_m515_span_differs_from_shape(self):
        def short(header):
            header["codes13"]["data_offsets"][1] -= 1

        self.refuse_file(
            "codes13 byte span differs from shape", encode(header_mut=short)
        )

    def test_m516_gap_before_first_span(self):
        def shift(header):
            for name in LAYOUT:
                header[name]["data_offsets"] = [
                    v + 1 for v in header[name]["data_offsets"]
                ]

        self.refuse_file(
            "tensor spans overlap or leave gaps",
            encode(header_mut=shift, data_prefix=b"\0"),
        )


# ------------------------------------------------------ load_stack post-checks


class LoadRefusals(Base):
    def test_m523_stale_expected_identity(self):
        info = make_info()
        stale = make_info(epoch=info["epoch"] + 1)
        self.refuse_file(
            "artifact identity differs from expected metadata",
            encode(info),
            expected=stale,
        )

    def test_m518_file_changed_while_reading(self):
        from safetensors import safe_open as real_safe_open

        path = self.write(encode())
        before = os.stat(path)

        def touching(*args, **kwargs):
            later = before.st_mtime_ns + 10**9
            os.utime(path, ns=(later, later))
            return real_safe_open(*args, **kwargs)

        with mock.patch("safetensors.safe_open", touching):
            self.refuse(
                "source file changed while reading; quiescence required",
                ra.load_stack,
                path,
            )

    def test_m527_scales13_not_unsigned_finite(self):
        tensors = make_tensors()
        tensors["scales13"][5] = 0x7F
        self.refuse_file("scales must be unsigned finite E4M3", encode(tensors=tensors))

    def test_m528_scales2_not_unsigned_finite(self):
        tensors = make_tensors()
        tensors["scales2"][5] = 0x7F
        self.refuse_file("scales must be unsigned finite E4M3", encode(tensors=tensors))


class ReaderRefusals(Base):
    """Guards against a misbehaving Safetensors reader, driven by a fake one."""

    def load_with_reader(self, message, metadata=None, **tensor_updates):
        path = self.write(encode())
        if metadata is None:
            metadata = {
                "vf_schema": ra.SCHEMA,
                "vf_info": json.dumps(make_info(), separators=(",", ":")),
            }
        tensors = make_tensors()
        tensors.update(tensor_updates)
        with contextlib.ExitStack() as stack:
            stack.enter_context(
                mock.patch(
                    "safetensors.safe_open",
                    lambda *a, **k: FakeReader(metadata, tensors),
                )
            )
            self.refuse(message, ra.load_stack, path)

    def test_m529_reader_metadata_differs(self):
        self.load_with_reader(
            "parsed metadata differs from Safetensors metadata", metadata={}
        )

    def test_m530_tensor_off_cpu(self):
        self.load_with_reader(
            "tensor unexpectedly left CPU",
            codes13=torch.empty((4096,), dtype=torch.uint8, device="meta"),
        )

    def test_m524_loaded_dtype_differs(self):
        self.load_with_reader(
            "codes13 loaded tensor dtype/shape/layout differs",
            codes13=torch.full((4096,), 0x22, dtype=torch.int8),
        )

    def test_m525_loaded_shape_differs(self):
        self.load_with_reader(
            "codes13 loaded tensor dtype/shape/layout differs",
            codes13=torch.full((2048, 2), 0x22, dtype=torch.uint8),
        )


if __name__ == "__main__":
    unittest.main()
