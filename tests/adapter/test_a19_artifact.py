"""CPU-only tests for strict compressed resident Safetensors loading."""

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from scripts.resident_artifact import load_stack

RUN_TMP = Path(
    os.environ.get(
        "VF_A19_TEST_TMP",
        os.environ.get(
            "TMPDIR", str(Path(__file__).resolve().parents[2] / "evidence/adapter-temp")
        ),
    )
)


def fixture(e=1, *, globals13=None, scales13=None, info_updates=None):
    shape = [e, 128 if e else 0, 64 if e else 0, 64 if e else 0, 64 if e else 0]
    physical = (
        [{"id": 90, "local": 7}]
        if e == 1
        else [{"id": 90, "local": 7}, {"id": 11, "local": 2}]
        if e == 2
        else []
    )
    order = [entry["local"] for entry in physical]
    e0, n13, k13, n2, k2 = shape
    info = {
        "model_revision": "test-rev",
        "layer": 3,
        "rank": 2,
        "epoch": 9,
        "payload_bytes": e0 * n13 * k13 // 2
        + e0 * n13 * k13 // 16
        + 4 * e0 * (n13 // 32)
        + e0 * n2 * k2 // 2
        + e0 * n2 * k2 // 16
        + 4 * e0 * (n2 // 32),
        "physical_experts": physical,
        "resident_order": order,
        "shape": shape,
    }
    if info_updates:
        info.update(info_updates)
    gs13 = globals13 if globals13 is not None else [0.25] * (e * (n13 // 32))
    gs2 = [0.25] * (e * (n2 // 32))
    tensors = {
        "codes13": torch.full((e * n13 * k13 // 2,), 0x22, dtype=torch.uint8),
        "scales13": torch.full((e * n13 * k13 // 16,), 0x38, dtype=torch.uint8)
        if scales13 is None
        else torch.tensor(scales13, dtype=torch.uint8)
        .expand(e * n13 * k13 // 16)
        .clone(),
        "gscales13": torch.tensor(gs13, dtype=torch.float32).reshape(e, n13 // 32),
        "codes2": torch.full((e * n2 * k2 // 2,), 0x22, dtype=torch.uint8),
        "scales2": torch.full((e * n2 * k2 // 16,), 0x38, dtype=torch.uint8),
        "gscales2": torch.tensor(gs2, dtype=torch.float32).reshape(e, n2 // 32),
    }
    return info, tensors


def save(path, info, tensors):
    save_file(
        tensors,
        str(path),
        metadata={
            "vf_schema": "QPN2_RESIDENT_V1",
            "vf_info": json.dumps(info, separators=(",", ":")),
        },
    )


def replace_header(path, transform):
    data = path.read_bytes()
    size = struct.unpack("<Q", data[:8])[0]
    raw = data[8 : 8 + size].decode("utf-8").rstrip(" ")
    new = transform(raw)
    header = new.encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + data[8 + size :])


class ResidentArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        RUN_TMP.mkdir(parents=True, exist_ok=True)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="a19-", dir=RUN_TMP)
        self.path = Path(self.temp.name) / "resident.safetensors"

    def tearDown(self):
        self.temp.cleanup()

    def test_temporary_files_stay_in_configured_project_tmpdir(self):
        configured = os.environ.get("VF_A19_TEST_TMP") or os.environ.get("TMPDIR")
        if configured:
            self.assertTrue(
                self.path.resolve().is_relative_to(Path(configured).resolve())
            )

    def test_valid_preserves_sparse_ids_and_reverse_physical_order(self):
        info, tensors = fixture(2)
        info["physical_experts"] = [{"id": 90, "local": 7}, {"id": 11, "local": 2}]
        info["resident_order"] = [7, 2]
        save(self.path, info, tensors)
        loaded = load_stack(self.path, expected=info)
        self.assertEqual(loaded.metadata, info)
        self.assertEqual(loaded.metadata["physical_experts"], info["physical_experts"])
        self.assertEqual(loaded.metadata["resident_order"], [7, 2])
        for name in tensors:
            self.assertTrue(torch.equal(loaded.tensors[name], tensors[name]))
            if name.startswith("gscales"):
                self.assertTrue(
                    torch.equal(
                        loaded.tensors[name].view(torch.int32),
                        tensors[name].view(torch.int32),
                    )
                )
            self.assertEqual(loaded.tensors[name].device.type, "cpu")
            self.assertTrue(loaded.tensors[name].is_contiguous())

    def test_empty_rank_uses_exact_zero_sentinel(self):
        info, tensors = fixture(0)
        save(self.path, info, tensors)
        loaded = load_stack(self.path, expected=info)
        self.assertEqual(loaded.metadata["shape"], [0, 0, 0, 0, 0])
        self.assertEqual(loaded.tensors["gscales13"].shape, (0, 0))
        self.assertEqual(sum(t.numel() for t in loaded.tensors.values()), 0)

    def test_stale_expected_identity_rejected(self):
        info, tensors = fixture()
        save(self.path, info, tensors)
        stale = json.loads(json.dumps(info))
        for field in ("epoch", "physical_experts"):
            stale = json.loads(json.dumps(info))
            if field == "epoch":
                stale[field] += 1
            else:
                stale[field][0]["id"] += 1
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, "identity"),
            ):
                load_stack(self.path, expected=stale)

    def test_duplicate_header_key_rejected_before_safe_open(self):
        info, tensors = fixture()
        save(self.path, info, tensors)

        def duplicate(raw):
            desc = json.loads(raw)["codes13"]
            return (
                raw[:-1] + ',"codes13":' + json.dumps(desc, separators=(",", ":")) + "}"
            )

        replace_header(self.path, duplicate)
        with patch(
            "safetensors.safe_open",
            side_effect=AssertionError("must reject before safe_open"),
        ):
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_stack(self.path)

    def test_bad_dtype_shape_payload_and_bool_integer_rejected_before_safe_open(self):
        cases = [
            (lambda h: h["codes13"].update(dtype="F32"), "dtype"),
            (lambda h: h["codes13"].update(shape=[1]), "shape"),
        ]
        for mutate, text in cases:
            info, tensors = fixture()
            save(self.path, info, tensors)

            def change(raw, mutate=mutate):
                header = json.loads(raw)
                mutate(header)
                return json.dumps(header, separators=(",", ":"))

            replace_header(self.path, change)
            with patch(
                "safetensors.safe_open",
                side_effect=AssertionError("must reject before safe_open"),
            ):
                with self.assertRaisesRegex(ValueError, text):
                    load_stack(self.path)
        info, tensors = fixture(info_updates={"payload_bytes": 1})
        save(self.path, info, tensors)
        with patch(
            "safetensors.safe_open",
            side_effect=AssertionError("must reject before safe_open"),
        ):
            with self.assertRaisesRegex(ValueError, "payload"):
                load_stack(self.path)
        info, tensors = fixture(info_updates={"layer": True})
        save(self.path, info, tensors)
        with patch(
            "safetensors.safe_open",
            side_effect=AssertionError("must reject before safe_open"),
        ):
            with self.assertRaisesRegex(ValueError, "layer"):
                load_stack(self.path)

    def test_wrong_global_scale_and_d04_intermediates_rejected(self):
        info, tensors = fixture(globals13=[4.0] * 4)
        save(self.path, info, tensors)
        with self.assertRaisesRegex(ValueError, "SM70"):
            load_stack(self.path)
        info, tensors = fixture(scales13=[0x7E])
        save(self.path, info, tensors)
        with self.assertRaisesRegex(ValueError, "SM70|unsigned"):
            load_stack(self.path)
        info, tensors = fixture(globals13=[0.5] * 4, scales13=[0x7E])
        save(self.path, info, tensors)
        with self.assertRaisesRegex(ValueError, "SM70|unsigned"):
            load_stack(self.path)
        info, tensors = fixture()
        tensors["gscales2"].fill_(4.0)
        save(self.path, info, tensors)
        with self.assertRaisesRegex(ValueError, "SM70"):
            load_stack(self.path)
        info, tensors = fixture()
        tensors["scales2"].fill_(0x7E)
        tensors["gscales2"].fill_(0.5)
        save(self.path, info, tensors)
        with self.assertRaisesRegex(ValueError, "SM70|unsigned"):
            load_stack(self.path)

    def test_mutated_loaded_tensors_are_not_cached(self):
        info, tensors = fixture()
        save(self.path, info, tensors)
        first = load_stack(self.path)
        first.tensors["codes13"].fill_(0)
        second = load_stack(self.path)
        self.assertTrue(torch.equal(second.tensors["codes13"], tensors["codes13"]))

    def test_truncated_and_trailing_payload_rejected_before_safe_open(self):
        info, tensors = fixture()
        save(self.path, info, tensors)
        original = self.path.read_bytes()
        for bad in (original[:-1], original + b"\0"):
            self.path.write_bytes(bad)
            with patch(
                "safetensors.safe_open",
                side_effect=AssertionError("must reject before safe_open"),
            ):
                with self.assertRaisesRegex(
                    ValueError, "file has truncated or trailing"
                ):
                    load_stack(self.path)

    def test_model_revision_matches_rust_utf8_nonblank_256_byte_bound(self):
        for revision, accepted in [
            ("é" * 128, True),
            ("é" * 129, False),
            ("   ", False),
        ]:
            info, tensors = fixture(info_updates={"model_revision": revision})
            save(self.path, info, tensors)
            with self.subTest(bytes=len(revision.encode("utf-8")), accepted=accepted):
                if accepted:
                    self.assertEqual(
                        load_stack(self.path, expected=info).metadata["model_revision"],
                        revision,
                    )
                else:
                    with patch(
                        "safetensors.safe_open",
                        side_effect=AssertionError("must reject before safe_open"),
                    ):
                        with self.assertRaisesRegex(ValueError, "model_revision"):
                            load_stack(self.path)

    def test_descriptor_shape_rejects_bool_and_float_equal_to_integer_before_safe_open(
        self,
    ):
        for value in (True, 1.0):
            info, tensors = fixture()
            save(self.path, info, tensors)

            def change(raw, value=value):
                header = json.loads(raw)
                header["gscales13"]["shape"][0] = value
                return json.dumps(header, separators=(",", ":"))

            replace_header(self.path, change)
            with (
                self.subTest(value=value),
                patch(
                    "safetensors.safe_open",
                    side_effect=AssertionError("must reject before safe_open"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, "shape"):
                    load_stack(self.path)

    def test_rust_plane_limit_accepts_exact_1mib_elements(self):
        from scripts.resident_artifact import _validate_info

        info, _ = fixture()
        # Per-expert gate/up and down plane sizes are exactly 1<<20 elements.
        info["shape"] = [1, 512, 4096, 4096, 256]
        info["payload_bytes"] = (
            512 * 4096 // 2
            + 512 * 4096 // 16
            + 4 * (512 // 32)
            + 4096 * 256 // 2
            + 4096 * 256 // 16
            + 4 * (4096 // 32)
        )
        _validate_info(info)

    def test_gate_rows32_rejected_by_down_column_alignment(self):
        info, tensors = fixture()
        # Gate rows are down columns. QPN2/resident require every plane's
        # columns divisible by 64, so this producer-invalid shape must fail.
        info["shape"] = [1, 64, 64, 64, 32]
        with self.assertRaisesRegex(ValueError, "tile aligned"):
            from scripts.resident_artifact import _validate_info

            _validate_info(info)

    def test_nonregular_input_is_rejected(self):
        with self.assertRaises(ValueError):
            load_stack(Path(self.temp.name))


if __name__ == "__main__":
    unittest.main()
