"""Independent rounded-domain oracle and a bounded host extraction contract."""

import itertools
import math
import struct
import unittest
from unittest.mock import patch

import torch
from adapters.skinny_moe import _validate_sm70_scale_domain as validate


def rounded(value, fmt):
    try:
        return struct.unpack("<" + fmt, struct.pack("<" + fmt, value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def scalar_error(byte, factor):
    if byte >= 127:
        return "SM70 group scales must be unsigned finite E4M3"
    factor = rounded(factor, "f")
    if not math.isfinite(factor) or factor <= 0:
        return "SM70 global factors must be positive finite"
    exponent, mantissa = divmod(byte, 8)
    scale = (
        mantissa / 512 if exponent == 0 else (1 + mantissa / 8) * 2 ** (exponent - 7)
    )
    gm = rounded(rounded(factor * 16384, "f"), "e")
    if not math.isfinite(gm) or scale * gm >= 65520:
        return "SM70 half-scale intermediate overflow"
    return None


class AdmissionFlagsTests(unittest.TestCase):
    def error(self, scales, globals, k=64):
        try:
            validate(scales, globals, k)
            return None
        except ValueError as error:
            return str(error)

    def test_all_bytes_against_independent_rounded_scalar_oracle(self):
        boundary = torch.tensor(65520 / 16384, dtype=torch.float32)
        factors = [
            0.0,
            -0.0,
            -1.0,
            math.nan,
            math.inf,
            -math.inf,
            2**-149,
            2**-140,
            2**-50,
            2**-40,
            2**-39,
            2**-38,
            2**-24,
            2**-16,
            0.25,
            0.5,
            1.0,
            2.666015625,
            float(torch.nextafter(boundary, torch.tensor(-math.inf))),
            float(boundary),
            float(torch.nextafter(boundary, torch.tensor(math.inf))),
            4.0,
            100000.0,
            3e38,
        ]
        for byte, factor in itertools.product(range(256), factors):
            with self.subTest(byte=byte, factor=factor):
                self.assertEqual(
                    self.error(
                        torch.full((128,), byte, dtype=torch.uint8),
                        torch.tensor([[factor]], dtype=torch.float32),
                    ),
                    scalar_error(byte, factor),
                )

    def test_combined_invalid_tiles_preserve_error_priority_and_order(self):
        rows = [(0x38, 4.0), (0x7F, -1.0), (0x3C, 2.666015625)]
        for order in itertools.permutations(rows):
            scales = torch.cat(
                [torch.full((128,), byte, dtype=torch.uint8) for byte, _ in order]
            )
            globals = torch.tensor([[factor] for _, factor in order])
            self.assertEqual(self.error(scales, globals), scalar_error(127, 1.0))
        scales = torch.zeros(256, dtype=torch.uint8)
        globals = torch.tensor([[math.nan], [4.0]])
        self.assertEqual(self.error(scales, globals), scalar_error(0, math.nan))

    def test_valid_admission_has_at_most_one_explicit_host_extraction(self):
        # CPU instrumentation counts host extraction APIs, not CUDA sync events.
        # The GPU ablation separately measures their cost on the real device.
        events = []
        tensor_bool, tensor_cpu = torch.Tensor.__bool__, torch.Tensor.cpu

        def observed_bool(tensor):
            events.append("scalar")
            return tensor_bool(tensor)

        def observed_cpu(tensor, *args, **kwargs):
            events.append("copy")
            return tensor_cpu(tensor, *args, **kwargs)

        scales = torch.full((256,), 0x38, dtype=torch.uint8)
        globals = torch.ones((2, 1), dtype=torch.float32)
        with patch.object(torch.Tensor, "__bool__", observed_bool):
            with patch.object(torch.Tensor, "cpu", observed_cpu):
                validate(scales, globals, 64)
        self.assertLessEqual(len(events), 1, events)

    def test_same_mutable_buffers_are_revalidated(self):
        scales = torch.full((128,), 0x38, dtype=torch.uint8)
        globals = torch.ones((1, 1), dtype=torch.float32)
        addresses = scales.data_ptr(), globals.data_ptr()
        self.assertIsNone(self.error(scales, globals))
        scales.fill_(0x7F)
        self.assertEqual(self.error(scales, globals), scalar_error(127, 1))
        scales.fill_(0x38)
        globals.fill_(4)
        self.assertEqual(self.error(scales, globals), scalar_error(0x38, 4))
        self.assertEqual(addresses, (scales.data_ptr(), globals.data_ptr()))

    def test_malformed_and_empty_planes_preserve_domain(self):
        scales = torch.zeros(128, dtype=torch.uint8)
        globals = torch.ones((1, 1), dtype=torch.float32)
        for data, factors, k in [
            (scales, globals, 63),
            (scales, globals, 65537),
            (scales.to(torch.int16), globals, 64),
            (scales, globals.double(), 64),
            (scales[:-1], globals, 64),
        ]:
            with self.assertRaises(ValueError):
                validate(data, factors, k)
        self.assertIsNone(self.error(torch.empty(0, dtype=torch.uint8), torch.empty(0)))


if __name__ == "__main__":
    unittest.main()
