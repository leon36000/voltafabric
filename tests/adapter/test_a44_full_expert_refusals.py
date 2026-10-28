"""A44: one refusal per guard operand of scripts/full_expert_canary.py.

Each case starts from a baseline the module accepts (the pinned full-expert sample,
a real CPU export through the release ``vf`` binary, a real loaded artifact) and
breaks exactly one thing, so that exactly one operand of one ``if ...: raise``
guard is true while every earlier check passes. Each refusal pins the exact
exception type and the full message with an anchored pattern, so a neighbouring
or later guard cannot satisfy the assertion instead. OS-level guards (memfd,
seals, short writes, concurrent file changes) are provoked through a per-module
``os``/``fcntl`` proxy, never by patching the global modules.

CPU only: no CUDA runtime, library or device is used. The provider is a recorded
fake on the CPU device, and Torch CUDA / ctypes entry points are replaced by
fakes that fail the test if anything reaches them.
"""

import copy
import ctypes
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import re
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from scripts import full_expert_canary as full
from scripts.hardware_canary import rust_call as real_rust_call

ROOT = Path(__file__).resolve().parents[2]
BINARY = Path(os.environ["VF_BINARY"])
SAMPLE = ROOT / "fixtures/checkpoint_qwen36_expert0_full.safetensors"
CAP = 2 * 1024 * 1024
REQUIRED_SEALS = (
    full._F_SEAL_WRITE | full._F_SEAL_SHRINK | full._F_SEAL_GROW | full._F_SEAL_SEAL
)


def _module_proxy(module, name, **overrides):
    """Copy of ``module``'s namespace with selected attributes replaced."""
    proxy = types.ModuleType(name)
    proxy.__dict__.update(vars(module))
    proxy.__dict__.update(overrides)
    return proxy


class _NoCuda:
    """Fails loudly if anything tries to reach CUDA or a native library."""

    def __init__(self, test):
        self.test = test

    def __enter__(self):
        def forbidden(*_args, **_kwargs):
            raise AssertionError("CUDA/native entry point reached in CPU-only test")

        self.patches = [
            patch.object(torch.cuda, "init", side_effect=forbidden),
            patch.object(torch.cuda, "current_stream", side_effect=forbidden),
            patch.object(torch.cuda, "synchronize", side_effect=forbidden),
            patch.object(torch.cuda, "set_device", side_effect=forbidden),
            patch.object(ctypes, "CDLL", side_effect=forbidden),
        ]
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc):
        for item in reversed(self.patches):
            item.stop()
        return False


class RefusalTest(unittest.TestCase):
    def setUp(self):
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES", ""), "")
        self._temp = tempfile.TemporaryDirectory(prefix="a44-full-")
        self.addCleanup(self._temp.cleanup)
        self.directory = Path(self._temp.name)
        guard = _NoCuda(self)
        guard.__enter__()
        self.addCleanup(guard.__exit__, None, None, None)

    def assertRefused(self, exc_type, message, function, *args):
        """Exact exception type and full anchored message."""
        pattern = r"\A" + re.escape(message) + r"\Z"
        with self.assertRaisesRegex(exc_type, pattern) as caught:
            result = function(*args)
            if isinstance(result, int) and not isinstance(result, bool):
                # A guard that failed open on a memfd would leak its descriptor.
                try:
                    os.close(result)
                except OSError:
                    pass
        self.assertIs(type(caught.exception), exc_type)
        return caught.exception

    def sample_copy(self, name="sample.safetensors"):
        path = self.directory / name
        path.write_bytes(SAMPLE.read_bytes())
        return path

    def prepared(self):
        return full.prepare_full_expert(BINARY, SAMPLE, self.directory)


class HalfAndStagedWeightTests(RefusalTest):
    def test_baseline_half_and_staged_weight_accept_finite_domain(self):
        self.assertEqual(full._half(1.5), 1.5)
        self.assertEqual(full.staged_weight(2, 0x38, 0.25), 0.25)
        self.assertEqual(full.staged_weight(15, 126, 2**-10), -2.625)

    def test_nonfinite_half_without_overflow_is_refused(self):
        # inf/nan pack into half without OverflowError; only the isfinite guard.
        for value in [float("inf"), float("-inf"), float("nan")]:
            with self.subTest(value=value):
                self.assertRefused(
                    ValueError,
                    "nonfinite half intermediate in strict full-expert reference",
                    full._half,
                    value,
                )

    def test_non_int_nibble_in_range_is_refused(self):
        # bool and float values satisfy 0 <= nibble < 16; only the type operand.
        for nibble in [True, False, 2.0]:
            with self.subTest(nibble=nibble):
                self.assertRefused(
                    ValueError,
                    "FP4 nibble outside domain",
                    full.staged_weight,
                    nibble,
                    0x38,
                    0.25,
                )

    def test_int_nibble_out_of_range_is_refused(self):
        for nibble in [-1, 16, 255]:
            with self.subTest(nibble=nibble):
                self.assertRefused(
                    ValueError,
                    "FP4 nibble outside domain",
                    full.staged_weight,
                    nibble,
                    0x38,
                    0.25,
                )

    def test_non_int_scale_byte_in_range_is_refused(self):
        for scale in [True, 56.0]:
            with self.subTest(scale=scale):
                self.assertRefused(
                    ValueError,
                    "unsigned finite E4M3 scale required",
                    full.staged_weight,
                    2,
                    scale,
                    0.25,
                )

    def test_int_scale_byte_out_of_range_is_refused(self):
        for scale in [-1, 127, 128, 255]:
            with self.subTest(scale=scale):
                self.assertRefused(
                    ValueError,
                    "unsigned finite E4M3 scale required",
                    full.staged_weight,
                    2,
                    scale,
                    0.25,
                )

    def test_nonfinite_positive_global_factor_is_refused(self):
        # nan <= 0 and inf <= 0 are both False: only the isfinite operand.
        for factor in [float("nan"), float("inf")]:
            with self.subTest(factor=factor):
                self.assertRefused(
                    ValueError,
                    "positive finite global factor required",
                    full.staged_weight,
                    2,
                    0x38,
                    factor,
                )

    def test_finite_nonpositive_global_factor_is_refused(self):
        for factor in [0.0, -0.0, -0.25]:
            with self.subTest(factor=factor):
                self.assertRefused(
                    ValueError,
                    "positive finite global factor required",
                    full.staged_weight,
                    2,
                    0x38,
                    factor,
                )


class SampleReadTests(RefusalTest):
    def read_proxy(self, on_first_read):
        """``os`` proxy whose first read triggers ``on_first_read`` once."""
        state = {"fired": False}

        def read(fd, size):
            if not state["fired"]:
                state["fired"] = True
                on_first_read()
            return os.read(fd, size)

        return _module_proxy(os, "os_read_proxy", read=read), state

    def test_baseline_pinned_sample_is_read(self):
        raw, sha, header, data = full._sample(self.sample_copy())
        self.assertEqual(sha, full.SAMPLE_SHA)
        self.assertEqual(raw, SAMPLE.read_bytes())
        self.assertTrue(8 <= len(raw) <= CAP)
        self.assertIn(
            "model.language_model.layers.0.mlp.experts.0.gate_proj.weight", header
        )
        self.assertEqual(len(data), len(raw) - 8 - int.from_bytes(raw[:8], "little"))

    def test_directory_with_in_range_size_is_refused(self):
        # Not a regular file, yet its st_size is inside [8, 2 MiB]: only S_ISREG.
        directory = self.directory / "not-a-file"
        directory.mkdir()
        size = os.stat(directory).st_size
        self.assertTrue(8 <= size <= CAP, size)
        self.assertRefused(
            ValueError,
            "full-expert sample must be a bounded regular file",
            full._sample,
            directory,
        )

    def test_regular_file_outside_size_bounds_is_refused(self):
        for size in [0, 7, CAP + 1]:
            with self.subTest(size=size):
                path = self.directory / ("size-%d" % size)
                path.write_bytes(b"\x01" * size)
                self.assertRefused(
                    ValueError,
                    "full-expert sample must be a bounded regular file",
                    full._sample,
                    path,
                )

    def test_file_growing_past_cap_while_reading_is_refused(self):
        path = self.directory / "growing"
        path.write_bytes(b"\x00" * CAP)

        # Control: exactly at the cap the read finishes and only the pin refuses.
        self.assertRefused(
            ValueError,
            "full-expert sample differs from pinned content",
            full._sample,
            path,
        )

        def grow():
            with open(path, "ab") as handle:
                handle.write(b"\x00")

        proxy, state = self.read_proxy(grow)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                ValueError, "full-expert sample grew beyond cap", full._sample, path
            )
        self.assertTrue(state["fired"])
        self.assertEqual(path.stat().st_size, CAP + 1)

    def test_pinned_sample_touched_while_reading_is_refused(self):
        # Same bytes (so the pin would still match) but a changed inode stamp.
        path = self.sample_copy()
        before = path.stat()

        def touch():
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns - 10**9))

        proxy, state = self.read_proxy(touch)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                ValueError,
                "full-expert sample changed while reading",
                full._sample,
                path,
            )
        self.assertTrue(state["fired"])
        self.assertEqual(path.read_bytes(), SAMPLE.read_bytes())

    def test_same_size_different_content_is_refused(self):
        raw = bytearray(SAMPLE.read_bytes())
        header_size = int.from_bytes(raw[:8], "little")
        raw[8 + header_size] ^= 1
        path = self.directory / "flipped.safetensors"
        path.write_bytes(raw)
        with patch.object(full, "rust_call") as called:
            self.assertRefused(
                ValueError,
                "full-expert sample differs from pinned content",
                full.prepare_full_expert,
                BINARY,
                path,
                self.directory,
            )
        called.assert_not_called()


class DigestTests(RefusalTest):
    def setUp(self):
        super().setUp()
        self.artifact = self.prepared().artifact
        self.baseline = full._digest(self.artifact)
        self.assertRegex(self.baseline, r"\A[0-9a-f]{64}\Z")

    def broken(self, **tensors):
        tensors = {**self.artifact.tensors, **tensors}
        return dataclasses.replace(self.artifact, tensors=tensors)

    def assertDigestRefused(self, artifact, message):
        self.assertRefused(ValueError, message, full._digest, artifact)

    def test_static_identity_differs_is_refused(self):
        for key, value in [("epoch", 2), ("payload_bytes", 1769857)]:
            with self.subTest(key=key):
                metadata = copy.deepcopy(self.artifact.metadata)
                metadata[key] = value
                artifact = dataclasses.replace(self.artifact, metadata=metadata)
                self.assertDigestRefused(
                    artifact, "full-expert static identity differs"
                )

    def test_extra_tensor_is_refused(self):
        artifact = self.broken(extra=torch.zeros(1, dtype=torch.uint8))
        self.assertDigestRefused(artifact, "full-expert tensor set differs")

    def test_missing_tensor_is_refused(self):
        tensors = dict(self.artifact.tensors)
        del tensors["gscales2"]
        artifact = dataclasses.replace(self.artifact, tensors=tensors)
        self.assertDigestRefused(artifact, "full-expert tensor set differs")

    def test_non_cpu_tensor_is_refused(self):
        tensor = torch.zeros((1048576,), dtype=torch.uint8, device="meta")
        self.assertTrue(tensor.is_contiguous())
        self.assertDigestRefused(
            self.broken(codes13=tensor), "full-expert CPU tensor dtype/shape differs"
        )

    def test_wrong_dtype_is_refused(self):
        for name, tensor in [
            ("codes13", self.artifact.tensors["codes13"].to(torch.int8)),
            ("gscales2", self.artifact.tensors["gscales2"].to(torch.int32)),
        ]:
            with self.subTest(name=name):
                self.assertDigestRefused(
                    self.broken(**{name: tensor}),
                    "full-expert CPU tensor dtype/shape differs",
                )

    def test_wrong_shape_is_refused(self):
        codes = self.artifact.tensors["codes13"]
        for name, tensor in [
            ("codes13", codes.reshape(2, -1)),
            ("codes13", codes[:-1].clone()),
            ("gscales13", self.artifact.tensors["gscales13"].reshape(32)),
        ]:
            with self.subTest(name=name, shape=tuple(tensor.shape)):
                self.assertTrue(tensor.is_contiguous())
                self.assertDigestRefused(
                    self.broken(**{name: tensor}),
                    "full-expert CPU tensor dtype/shape differs",
                )

    def test_non_contiguous_tensor_is_refused(self):
        codes = self.artifact.tensors["codes13"]
        tensor = torch.stack([codes, codes], dim=1)[:, 0]
        self.assertEqual(tuple(tensor.shape), tuple(codes.shape))
        self.assertEqual(tensor.dtype, torch.uint8)
        self.assertFalse(tensor.is_contiguous())
        self.assertDigestRefused(
            self.broken(codes13=tensor), "full-expert CPU tensor dtype/shape differs"
        )


class MemfdTests(RefusalTest):
    RAW = bytes(range(256)) * 700  # Several 64 KiB read chunks.

    def os_proxy(self, **overrides):
        return _module_proxy(os, "os_proxy", **overrides)

    def test_baseline_memfd_is_sealed_and_exact(self):
        fd = full._sealed_checkpoint_memfd(self.RAW)
        try:
            self.assertEqual(
                fcntl.fcntl(fd, full._F_GET_SEALS) & REQUIRED_SEALS, REQUIRED_SEALS
            )
            self.assertEqual(os.fstat(fd).st_size, len(self.RAW))
            self.assertEqual(os.pread(fd, len(self.RAW) + 1, 0), self.RAW)
        finally:
            os.close(fd)

    def test_missing_memfd_support_is_refused(self):
        for missing in ["memfd_create", "MFD_CLOEXEC", "MFD_ALLOW_SEALING"]:
            with self.subTest(missing=missing):
                proxy = self.os_proxy()
                del proxy.__dict__[missing]
                with patch.object(full, "os", proxy):
                    self.assertRefused(
                        RuntimeError,
                        "sealed Linux memfd support is required for full-expert export",
                        full._sealed_checkpoint_memfd,
                        self.RAW,
                    )

    def recorded_proxy(self, **overrides):
        created = []

        def memfd_create(*args, **kwargs):
            fd = os.memfd_create(*args, **kwargs)
            created.append(fd)
            return fd

        return self.os_proxy(memfd_create=memfd_create, **overrides), created

    def assertClosed(self, created):
        self.assertEqual(len(created), 1)
        with self.assertRaises(OSError):
            os.fstat(created[0])

    def test_zero_byte_write_is_refused_and_closes_memfd(self):
        calls = []

        def write(fd, data):
            calls.append(len(data))
            if len(calls) == 1:
                return 0
            raise AssertionError("write retried after a zero-byte write")

        proxy, created = self.recorded_proxy(write=write)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                OSError,
                "short write creating full-expert memfd",
                full._sealed_checkpoint_memfd,
                self.RAW,
            )
        self.assertEqual(calls, [len(self.RAW)])
        self.assertClosed(created)

    def test_missing_seal_is_refused_and_closes_memfd(self):
        for dropped in [
            full._F_SEAL_SEAL,
            full._F_SEAL_WRITE,
            full._F_SEAL_SHRINK,
            full._F_SEAL_GROW,
        ]:
            with self.subTest(dropped=dropped):

                def fake_fcntl(fd, command, *args, dropped=dropped):
                    value = fcntl.fcntl(fd, command, *args)
                    if command == full._F_GET_SEALS:
                        return value & ~dropped
                    return value

                fcntl_proxy = _module_proxy(fcntl, "fcntl_proxy", fcntl=fake_fcntl)
                proxy, created = self.recorded_proxy()
                with (
                    patch.object(full, "os", proxy),
                    patch.object(full, "fcntl", fcntl_proxy),
                ):
                    self.assertRefused(
                        RuntimeError,
                        "full-expert memfd did not retain all required seals",
                        full._sealed_checkpoint_memfd,
                        self.RAW,
                    )
                self.assertClosed(created)

    def test_memfd_size_mismatch_is_refused(self):
        def fstat(fd):
            real = os.fstat(fd)
            return types.SimpleNamespace(st_size=real.st_size + 1)

        proxy, created = self.recorded_proxy(fstat=fstat)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                ValueError,
                "full-expert memfd size differs from pinned bytes",
                full._sealed_checkpoint_memfd,
                self.RAW,
            )
        self.assertClosed(created)

    def test_memfd_read_length_mismatch_with_equal_digest_is_refused(self):
        class LongBytes(bytes):
            """Same buffer (so same SHA-256) but reports one extra byte."""

            def __len__(self):
                return bytes.__len__(self) + 1

        state = {"wrapped": 0}

        def read(fd, size):
            chunk = os.read(fd, size)
            if chunk and not state["wrapped"]:
                state["wrapped"] += 1
                return LongBytes(chunk)
            return chunk

        proxy, created = self.recorded_proxy(read=read)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                ValueError,
                "full-expert memfd differs from pinned bytes",
                full._sealed_checkpoint_memfd,
                self.RAW,
            )
        self.assertEqual(state["wrapped"], 1)
        self.assertClosed(created)

    def test_memfd_read_content_mismatch_with_equal_length_is_refused(self):
        state = {"flipped": 0}

        def read(fd, size):
            chunk = os.read(fd, size)
            if chunk and not state["flipped"]:
                state["flipped"] += 1
                return bytes([chunk[0] ^ 1]) + chunk[1:]
            return chunk

        proxy, created = self.recorded_proxy(read=read)
        with patch.object(full, "os", proxy):
            self.assertRefused(
                ValueError,
                "full-expert memfd differs from pinned bytes",
                full._sealed_checkpoint_memfd,
                self.RAW,
            )
        self.assertEqual(state["flipped"], 1)
        self.assertClosed(created)


class ExportReceiptTests(RefusalTest):
    def run_edit(self, edit):
        calls = []

        def edited(binary, operation, request, *, pass_fds=()):
            receipt = real_rust_call(binary, operation, request, pass_fds=pass_fds)
            calls.append(copy.deepcopy(receipt))
            edit(receipt)
            return receipt

        with patch.object(full, "rust_call", side_effect=edited):
            self.assertRefused(
                ValueError,
                "unexpected full-expert export identity/scope",
                full.prepare_full_expert,
                BINARY,
                SAMPLE,
                self.directory,
            )
        self.assertEqual(len(calls), 1)
        # The genuine receipt was valid; only the injected edit is wrong.
        self.assertIs(calls[0]["cuda_executed"], False)
        self.assertEqual(calls[0]["scope"], "CPU_COMPRESSED_EXPORT_ONLY")
        self.assertEqual(calls[0]["meta"], full.META)

    def test_baseline_real_export_is_accepted(self):
        prepared = self.prepared()
        self.assertEqual(prepared.sample_sha256, full.SAMPLE_SHA)
        self.assertEqual(prepared.digest, full._digest(prepared.artifact))
        self.assertTrue(prepared.path.is_file())

    def test_cuda_executed_not_false_is_refused(self):
        for value in [True, None, 0, "false"]:
            with self.subTest(value=value):
                self.run_edit(lambda receipt: receipt.update(cuda_executed=value))

    def test_wrong_scope_is_refused(self):
        for value in ["GPU_EXECUTED", "CPU_COMPRESSED_EXPORT_ONLY ", None]:
            with self.subTest(value=value):
                self.run_edit(lambda receipt: receipt.update(scope=value))

    def test_wrong_meta_is_refused(self):
        def edit(receipt):
            receipt["meta"] = dict(receipt["meta"], epoch=2)

        self.run_edit(edit)


class SentinelProvider:
    """CPU fake provider: records calls; never reaches a GPU."""

    torch = torch
    device = torch.device("cpu")

    def __init__(self, output=None):
        self.output = output
        self.launches = 0
        self.synchronized_launches = 0
        self.synchronize_calls = 0

    def gemm_qpn2_moe(
        self, x, codes, scales, gs, blocks, slots, padded, s, n, k, in_div, splitk, nacc
    ):
        self.launches += 1
        if self.output is None:
            raise AssertionError("provider reached despite changed snapshot")
        return self.output(n)

    def synchronize(self):
        self.synchronize_calls += 1
        raise RuntimeError("sentinel: synchronize reached")


class ExecuteTests(RefusalTest):
    def setUp(self):
        super().setUp()
        self.prepared_expert = self.prepared()

    def execute(self, provider, cases):
        return full.execute_full_expert(provider, self.prepared_expert, cases)

    def test_baseline_valid_output_passes_guard_and_reaches_synchronize(self):
        provider = SentinelProvider(lambda n: torch.zeros((6, n), dtype=torch.float16))
        cases = []
        self.assertRefused(
            RuntimeError,
            "sentinel: synchronize reached",
            self.execute,
            provider,
            cases,
        )
        self.assertEqual((provider.launches, provider.synchronize_calls), (1, 1))
        self.assertEqual(cases, [])

    def assertSnapshotRefused(self):
        provider = SentinelProvider()
        cases = []
        self.assertRefused(
            ValueError,
            "full-expert compressed snapshot changed",
            self.execute,
            provider,
            cases,
        )
        self.assertEqual(provider.launches, 0)
        self.assertEqual(cases, [])

    def test_in_memory_prepared_artifact_changed_is_refused(self):
        # File on disk unchanged: fresh digest still equals the prepared digest.
        self.prepared_expert.artifact.tensors["codes13"][0] ^= 1
        self.assertSnapshotRefused()

    def test_artifact_file_changed_after_prepare_is_refused(self):
        # Prepared in-memory artifact unchanged; one code byte flipped on disk
        # (codes accept any byte, so load_stack still accepts the file).
        path = self.prepared_expert.path
        raw = bytearray(path.read_bytes())
        header_size = int.from_bytes(raw[:8], "little")
        header = json.loads(raw[8 : 8 + header_size])
        start, _ = header["codes13"]["data_offsets"]
        raw[8 + header_size + start] ^= 1
        path.write_bytes(raw)
        fresh = full.load_stack(path, expected=full.META)
        self.assertNotEqual(full._digest(fresh), self.prepared_expert.digest)
        self.assertEqual(
            full._digest(self.prepared_expert.artifact), self.prepared_expert.digest
        )
        self.assertSnapshotRefused()

    def assertOutputRefused(self, output):
        provider = SentinelProvider(output)
        cases = []
        self.assertRefused(
            ValueError,
            "full-expert provider output differs from prepared shape",
            self.execute,
            provider,
            cases,
        )
        self.assertEqual((provider.launches, provider.synchronize_calls), (1, 0))
        self.assertEqual(cases, [])

    def test_wrong_output_dtype_is_refused(self):
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.assertOutputRefused(lambda n: torch.zeros((6, n), dtype=dtype))

    def test_wrong_output_device_is_refused(self):
        self.assertOutputRefused(
            lambda n: torch.zeros((6, n), dtype=torch.float16, device="meta")
        )

    def test_wrong_output_shape_is_refused(self):
        for shape in [
            lambda n: (6, n + 1),
            lambda n: (n, 6),
            lambda n: (7, n),
            lambda n: (6 * n,),
        ]:
            with self.subTest(shape=shape(512)):
                self.assertOutputRefused(
                    lambda n, shape=shape: torch.zeros(shape(n), dtype=torch.float16)
                )


if __name__ == "__main__":
    unittest.main()
