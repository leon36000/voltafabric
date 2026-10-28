from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request

from test_a106_expert383_extractor import FakeResponse, extractor


ROOT = Path(__file__).resolve().parents[2]
SELECTED = json.loads(
    (ROOT / "evidence/continuation_a120_claude/A120_SELECTED_METADATA.json").read_text()
)
DATA_START = 8 + 258_136


def synthetic_bytes(start: int, length: int) -> bytes:
    """Deterministic stand-in for shard bytes; no real payload is used."""
    block = hashlib.sha256(str(start).encode()).digest()
    return (block * (length // len(block) + 1))[:length]


class FakeServer:
    """Serves exact ranges of a virtual shard; `mutate` corrupts one response."""

    def __init__(self, mutate=None) -> None:
        self.requests: list[str] = []
        self.mutate = mutate

    def __call__(self, request, timeout=None):
        header = request.get_header("Range")
        self.requests.append(header)
        start, end = (int(x) for x in header.removeprefix("bytes=").split("-"))
        body = synthetic_bytes(start, end - start + 1)
        status, content_range = 206, f"bytes {start}-{end}/{extractor.SHARD_BYTES}"
        if self.mutate is not None:
            body, status, content_range = self.mutate(body, status, content_range)
        return FakeResponse(body, status=status, content_range=content_range)


class A120SelectedExpertTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(extractor, "ARTIFACT_ROOT", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_main(self, server, *argv: str):
        def open_range(request, timeout=30):
            return server(request, timeout), []

        with patch.object(extractor, "open_range_request", side_effect=open_range):
            return extractor.main(list(argv))

    def test_selected_96_192_match_pinned_header_and_index(self) -> None:
        _, ids, specs = extractor.selected_ranges([192, 96])
        self.assertEqual(ids, (96, 192))
        expected = [t for e in SELECTED["selected_experts"] for t in e["tensors"]]
        self.assertEqual([s["name"] for s in specs], [t["name"] for t in expected])
        for spec, tensor in zip(specs, expected):
            self.assertEqual(spec["dtype"], tensor["dtype"])
            self.assertEqual(spec["shape"], tensor["shape"])
            self.assertEqual(spec["data_offsets"], tensor["data_offsets"])
            self.assertEqual(
                spec["range_start"], DATA_START + tensor["data_offsets"][0]
            )
            self.assertEqual(
                spec["range_end"], DATA_START + tensor["data_offsets"][1] - 1
            )
            self.assertLess(spec["range_end"], extractor.SHARD_BYTES)
        self.assertEqual(sum(s["expected_bytes"] for s in specs), 37_601_280)
        for expert_id in (96, 192):
            _, single = extractor.pinned_ranges(expert_id)
            self.assertEqual(sum(s["expected_bytes"] for s in single), 18_800_640)

    def test_invalid_selections_are_rejected(self) -> None:
        for bad in (
            [96, 96],
            [5],
            [384],
            [-1],
            [True],
            ["96"],
            [96.0],
            [],
            [96, 192, 383],
            "96",
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                extractor.validate_expert_ids(bad)
        self.assertEqual(extractor.validate_expert_ids(None), (383,))

    def test_size_budget_overflow_is_rejected(self) -> None:
        with patch.object(extractor, "PAYLOAD_CAP_BYTES", 18_800_639):
            with self.assertRaises(ValueError):
                extractor.validate_expert_ids([96])
        with self.assertRaises(ValueError):
            extractor.make_safetensors_fixture([], {}, 37_601_281)
        with self.assertRaises(ValueError):
            extractor.validate_range_spec(
                {
                    "range_start": 0,
                    "range_end": 37_601_280,
                    "expected_bytes": 37_601_281,
                }
            )
        for spec in (
            {"range_start": 5, "range_end": 4, "expected_bytes": 0},
            {
                "range_start": extractor.SHARD_BYTES - 1,
                "range_end": extractor.SHARD_BYTES,
                "expected_bytes": 2,
            },
            {"range_start": 0, "range_end": 3, "expected_bytes": 3},
        ):
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                extractor.validate_range_spec(spec)

    def _patched_metadata(self, header_edit=None, index_edit=None):
        header = json.loads(extractor.HEADER_PATH.read_bytes())
        index = json.loads(extractor.INDEX_PATH.read_bytes())
        if header_edit:
            header_edit(header)
        if index_edit:
            index_edit(index["weight_map"])
        tmp = Path(self.tmp.name)
        header_bytes = json.dumps(header).encode()
        index_bytes = json.dumps(index).encode()
        (tmp / "h.json").write_bytes(header_bytes)
        (tmp / "i.json").write_bytes(index_bytes)
        return [
            patch.object(extractor, "HEADER_PATH", tmp / "h.json"),
            patch.object(extractor, "HEADER_SHA256", extractor.sha256(header_bytes)),
            patch.object(extractor, "INDEX_PATH", tmp / "i.json"),
            patch.object(extractor, "INDEX_SHA256", extractor.sha256(index_bytes)),
        ]

    def test_wrong_metadata_shard_and_out_of_domain_tensors_rejected(self) -> None:
        name = "layers.0.ffn.experts.96.w2.scale"
        cases = {
            "extra_header_tensor": dict(
                header_edit=lambda h: h.setdefault(
                    "layers.0.ffn.experts.96.w4.weight", h[name]
                )
            ),
            "missing_header_tensor": dict(header_edit=lambda h: h.pop(name)),
            "wrong_dtype": dict(header_edit=lambda h: h[name].update(dtype="I8")),
            "wrong_shape": dict(header_edit=lambda h: h[name].update(shape=[72, 5120])),
            "wrong_offsets": dict(
                header_edit=lambda h: h[name].update(data_offsets=[0, 1])
            ),
            "wrong_shard": dict(
                index_edit=lambda m: m.update(
                    {name: "model-00004-of-00048.safetensors"}
                )
            ),
            "extra_index_tensor": dict(
                index_edit=lambda m: m.update(
                    {"layers.0.ffn.experts.192.gate": extractor.SHARD}
                )
            ),
        }
        for label, edits in cases.items():
            patches = self._patched_metadata(**edits)
            with self.subTest(label), patches[0], patches[1], patches[2], patches[3]:
                with self.assertRaises(ValueError):
                    extractor.selected_ranges([96, 192])
        # Provenance: content change without a pinned-hash update is rejected.
        patches = self._patched_metadata()
        for target in (patches[0], patches[2]):
            with target, self.assertRaisesRegex(ValueError, "SHA-256"):
                extractor.selected_ranges([96])

    def test_end_to_end_fixture_preserves_bytes_and_offsets(self) -> None:
        server = FakeServer()
        self.assertEqual(
            self.run_main(server, "--expert-id", "192", "--expert-id", "96"), 0
        )
        _, _, specs = extractor.selected_ranges([96, 192])
        self.assertEqual(server.requests, [s["range"] for s in specs])
        paths = extractor.artifact_paths((96, 192))
        fixture = paths["output"].read_bytes()
        header_size = struct.unpack("<Q", fixture[:8])[0]
        self.assertEqual(header_size % 8, 0)
        header = json.loads(fixture[8 : 8 + header_size])
        payload = fixture[8 + header_size :]
        self.assertEqual(len(payload), 37_601_280)
        offset = 0
        for spec in specs:
            entry = header[spec["name"]]
            self.assertEqual(
                entry["data_offsets"], [offset, offset + spec["expected_bytes"]]
            )
            self.assertEqual(
                (entry["dtype"], entry["shape"]), (spec["dtype"], spec["shape"])
            )
            self.assertEqual(
                payload[offset : offset + spec["expected_bytes"]],
                synthetic_bytes(spec["range_start"], spec["expected_bytes"]),
            )
            offset += spec["expected_bytes"]
        self.assertEqual(set(header), {s["name"] for s in specs})
        receipt = json.loads(paths["receipt"].read_text())
        self.assertEqual(receipt["status"], "PASS_PINNED_EXPERTS96_192_RANGES_ONLY")
        self.assertEqual(receipt["expert_ids"], [96, 192])
        self.assertEqual(receipt["index_sha256"], extractor.INDEX_SHA256)
        self.assertEqual(receipt["fixture_sha256"], extractor.sha256(fixture))
        self.assertTrue(
            all("redirect_chain" in entry for entry in receipt["completed_tensors"])
        )
        # Duplicate destination: a second run never requests or overwrites.
        again = FakeServer()
        with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
            self.run_main(again, "--expert-id", "96", "--expert-id", "192")
        self.assertEqual(again.requests, [])
        self.assertEqual(paths["output"].read_bytes(), fixture)

    def test_bad_responses_stop_after_one_request_without_retry(self) -> None:
        mutations = {
            "status_200": lambda b, s, c: (b, 200, c),
            "malformed_content_range": lambda b, s, c: (
                b,
                s,
                c.replace("bytes ", "bytes="),
            ),
            "wrong_total": lambda b, s, c: (b, s, c.rsplit("/", 1)[0] + "/*"),
            "truncated": lambda b, s, c: (b[:-1], s, c),
            "overlong": lambda b, s, c: (b + b"x", s, c),
        }
        for label, mutate in mutations.items():
            with self.subTest(label):
                with (
                    tempfile.TemporaryDirectory() as tmp,
                    patch.object(extractor, "ARTIFACT_ROOT", Path(tmp)),
                ):
                    server = FakeServer(mutate)
                    with self.assertRaisesRegex(SystemExit, "receipt saved"):
                        self.run_main(server, "--expert-id", "96")
                    self.assertEqual(len(server.requests), 1)
                    paths = extractor.artifact_paths((96,))
                    receipt = json.loads(paths["receipt"].read_text())
                    self.assertEqual(
                        receipt["status"], "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
                    )
                    self.assertFalse(receipt["last_failure"]["automatic_retry"])
                    self.assertEqual(receipt["completed_tensors"], [])
                    self.assertFalse(paths["output"].exists())
                    self.assertEqual(list(paths["work"].iterdir()), [])
                    rerun = FakeServer()
                    with self.assertRaisesRegex(SystemExit, "--resume"):
                        self.run_main(rerun, "--expert-id", "96")
                    self.assertEqual(rerun.requests, [])

    def test_content_length_mismatch_is_rejected_before_body_read(self) -> None:
        spec = {"range_start": 10, "range_end": 12, "expected_bytes": 3}
        response = FakeResponse(
            b"abc", status=206, content_range=f"bytes 10-12/{extractor.SHARD_BYTES}"
        )
        response.headers["Content-Length"] = "4"
        with patch.object(
            extractor, "open_range_request", return_value=(response, [])
        ) as open_range:
            with self.assertRaises(extractor.RangeReadError) as caught:
                extractor.read_range(spec)
        self.assertEqual(
            (
                open_range.call_count,
                caught.exception.body_bytes_read,
                response.read_sizes,
            ),
            (1, 0, []),
        )

    def test_resume_after_midway_failure_fetches_only_remaining(self) -> None:
        calls = []

        def fail_third(body, status, content_range):
            calls.append(1)
            return (body[:-1] if len(calls) == 3 else body), status, content_range

        with self.assertRaises(SystemExit):
            self.run_main(FakeServer(fail_third), "--expert-id", "192")
        self.assertEqual(len(calls), 3)
        server = FakeServer()
        self.assertEqual(self.run_main(server, "--expert-id", "192", "--resume"), 0)
        _, specs = extractor.pinned_ranges(192)
        self.assertEqual(server.requests, [s["range"] for s in specs[2:]])
        # Unreceipted leftovers after uncertain I/O block resume with no request.
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(extractor, "ARTIFACT_ROOT", Path(tmp)),
        ):
            with self.assertRaises(SystemExit):
                self.run_main(
                    FakeServer(lambda b, s, c: (b, 200, c)), "--expert-id", "96"
                )
            paths = extractor.artifact_paths((96,))
            (paths["work"] / "expert96_w1_weight.bin.tmp").write_bytes(b"?")
            probe = FakeServer()
            with self.assertRaisesRegex(SystemExit, "invalid"):
                self.run_main(probe, "--expert-id", "96", "--resume")
            self.assertEqual(probe.requests, [])

    def test_output_and_receipt_identities_are_unique_and_legacy_unchanged(
        self,
    ) -> None:
        selections = [(96,), (192,), (96, 192), (96, 383), (383,)]
        paths = [extractor.artifact_paths(s) for s in selections]
        for key in ("output", "receipt", "work"):
            self.assertEqual(len({p[key] for p in paths}), len(selections))
        legacy = extractor.artifact_paths((383,))
        self.assertEqual(
            legacy["output"],
            ROOT / "fixtures/deepseek_v41_layer0_expert383_314113f8.safetensors",
        )
        self.assertEqual(
            legacy["receipt"],
            ROOT
            / "evidence/continuation_a106_real_second_expert/A106_EXPERT383_RANGE_RECEIPT.json",
        )
        self.assertEqual(
            extractor.payload_filename("layers.0.ffn.experts.383.w1.weight", (383,)),
            "w1_weight.bin",
        )
        self.assertEqual(
            extractor.payload_filename("layers.0.ffn.experts.96.w1.weight", (96, 192)),
            "expert96_w1_weight.bin",
        )
        # Default selection still matches the committed A106 receipt exactly.
        receipt = json.loads(legacy["receipt"].read_text())
        _, specs = extractor.pinned_ranges()
        self.assertEqual(
            receipt["requested_tensors"],
            [{k: s[k] for k in receipt["requested_tensors"][0]} for s in specs],
        )

    def test_destination_race_never_overwrites(self) -> None:
        target = Path(self.tmp.name) / "out.bin"
        target.write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            extractor.write_new_file(target, b"new")
        self.assertEqual(target.read_bytes(), b"keep")
        self.assertFalse(target.with_suffix(".bin.tmp").exists())

    def test_atomic_receipt_writer_never_clobbers_orphan_temp(self) -> None:
        target = Path(self.tmp.name) / "receipt.json"
        temp = target.with_suffix(target.suffix + ".tmp")
        sentinel = b"possible prior receipt state"
        temp.write_bytes(sentinel)

        with self.assertRaises(FileExistsError):
            extractor.atomic_json(target, {"status": "new"})

        self.assertEqual(temp.read_bytes(), sentinel)
        self.assertFalse(target.exists())

    def test_orphaned_receipt_temp_blocks_before_any_new_range(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(extractor, "ARTIFACT_ROOT", Path(tmp)),
        ):
            paths = extractor.artifact_paths((96,))
            paths["evidence"].mkdir(parents=True)
            temp_receipt = paths["receipt"].with_suffix(
                paths["receipt"].suffix + ".tmp"
            )
            sentinel = b"preexisting possibly-uncertain receipt"
            temp_receipt.write_bytes(sentinel)
            server = FakeServer()

            with self.assertRaisesRegex(SystemExit, "receipt temp"):
                self.run_main(server, "--expert-id", "96")

            self.assertEqual(server.requests, [])
            self.assertEqual(temp_receipt.read_bytes(), sentinel)
            self.assertFalse(paths["receipt"].exists())

    def test_timeout_records_unknown_and_does_not_retry(self) -> None:
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(extractor, "ARTIFACT_ROOT", Path(tmp)),
        ):
            paths = extractor.artifact_paths((96,))
            with patch.object(
                extractor,
                "open_range_request",
                side_effect=TimeoutError("synthetic uncertain response"),
            ) as open_range:
                with self.assertRaisesRegex(SystemExit, "receipt saved"):
                    extractor.main(["--expert-id", "96"])

            self.assertEqual(open_range.call_count, 1)
            receipt = json.loads(paths["receipt"].read_text())
            self.assertEqual(
                receipt["status"], "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY"
            )
            self.assertEqual(receipt["last_failure"]["exception_type"], "TimeoutError")
            self.assertFalse(receipt["last_failure"]["automatic_retry"])
            self.assertFalse(paths["output"].exists())


class A120RedirectPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = extractor.PinnedHfRedirectHandler()

    def redirect(self, request, location: str):
        return self.policy.redirect_request(request, None, 302, "Found", {}, location)

    def test_known_hugging_face_storage_domains_and_relative_paths_are_allowed(
        self,
    ) -> None:
        request = Request(
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/resolve/pinned/shard",
            headers={"Range": "bytes=1-3"},
        )
        redirected = self.redirect(
            request, "https://cdn-lfs-us-1.hf.co:443/signed-object?signature=secret"
        )
        self.assertEqual(
            redirected.full_url,
            "https://cdn-lfs-us-1.hf.co:443/signed-object?signature=secret",
        )
        self.assertEqual(redirected.get_header("Range"), "bytes=1-3")
        self.assertEqual(self.policy.redirects[0]["host"], "cdn-lfs-us-1.hf.co")
        self.assertNotIn("secret", json.dumps(self.policy.redirects))

        same_host = self.redirect(redirected, "/another-path")
        self.assertEqual(
            same_host.full_url, "https://cdn-lfs-us-1.hf.co:443/another-path"
        )

    def test_untrusted_redirect_targets_are_rejected_before_following(self) -> None:
        bad_locations = (
            "http://cdn-lfs-us-1.hf.co/payload",
            "ftp://cdn-lfs-us-1.hf.co/payload",
            "https://huggingface.co.attacker.invalid/payload",
            "https://hf.co.attacker.invalid/payload",
            "https://user:password@cdn-lfs-us-1.hf.co/payload",
            "https://cdn-lfs-us-1.hf.co:8443/payload",
        )
        for location in bad_locations:
            with self.subTest(location=location):
                policy = extractor.PinnedHfRedirectHandler()
                request = Request("https://huggingface.co/repo/resolve/pinned/shard")
                with self.assertRaises(HTTPError):
                    policy.redirect_request(request, None, 302, "Found", {}, location)
                self.assertEqual(policy.redirects, [])

    def test_redirect_limit_is_enforced(self) -> None:
        request = Request("https://huggingface.co/repo/resolve/pinned/shard")
        for index in range(extractor.MAX_HF_REDIRECTS):
            request = self.redirect(request, f"https://cdn-lfs-us-1.hf.co/hop-{index}")
        with self.assertRaises(HTTPError):
            self.redirect(request, "https://huggingface.co/final")

    def test_range_reader_uses_policy_opener_instead_of_default_urlopen(self) -> None:
        request = Request("https://huggingface.co/repo/resolve/pinned/shard")
        observed = {}

        class FakeOpener:
            def __init__(self, handlers):
                observed["handlers"] = handlers

            def open(self, req, timeout):
                policy = next(
                    handler
                    for handler in observed["handlers"]
                    if isinstance(handler, extractor.PinnedHfRedirectHandler)
                )
                try:
                    policy.redirect_request(
                        req,
                        None,
                        302,
                        "Found",
                        {},
                        "http://127.0.0.1:8080/escape",
                    )
                except HTTPError:
                    observed["followed"] = False
                    raise
                observed["followed"] = True

        with (
            patch.object(
                extractor.urllib.request,
                "build_opener",
                side_effect=lambda *handlers: FakeOpener(handlers),
            ) as build_opener,
            patch.object(
                extractor.urllib.request,
                "urlopen",
                side_effect=AssertionError("default opener bypasses redirect policy"),
            ),
        ):
            with self.assertRaises(HTTPError):
                extractor.open_range_request(request)

        self.assertEqual(build_opener.call_count, 1)
        self.assertFalse(observed["followed"])


if __name__ == "__main__":
    unittest.main()
