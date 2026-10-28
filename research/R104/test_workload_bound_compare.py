"""CPU-only contract checks for the private R104 adapter; fixtures are synthetic."""
import hashlib
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
PRIVATE_TMP = HERE / ".private-tmp"
SOURCE = HERE / "workload_bound_compare.py"
_SPEC = importlib.util.spec_from_file_location("workload_bound_compare_tested", SOURCE)
adapter = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(adapter)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _dump(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")


class BoundCompareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="r104-test-", dir=PRIVATE_TMP)
        self.root = pathlib.Path(self.temp.name)
        self.corpus = self.root / "corpus.json"
        self.input_counts = self.root / "input.json"
        self.output_counts = self.root / "output.json"
        self.quality_a = self.root / "quality-a.json"
        self.quality_b = self.root / "quality-b.json"
        self.left = self.root / "left.json"
        self.right = self.root / "right.json"
        self.payloads = [
            b'{"schema":"r100-prompt-corpus-v1","requests":[{"id":"r1","prompt":"Hello"}]}',
            b'{"schema":"r100-token-counts-v1","requests":[{"id":"r1","count":2}]}',
            b'{"schema":"r100-token-counts-v1","requests":[{"id":"r1","count":3}]}',
        ]
        self.corpus.write_bytes(self.payloads[0])
        self.input_counts.write_bytes(self.payloads[1])
        self.output_counts.write_bytes(self.payloads[2])
        self._write_quality(self.quality_a, "d" * 64)
        self._write_quality(self.quality_b, "d" * 64)
        self._write_runs()

    def tearDown(self):
        self.temp.cleanup()

    def _write_quality(self, path, package_hash):
        raw = _dump({"schema": "r97-quality-v1", "package_sha256": package_hash,
                     "dataset_sha256": "e" * 64, "reference_sha256": "f" * 64,
                     "metric": "fixture-score", "higher_is_better": True,
                     "threshold": 0.5, "score": 0.9, "sample_count": 2})
        path.write_bytes(raw)
        return _sha(raw)

    def _run(self, engine, quality_path, quality_hash, package="d" * 64):
        return {
            "schema": "r97-run-v1",
            "engine": {"name": engine, "revision": "synthetic-fixture"},
            "model": {"logical_id": "fixture/model", "base_sha256": "a" * 64,
                      "tokenizer_sha256": "b" * 64, "package_sha256": package,
                      "quantization": "fp16", "layout": "dense"},
            "environment": {"hardware_id": "synthetic-only", "driver": "fixture",
                            "device_count": 1, "parallelism": "single",
                            "software_boundary": "synthetic fixture"},
            "workload": {"prompt_corpus_sha256": _sha(self.payloads[0]),
                         "token_counts_sha256": _sha(self.payloads[1]),
                         "sampling_sha256": "c" * 64, "concurrency": 1,
                         "max_new_tokens": 4, "prefix_cache": False,
                         "warmup_policy": "none", "measurement_boundary": "fixture"},
            "metrics": {"requests": 1, "output_tokens": 3,
                         "output_token_counts_sha256": _sha(self.payloads[2]),
                         "wall_seconds": 1.0, "ttft_p50_ms": 1.0,
                         "ttft_p95_ms": 2.0, "tpot_p50_ms": 1.0, "tpot_p95_ms": 2.0},
            "quality": {"evidence_path": str(quality_path), "evidence_sha256": quality_hash},
        }

    def _write_runs(self, left=None, right=None):
        qa_hash = _sha(self.quality_a.read_bytes())
        qb_hash = _sha(self.quality_b.read_bytes())
        a = left or self._run("synthetic-A", self.quality_a, qa_hash)
        b = right or self._run("synthetic-B", self.quality_b, qb_hash)
        self.left.write_bytes(_dump(a))
        self.right.write_bytes(_dump(b))

    def _compare(self):
        return adapter.compare_bound_files(str(self.left), str(self.right),
                                           str(self.corpus), str(self.input_counts),
                                           str(self.output_counts))

    def test_positive_fixture_binds_same_workload_to_two_runs(self):
        result = self._compare()
        self.assertEqual(result["verdict"], "ENGINE_COMPARABLE_SAME_PACKAGE")
        self.assertEqual(result["workload_admission"]["input_tokens"], 2)
        self.assertEqual(result["workload_admission"]["output_tokens"], 3)
        self.assertEqual(result["workload_admission"]["prompt_corpus_sha256"], _sha(self.payloads[0]))
        self.assertTrue(any("producer declarations" in item for item in result["limitations"]))
        self.assertTrue(any("Synthetic records" in item for item in result["limitations"]))

    def test_right_run_must_link_to_same_workload_hashes(self):
        right = self._run("synthetic-B", self.quality_b, _sha(self.quality_b.read_bytes()))
        right["workload"]["prompt_corpus_sha256"] = "0" * 64
        self._write_runs(right=right)
        self.assertEqual(self._compare()["verdict"], "NON_COMPARABLE")

    def test_substitution_after_artifact_snapshot_uses_hashed_bytes(self):
        original = adapter._r103._path_bytes
        changed = b'{"schema":"r100-prompt-corpus-v1","requests":[{"id":"r1","prompt":"After"}]}'
        def snapshot_then_substitute(path):
            result = original(path)
            if path == str(self.corpus):
                self.corpus.write_bytes(changed)
            return result
        with mock.patch.object(adapter._r103, "_path_bytes", side_effect=snapshot_then_substitute):
            result = self._compare()
        self.assertEqual(result["verdict"], "ENGINE_COMPARABLE_SAME_PACKAGE")
        self.assertEqual(result["workload_admission"]["prompt_corpus_sha256"], _sha(self.payloads[0]))
        self.assertEqual(_sha(self.corpus.read_bytes()), _sha(changed))

    def test_quality_substitution_after_hash_uses_hashed_bytes(self):
        original = adapter._read_quality
        changed = b'{"schema":"r97-quality-v1","replacement":true}'
        def snapshot_then_substitute(path):
            result = original(path)
            if path == str(self.quality_a):
                self.quality_a.write_bytes(changed)
            return result
        with mock.patch.object(adapter, "_read_quality", side_effect=snapshot_then_substitute):
            result = self._compare()
        self.assertEqual(result["verdict"], "ENGINE_COMPARABLE_SAME_PACKAGE")
        self.assertEqual(_sha(self.quality_a.read_bytes()), _sha(changed))

    def test_duplicate_json_key_is_rejected_even_with_matching_run_hash(self):
        duplicate = b'{"schema":"r100-prompt-corpus-v1","schema":"r100-prompt-corpus-v1","requests":[{"id":"r1","prompt":"Hello"}]}'
        self.corpus.write_bytes(duplicate)
        self.payloads[0] = duplicate
        self._write_runs()
        self.assertEqual(self._compare()["verdict"], "NON_COMPARABLE")

    def test_boolean_token_count_is_rejected(self):
        invalid = b'{"schema":"r100-token-counts-v1","requests":[{"id":"r1","count":true}]}'
        self.input_counts.write_bytes(invalid)
        self.payloads[1] = invalid
        self._write_runs()
        self.assertEqual(self._compare()["verdict"], "NON_COMPARABLE")

    def test_false_quality_hash_is_rejected(self):
        a = self._run("synthetic-A", self.quality_a, "0" * 64)
        b = self._run("synthetic-B", self.quality_b, _sha(self.quality_b.read_bytes()))
        self._write_runs(left=a, right=b)
        self.assertEqual(self._compare()["verdict"], "NON_COMPARABLE")

    def test_quality_fifo_is_rejected_without_blocking(self):
        fifo = self.root / "quality.fifo"
        os.mkfifo(fifo)
        a = self._run("synthetic-A", fifo, "0" * 64)
        b = self._run("synthetic-B", fifo, "0" * 64)
        self._write_runs(left=a, right=b)
        child = ("import importlib.util,json,sys;"
                 "p=sys.argv[1];s=importlib.util.spec_from_file_location('bound',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                 "r=m.compare_bound_files(*sys.argv[2:]);print(json.dumps(r));sys.exit(0 if r.get('verdict')=='NON_COMPARABLE' else 1)")
        argv = [sys.executable, "-c", child, str(SOURCE), str(self.left), str(self.right),
                str(self.corpus), str(self.input_counts), str(self.output_counts)]
        started = time.monotonic()
        try:
            completed = subprocess.run(argv, capture_output=True, text=True, timeout=3)
        except subprocess.TimeoutExpired as exc:
            (HERE / "logs").mkdir(exist_ok=True)
            (HERE / "logs" / "fifo-check.log").write_text(
                "ARGV=" + json.dumps(argv) + "\nTIMEOUT_SECONDS=3\nEXIT_CODE=TIMEOUT\n",
                encoding="utf-8")
            self.fail("quality FIFO check blocked: " + str(exc))
        elapsed = time.monotonic() - started
        (HERE / "logs").mkdir(exist_ok=True)
        (HERE / "logs" / "fifo-check.log").write_text(
            "ARGV=" + json.dumps(argv) + "\nELAPSED_SECONDS=" + str(elapsed) +
            "\nSTDOUT=" + completed.stdout + "\nSTDERR=" + completed.stderr +
            "\nEXIT_CODE=" + str(completed.returncode) + "\n",
            encoding="utf-8")
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("regular file", completed.stdout)

    def test_incomparable_package_pair_remains_non_comparable(self):
        self._write_quality(self.quality_b, "1" * 64)
        a = self._run("synthetic-A", self.quality_a, _sha(self.quality_a.read_bytes()))
        b = self._run("synthetic-B", self.quality_b, _sha(self.quality_b.read_bytes()), package="1" * 64)
        self._write_runs(left=a, right=b)
        result = self._compare()
        self.assertEqual(result["verdict"], "NON_COMPARABLE")
        self.assertTrue(any("ambiguous" in reason for reason in result["reasons"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
