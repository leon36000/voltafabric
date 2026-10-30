import json
import hashlib
import struct
import subprocess
import sys
import unittest
from unittest.mock import patch

from adapters.strata_tokenizer.prompt_runner import _run_bounded, run_prompt as _run_prompt
from adapters.strata_tokenizer.test_adapter import FakeTokenizer


def run_prompt(*args, **kwargs):
    # The old synthetic fixtures deliberately exercise legacy compatibility.
    kwargs.setdefault("require_bound_plan", False)
    return _run_prompt(*args, **kwargs)


class Tokenizer(FakeTokenizer):
    def encode(self, text, parse_special=False):
        return {"A": [0], "multi": [0, 1], "eight": [0] * 8, "long32": [0] * 32, "special": [3], "too many": [0] * 33}.get(text, [])


def receipt():
    return {"layer_count": 48, "run_id": "fresh", "workers_closed": 96, "eos_token_id": 3,
            "input_tokens": [0, 1], "generated_token_ids": [1, 2],
            "output_sha256": ["a" * 64] * 2, "logits_sha256": ["b" * 64] * 2}


class RunnerTests(unittest.TestCase):
    def test_expected_plan_is_checked_before_and_after_engine(self):
        value = dict(receipt(), plan_id="r512-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        def run():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": value["plan_id"]}):
            self.assertEqual(run()["output"]["text"], "😀")
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": "r512-" + "b" * 64}):
            with self.assertRaisesRegex(ValueError, "differs from expected plan"):
                run()
        count = len(calls)
        with patch.dict("os.environ", {"VF_EXPECTED_PLAN_ID": "malformed"}):
            with self.assertRaisesRegex(ValueError, "invalid expected plan"):
                run()
        self.assertEqual(len(calls), count)

    def test_strict_plan_rejects_missing_unknown_malformed_and_missing_digest(self):
        value = dict(receipt(), plan_id="r512-" + "a" * 64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        def run():
            return _run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(run()["output"]["text"], "😀")
        for bad in (None, "r508-" + "a" * 64, "r512-" + "A" * 64,
                    "r512-" + "a" * 63, 123):
            value["plan_id"] = bad
            with self.assertRaisesRegex(ValueError, "bound plan identity"):
                run()
        del value["plan_id"]
        with self.assertRaisesRegex(ValueError, "bound plan identity"):
            run()
        value["plan_id"] = "r512-" + "a" * 64
        for bad in (None, "0" * 64):
            value["prompt_tokens_sha256"] = bad
            with self.assertRaisesRegex(ValueError, "prompt digest"):
                run()
        del value["prompt_tokens_sha256"]
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run()

    def test_prompt_digest_is_checked_when_present_or_required_by_plan(self):
        value = dict(receipt(), plan_id="r506-" + "a"*64,
                     prompt_tokens_sha256=hashlib.sha256(struct.pack("<I", 0)).hexdigest())
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        self.assertEqual(run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)["output"]["text"], "😀")
        for bad in ("0"*64, None):
            value["prompt_tokens_sha256"] = bad
            with self.assertRaisesRegex(ValueError, "prompt digest"):
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        del value["prompt_tokens_sha256"]
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        value["plan_id"] = "r508-" + "a"*64
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        value["plan_id"] = "r512-" + "a"*64
        with self.assertRaisesRegex(ValueError, "prompt digest"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_eos_binding_is_passed_and_checked_before_returning_text(self):
        calls = []
        value = dict(receipt(), input_tokens=[0], generated_token_ids=[3],
                     output_sha256=["a" * 64], logits_sha256=["b" * 64])
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        result = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)
        self.assertEqual(result["output"]["stop_reason"], "eos")
        self.assertEqual(calls[0][-2:], ["2", "3"])
        for invalid in (None, 2, True):
            altered = dict(value, eos_token_id=invalid)
            with self.assertRaisesRegex(ValueError, "tokenizer EOS"):
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8,
                           runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(altered).encode(), b""))
        missing = dict(value)
        del missing["eos_token_id"]
        with self.assertRaisesRegex(ValueError, "tokenizer EOS"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(missing).encode(), b""))
        with self.assertRaisesRegex(ValueError, "invalid tokenizer EOS"):
            run_prompt("A", Tokenizer(), 0, "/engine", ["/input"]*8, runner=runner)

    def test_long_prompt_and_combined_position_bound(self):
        value = dict(receipt(), prompt_tokens=[0]*32, input_tokens=[0]*39,
                     requested_max_new_tokens=8, generated_token_ids=[0]*8,
                     output_sha256=["a"*64]*39, logits_sha256=["b"*64]*8)
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        result = run_prompt("long32", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=8, runner=runner)
        self.assertEqual(result["output"]["text"], "A"*8)
        with self.assertRaises(ValueError):
            run_prompt("long32", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=9, runner=runner)
        self.assertEqual(len(calls), 1)
    def test_bounded_capture_drains_both_pipes_and_marks_overflow(self):
        program = ('import sys; sys.stdout.buffer.write(b"x"*200000); sys.stdout.flush(); '
                   'sys.stderr.buffer.write(b"y"*200000); sys.stderr.flush()')
        result = _run_bounded([sys.executable, "-c", program], stdout_limit=1024, stderr_limit=512)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, b"x"*1024)
        self.assertEqual(result.stderr, b"y"*512)
        self.assertTrue(result.stdout_truncated)
        self.assertTrue(result.stderr_truncated)
        exact = _run_bounded([sys.executable, "-c", 'print("abc", end="")'], stdout_limit=3)
        self.assertEqual(exact.stdout, b"abc")
        self.assertFalse(exact.stdout_truncated)

    def test_large_complete_receipt_crosses_real_pipe(self):
        program = ('import json; r={"layer_count":48,"run_id":"fresh","workers_closed":96,'
                   '"prompt_tokens":[0]*8,"input_tokens":[0]*39,"generated_token_ids":[0]*32,'
                   '"requested_max_new_tokens":32,"eos_token_id":3,"output_sha256":["a"*64]*39,'
                   '"logits_sha256":["b"*64]*32,"padding":"x"*2000000}; print(json.dumps(r))')
        def runner(args, **kwargs):
            return _run_bounded([sys.executable, "-c", program], **kwargs)
        result = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8,
                            max_new_tokens=32, runner=runner)
        self.assertEqual(result["output"]["text"], "A"*32)

    def test_truncated_success_is_not_decoded(self):
        value = json.dumps(receipt()).encode()
        def runner(args, **kwargs):
            result = subprocess.CompletedProcess(args, 0, value, b"")
            result.stdout_truncated = True
            return result
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_truncated_error_is_never_misclassified(self):
        def runner(args, **kwargs):
            result = subprocess.CompletedProcess(args, 1, b"", b"prefill-greedy qualification refused: worker shutdown receipt\n")
            result.stderr_truncated = True
            return result
        with self.assertRaisesRegex(RuntimeError, "unclassified"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"]*8, runner=runner)

    def test_runs_once_and_returns_complete_text(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(receipt()).encode(), b"")
        out = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(out["output"]["text"], "😀")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-4:], ["48", "0", "2", "3"])

    def test_two_token_prompt_is_prefilled_before_generation(self):
        calls = []
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1, 0],
                     generated_token_ids=[0, 0], output_sha256=["a" * 64] * 3)
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(calls[0][-3:], ["0,1", "2", "3"])
        self.assertEqual(out["output"]["text"], "AA")
        bad = dict(value, input_tokens=[0, 0, 0])
        with self.assertRaises(ValueError):
            run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(bad).encode(), b""))

    def test_four_tokens_and_incomplete_receipt_are_checked(self):
        calls = []
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1, 0, 0, 0],
                     requested_max_new_tokens=4, generated_token_ids=[0, 0, 0, 0],
                     output_sha256=["a" * 64] * 5, logits_sha256=["b" * 64] * 4)
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=4, runner=runner)
        self.assertEqual(out["output"]["text"], "AAAA")
        self.assertEqual(calls[0][-4:], ["48", "0,1", "4", "3"])
        short = dict(value, input_tokens=[0, 1], generated_token_ids=[0],
                     output_sha256=["a" * 64] * 2, logits_sha256=["b" * 64])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=4,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(short).encode(), b""))
        with self.assertRaises(ValueError):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=33, runner=runner)

    def test_32_outputs_and_39_positions_are_bounded_and_atomic(self):
        value = dict(receipt(), prompt_tokens=[0]*8, input_tokens=[0]*39,
                     requested_max_new_tokens=32, generated_token_ids=[0]*32,
                     output_sha256=["a" * 64]*39, logits_sha256=["b" * 64]*32)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=32, runner=runner)
        self.assertEqual(out["output"]["text"], "A"*32)
        incomplete = dict(value, generated_token_ids=[0]*31, input_tokens=[0]*38,
                          output_sha256=["a"*64]*38, logits_sha256=["b"*64]*31)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"]*8, max_new_tokens=32,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(incomplete).encode(), b""))

    def test_eight_token_bound_and_prompt_bound(self):
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1] + [0]*7,
                     requested_max_new_tokens=8, generated_token_ids=[0]*8,
                     output_sha256=["a" * 64] * 9, logits_sha256=["b" * 64] * 8)
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=8, runner=runner)
        self.assertEqual(out["output"]["text"], "A" * 8)
        prefill = dict(receipt(), prompt_tokens=[0]*8, input_tokens=[0]*8,
                       requested_max_new_tokens=1, generated_token_ids=[0],
                       output_sha256=["a" * 64] * 8, logits_sha256=["b" * 64])
        out = run_prompt("eight", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=1,
                         runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(prefill).encode(), b""))
        self.assertEqual(out["prompt_token_ids"], [0]*8)

    def test_refuses_prompt_before_spawning_and_never_retries_failure(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, b"partial", b"failure")
        for prompt in ("", "too many", "special"):
            with self.assertRaises(ValueError):
                run_prompt(prompt, Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(calls, [])
        with self.assertRaisesRegex(RuntimeError, "do not replay"):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(len(calls), 1)

    def test_rejects_receipt_for_different_prompt(self):
        bad = dict(receipt(), input_tokens=[4, 1])
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, 0, json.dumps(bad).encode(), b"")
        with self.assertRaises(ValueError):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)

    def test_failure_diagnostic_is_allowlisted_and_never_returns_partial_output(self):
        cases = [
            (b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity\n", "rank_payload_capacity"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: expert payload budget exceeded\n", "global_payload_capacity"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: eviction target cannot be met safely\n", "eviction_target"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: eviction limit\n", "eviction_limit"),
            (b"prefill-greedy qualification refused: worker shutdown receipt\n", "worker_shutdown"),
            (b"prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity SECRET\n", "unclassified"),
            (b"private path or secret", "unclassified"),
        ]
        for stderr, category in cases:
            calls = []
            def runner(args, **kwargs):
                calls.append(args)
                return subprocess.CompletedProcess(args, 1, b"partial secret", stderr)
            with self.subTest(category=category), self.assertRaises(RuntimeError) as caught:
                run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
            self.assertIn(category, str(caught.exception))
            self.assertNotIn("secret", str(caught.exception).lower())
            self.assertEqual(len(calls), 1)

    def test_real_subprocess_reports_known_failure_without_stdout(self):
        def runner(args, **kwargs):
            program = ('import sys; print("private partial output"); '
                       'print("prefill-greedy qualification refused: INVALID_INPUT: no rank has expert payload capacity", file=sys.stderr); '
                       'sys.exit(1)')
            return subprocess.run([sys.executable, "-c", program], **kwargs)
        with self.assertRaises(RuntimeError) as caught:
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertIn("rank_payload_capacity", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))

    def test_signal_status_is_preserved_without_exposing_stderr(self):
        def runner(args, **kwargs):
            return subprocess.CompletedProcess(args, -9, b"secret stdout", b"secret stderr")
        with self.assertRaises(RuntimeError) as caught:
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertIn("unclassified, exit=-9", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
