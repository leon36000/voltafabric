import json
import subprocess
import sys
import unittest

from adapters.strata_tokenizer.prompt_runner import run_prompt
from adapters.strata_tokenizer.test_adapter import FakeTokenizer


class Tokenizer(FakeTokenizer):
    def encode(self, text, parse_special=False):
        return {"A": [0], "multi": [0, 1], "eight": [0] * 8, "special": [3], "too many": [0] * 9}.get(text, [])


def receipt():
    return {"layer_count": 48, "run_id": "fresh", "workers_closed": 96,
            "input_tokens": [0, 1], "generated_token_ids": [1, 2],
            "output_sha256": ["a" * 64] * 2, "logits_sha256": ["b" * 64] * 2}


class RunnerTests(unittest.TestCase):
    def test_runs_once_and_returns_complete_text(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(receipt()).encode(), b"")
        out = run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(out["output"]["text"], "😀")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][-2:], ["48", "0"])

    def test_two_token_prompt_is_prefilled_before_generation(self):
        calls = []
        value = dict(receipt(), prompt_tokens=[0, 1], input_tokens=[0, 1, 0],
                     generated_token_ids=[0, 0], output_sha256=["a" * 64] * 3)
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, json.dumps(value).encode(), b"")
        out = run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertEqual(calls[0][-1], "0,1")
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
        self.assertEqual(calls[0][-3:], ["48", "0,1", "4"])
        short = dict(value, input_tokens=[0, 1], generated_token_ids=[0],
                     output_sha256=["a" * 64] * 2, logits_sha256=["b" * 64])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            run_prompt("multi", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=4,
                       runner=lambda args, **kw: subprocess.CompletedProcess(args, 0, json.dumps(short).encode(), b""))
        with self.assertRaises(ValueError):
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, max_new_tokens=9, runner=runner)

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
            (b"prefill-greedy qualification refused: no rank has expert payload capacity\n", "rank_payload_capacity"),
            (b"prefill-greedy qualification refused: expert payload budget exceeded\n", "global_payload_capacity"),
            (b"prefill-greedy qualification refused: eviction target cannot be met safely\n", "eviction_target"),
            (b"prefill-greedy qualification refused: eviction limit\n", "eviction_limit"),
            (b"prefill-greedy qualification refused: no rank has expert payload capacity SECRET\n", "unclassified"),
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
                       'print("prefill-greedy qualification refused: no rank has expert payload capacity", file=sys.stderr); '
                       'sys.exit(1)')
            return subprocess.run([sys.executable, "-c", program], **kwargs)
        with self.assertRaises(RuntimeError) as caught:
            run_prompt("A", Tokenizer(), 3, "/engine", ["/input"] * 8, runner=runner)
        self.assertIn("rank_payload_capacity", str(caught.exception))
        self.assertNotIn("private", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
