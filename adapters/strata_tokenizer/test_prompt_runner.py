import json
import subprocess
import unittest

from adapters.strata_tokenizer.prompt_runner import run_prompt
from adapters.strata_tokenizer.test_adapter import FakeTokenizer


class Tokenizer(FakeTokenizer):
    def encode(self, text, parse_special=False):
        return {"A": [0], "multi": [0, 1], "special": [3]}.get(text, [])


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

    def test_refuses_prompt_before_spawning_and_never_retries_failure(self):
        calls = []
        def runner(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 1, b"partial", b"failure")
        for prompt in ("", "multi", "special"):
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


if __name__ == "__main__":
    unittest.main()
