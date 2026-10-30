import json
import os
from pathlib import Path
import unittest

from adapters.strata_tokenizer.adapter import GreedyDecoder, _unique_object, load_sc117, decode_greedy_receipt


class FakeTokenizer:
    tokens = ["start", "partial", "end", "eos", "control", "bad", "unused"]
    token_types = [1, 1, 1, 3, 3, 1, 5]
    pieces = [b"A", b"\xf0\x9f", b"\x98\x80", b"", b"", b"\xff", b"x"]

    def token_bytes(self, token):
        return self.pieces[token]


class AdapterTests(unittest.TestCase):
    def test_streams_split_utf8_and_stops_at_eos(self):
        d = GreedyDecoder(FakeTokenizer(), 3)
        self.assertEqual(d.push(0), ("A", False))
        self.assertEqual(d.push(1), ("", False))
        self.assertEqual(d.push(2), ("😀", False))
        self.assertEqual(d.push(3), ("", True))
        with self.assertRaises(ValueError):
            d.push(0)

    def test_incomplete_bytes_invalid_control_and_invalid_ids_refuse(self):
        d = GreedyDecoder(FakeTokenizer(), 3)
        d.push(1)
        with self.assertRaises(UnicodeDecodeError):
            d.push(3)
        self.assertTrue(d.stopped)
        for bad in (-1, 7, True, "1", 4, 6):
            d = GreedyDecoder(FakeTokenizer(), 3)
            with self.assertRaises(ValueError):
                d.push(bad)
            self.assertTrue(d.stopped)
            with self.assertRaises(ValueError):
                d.push(0)
        d = GreedyDecoder(FakeTokenizer(), 3)
        with self.assertRaises(UnicodeDecodeError):
            d.push(5)
        self.assertTrue(d.stopped)

    def test_receipt_decodes_atomically_and_checks_autoregression(self):
        receipt = {"layer_count": 48, "run_id": "r", "workers_closed": 96,
                   "input_tokens": [0, 1], "generated_token_ids": [1, 2],
                   "output_sha256": ["a" * 64] * 2, "logits_sha256": ["b" * 64] * 2}
        self.assertEqual(decode_greedy_receipt(FakeTokenizer(), 3, receipt)["text"], "😀")
        eos = dict(receipt, input_tokens=[0], generated_token_ids=[3],
                   output_sha256=["a" * 64], logits_sha256=["b" * 64])
        self.assertEqual(decode_greedy_receipt(FakeTokenizer(), 3, eos)["stop_reason"], "eos")
        bad = dict(receipt, input_tokens=[True, 1])
        with self.assertRaises(ValueError):
            decode_greedy_receipt(FakeTokenizer(), 3, bad)
        bad = dict(receipt, input_tokens=[0, 0])
        with self.assertRaises(ValueError):
            decode_greedy_receipt(FakeTokenizer(), 3, bad)
        bad = dict(receipt, input_tokens=[0, 3], generated_token_ids=[3, 0])
        with self.assertRaises(ValueError):
            decode_greedy_receipt(FakeTokenizer(), 3, bad)
        bad = dict(receipt, output_sha256=["bad"])
        with self.assertRaises(ValueError):
            decode_greedy_receipt(FakeTokenizer(), 3, bad)

    def test_duplicate_json_keys_refused(self):
        with self.assertRaises(ValueError):
            _unique_object([("a", 1), ("a", 2)])


class ObservedPackTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("VF_SC117_TOKENIZER_DIR"), "observed tokenizer pack not configured")
    def test_pinned_pack_and_wrong_digest(self):
        manifest = json.loads((Path(__file__).parent / "observed-sc117.json").read_text(encoding="utf-8"))
        hashes = {name: manifest[name] for name in ("vocab.json", "merges.txt", "token_type.json", "tokenizer.json")}
        tokenizer, eos = load_sc117(os.environ["VF_SC117_TOKENIZER_DIR"], hashes)
        self.assertEqual(eos, 248046)
        decoder = GreedyDecoder(tokenizer, eos)
        self.assertEqual("".join(decoder.push(i)[0] for i in (42, 3649, 1277)) + decoder.finish(), "Kıbr")
        hashes["vocab.json"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "digest mismatch"):
            load_sc117(os.environ["VF_SC117_TOKENIZER_DIR"], hashes)
        with self.assertRaises(ValueError):
            load_sc117(os.environ["VF_SC117_TOKENIZER_DIR"], {})


if __name__ == "__main__":
    unittest.main()
