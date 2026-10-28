from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
FETCH_SCRIPT = ROOT / "evidence/continuation_a105_all_backbone_headers/fetch_headers.py"
SPEC = importlib.util.spec_from_file_location("a105_fetch_headers", FETCH_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fetch_headers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_headers)


class A105BackboneFetchTests(unittest.TestCase):
    def test_only_missing_backbone_headers_are_selected_for_fetch(self) -> None:
        _, shards = fetch_headers.index_shards()

        self.assertEqual(len(shards), 39)
        self.assertNotIn("model-00003-of-00048.safetensors", shards)
        selected_layers = {
            int(fetch_headers.EXPERT_KEY.fullmatch(name).group(1))
            for names in shards.values()
            for name in names
            if fetch_headers.EXPERT_KEY.fullmatch(name)
        }
        self.assertEqual(selected_layers, set(range(1, 40)))

    def test_resume_receipt_rejects_duplicate_shard_before_network_work(self) -> None:
        index_bytes, shards = fetch_headers.index_shards()
        repeated = sorted(shards)[0]
        receipt = {
            "schema_version": 1,
            "status": "INCOMPLETE_DO_NOT_REPLAY_AUTOMATICALLY",
            "repository": fetch_headers.REPOSITORY,
            "revision": fetch_headers.REVISION,
            "index_sha256": fetch_headers.INDEX_SHA256,
            "max_header_bytes": fetch_headers.MAX_HEADER_BYTES,
            "full_checkpoint_downloaded": False,
            "payload_downloaded": False,
            "requested_shards": sorted(shards),
            "network_response_body_bytes": 0,
            "shards": [{"file": repeated}, {"file": repeated}],
        }

        with self.assertRaisesRegex(ValueError, "duplicate shard"):
            fetch_headers._validate_resume_receipt(receipt, index_bytes, shards)

    def test_layer_lookup_ignores_non_expert_names_in_the_same_shard(self) -> None:
        names = {
            "layers.30.ffn.experts.7.w3.scale",
            "layers.30.attention.weight",
        }

        self.assertEqual(fetch_headers._layer_for_shard(names), 30)

    def test_layer_lookup_rejects_a_shard_split_across_layers(self) -> None:
        names = {
            "layers.30.ffn.experts.7.w3.scale",
            "layers.31.ffn.experts.9.w1.weight",
        }

        with self.assertRaisesRegex(ValueError, "exactly one backbone layer"):
            fetch_headers._layer_for_shard(names)


if __name__ == "__main__":
    unittest.main()
