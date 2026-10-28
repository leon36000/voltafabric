"""A45: the exception translations of scripts/vf_agents.py.

guard_schemata.py --sites py-handlers makes each `except ..: raise X(..)` re-raise
the original exception instead. Here each translation is reached by an input that
only it handles. The tests pin the type and exact message and, for RequestFailure,
`uncertain`, which decides whether the ledger records a call as rejected or as
unknown (never replayed). Ids in comments are py-handlers catalog ids.

No network: the Gateway's opener is an in-process fake that never connects.
"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from vf_agents import Gateway, RequestFailure, safe_context  # noqa: E402

ALIAS = "a45-alias"


class Response:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, n=-1):
        return self.body if n < 0 else self.body[:n]


class Opener:
    """Returns one canned body for every request and records the URLs."""

    def __init__(self, body):
        self.body = body
        self.urls = []

    def open(self, req, timeout=None):
        self.urls.append(req.full_url)
        return Response(self.body)


def gateway(body):
    g = Gateway("https://proxy.example", "dummy-test-key")
    g.opener = Opener(body)
    return g


def chat(content):
    return json.dumps(
        {
            "model": "upstream",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    ).encode()


class GatewayTranslations(unittest.TestCase):
    def failure(self, message, fn, *args):
        with self.assertRaisesRegex(RequestFailure, "^" + message + "$") as caught:
            fn(*args)
        self.assertIs(type(caught.exception), RequestFailure)
        self.assertIs(caught.exception.uncertain, True)

    def test_controls_pass(self):
        self.assertEqual(
            gateway(json.dumps({"data": [{"id": ALIAS}]}).encode()).models(), [ALIAS]
        )

    def test_response_not_json(self):  # 23: JSONDecodeError
        self.failure("INVALID_RESPONSE_JSON", gateway(b"{not json").models)

    def test_response_not_utf8(self):  # 23: UnicodeDecodeError
        self.failure("INVALID_RESPONSE_JSON", gateway(b'{"data": "\xff"}').models)

    def test_proposal_content_not_json(self):  # 24: JSONDecodeError
        g = gateway(chat("not json"))
        self.failure(
            "INVALID_PROPOSAL_SCHEMA",
            g.propose,
            ALIAS,
            [{"role": "user", "content": "x"}],
            64,
        )
        self.assertEqual(g.opener.urls, ["https://proxy.example/v1/chat/completions"])

    def test_proposal_without_choices(self):  # 24: KeyError
        g = gateway(json.dumps({"model": "upstream"}).encode())
        self.failure(
            "INVALID_PROPOSAL_SCHEMA",
            g.propose,
            ALIAS,
            [{"role": "user", "content": "x"}],
            64,
        )


class ContextTranslation(unittest.TestCase):
    def test_context_not_utf8(self):  # 20: UnicodeDecodeError
        with tempfile.TemporaryDirectory(prefix="a45-ctx-") as tmp:
            root = Path(tmp)
            (root / "ok.txt").write_text("plain\n")
            (root / "bad.txt").write_bytes(b"caf\xe9\n")
            self.assertEqual(safe_context(root, ["ok.txt"]), {"ok.txt": "plain\n"})
            with self.assertRaisesRegex(
                ValueError, "^context must be UTF-8 text$"
            ) as caught:
                safe_context(root, ["bad.txt"])
            self.assertIs(type(caught.exception), ValueError)


if __name__ == "__main__":
    unittest.main()
