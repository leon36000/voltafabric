"""A53-R39: the gateway must refuse a direct provider endpoint however its host is spelled.

Gateway compared the lowered host with the provider domains, but http.client, getaddrinfo
and the TLS server name all IDNA-encode the host: `api.openai\\uff0ecom` (fullwidth full stop),
`\\u3002` / `\\uff61` dots or fullwidth letters reach api.openai.com, with the gateway key.
Only the constructor runs here: no request is sent.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from vf_agents import Gateway  # noqa: E402

DIRECT = "direct provider endpoint forbidden; use the existing LiteLLM gateway"


class GatewayIdna(unittest.TestCase):
    def test_unicode_spellings_of_provider_hosts_are_refused(self):
        for url in (
            "https://api.openai．com/v1",
            "https://api.openai。com/v1",
            "https://api.openai｡com/v1",
            "https://API.ＯＰＥＮＡＩ.com/v1",
            "https://api.anthropic．com/v1",
            "https://openrouter。ai/api/v1",
            "https://api.openai．com．/v1",
        ):
            with self.assertRaisesRegex(ValueError, DIRECT, msg=ascii(url)):
                Gateway(url, "dummy-key")

    def test_a_host_without_an_idna_form_is_refused(self):
        with self.assertRaisesRegex(ValueError, "invalid explicit gateway URL"):
            Gateway("https://" + "a" * 64 + ".example.com/v1", "dummy-key")

    def test_ordinary_gateways_are_still_accepted(self):
        Gateway("https://gateway.example.com/v1", "dummy-key")
        Gateway("http://127.0.0.1:4000/v1", "dummy-key", allow_local_http=True)
        Gateway("http://[::1]:4000/v1", "dummy-key", allow_local_http=True)


if __name__ == "__main__":
    unittest.main()
