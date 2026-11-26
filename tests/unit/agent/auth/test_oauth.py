import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest


@pytest.mark.parametrize("valid", [True, False])
def test_openrouter_pkce(valid):
    from mobilerun_agents.agent.auth.oauth import OAuthError, browser_key

    params = {}

    def callback(make_url):
        parts = urlsplit(make_url("http://127.0.0.1:1234/callback"))
        assert (parts.netloc, parts.path) == ("openrouter.ai", "/auth")
        params.update(parse_qs(parts.query))
        return {"code": ["fixture-code"], "state": params["state"] if valid else ["wrong"]}

    def exchange(req):
        assert valid and str(req.url) == "https://openrouter.ai/api/v1/auth/keys"
        body = json.loads(req.content)
        assert body["code"] == "fixture-code" and body["code_challenge_method"] == "S256"
        assert (
            base64.urlsafe_b64encode(hashlib.sha256(body["code_verifier"].encode()).digest())
            .rstrip(b"=")
            .decode()
            == params["code_challenge"][0]
        )
        return httpx.Response(200, json={"key": "fixture-key"})

    if valid:
        assert (
            browser_key(callback=callback, transport=httpx.MockTransport(exchange)) == "fixture-key"
        )
    else:
        with pytest.raises(OAuthError):
            browser_key(callback=callback, transport=httpx.MockTransport(exchange))
