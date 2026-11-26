"""OpenRouter's user-approved API key flow. No registered client ID is required."""

import base64
import hashlib
import secrets
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlencode, urlsplit

import httpx
from pydantic import BaseModel, SecretStr, ValidationError


class OAuthError(Exception):
    """Safe to display; never includes request or response contents."""


class _Key(BaseModel):
    key: SecretStr


def _callback(make_url: Callable[[str], str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}

    class Handler(BaseHTTPRequestHandler):
        timeout = 2

        def log_message(self, format: str, *args: object) -> None:
            pass  # Request targets contain authorization codes.

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            if url.path != "/callback":
                self.send_error(404)
                return
            result.update(parse_qs(url.query) or {"error": ["cancelled"]})
            self.send_response(200)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(b"Return to the terminal to finish setup.")

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        target = make_url(f"http://127.0.0.1:{server.server_port}/callback")
        if not webbrowser.open(target):
            raise OAuthError("Could not open the browser. Use an API key instead.")
        deadline = time.monotonic() + 120
        while not result:
            server.timeout = deadline - time.monotonic()
            if server.timeout <= 0:
                raise OAuthError("Browser sign-in timed out.")
            server.handle_request()
    return result


def browser_key(
    callback: Callable[[Callable[[str], str]], dict[str, list[str]]] = _callback,
    transport: httpx.BaseTransport | None = None,
) -> str:
    verifier, state = secrets.token_urlsafe(48), secrets.token_urlsafe(32)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    )

    def auth_url(callback_url: str) -> str:
        return "https://openrouter.ai/auth?" + urlencode(
            dict(
                callback_url=callback_url,
                state=state,
                code_challenge=challenge,
                code_challenge_method="S256",
            )
        )

    try:
        response = callback(auth_url)
        codes = response.get("code", [])
        if response.get("state") != [state] or len(codes) != 1 or not codes[0]:
            raise OAuthError("Browser sign-in cancelled or invalid callback.")
        with httpx.Client(transport=transport, timeout=5, trust_env=False) as client:
            reply = client.post(
                "https://openrouter.ai/api/v1/auth/keys",
                json=dict(code=codes[0], code_verifier=verifier, code_challenge_method="S256"),
            )
            reply.raise_for_status()
            key = _Key.model_validate_json(reply.content).key.get_secret_value().strip()
            if not key:
                raise OAuthError("No key returned by OpenRouter.")
            return key
    except (OSError, httpx.HTTPError, ValidationError):
        raise OAuthError("Browser sign-in failed. Try again or use an API key.") from None
