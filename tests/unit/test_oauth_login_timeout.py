from __future__ import annotations

import math
from types import SimpleNamespace

import httpx
import pytest

from mobilerun.agents.clients.oauth import anthropic, gemini, openai
from mobilerun.agents.clients.oauth.anthropic import AnthropicOAuthSession
from mobilerun.agents.clients.oauth.gemini import GeminiOAuthSession
from mobilerun.agents.clients.oauth.login_timeout import OAuthLoginDeadline
from mobilerun.agents.clients.oauth.openai import OpenAIOAuthSession


class _FakeClock:
    def __init__(self) -> None:
        self.now = 10.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize("timeout", [0, -1, math.nan, math.inf, -math.inf])
def test_oauth_login_deadline_rejects_non_positive_or_non_finite_timeout(
    timeout: float,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        OAuthLoginDeadline(timeout)


def test_oauth_login_deadline_caps_requests_and_never_resets() -> None:
    clock = _FakeClock()
    deadline = OAuthLoginDeadline(10, clock=clock, sleeper=clock.sleep)

    assert deadline.remaining(cap=3) == 3
    clock.now += 4
    assert deadline.remaining() == 6

    limited = deadline.limited_to(2)
    limited.sleep(1)
    assert clock.sleeps == [1]
    assert limited.remaining() == 1

    with pytest.raises(TimeoutError, match="OAuth login timed out"):
        limited.sleep(2)
    assert clock.sleeps == [1, 1]
    assert deadline.remaining() == 4


@pytest.mark.parametrize("open_browser", [False, True])
def test_openai_device_flow_preserves_deadline_and_browser_preference(
    monkeypatch,
    tmp_path,
    open_browser: bool,
) -> None:
    clock = _FakeClock()
    deadline = OAuthLoginDeadline(20, clock=clock, sleeper=clock.sleep)
    opened: list[str] = []
    requests: list[tuple[str, float]] = []
    responses = iter(
        [
            httpx.Response(503),
            httpx.Response(
                200,
                json={
                    "device_auth_id": "device",
                    "user_code": "ABCD",
                    "verification_uri": "https://auth.openai.test/device",
                    "expires_in": 60,
                    "interval": 1,
                },
            ),
            httpx.Response(403),
            httpx.Response(
                200,
                json={"authorization_code": "code", "code_verifier": "verifier"},
            ),
            httpx.Response(
                200,
                json={
                    "access_token": "access",
                    "refresh_token": "refresh",
                    "expires_in": 3600,
                },
            ),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.path, float(request.extensions["timeout"]["read"])))
        clock.now += 0.5
        return next(responses)

    llm = OpenAIOAuthSession(
        model="gpt-5.5",
        oauth_credential_path=str(tmp_path / "auth.json"),
    )
    llm._oauth_manager.http_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(
        openai,
        "open_browser_async",
        lambda url, opener: opened.append(url),
    )

    credentials = llm._login_device_code(
        open_browser=open_browser,
        deadline=deadline,
    )

    assert credentials.access_token == "access"
    assert llm._oauth_manager.credential_store.load() == credentials
    assert [path for path, _ in requests] == [
        "/api/accounts/deviceauth/usercode",
        "/api/accounts/deviceauth/usercode",
        "/api/accounts/deviceauth/token",
        "/api/accounts/deviceauth/token",
        "/oauth/token",
    ]
    assert [timeout for _, timeout in requests] == pytest.approx([20.0, 17.5, 17.0, 15.5, 15.0])
    assert clock.sleeps == [2, 1]
    assert opened == (["https://auth.openai.test/device"] if open_browser else [])


@pytest.mark.parametrize(
    ("llm_class", "exchange_kwargs"),
    (
        (
            AnthropicOAuthSession,
            {
                "code": "code",
                "redirect_uri": "https://example.test/callback",
                "code_verifier": "verifier",
                "state": "state",
            },
        ),
        (
            GeminiOAuthSession,
            {
                "code": "code",
                "redirect_uri": "https://example.test/callback",
                "code_verifier": "verifier",
            },
        ),
    ),
)
def test_anthropic_and_gemini_exchange_use_one_remaining_budget(
    llm_class,
    exchange_kwargs,
    tmp_path,
) -> None:
    llm = llm_class(credential_path=str(tmp_path / "auth-profiles.json"), timeout=30)
    clock = _FakeClock()
    deadline = OAuthLoginDeadline(5, clock=clock, sleeper=clock.sleep)
    timeouts: list[float] = []

    def post(*args, **kwargs):  # type: ignore[no-untyped-def]
        timeouts.append(kwargs["timeout"])
        clock.now += 1
        return SimpleNamespace(
            raise_for_status=lambda: None,
            json=lambda: {
                "access_token": "access",
                "refresh_token": "refresh",
                "expires_in": 3600,
            },
        )

    llm._session = SimpleNamespace(post=post)

    assert (
        llm._exchange_authorization_code(
            **exchange_kwargs,
            deadline=deadline,
        )
        == "access"
    )
    assert timeouts == [5]


@pytest.mark.parametrize("open_browser", [False, True])
@pytest.mark.parametrize(
    ("module", "llm_class", "constructor_kwargs"),
    (
        (anthropic, AnthropicOAuthSession, {"credential_path": None}),
        (
            gemini,
            GeminiOAuthSession,
            {"credential_path": None},
        ),
    ),
)
def test_anthropic_and_gemini_bind_fallback_preserves_deadline_and_browser(
    monkeypatch,
    module,
    llm_class,
    constructor_kwargs,
    open_browser: bool,
) -> None:
    deadline = OAuthLoginDeadline(5)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(module, "_is_headless_environment", lambda: False)
    monkeypatch.setattr(
        module,
        "HTTPServer",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("unavailable")),
    )
    monkeypatch.setattr(
        llm_class,
        "login_headless",
        lambda self, **kwargs: (calls.append(kwargs), "access")[1],
    )

    assert (
        llm_class(**constructor_kwargs).login(
            open_browser=open_browser,
            deadline=deadline,
        )
        == "access"
    )

    assert calls[0]["deadline"] is deadline
    assert calls[0]["open_browser"] is open_browser
