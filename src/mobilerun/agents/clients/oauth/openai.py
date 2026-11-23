from __future__ import annotations

import base64
import hashlib
import json
import os  # Host session variables determine whether browser login is available.
import secrets
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from mobilerun.agents.clients.oauth.login_timeout import (
    OAuthLoginDeadline,
    open_browser_async,
)
from mobilerun.agents.clients.oauth.profiles import AuthProfileStore
from mobilerun.agents.config.settings import CODEX_CONTEXT_WINDOW, get_agent_settings

DEFAULT_OPENAI_OAUTH_ISSUER = "https://auth.openai.com"

DEFAULT_OPENAI_OAUTH_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"

DEFAULT_AUTH_MODEL = "openai-codex/gpt-6-astra"

DEFAULT_OPENAI_API_BASE = "https://api.openai.com/v1"

DEFAULT_CODEX_API_BASE = "https://chatgpt.com/backend-api/codex"

DEFAULT_BACKEND_API_BASE = "https://chatgpt.com/backend-api"

DEFAULT_OPENAI_OAUTH_CALLBACK_HOST = "localhost"

DEFAULT_OPENAI_OAUTH_CALLBACK_PATH = "/auth/callback"

DEFAULT_OPENAI_OAUTH_SCOPE = (
    "openid profile email offline_access api.connectors.read api.connectors.invoke"
)

_OPENAI_LOGIN_TIMEOUT_MESSAGE = "OpenAI OAuth login timed out."

_GPT_6_ASTRA_MODEL = "gpt-6-astra"

_GPT_6_UNSUPPORTED_PARAMS = frozenset({"temperature", "top_p", "logprobs", "top_logprobs"})

_GPT_6_UNSUPPORTED_INCLUDE = "message.output_text.logprobs"

_CODEX_CONTEXT_WINDOW = CODEX_CONTEXT_WINDOW


def _b64_no_pad(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("utf-8").rstrip("=")


def _pkce_pair() -> tuple[str, str]:
    verifier = _b64_no_pad(secrets.token_bytes(64))
    challenge = _b64_no_pad(hashlib.sha256(verifier.encode("utf-8")).digest())
    return verifier, challenge


def _is_headless_environment() -> bool:
    """Detect SSH, WSL, or missing display where browser popups won't work."""
    if os.environ.get("SSH_CONNECTION") or os.environ.get("SSH_TTY"):
        return True
    if os.environ.get("WSL_DISTRO_NAME"):
        return True
    if sys.platform.startswith("linux"):
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    return False


def _tls_preflight(
    issuer: str,
    timeout: float = 5.0,
    *,
    deadline: OAuthLoginDeadline | None = None,
    on_output: Callable[[str], None] | None = None,
) -> None:
    """Probe the OAuth issuer to detect TLS/certificate issues before login.

    Raises RuntimeError on TLS certificate errors (with fix suggestions).
    Prints warnings for non-TLS connection errors but does not block.
    """
    probe_url = f"{issuer.rstrip('/')}/oauth/authorize"
    try:
        request_timeout = deadline.remaining(cap=timeout) if deadline else timeout
        httpx.head(probe_url, follow_redirects=False, timeout=request_timeout)
        if deadline:
            deadline.check()
    except httpx.ConnectError as exc:
        if deadline:
            deadline.check()
        err_str = str(exc).lower()
        tls_indicators = (
            "certificate",
            "ssl",
            "tls",
            "unable_to_get_issuer_cert",
            "cert_has_expired",
            "self_signed_cert",
            "verify_leaf_signature",
            "altname_invalid",
        )
        if any(indicator in err_str for indicator in tls_indicators):
            raise RuntimeError(
                f"TLS certificate error connecting to {probe_url}: {exc}\n"
                "Possible fixes:\n"
                "  - Update CA certificates "
                "(e.g. `sudo update-ca-certificates` on Linux, "
                "`brew postinstall ca-certificates` on macOS)\n"
                "  - Update OpenSSL (e.g. `brew postinstall openssl@3`)\n"
                "  - Check if a corporate proxy is intercepting HTTPS traffic"
            ) from exc
        if on_output is not None:
            on_output(
                f"Warning: Could not connect to {probe_url}: {exc}\n"
                "The login flow may fail if there is a DNS or firewall issue."
            )
    except httpx.TimeoutException:
        if deadline:
            deadline.check()
        if on_output is not None:
            on_output(
                f"Warning: Connection to {probe_url} timed out.\n"
                "The login flow may fail if there is a network issue."
            )
    except Exception as exc:
        if deadline:
            deadline.check()
        # Unexpected error — warn but don't block.
        if on_output is not None:
            on_output(f"Warning: TLS preflight check encountered an error: {exc}")


def _request_device_code(
    issuer: str,
    client_id: str,
    http_client: Optional[httpx.Client] = None,
    request_timeout: float | None = None,
    retries: int | None = None,
    *,
    deadline: OAuthLoginDeadline | None = None,
) -> dict:
    url = f"{issuer.rstrip('/')}/api/accounts/deviceauth/usercode"
    settings = get_agent_settings()
    request_timeout = (
        settings.MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS
        if request_timeout is None
        else request_timeout
    )
    retries = settings.MOBILERUN_OPENAI_OAUTH_DEVICE_CODE_RETRIES if retries is None else retries
    post = http_client.post if http_client is not None else httpx.post
    for attempt in range(1 + retries):
        try:
            timeout = deadline.remaining(cap=request_timeout) if deadline else request_timeout
            response = post(
                url,
                headers={"Content-Type": "application/json"},
                json={"client_id": client_id},
                timeout=timeout,
            )
            if deadline:
                deadline.check()
        except (httpx.ConnectError, httpx.TimeoutException):
            if deadline:
                deadline.check()
            if attempt < retries:
                if deadline:
                    deadline.sleep(2)
                else:
                    time.sleep(2)
                continue
            raise
        if response.status_code == 404:
            raise RuntimeError("Device code login is not enabled for this server.")
        if response.status_code >= 500 and attempt < retries:
            if deadline:
                deadline.sleep(2)
            else:
                time.sleep(2)
            continue
        response.raise_for_status()
        payload = response.json()
        if deadline:
            deadline.check()
        return payload
    raise RuntimeError("Device code request failed after retries.")


def _poll_device_code(
    issuer: str,
    device_auth_id: str,
    user_code: str,
    interval: int,
    http_client: Optional[httpx.Client] = None,
    request_timeout: float | None = None,
    timeout_seconds: float | None = None,
    *,
    deadline: OAuthLoginDeadline | None = None,
) -> dict:
    url = f"{issuer.rstrip('/')}/api/accounts/deviceauth/token"
    request_timeout = (
        get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS
        if request_timeout is None
        else request_timeout
    )
    device_code_timeout = get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEVICE_CODE_TIMEOUT
    if timeout_seconds is None:
        timeout_seconds = device_code_timeout
    login_deadline = deadline or OAuthLoginDeadline(
        min(timeout_seconds, device_code_timeout),
        timeout_message=_OPENAI_LOGIN_TIMEOUT_MESSAGE,
    )
    post = http_client.post if http_client is not None else httpx.post

    last_error: Optional[str] = None

    try:
        while True:
            try:
                response = post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "device_auth_id": device_auth_id,
                        "user_code": user_code,
                    },
                    timeout=login_deadline.remaining(cap=request_timeout),
                )
                login_deadline.check()
            except (httpx.ConnectError, httpx.TimeoutException) as exc:
                login_deadline.check()
                last_error = str(exc)
                login_deadline.sleep(interval)
                continue
            if response.status_code == 200:
                payload = response.json()
                login_deadline.check()
                return payload
            # OpenAI returns 403/404 while the user hasn't completed browser auth.
            # This differs from RFC 8628's 400 + authorization_pending body, but
            # matches the observed behaviour of
            # auth.openai.com/api/accounts/deviceauth/token.
            # TODO: handle slow_down (RFC 8628 §3.5) by increasing interval.
            if response.status_code in (403, 404):
                login_deadline.sleep(interval)
                continue
            if response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
                login_deadline.sleep(interval)
                continue
            response.raise_for_status()
    except TimeoutError:
        if last_error:
            raise TimeoutError(
                f"{_OPENAI_LOGIN_TIMEOUT_MESSAGE} Last response: {last_error}."
            ) from None
        raise


@dataclass
class OpenAIOAuthCredentials:
    access_token: str
    refresh_token: Optional[str] = None
    expires_at_ms: Optional[int] = None
    account_id: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "OpenAIOAuthCredentials":
        access_token = payload.get("access") or payload.get("access_token")
        if not access_token:
            raise ValueError("OpenAI OAuth payload missing access token.")

        refresh_token = payload.get("refresh") or payload.get("refresh_token")
        account_id = payload.get("account_id")

        raw_expires = payload.get("expires")
        expires_at_ms: Optional[int]
        if raw_expires is None:
            expires_at_ms = None
        else:
            try:
                expires_at_ms = int(raw_expires)
            except (TypeError, ValueError):
                expires_at_ms = None

        return cls(
            access_token=str(access_token),
            refresh_token=str(refresh_token) if refresh_token else None,
            expires_at_ms=expires_at_ms,
            account_id=str(account_id) if account_id else None,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "oauth",
            "provider": "openai-chatgpt",
            "access": self.access_token,
            "refresh": self.refresh_token,
            "expires": self.expires_at_ms,
            "account_id": self.account_id,
        }

    def is_valid(self, skew_ms: int | None = None) -> bool:
        if skew_ms is None:
            skew_ms = int(get_agent_settings().MOBILERUN_OPENAI_OAUTH_REFRESH_SKEW_SECONDS * 1000)
        if not self.access_token:
            return False
        if self.expires_at_ms is None:
            return True
        return int(time.time() * 1000) + skew_ms < self.expires_at_ms


class OpenAIOAuthCredentialStore:
    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            path = get_agent_settings().openai_oauth_credential_path
        self.path = Path(path).expanduser()
        self._store = AuthProfileStore(self.path)

    _NESTED_KEY = "openaiOauth"

    def load(self) -> Optional[OpenAIOAuthCredentials]:
        if not self.path.exists():
            return None

        raw = self._store.read_profile()
        nested = raw.get(self._NESTED_KEY)
        payload = nested if isinstance(nested, dict) else raw
        try:
            return OpenAIOAuthCredentials.from_payload(payload)
        except ValueError:
            return None

    def save(
        self,
        credentials: OpenAIOAuthCredentials,
        *,
        deadline: OAuthLoginDeadline | None = None,
    ) -> None:
        self._store.update_slot(
            self._NESTED_KEY,
            credentials.to_dict(),
            lock_timeout=deadline.remaining() if deadline is not None else None,
            before_commit=deadline.check if deadline is not None else None,
        )


class OpenAIOAuthSessionManager:
    def __init__(
        self,
        *,
        issuer: str | None = None,
        client_id: str = DEFAULT_OPENAI_OAUTH_CLIENT_ID,
        credential_store: Optional[OpenAIOAuthCredentialStore] = None,
        request_timeout: float | None = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        request_timeout = (
            get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS
            if request_timeout is None
            else request_timeout
        )
        issuer = issuer or get_agent_settings().MOBILERUN_OPENAI_OAUTH_ISSUER
        self.issuer = issuer.rstrip("/")
        self.client_id = client_id
        self.request_timeout = request_timeout
        self.credential_store = credential_store or OpenAIOAuthCredentialStore()
        self.http_client = http_client
        self._lock = threading.Lock()
        self._credentials: Optional[OpenAIOAuthCredentials] = None

    @staticmethod
    def _jwt_claims(token: str) -> Dict[str, Any]:
        parts = token.split(".")
        if len(parts) != 3:
            return {}

        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        try:
            raw = base64.urlsafe_b64decode(payload + padding).decode("utf-8")
            claims = json.loads(raw)
            return claims if isinstance(claims, dict) else {}
        except Exception:
            return {}

    @classmethod
    def _extract_account_id(cls, token: str) -> Optional[str]:
        claims = cls._jwt_claims(token)

        auth_claims = claims.get("https://api.openai.com/auth")
        if isinstance(auth_claims, dict):
            nested = auth_claims.get("chatgpt_account_id")
            if isinstance(nested, str) and nested:
                return nested

        direct = claims.get("chatgpt_account_id")
        if isinstance(direct, str) and direct:
            return direct

        orgs = claims.get("organizations")
        if isinstance(orgs, list) and orgs:
            first = orgs[0]
            if isinstance(first, dict):
                org_id = first.get("id")
                if isinstance(org_id, str) and org_id:
                    return org_id

        return None

    @classmethod
    def _compute_expiry_ms(cls, token: str, expires_in: Optional[Any]) -> Optional[int]:
        if expires_in is not None:
            try:
                return int(time.time() * 1000) + int(expires_in) * 1000
            except (TypeError, ValueError):
                pass

        claims = cls._jwt_claims(token)
        exp = claims.get("exp")
        if isinstance(exp, (int, float)):
            return int(exp * 1000)
        return None

    def set_initial_credentials(
        self,
        credentials: OpenAIOAuthCredentials,
        *,
        deadline: OAuthLoginDeadline | None = None,
    ) -> None:
        if deadline is None:
            with self._lock:
                self._credentials = credentials
                self.credential_store.save(credentials)
            return

        deadline.check()
        if not self._lock.acquire(timeout=deadline.remaining()):
            raise TimeoutError(_OPENAI_LOGIN_TIMEOUT_MESSAGE)
        try:
            deadline.check()
            self.credential_store.save(credentials, deadline=deadline)
            self._credentials = credentials
        finally:
            self._lock.release()

    def _load_cached_credentials(self) -> Optional[OpenAIOAuthCredentials]:
        if self._credentials is not None:
            return self._credentials

        cached = self.credential_store.load()
        if cached is not None:
            self._credentials = cached
        return cached

    def _refresh(self, refresh_token: str) -> OpenAIOAuthCredentials:
        data = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": self.client_id,
        }

        if self.http_client is not None:
            response = self.http_client.post(
                get_agent_settings().MOBILERUN_OPENAI_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=self.request_timeout,
            )
        else:
            response = httpx.post(
                get_agent_settings().MOBILERUN_OPENAI_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/json"},
                json=data,
                timeout=self.request_timeout,
            )

        response.raise_for_status()
        payload = response.json()

        access = payload.get("access_token")
        if not isinstance(access, str) or not access:
            raise RuntimeError("OpenAI OAuth refresh response missing access_token.")

        new_refresh = payload.get("refresh_token")
        if not isinstance(new_refresh, str) or not new_refresh:
            new_refresh = refresh_token

        expires_at_ms = self._compute_expiry_ms(access, payload.get("expires_in"))
        id_or_access = (
            payload.get("id_token") if isinstance(payload.get("id_token"), str) else access
        )
        account_id = self._extract_account_id(id_or_access)

        return OpenAIOAuthCredentials(
            access_token=access,
            refresh_token=new_refresh,
            expires_at_ms=expires_at_ms,
            account_id=account_id,
        )

    def get_valid_credentials(self, skew_ms: int | None = None) -> OpenAIOAuthCredentials:
        with self._lock:
            cached = self._load_cached_credentials()
            if cached is not None and cached.is_valid(skew_ms=skew_ms):
                return cached

            if cached is None or not cached.refresh_token:
                raise ValueError(
                    "No valid OpenAI OAuth credentials found. Provide a refresh token or a cached credential file."
                )

            refreshed = self._refresh(cached.refresh_token)
            self._credentials = refreshed
            self.credential_store.save(refreshed)
            return refreshed

    def exchange_authorization_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        deadline: OAuthLoginDeadline | None = None,
    ) -> OpenAIOAuthCredentials:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
        }

        request_timeout = (
            deadline.remaining(cap=self.request_timeout) if deadline else self.request_timeout
        )
        if self.http_client is not None:
            response = self.http_client.post(
                get_agent_settings().MOBILERUN_OPENAI_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=data,
                timeout=request_timeout,
            )
        else:
            response = httpx.post(
                get_agent_settings().MOBILERUN_OPENAI_OAUTH_TOKEN_URL,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                data=data,
                timeout=request_timeout,
            )

        if deadline:
            deadline.check()
        response.raise_for_status()
        payload = response.json()
        if deadline:
            deadline.check()

        access = payload.get("access_token")
        if not isinstance(access, str) or not access:
            raise RuntimeError("OpenAI OAuth code exchange response missing access_token.")

        refresh = payload.get("refresh_token")
        credentials = OpenAIOAuthCredentials(
            access_token=access,
            refresh_token=refresh if isinstance(refresh, str) and refresh else None,
            expires_at_ms=self._compute_expiry_ms(access, payload.get("expires_in")),
            account_id=self._extract_account_id(
                payload.get("id_token") if isinstance(payload.get("id_token"), str) else access
            ),
        )
        self.set_initial_credentials(credentials, deadline=deadline)
        return credentials


class OpenAIOAuthSession:
    """Provider-owned OAuth login, credentials, and refresh lifecycle."""

    def __init__(
        self,
        *,
        model: str | None = None,
        oauth_credential_path: str | Path | None = None,
        timeout: float | None = None,
    ) -> None:
        timeout = (
            get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEFAULT_TIMEOUT_SECONDS
            if timeout is None
            else timeout
        )
        self.model = model or "gpt-6-astra"
        self._oauth_manager = OpenAIOAuthSessionManager(
            credential_store=OpenAIOAuthCredentialStore(oauth_credential_path),
            request_timeout=timeout,
        )

    @staticmethod
    def _build_auth_url(
        *,
        issuer: str,
        client_id: str,
        redirect_uri: str,
        code_challenge: str,
        state: str,
        scope: str = DEFAULT_OPENAI_OAUTH_SCOPE,
    ) -> str:
        query = {
            "response_type": "code",
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "id_token_add_organizations": "true",
            "codex_cli_simplified_flow": "true",
            "state": state,
            "originator": "codex_cli_rs",
        }
        return f"{get_agent_settings().MOBILERUN_OPENAI_OAUTH_AUTHORIZE_URL}?{urlencode(query)}"

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float | None = None,
        callback_host: str = DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
        callback_port: int | None = None,
        callback_path: str = DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
        redirect_host: str = DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
        scope: str = DEFAULT_OPENAI_OAUTH_SCOPE,
        deadline: OAuthLoginDeadline | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> OpenAIOAuthCredentials:
        settings = get_agent_settings()
        timeout_seconds = (
            settings.MOBILERUN_OPENAI_OAUTH_LOGIN_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        callback_port = (
            settings.MOBILERUN_OPENAI_OAUTH_CALLBACK_PORT
            if callback_port is None
            else callback_port
        )
        login_deadline = deadline or OAuthLoginDeadline(
            timeout_seconds,
            timeout_message=_OPENAI_LOGIN_TIMEOUT_MESSAGE,
        )
        login_deadline.check()
        _tls_preflight(
            self._oauth_manager.issuer,
            deadline=login_deadline,
            on_output=on_output,
        )

        # Headless environments: use device code flow (no local server needed)
        use_device_code = _is_headless_environment() or get_agent_settings().oauth_manual
        if use_device_code:
            return self._login_device_code(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                deadline=login_deadline,
                **({"on_output": on_output} if on_output is not None else {}),
            )

        # Desktop: browser callback server
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        done = threading.Event()
        callback_lock = threading.Lock()
        code_verifier, code_challenge = _pkce_pair()
        state = _b64_no_pad(secrets.token_bytes(32))

        class _OAuthHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path != callback_path:
                    self.send_response(404)
                    self.end_headers()
                    return

                with callback_lock:
                    if done.is_set():
                        self.send_response(409)
                        self.end_headers()
                        return
                    params = parse_qs(parsed.query)
                    result["code"] = params.get("code", [None])[0]
                    result["state"] = params.get("state", [None])[0]
                    result["error"] = params.get("error", [None])[0]
                    ok = result["code"] is not None and result["error"] is None
                    done.set()
                self.send_response(200 if ok else 400)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                if ok:
                    self.wfile.write(
                        b"<html><body><h3>Login complete. You can close this tab.</h3></body></html>"
                    )
                else:
                    self.wfile.write(
                        b"<html><body><h3>Login failed. Return to your terminal.</h3></body></html>"
                    )

            def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
                return

        try:
            httpd = HTTPServer((callback_host, callback_port), _OAuthHandler)
        except OSError as exc:
            login_deadline.check()
            if on_output is not None:
                on_output(
                    f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                    "Falling back to device code login."
                )
            return self._login_device_code(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                deadline=login_deadline,
                **({"on_output": on_output} if on_output is not None else {}),
            )
        login_deadline.check()

        actual_port = httpd.server_address[1]
        redirect_uri = f"http://{redirect_host}:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            issuer=self._oauth_manager.issuer,
            client_id=self._oauth_manager.client_id,
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            state=state,
            scope=scope,
        )

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        httpd.daemon_threads = True
        server_thread.start()

        try:
            if on_output is not None:
                on_output(f"Open this URL to login:\n{auth_url}\n")
            if open_browser:
                open_browser_async(auth_url, webbrowser.open)

            if not done.wait(timeout=login_deadline.remaining()):
                raise TimeoutError("OpenAI OAuth login timed out before callback was received.")
            login_deadline.check()

            if result["error"]:
                raise RuntimeError(f"OAuth callback returned error: {result['error']}")
            if result["state"] != state:
                raise RuntimeError("OAuth callback state mismatch.")
            if not result["code"]:
                raise RuntimeError("OAuth callback did not include an authorization code.")

            creds = self._oauth_manager.exchange_authorization_code(
                code=result["code"],
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                deadline=login_deadline,
            )
            if creds.account_id:
                object.__setattr__(self, "_oauth_account_id", creds.account_id)
            return creds
        finally:
            httpd.shutdown()
            httpd.server_close()

    def _login_device_code(
        self,
        *,
        open_browser: bool = False,
        timeout_seconds: float | None = None,
        deadline: OAuthLoginDeadline | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> OpenAIOAuthCredentials:
        """Device Code login for headless/SSH environments.

        timeout_seconds is capped at 15 min (device code expiry).
        """
        mgr = self._oauth_manager
        http_client = mgr.http_client
        device_code_timeout = get_agent_settings().MOBILERUN_OPENAI_OAUTH_DEVICE_CODE_TIMEOUT
        timeout_seconds = device_code_timeout if timeout_seconds is None else timeout_seconds
        login_deadline = deadline or OAuthLoginDeadline(
            timeout_seconds,
            timeout_message=_OPENAI_LOGIN_TIMEOUT_MESSAGE,
        )
        login_deadline.check()

        device_resp = _request_device_code(
            mgr.issuer,
            mgr.client_id,
            http_client=http_client,
            request_timeout=mgr.request_timeout,
            deadline=login_deadline,
        )
        device_auth_id = device_resp.get("device_auth_id")
        if not device_auth_id:
            raise RuntimeError("Device code response missing 'device_auth_id'.")

        user_code = device_resp.get("user_code") or device_resp.get("usercode")
        if not user_code:
            raise RuntimeError("Device code response missing 'user_code'.")
        try:
            interval = max(1, int(str(device_resp.get("interval", "5")).strip()))
        except (TypeError, ValueError):
            interval = 5
        try:
            server_expires = int(device_resp["expires_in"])
        except (KeyError, TypeError, ValueError):
            server_expires = device_code_timeout
        if server_expires <= 0:
            raise TimeoutError("OpenAI OAuth device authorization expired.")
        device_deadline = login_deadline.limited_to(min(device_code_timeout, server_expires))
        effective_timeout = device_deadline.remaining()
        verification_url = (
            device_resp.get("verification_uri")
            or device_resp.get("verification_url")
            or f"{mgr.issuer}/codex/device"
        )

        if effective_timeout >= 60:
            mins = int(effective_timeout // 60)
            expires_str = f"{mins} minute{'s' if mins != 1 else ''}"
        else:
            secs = int(effective_timeout)
            expires_str = f"{secs} second{'s' if secs != 1 else ''}"
        if on_output is not None:
            on_output(
                f"\nSign in with your ChatGPT account:\n"
                f"\n1. Open this link in your browser:\n   {verification_url}\n"
                f"\n2. Enter this code (expires in {expires_str}):\n   {user_code}\n"
                f"\nDevice codes are a common phishing target. Never share this code.\n"
            )
        if open_browser:
            open_browser_async(str(verification_url), webbrowser.open)

        token_resp = _poll_device_code(
            mgr.issuer,
            device_auth_id,
            user_code,
            interval,
            http_client=http_client,
            request_timeout=mgr.request_timeout,
            deadline=device_deadline,
        )

        auth_code = token_resp.get("authorization_code")
        code_verifier = token_resp.get("code_verifier")
        if not auth_code or not code_verifier:
            raise RuntimeError(
                "Device code token response missing required fields "
                "('authorization_code' / 'code_verifier')."
            )

        redirect_uri = f"{mgr.issuer}/deviceauth/callback"
        creds = self._oauth_manager.exchange_authorization_code(
            code=auth_code,
            redirect_uri=redirect_uri,
            code_verifier=code_verifier,
            deadline=device_deadline,
        )
        if creds.account_id:
            object.__setattr__(self, "_oauth_account_id", creds.account_id)
        return creds


def run_openai_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float | None = None,
    callback_host: str = DEFAULT_OPENAI_OAUTH_CALLBACK_HOST,
    callback_port: int | None = None,
    callback_path: str = DEFAULT_OPENAI_OAUTH_CALLBACK_PATH,
    open_browser: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> None:
    class OutputOption(TypedDict, total=False):
        on_output: Callable[[str], None]

    output_option: OutputOption = {}
    if on_output is not None:
        output_option["on_output"] = on_output
    timeout = (
        get_agent_settings().MOBILERUN_OPENAI_OAUTH_LOGIN_TIMEOUT_SECONDS
        if timeout is None
        else timeout
    )
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="OpenAI OAuth login timed out.",
    )
    llm = OpenAIOAuthSession(
        model=model,
        oauth_credential_path=credential_path,
        timeout=timeout,
    )
    llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        redirect_host=callback_host,
        deadline=deadline,
        **output_option,
    )


def get_default_openai_credential_path() -> str:
    return str(get_agent_settings().openai_oauth_credential_path)
