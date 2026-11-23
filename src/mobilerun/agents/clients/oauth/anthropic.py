import base64
import hashlib
import json
import os  # Host session variables determine whether browser login is available.
import queue
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from urllib.parse import parse_qs, urlencode, urlparse

import requests
from mobilerun.config.env import secret_value

from mobilerun.agents.clients.oauth.login_timeout import (
    OAuthLoginDeadline,
    open_browser_async,
)
from mobilerun.agents.clients.oauth.profiles import AuthProfileStore
from mobilerun.agents.config.env import get_agent_secrets
from mobilerun.agents.config.settings import (
    ANTHROPIC_OAUTH_REDIRECT_URI,
    SETUP_TOKEN_EXPIRES_IN_SECONDS,
    get_agent_settings,
)

DEFAULT_MODEL = "claude-sonnet-5"

DEFAULT_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

DEFAULT_LOGIN_SCOPE = "org:create_api_key user:profile"

DEFAULT_MODERN_LOGIN_SCOPE = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)

DEFAULT_SETUP_TOKEN_SCOPE = "user:inference"

DEFAULT_REFRESH_SCOPE = (
    "user:inference user:profile user:file_upload user:mcp_servers user:sessions:claude_code"
)

DEFAULT_OAUTH_BETA = "oauth-2025-04-20"

DEFAULT_ANTHROPIC_VERSION = "2023-06-01"

DEFAULT_CLAUDE_CODE_VERSION = "2.1.281"

DEFAULT_USER_AGENT = f"claude-cli/{DEFAULT_CLAUDE_CODE_VERSION}"

DEFAULT_CC_VERSION = f"{DEFAULT_CLAUDE_CODE_VERSION}.000"

DEFAULT_CC_ENTRYPOINT = "cli"

_IGNORED_REQUEST_KWARGS = {
    "formatted",
}

_ERROR_TYPE_STATUS = {
    "invalid_request_error": 400,
    "authentication_error": 401,
    "billing_error": 402,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_error": 429,
    "api_error": 500,
    "timeout_error": 504,
    "overloaded_error": 529,
}


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


def _normalize_manual_code(raw: str, expected_state: str) -> str:
    value = raw.strip()
    if not value:
        return value

    first_token = value.split()[0]

    if "error=" in first_token or "code=" in first_token:
        parsed = urlparse(first_token)
        params = parse_qs(parsed.query)
        error = params.get("error", [None])[0]
        if error:
            desc = params.get("error_description", [error])[0]
            raise RuntimeError(f"OAuth error: {desc}")
        code = params.get("code", [None])[0]
        state_from_url = params.get("state", [None])[0]
        if state_from_url and state_from_url != expected_state:
            raise RuntimeError("OAuth manual code state mismatch.")
        if isinstance(code, str) and code:
            return code

    if "#" in first_token:
        code_part, fragment = first_token.split("#", 1)
        if fragment and fragment != expected_state:
            raise RuntimeError("OAuth manual code state mismatch.")
        return code_part

    return first_token


class AnthropicOAuthSession:
    """Provider-owned OAuth login, credentials, and refresh lifecycle."""

    def __init__(
        self,
        *,
        credential_path: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: float | None = None,
        access_token: str | None = None,
        refresh_token: str | None = None,
        client_id: str = DEFAULT_CLIENT_ID,
        authorize_url: str | None = None,
        token_url: str | None = None,
        refresh_buffer_seconds: float | None = None,
        login_scope: str = DEFAULT_LOGIN_SCOPE,
        refresh_scope: str = DEFAULT_REFRESH_SCOPE,
        oauth_beta: str = DEFAULT_OAUTH_BETA,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        timeout = (
            get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_DEFAULT_TIMEOUT_SECONDS
            if timeout is None
            else timeout
        )
        refresh_buffer_seconds = (
            get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_REFRESH_BUFFER_SECONDS
            if refresh_buffer_seconds is None
            else refresh_buffer_seconds
        )
        settings = get_agent_settings()
        self.credential_path = credential_path or str(
            get_agent_settings().anthropic_oauth_credential_path
        )
        self.model = model
        self.timeout = timeout
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.authorize_url = authorize_url or settings.MOBILERUN_ANTHROPIC_OAUTH_AUTHORIZE_URL
        self.token_url = token_url or settings.MOBILERUN_ANTHROPIC_OAUTH_TOKEN_URL
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self.login_scope = login_scope
        self.refresh_scope = refresh_scope
        self.oauth_beta = oauth_beta
        self.user_agent = user_agent
        self._session = requests.Session()
        self._cached_access_token = access_token
        self._cached_refresh_token = refresh_token
        self._access_token_expiry = None
        self._load_credentials_from_file(self.credential_path)

    def _load_credentials_from_file(self, credential_path: str) -> None:
        path = Path(credential_path).expanduser()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        nested = payload.get("claudeAiOauth")
        if isinstance(nested, dict):
            file_access = nested.get("accessToken")
            file_refresh = nested.get("refreshToken")
            expires_at = nested.get("expiresAt")
        else:
            file_access = payload.get("access_token")
            file_refresh = payload.get("refresh_token")
            expires_at = payload.get("expires_at")

        if not self._cached_access_token and isinstance(file_access, str):
            self._cached_access_token = file_access
        if not self._cached_refresh_token and isinstance(file_refresh, str):
            self._cached_refresh_token = file_refresh
        if isinstance(expires_at, (int, float)):
            self._access_token_expiry = float(expires_at) / 1000.0

    def _persist_credentials(
        self,
        *,
        deadline: OAuthLoginDeadline | None = None,
    ) -> None:
        if not self.credential_path:
            return
        path = Path(self.credential_path).expanduser()
        AuthProfileStore(path).update_slot(
            "claudeAiOauth",
            {
                "accessToken": self._cached_access_token,
                "refreshToken": self._cached_refresh_token,
                "expiresAt": (
                    int(self._access_token_expiry * 1000) if self._access_token_expiry else None
                ),
                "scopes": self.refresh_scope.split(),
            },
            lock_timeout=deadline.remaining() if deadline is not None else None,
            before_commit=deadline.check if deadline is not None else None,
        )

    def _token_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "anthropic-beta": self.oauth_beta,
        }
        if self.user_agent:
            headers["User-Agent"] = self.user_agent
        return headers

    def _access_token_is_stale(self) -> bool:
        if not self._access_token_expiry:
            return False
        return time.time() >= (self._access_token_expiry - self.refresh_buffer_seconds)

    def _refresh_access_token(self) -> str:
        refresh_token = self._cached_refresh_token or self.refresh_token
        if not refresh_token:
            raise ValueError(
                "No refresh token available. Provide `refresh_token` or credentials file."
            )

        res = self._session.post(
            self.token_url,
            headers=self._token_headers(),
            json={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "scope": self.refresh_scope,
            },
            timeout=self.timeout,
        )
        res.raise_for_status()
        data = res.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"Token refresh succeeded but no access_token returned: {data}")

        expires_in = data.get("expires_in", 28_800)
        try:
            expires_in_s = int(expires_in) if expires_in is not None else 28_800
        except (TypeError, ValueError):
            expires_in_s = 28_800

        self._cached_access_token = access_token
        self._cached_refresh_token = data.get("refresh_token") or refresh_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()
        return access_token

    def _exchange_authorization_code(
        self,
        *,
        code: str,
        redirect_uri: str,
        code_verifier: str,
        state: str,
        expires_in: Optional[int] = None,
        deadline: OAuthLoginDeadline | None = None,
    ) -> str:
        request_body: Dict[str, Any] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "code_verifier": code_verifier,
            "state": state,
        }
        if expires_in is not None:
            request_body["expires_in"] = expires_in

        request_timeout = self.timeout if deadline is None else deadline.remaining(cap=self.timeout)
        res = self._session.post(
            self.token_url,
            headers=self._token_headers(),
            json=request_body,
            timeout=request_timeout,
        )
        if deadline is not None:
            deadline.check()
        res.raise_for_status()
        data = res.json()
        if deadline is not None:
            deadline.check()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(
                f"OAuth code exchange succeeded but no access_token returned: {data}"
            )

        refresh_token = data.get("refresh_token")
        cached_refresh_token = (
            refresh_token
            if isinstance(refresh_token, str) and refresh_token
            else self._cached_refresh_token
        )

        expires_in = data.get("expires_in", 28_800)
        try:
            expires_in_s = int(expires_in) if expires_in is not None else 28_800
        except (TypeError, ValueError):
            expires_in_s = 28_800

        if deadline is not None:
            deadline.check()
        previous = (
            self._cached_access_token,
            self._cached_refresh_token,
            self._access_token_expiry,
        )
        self._cached_refresh_token = cached_refresh_token
        self._cached_access_token = access_token
        self._access_token_expiry = time.time() + expires_in_s
        try:
            self._persist_credentials(deadline=deadline)
        except BaseException:
            (
                self._cached_access_token,
                self._cached_refresh_token,
                self._access_token_expiry,
            ) = previous
            raise
        return access_token

    def _build_auth_url(
        self,
        *,
        redirect_uri: str,
        code_challenge: str,
        state: str,
    ) -> str:
        query = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": self.login_scope,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "state": state,
        }
        if "/cai/oauth/authorize" in self.authorize_url:
            query["code"] = "true"
        return f"{self.authorize_url}?{urlencode(query)}"

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float | None = None,
        callback_host: str = "127.0.0.1",
        callback_port: int | None = None,
        callback_path: str = "/callback",
        expires_in: Optional[int] = None,
        deadline: OAuthLoginDeadline | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        timeout_seconds = (
            get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_LOGIN_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        callback_port = (
            get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_CALLBACK_PORT
            if callback_port is None
            else callback_port
        )
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        # Headless environments: skip local server, use hosted callback page
        use_headless = _is_headless_environment() or get_agent_settings().oauth_manual
        if use_headless:
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                expires_in=expires_in,
                deadline=active_deadline,
                **({"on_output": on_output} if on_output is not None else {}),
            )

        # Desktop: browser callback server
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        done = threading.Event()
        callback_lock = threading.Lock()

        code_verifier, code_challenge = _pkce_pair()
        state = _b64_no_pad(secrets.token_bytes(32))
        original_authorize_url = self.authorize_url

        if "/cai/oauth/authorize" not in self.authorize_url:
            self.authorize_url = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_MODERN_AUTHORIZE_URL

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
            self.authorize_url = original_authorize_url
            if on_output is not None:
                on_output(
                    f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                    "Falling back to headless code entry."
                )
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                expires_in=expires_in,
                deadline=active_deadline,
                **({"on_output": on_output} if on_output is not None else {}),
            )

        active_deadline.check()
        actual_port = httpd.server_address[1]
        redirect_uri = f"http://localhost:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            state=state,
        )

        server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        httpd.daemon_threads = True
        server_thread.start()

        try:
            if on_output is not None:
                on_output(f"Open this URL to login:\n{auth_url}\n")
            if open_browser:
                open_browser_async(auth_url, webbrowser.open)

            if not done.wait(timeout=active_deadline.remaining()):
                raise TimeoutError("OAuth login timed out before callback was received.")
            active_deadline.check()

            if result["error"]:
                raise RuntimeError(f"OAuth callback returned error: {result['error']}")
            if result["state"] != state:
                raise RuntimeError("OAuth callback state mismatch.")
            if not result["code"]:
                raise RuntimeError("OAuth callback did not include an authorization code.")

            access_token = self._exchange_authorization_code(
                code=result["code"],
                redirect_uri=redirect_uri,
                code_verifier=code_verifier,
                state=state,
                expires_in=expires_in,
                deadline=active_deadline,
            )
        finally:
            self.authorize_url = original_authorize_url
            httpd.shutdown()
            httpd.server_close()
        return access_token

    def login_headless(
        self,
        *,
        open_browser: bool = False,
        timeout_seconds: float | None = None,
        input_fn: Any = input,
        expires_in: Optional[int] = None,
        deadline: OAuthLoginDeadline | None = None,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        """Headless OAuth flow for SSH/WSL environments."""
        timeout_seconds = (
            get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_LOGIN_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        max_code_attempts = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_MAX_CODE_ATTEMPTS
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        code_verifier, code_challenge = _pkce_pair()
        state = _b64_no_pad(secrets.token_bytes(32))
        redirect_uri = ANTHROPIC_OAUTH_REDIRECT_URI
        original_authorize_url = self.authorize_url

        if "/cai/oauth/authorize" not in self.authorize_url:
            self.authorize_url = get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_MODERN_AUTHORIZE_URL

        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            code_challenge=code_challenge,
            state=state,
        )

        try:
            if on_output is not None:
                on_output(
                    f"\nSign in with your Anthropic account:\n"
                    f"\n1. Open this link in your browser:\n   {auth_url}\n"
                    f"\n2. Complete sign-in, then paste the authorization code shown on the page.\n"
                )
            if open_browser:
                open_browser_async(auth_url, webbrowser.open)

            input_queue: queue.Queue[Optional[str]] = queue.Queue()
            stop = threading.Event()
            need_more = threading.Event()

            def _reader() -> None:
                for _ in range(max_code_attempts):
                    need_more.wait()
                    need_more.clear()
                    if stop.is_set():
                        return
                    try:
                        input_queue.put(str(input_fn("Enter the authorization code: ")))
                    except (EOFError, OSError):
                        input_queue.put(None)
                        return

            threading.Thread(target=_reader, daemon=True).start()

            try:
                for attempt in range(max_code_attempts):
                    need_more.set()

                    try:
                        raw = input_queue.get(timeout=active_deadline.remaining())
                    except queue.Empty:
                        raise TimeoutError("OAuth login timed out.") from None
                    active_deadline.check()

                    if raw is None:
                        raise RuntimeError("Login failed — stdin closed.")
                    if not raw.strip():
                        if attempt == 0:
                            if on_output is not None:
                                on_output("No code entered. Try again.")
                            continue
                        raise RuntimeError("Login failed.") from None
                    try:
                        code = _normalize_manual_code(raw, state)
                    except Exception:  # noqa: BLE001
                        if attempt == 0:
                            if on_output is not None:
                                on_output("Invalid code. Try again.")
                            continue
                        raise RuntimeError("Login failed.") from None
                    if code:
                        return self._exchange_authorization_code(
                            code=code,
                            redirect_uri=redirect_uri,
                            code_verifier=code_verifier,
                            state=state,
                            expires_in=expires_in,
                            deadline=active_deadline,
                        )
                    if attempt == 0:
                        if on_output is not None:
                            on_output("Invalid code. Try again.")
                        continue
                    raise RuntimeError("Login failed.")
                raise RuntimeError("Login failed.")
            finally:
                # stop.set() prevents the reader from starting a new input() call.
                # It cannot interrupt an input() already in progress (Python limitation);
                # the daemon thread dies with the process.
                stop.set()
                need_more.set()
        finally:
            self.authorize_url = original_authorize_url

    def _resolve_access_token(self) -> str:
        env_access_token = secret_value(get_agent_secrets().ANTHROPIC_OAUTH_TOKEN)
        if env_access_token:
            return env_access_token

        if self._cached_access_token and not self._access_token_is_stale():
            return self._cached_access_token

        if self._cached_access_token and not self._cached_refresh_token:
            return self._cached_access_token

        if self._cached_refresh_token or self.refresh_token:
            return self._refresh_access_token()

        raise ValueError(
            "No OAuth token available. Provide `access_token`, `refresh_token`, "
            "or a valid credential_path."
        )


def run_anthropic_setup_token_oauth(
    *,
    credential_path: str | None = None,
    timeout: float | None = None,
    callback_host: str = "127.0.0.1",
    callback_port: int | None = None,
    callback_path: str = "/callback",
    open_browser: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> str:
    timeout = (
        get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_LOGIN_TIMEOUT_SECONDS
        if timeout is None
        else timeout
    )
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="Anthropic OAuth login timed out.",
    )
    llm = AnthropicOAuthSession(
        credential_path=None,
        authorize_url=get_agent_settings().MOBILERUN_ANTHROPIC_OAUTH_MODERN_AUTHORIZE_URL,
        login_scope=DEFAULT_SETUP_TOKEN_SCOPE,
        timeout=timeout,
    )
    token = llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        expires_in=SETUP_TOKEN_EXPIRES_IN_SECONDS,
        deadline=deadline,
        **({"on_output": on_output} if on_output is not None else {}),
    )
    if credential_path is not None:
        save_anthropic_setup_token(
            credential_path,
            token,
            deadline=deadline,
        )
    return token


def save_anthropic_setup_token(
    credential_path: str,
    token: str,
    *,
    deadline: OAuthLoginDeadline | None = None,
) -> None:
    AuthProfileStore(credential_path).update_slot(
        "claudeAiOauth",
        {
            "accessToken": token,
            "refreshToken": None,
            "expiresAt": None,
            "scopes": [],
        },
        lock_timeout=deadline.remaining() if deadline is not None else None,
        before_commit=deadline.check if deadline is not None else None,
    )


def run_anthropic_oauth_setup(credential_path: str) -> None:
    run_anthropic_setup_token_oauth(credential_path=credential_path)
