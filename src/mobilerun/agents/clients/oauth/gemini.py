import base64
import hashlib
import json
import os  # Host session variables determine whether browser login is available.
import platform
import queue
import secrets
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer as HTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict
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
    GEMINI_OAUTH_REDIRECT_URI,
    get_agent_settings,
)

DEFAULT_MODEL = "gemini-3.8-flash-tiered"
GEMINI_OAUTH_DEFAULT_MODEL = DEFAULT_MODEL

DEFAULT_CODE_ASSIST_API_VERSION = "v1internal"

DEFAULT_CODE_ASSIST_MODELS_METHOD = "fetchAvailableModels"

DEFAULT_CLIENT_ID = "1071006060591-tmhssin2h21lcre235vtolojh4g403ep.apps.googleusercontent.com"

DEFAULT_CLIENT_SECRET = "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"

DEFAULT_OAUTH_SCOPES = (
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
)

DEFAULT_CREDENTIAL_SLOT = "geminiAntigravityOauth"

ANTIGRAVITY_CLIENT_VERSION = "2.14.0"


def _antigravity_user_agent() -> str:
    os_name = {"win32": "windows"}.get(sys.platform, sys.platform)
    arch = platform.machine().lower()
    arch = {"x86_64": "amd64", "aarch64": "arm64"}.get(arch, arch)
    return f"antigravity/{ANTIGRAVITY_CLIENT_VERSION} {os_name}/{arch}"


_IGNORED_REQUEST_KWARGS = {"formatted", "project", "project_id"}

_GEMINI_SAMPLING_CONFIG_KEYS = {"temperature", "topP", "topK", "top_p", "top_k"}

_UNSET = object()


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
    """Parse pasted input: full URL with code= param, code#state, or bare code."""
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


GEMINI_OAUTH_RETIRED_MODELS = frozenset(
    {"gemini-3.5-flash-low", "gemini-3.5-flash-extra-low", "gemini-3-flash-agent"}
)


def model_display_name_for_variant(family: str, variant: str, model_id: str) -> str:
    return {
        "gemini-3.8-flash-tiered": "gemini-3.8-flash",
        "gemini-3.7-flash-tiered": "gemini-3.7-flash",
    }.get(model_id, model_id)


class GeminiOAuthSession:
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
        client_secret: str = DEFAULT_CLIENT_SECRET,
        scopes: tuple[str, ...] = DEFAULT_OAUTH_SCOPES,
        credential_slot: str = DEFAULT_CREDENTIAL_SLOT,
        code_assist_endpoint: str | None = None,
        code_assist_api_version: str = DEFAULT_CODE_ASSIST_API_VERSION,
    ) -> None:
        timeout = (
            get_agent_settings().MOBILERUN_GEMINI_OAUTH_DEFAULT_TIMEOUT_SECONDS
            if timeout is None
            else timeout
        )
        refresh_buffer_seconds = (
            get_agent_settings().MOBILERUN_GEMINI_OAUTH_REFRESH_BUFFER_SECONDS
            if refresh_buffer_seconds is None
            else refresh_buffer_seconds
        )
        settings = get_agent_settings()
        self.credential_path = credential_path or str(
            get_agent_settings().gemini_oauth_credential_path
        )
        self.model = model
        self.timeout = timeout
        self.access_token = access_token
        self.refresh_token = refresh_token
        self.client_id = client_id
        self.authorize_url = authorize_url or settings.MOBILERUN_GEMINI_OAUTH_AUTHORIZE_URL
        self.token_url = token_url or settings.MOBILERUN_GEMINI_OAUTH_TOKEN_URL
        self.refresh_buffer_seconds = refresh_buffer_seconds
        self.client_secret = client_secret
        self.scopes = scopes
        self.credential_slot = credential_slot
        self.code_assist_endpoint = (
            code_assist_endpoint or settings.MOBILERUN_GEMINI_OAUTH_CODE_ASSIST_ENDPOINT
        )
        self.code_assist_api_version = code_assist_api_version
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
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return

        # Load only from the dedicated Antigravity slot. A bare/top-level or old
        # "geminiOauth" credential set has the wrong client/scopes and is ignored.
        payload = raw.get(self.credential_slot)
        if not isinstance(payload, dict):
            return

        file_access = payload.get("access_token")
        file_refresh = payload.get("refresh_token")
        expiry_ms = payload.get("expiry_date")

        if not self._cached_access_token and isinstance(file_access, str):
            self._cached_access_token = file_access
        if not self._cached_refresh_token and isinstance(file_refresh, str):
            self._cached_refresh_token = file_refresh
        if isinstance(expiry_ms, (int, float)):
            self._access_token_expiry = float(expiry_ms) / 1000.0

    def _persist_credentials(
        self,
        *,
        deadline: OAuthLoginDeadline | None = None,
    ) -> None:
        if not self.credential_path:
            return

        path = Path(self.credential_path).expanduser()
        AuthProfileStore(path).update_slot(
            self.credential_slot,
            {
                "access_token": self._cached_access_token,
                "refresh_token": self._cached_refresh_token,
                "token_type": "Bearer",
                "expiry_date": (
                    int(self._access_token_expiry * 1000) if self._access_token_expiry else None
                ),
            },
            lock_timeout=deadline.remaining() if deadline is not None else None,
            before_commit=deadline.check if deadline is not None else None,
        )

    def _metadata_payload(self) -> Dict[str, str]:
        return {
            "ideType": "IDE_UNSPECIFIED",
            "platform": "PLATFORM_UNSPECIFIED",
            "pluginType": "GEMINI",
        }

    def _build_headers(self, token: str) -> Dict[str, str]:
        # Identify as the Antigravity app. The gemini-cli/vscode User-Agent +
        # gl-node X-Goog-Api-Client make the backend 500 for an aicode token.
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "User-Agent": _antigravity_user_agent(),
            "Client-Metadata": json.dumps(self._metadata_payload()),
        }

    def fetch_available_models(
        self,
        *,
        deadline: OAuthLoginDeadline | None = None,
        access_token: str | None = None,
    ) -> list[Dict[str, Any]]:
        """Agent-usable Gemini models for the current entitlement.

        Calls Code Assist ``fetchAvailableModels`` and returns dicts with
        ``id``, ``display_name`` and ``supports_images``. Internal/tab/aux,
        image-generation and deprecated ids are filtered out. Used to verify
        login and to optionally discover the live catalog. Requires the
        Antigravity client headers.
        """
        token = access_token or self._resolve_access_token()
        effective_timeout = (
            self.timeout if deadline is None else deadline.remaining(cap=self.timeout)
        )
        response = self._session.post(
            self._method_url(DEFAULT_CODE_ASSIST_MODELS_METHOD),
            headers=self._build_headers(token),
            json={},
            timeout=effective_timeout,
        )
        if deadline is not None:
            deadline.check()
        response.raise_for_status()
        data = response.json()
        if deadline is not None:
            deadline.check()
        models = data.get("models")
        if not isinstance(models, dict):
            return []
        excluded = set((data.get("deprecatedModelIds") or {}).keys())
        excluded.update(GEMINI_OAUTH_RETIRED_MODELS)
        excluded.update(data.get("imageGenerationModelIds") or ())
        out: list[Dict[str, Any]] = []
        for model_id, meta in models.items():
            if not isinstance(meta, dict) or model_id in excluded:
                continue
            if meta.get("isInternal"):
                continue
            if "GEMINI" not in str(meta.get("apiProvider") or ""):
                continue
            display_name = meta.get("displayName")
            is_tiered_model = (
                isinstance(model_id, str)
                and model_id.startswith("gemini-")
                and model_id.endswith("-tiered")
            )
            # The live Antigravity catalog omits displayName for its public
            # tiered agent models. Keep requiring a name for every other model
            # so nameless tab/aux entries are not exposed as user choices.
            if not display_name and not is_tiered_model:
                continue
            friendly_name = model_display_name_for_variant("gemini", "oauth", model_id)
            if friendly_name == model_id:
                friendly_name = display_name or model_id
            out.append(
                {
                    "id": model_id,
                    "display_name": friendly_name,
                    "supports_images": bool(meta.get("supportsImages")),
                }
            )
        if deadline is not None:
            deadline.check()
        return out

    def _access_token_is_stale(self) -> bool:
        if not self._access_token_expiry:
            return False
        return time.time() >= (self._access_token_expiry - self.refresh_buffer_seconds)

    def _refresh_access_token(self) -> str:
        refresh_token = self._cached_refresh_token or self.refresh_token
        if not refresh_token:
            raise ValueError(
                "No refresh token available. Provide `refresh_token` or a credential file."
            )

        response = self._session.post(
            self.token_url,
            data={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()

        access_token = data.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise RuntimeError(f"Token refresh succeeded but no access_token returned: {data}")

        expires_in = data.get("expires_in", 3600)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 3600

        self._cached_access_token = access_token
        self._cached_refresh_token = data.get("refresh_token") or refresh_token
        self._access_token_expiry = time.time() + expires_in_s
        self._persist_credentials()

        return access_token

    def _exchange_authorization_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: Optional[str] = None,
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
    ) -> str:
        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if code_verifier:
            payload["code_verifier"] = code_verifier

        request_timeout = self.timeout if deadline is None else deadline.remaining(cap=self.timeout)
        response = self._session.post(
            self.token_url,
            data=payload,
            timeout=request_timeout,
        )
        if deadline is not None:
            deadline.check()
        response.raise_for_status()
        data = response.json()
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

        expires_in = data.get("expires_in", 3600)
        try:
            expires_in_s = int(expires_in)
        except (TypeError, ValueError):
            expires_in_s = 3600

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
            if persist_credentials:
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
        redirect_uri: str,
        state: str,
        prompt_consent: bool,
        code_challenge: Optional[str] = None,
    ) -> str:
        scope = " ".join(self.scopes)
        query = {
            "client_id": self.client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": scope,
            "access_type": "offline",
            "state": state,
        }
        if prompt_consent:
            query["prompt"] = "consent"
        if code_challenge:
            query["code_challenge"] = code_challenge
            query["code_challenge_method"] = "S256"
        return f"{self.authorize_url}?{urlencode(query)}"

    def login(
        self,
        *,
        open_browser: bool = True,
        timeout_seconds: float | None = None,
        callback_host: str = "127.0.0.1",
        callback_port: int | None = None,
        callback_path: str = "/oauth2callback",
        prompt_consent: bool = True,
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        settings = get_agent_settings()
        timeout_seconds = (
            settings.MOBILERUN_GEMINI_OAUTH_LOGIN_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        callback_port = (
            settings.MOBILERUN_GEMINI_OAUTH_CALLBACK_PORT
            if callback_port is None
            else callback_port
        )
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        # Headless environments: use authcode redirect flow (no local server)
        use_authcode = _is_headless_environment() or get_agent_settings().oauth_manual
        if use_authcode:
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                prompt_consent=prompt_consent,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
                **({"on_output": on_output} if on_output is not None else {}),
            )

        # Desktop: browser callback server
        result: Dict[str, Optional[str]] = {"code": None, "state": None, "error": None}
        done = threading.Event()
        callback_lock = threading.Lock()
        expected_state = secrets.token_hex(32)
        code_verifier, code_challenge = _pkce_pair()

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
            if on_output is not None:
                on_output(
                    f"Could not bind callback server on {callback_host}:{callback_port} ({exc}). "
                    "Falling back to headless code entry."
                )
            return self.login_headless(
                open_browser=open_browser,
                timeout_seconds=timeout_seconds,
                prompt_consent=prompt_consent,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
                **({"on_output": on_output} if on_output is not None else {}),
            )

        active_deadline.check()
        actual_port = httpd.server_address[1]
        redirect_uri = f"http://127.0.0.1:{actual_port}{callback_path}"
        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
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
            if result["state"] != expected_state:
                raise RuntimeError("OAuth callback state mismatch.")
            if not result["code"]:
                raise RuntimeError("OAuth callback did not include an authorization code.")

            access_token = self._exchange_authorization_code(
                result["code"],
                redirect_uri,
                code_verifier=code_verifier,
                deadline=active_deadline,
                persist_credentials=persist_credentials,
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        return access_token

    def login_headless(
        self,
        *,
        open_browser: bool = False,
        timeout_seconds: float | None = None,
        input_fn: Any = input,
        prompt_consent: bool = True,
        deadline: OAuthLoginDeadline | None = None,
        persist_credentials: bool = True,
        on_output: Callable[[str], None] | None = None,
    ) -> str:
        """Headless OAuth flow for SSH/WSL environments."""
        timeout_seconds = (
            get_agent_settings().MOBILERUN_GEMINI_OAUTH_LOGIN_TIMEOUT_SECONDS
            if timeout_seconds is None
            else timeout_seconds
        )
        max_code_attempts = get_agent_settings().MOBILERUN_GEMINI_OAUTH_MAX_CODE_ATTEMPTS
        active_deadline = deadline or OAuthLoginDeadline(timeout_seconds)
        active_deadline.check()

        code_verifier, code_challenge = _pkce_pair()
        expected_state = secrets.token_hex(32)
        redirect_uri = GEMINI_OAUTH_REDIRECT_URI

        auth_url = self._build_auth_url(
            redirect_uri=redirect_uri,
            state=expected_state,
            prompt_consent=prompt_consent,
            code_challenge=code_challenge,
        )

        if on_output is not None:
            on_output(
                f"\nSign in with your Google account:\n"
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
                    code = _normalize_manual_code(raw, expected_state)
                except Exception:  # noqa: BLE001
                    if attempt == 0:
                        if on_output is not None:
                            on_output("Invalid code. Try again.")
                        continue
                    raise RuntimeError("Login failed.") from None
                if code:
                    return self._exchange_authorization_code(
                        code,
                        redirect_uri,
                        code_verifier=code_verifier,
                        deadline=active_deadline,
                        persist_credentials=persist_credentials,
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

    def _resolve_access_token(self) -> str:
        env_access_token = secret_value(get_agent_secrets().GEMINI_OAUTH_ACCESS_TOKEN)
        if env_access_token:
            return env_access_token

        if self._cached_access_token and not self._access_token_is_stale():
            return self._cached_access_token

        if self._cached_access_token and not self._cached_refresh_token:
            # If we only have an access token and no expiry/refresh, still try to use it.
            return self._cached_access_token

        if self._cached_refresh_token or self.refresh_token:
            return self._refresh_access_token()

        raise ValueError(
            "No OAuth token available. Provide `access_token`, `refresh_token`, "
            "or a valid credential_path."
        )

    def _method_url(self, method: str) -> str:
        base = self.code_assist_endpoint.rstrip("/")
        version = self.code_assist_api_version.strip("/")
        return f"{base}/{version}:{method}"


def run_gemini_oauth_login(
    credential_path: str,
    model: str | None,
    timeout: float | None = None,
    callback_host: str = "127.0.0.1",
    callback_port: int | None = None,
    callback_path: str = "/oauth2callback",
    open_browser: bool = True,
    on_output: Callable[[str], None] | None = None,
) -> None:
    class OutputOption(TypedDict, total=False):
        on_output: Callable[[str], None]

    output_option: OutputOption = {}
    if on_output is not None:
        output_option["on_output"] = on_output
    timeout = (
        get_agent_settings().MOBILERUN_GEMINI_OAUTH_LOGIN_TIMEOUT_SECONDS
        if timeout is None
        else timeout
    )
    deadline = OAuthLoginDeadline(
        timeout,
        timeout_message="Gemini OAuth login timed out.",
    )
    llm = GeminiOAuthSession(
        model=model or DEFAULT_MODEL,
        credential_path=credential_path,
        timeout=timeout,
    )
    access_token = llm.login(
        open_browser=open_browser,
        timeout_seconds=deadline.remaining(),
        callback_host=callback_host,
        callback_port=callback_port,
        callback_path=callback_path,
        deadline=deadline,
        persist_credentials=False,
        **output_option,
    )
    # Verify the Antigravity consumer entitlement resolves before declaring
    # success (catches scope / header / endpoint problems at login time). Raise
    # on failure so the caller does not print a misleading success message.
    try:
        models = llm.fetch_available_models(
            deadline=deadline,
            access_token=access_token,
        )
    except TimeoutError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Gemini OAuth login received a token, but the Antigravity entitlement "
            f"check (fetchAvailableModels) failed: {exc}. The login is not "
            "usable; verify your Google One / AI access and retry."
        ) from exc
    if not models:
        raise RuntimeError(
            "Gemini OAuth login returned no usable models; credentials were not saved."
        )
    llm._persist_credentials(deadline=deadline)
    if on_output is not None:
        on_output(f"✓ Gemini (Antigravity) login OK — {len(models)} models available.")
