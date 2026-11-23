"""OAuth credential slot detection helpers."""

import json
from pathlib import Path

OAUTH_CREDENTIAL_SLOTS = {
    "gemini_oauth_code_assist": "geminiAntigravityOauth",
    "openai_oauth": "openaiOauth",
    "anthropic_oauth": "claudeAiOauth",
    "xai_oauth": "grokOauth",
}


def oauth_credentials_present(credential_path: str, variant_id: str) -> bool:
    path = Path(credential_path).expanduser()
    if not path.exists():
        return False
    slot = OAUTH_CREDENTIAL_SLOTS.get(variant_id)
    if not slot:
        return True  # unknown variant: fall back to file presence
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    nested = data.get(slot) if isinstance(data, dict) else None
    if not isinstance(nested, dict):
        return False
    # Token field names vary by provider: gemini access_token/refresh_token,
    # anthropic accessToken/refreshToken, openai access/refresh. Accept any.
    for key, val in nested.items():
        kl = key.lower()
        if val and ("token" in kl or kl in ("access", "refresh")):
            return True
    return False
