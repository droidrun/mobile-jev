# agents/clients/oauth/

Implements provider OAuth flows and stores agent authentication profiles.

Keep provider authentication and profile storage here; CLI prompts live in `agents/cli/`.

## Files

| File | Purpose |
| --- | --- |
| `__init__.py` | Package marker; no public names defined. |
| `anthropic.py` | `AnthropicOAuthSession`, `run_anthropic_setup_token_oauth`, `save_anthropic_setup_token`, `run_anthropic_oauth_setup`. |
| `credentials.py` | `oauth_credentials_present` — oauth credential slot detection helpers. |
| `gemini.py` | `model_display_name_for_variant`, `GeminiOAuthSession`, `run_gemini_oauth_login`. |
| `grok.py` | `GrokOAuthError`, `GrokOAuthReloginRequired`, `GrokOAuthCredentials`, `GrokOAuthCredentialStore`, `GrokIDTokenValidator`, `GrokOAuthSessionManager`, `GrokOAuthAuth`, `GrokOAuthSession`, `run_grok_oauth_login`. |
| `login_timeout.py` | `OAuthLoginDeadline`, `open_browser_async`. |
| `openai.py` | `OpenAIOAuthCredentials`, `OpenAIOAuthCredentialStore`, `OpenAIOAuthSessionManager`, `OpenAIOAuthSession`, `run_openai_oauth_login`, `get_default_openai_credential_path`. |
| `profiles.py` | `AuthProfileFormatError`, `AuthProfileTransaction`, `AuthProfileStore`, `ApiKeySources`, `load_env_key_sources`, `load_env_keys`, `resolve_env_key`, `save_env_keys` — safe shared storage for mobilerun authentication profiles. |


See [agent clients](../README.md).
