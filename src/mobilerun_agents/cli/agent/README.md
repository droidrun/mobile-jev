# cli/agent/

**Purpose:** Configure a model, run an agent on a device, or inspect a cloud task.

**What belongs here / what doesn't:** Click commands and terminal prompts belong here. Model metadata, key storage and workflow execution live in the agent package.

**Contents:** `__init__.py` exports the group. `group.py` registers commands lazily. `setup.py` implements setup and logout. `wizard.py` handles the interactive provider menu. `profiles.py` manages profiles and the default LLM role. `run.py` runs goals. `task.py` provides cloud task status, trajectory and stop commands.

**Examples:**

```sh
mobilerun agent setup
mobilerun agent setup --provider ollama --model llama3.2:3b
mobilerun agent setup --provider openai --default
mobilerun agent setup --provider typesafe --default
mobilerun agent setup --provider typesafe --default --jev-only
mobilerun agent setup --default ollama
mobilerun agent run "Open settings" DEVICE --vision on
mobilerun agent task status TASK_ID --json
mobilerun agent logout openai
```

Add multiple providers, change models, or set a default. Done saves the session; Esc or Ctrl-C discards local changes. Choose an installed LlamaIndex model or **other…** to enter an ID. Model selection is offline. For a new provider without a local registry, pass `--model` in scripts. Only OpenRouter offers browser sign-in. Set the provider environment variable or pipe a key with `--api-key-stdin`. Logout removes the profile and saved key, without revoking keys or unsetting environment variables.

`agent run --vision off|on|only` overrides the selected profiles for one run without changing saved settings.

Changing models keeps that provider's endpoint and adapter options. Adding another provider never copies an endpoint from an existing profile. Jev uses the default LLM as its delegate unless a different delegate was explicitly configured.

**Depends on / used by:** Uses base CLI prompts and device selection, provider credentials, model metadata and configuration. Registered through the base CLI extension entry point.

**TypeSafe (Jev):** Set `TYPESAFE_API_KEY` or supply a key with `--api-key-stdin`. Choose Jev in the menu to use it for the agent and executor. By default, Jev delegates uncertain decisions to the default LLM; configure that LLM first. Choose **Jev only** to run without a delegate. Planning still requires an LLM. `mobilerun agent logout typesafe` restores the default LLM and removes the saved TypeSafe key.
