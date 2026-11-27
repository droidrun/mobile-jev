# Agent CLI

**Purpose:** Set up and run mobile agents, and inspect cloud tasks.

**What belongs here / what doesn't:** Click commands and terminal prompts belong here. Credential storage, model construction and workflow execution do not.

**Contents:** `__init__.py` exports the command group. `group.py` registers commands; `providers.py`, `profiles.py` and `configure.py` manage configuration; `configuration.py` provides shared prompt and persistence helpers; `run.py` launches runs; `task.py` inspects and stops cloud tasks.

**Examples:**

## Provider credentials

```sh
mobilerun agent providers add openai
mobilerun agent providers add openrouter
mobilerun agent providers list
mobilerun agent providers remove openai
```

`add` accepts a pasted key in a terminal or a piped key with `--api-key-stdin`. OpenRouter also offers browser sign-in. An environment key takes precedence over a saved key. `list` shows the source, never the value. `remove` deletes only the saved key: it does not unset environment variables, revoke the key, or remove profiles.

## Profiles and setups

```sh
mobilerun agent profiles add luna --provider openai --model gpt-6-luna --reasoning low
mobilerun agent profiles add sol --provider openai --model gpt-6.1-sol --reasoning high
mobilerun agent profiles add jev-on --provider typesafe --vision on --min-confidence 0.7
mobilerun agent profiles list
mobilerun agent configure fast --selector luna
mobilerun agent configure deep --planner sol --executor jev-on --delegate luna
mobilerun agent configure show deep
mobilerun agent configure default fast
mobilerun agent configure default --unset
mobilerun agent run "Open Settings" DEVICE --setup deep --vision off
mobilerun agent task status TASK_ID --json
mobilerun agent configure remove deep
mobilerun agent profiles remove jev-on
```

LLM profiles require `--model` and can specify a credential-free `--base-url`. Model-ID validation for OpenAI and Anthropic is offline. Decision model profiles (currently TypeSafe/Jev) don't take a model ID or reasoning setting. To use Jev, set `TYPESAFE_API_KEY` or save a key with `agent providers add typesafe`. Only a decision model selector or executor can have a delegate. The planner and delegate must be LLM profiles.

In a terminal, `agent` opens a menu. The `providers` and `profiles` submenus offer add, list and remove. In `configure [NAME]`, choose **Straight** or **Manager + executor**, then pick profiles for the roles. The menu asks about delegation only if the selected action profile is a decision model. Choose a name and whether to make it the default; **None (no default)** clears the current default. **Done** saves both choices; **Cancel** or Esc writes nothing. `configure default` lets you choose a named setup or **None (no default)**, while `configure default --unset` clears it directly. `run --setup NAME` chooses a setup for one run, and `--vision off|on|only` overrides its action selector and delegate for that run. Without a default setup, runs use a straight agent with the first LLM profile, or a credential-based fallback if there are no LLM profiles.

`run --no-record` skips saving the run's recording, including screenshots and debug memory files. Console output still works. This flag overrides both `MOBILERUN_AGENT_RECORDING_ENABLED` and `recording.enabled` in [the YAML schema](../../config/README.md).

Profiles referenced by any setup cannot be removed until those setups are updated or removed.

Cloud agent runs do not prompt separately for device charges; the base CLI enforces the `MOBILERUN_BILLING` setting. The run command has no `--yes` option. Stopping a cloud task still requires confirmation, or `--yes` in non-interactive use.

**Depends on / used by:** Uses base CLI prompts and device selection, provider credentials, model metadata and agent configuration. Base loads it through a CLI entry point.
