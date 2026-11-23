# mobilerun/agents/cli/agent/llm

Configure the agent's LLM provider, authentication and model. Provider requests may cost money.

## Quick example

```sh
mobilerun agent llm configure
mobilerun agent llm login anthropic
mobilerun agent llm login openai
```

The configure wizard supports providers including OpenAI, Anthropic, Gemini, Ollama and OpenAI-compatible endpoints such as OpenRouter. Select a provider and authentication mode in the wizard. OAuth login commands also exist for Gemini and XAI; use `mobilerun agent llm login --help` for the current list. API-key environment variables include `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` and `GOOGLE_API_KEY`; keep keys out of tasks and source files. Ollama requires a running server and a suitable local model.

Provider and model can also be overridden per run with `--provider` and `--model`; `--llm-base-url` and `--llm-api-base` configure LLM endpoints, not the cloud device API.

| Path | Purpose |
| --- | --- |
| `configure.py`, `wizard.py` | Profile configuration wizard |
| [login/](login/README.md) | Provider OAuth commands |
| `group.py` | LLM command group |
