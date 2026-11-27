# Chat-based selection

**Purpose:** Build a provider LLM and turn its response into the workflow’s next selection.

**What belongs here / what doesn't:** Keep adapter capabilities, chat transport and reply parsing together. Selecting a tool does not execute it.

**Contents:**

- `__init__.py` — package entry.
- `base.py` — `LLMSelector` chat and selection pipeline.
- `calls.py` — bounded calls, response validation and usage.
- `factory.py` — build an LLM and resolve its capabilities.
- `image.py` — prepare the model’s image and map its coordinate arguments back to device pixels.
- `llama_index.py` — `LlamaIndexLLMSelector` transport adapter.
- `models.py` — supported adapter and grounding metadata.
- `parsing.py` — parse XML/native calls and coerce wire arguments against the selected action’s schema. Parsing alone leaves argument values unchanged.

**Examples:** `llm = await build_llm("openai", "gpt-4.1-mini")` (from `.factory`); `LlamaIndexLLMSelector(llm)` uses it for selection. Native chat passes LlamaIndex tools to the provider adapter. For an explicitly native Anthropic or Gemini adapter, use `supports_native=True`. Their adapters extract tool calls; OpenAI paths keep the strict parser because partial JSON is not an executable action. XML models read the same tool metadata.

Without an explicit `AgentConfig`, the selector loads the effective agent config at selection time. The typed output schema is advertised by `complete`; the workflow validates the contents of `output` during execution. Remove any `output_schema` references from custom `agent.system_prompt` templates: that variable is no longer passed.

**Depends on / used by:** Uses LlamaIndex, selector types, API keys and typed config. The profile loader builds these selectors.

`MODELS` maps provider names to four fields:

| Field | Why the agent needs it |
|---|---|
| `adapter` | Select the LlamaIndex transport. OpenAI-compatible wrappers disable nested SDK retries; the adapter owns retries. |
| `env_var` | Find the provider-specific credential, or omit credentials for local Ollama. |
| `capabilities` | Select the agent's native-tool parser or XML protocol, and size screenshots and map coordinates for grounding. These are agent decisions, not LLM token limits. |
| `kwargs` | Route compatible APIs to the correct host and chat/tool protocol. Ollama uses its configured local context allocation instead of reserving the model maximum. |

LlamaIndex owns model metadata and request parameters. Setup reads the installed registries or adapter defaults; it does not send keys to a model-list endpoint. Providers without a registry accept a model ID directly.
