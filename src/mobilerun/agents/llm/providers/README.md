# mobilerun/agents/llm/providers

Defines provider families, authentication variants and model selection policy. The LLM loader is in the sibling `utils` package.

## Quick example

An Ollama server and the selected model must be available to make requests. Model inference consumes local resources.

```python
from mobilerun.agents.llm.llm_picker import load_llm

llm = load_llm("Ollama", model="llama3.2:3b", base_url="http://localhost:11434")
```

For a user-facing setup wizard, run `mobilerun agent llm configure`; see [LLM setup](../../cli/agent/llm/README.md). API providers can incur charges and may require optional provider dependencies.

| File | Purpose |
| --- | --- |
| `registry.py`, `types.py` | Provider families, variants and capabilities |
| `setup_service.py`, `model_selection.py` | Profile and model selection |
| `anthropic.py`, `grok.py`, `minimax.py` | Provider-specific metadata and handling |
