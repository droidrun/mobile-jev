# Mobilerun Agents

The optional advanced-agents add-on for [Mobilerun](https://github.com/droidrun/mobilerun). It adds LLM-backed device operation, multi-provider LLM configuration, and `mobilerun agent ...` commands to the base CLI.

- `AgentOperator`: fast, single-LLM operation by default.
- `DelegationOperator`: a manager LLM plans and an executor LLM selects actions.
- `AgentMacroOperator`: replay a macro and fall back to an agent on divergence.

## Install

Requires Python 3.11–3.13. The add-on is **not yet published on PyPI**. After publication, install it with:

```sh
pip install mobilerun-agents
# Optional provider and tracing integrations:
pip install 'mobilerun-agents[anthropic,langfuse]'
```

These are planned public install commands, not available until publication. The `anthropic` and `langfuse` extras can also be selected separately.

For a base checkout whose metadata has a direct-URL routines dependency, `uv pip install` requires that dependency explicitly:

```sh
uv pip install mobilerun-agents 'mobilerun-routines[device] @ git+https://github.com/droidrun/mobilerun-routines.git@b4444f83fa9c52e680bf09b17635251c818c2ad8'
```

Plain `pip install mobilerun-agents` works without this workaround. This is temporary until the base publishes routines to PyPI; the add-on itself must also be published before these commands can install it from PyPI.

## Python API

```python
from mobilerun.agents import AgentMacroOperator, AgentOperator, DelegationOperator, MobileConfig

config = MobileConfig()
config.agent.reasoning = True
operator = AgentOperator(config=config)
planner = DelegationOperator()
macro = AgentMacroOperator()
```

Reasoning is set on `config.agent.reasoning`, not on the operator constructor. `DelegationOperator` enables it without changing the caller's configuration. Pass an operator to a Mobilerun device to run a task. Configure a provider first with `mobilerun agent llm configure`. Agent runs can incur LLM provider and device or cloud charges.

Once the add-on is installed, the base CLI exposes `mobilerun agent --help`, including `run`, `replay`, and `llm` commands. See the [package overview](src/mobilerun/agents/README.md) and [operator guide](src/mobilerun/agents/operators/README.md).

## Installation boundary and follow-ups

The base `mobilerun` distribution owns the regular `mobilerun` package. Non-editable wheels share its installed `site-packages/mobilerun/` directory. Editable add-on installs are not supported until a base-package change enables and tests discovery. Use the installed-wheel development loop in [CONTRIBUTING.md](CONTRIBUTING.md).

Publishing this add-on, adding a base `mobilerun[agents]` extra, and removing bundled agents and agent-only dependencies from base are follow-ups. The split does not yet make the published base dependency graph lean. Release also depends on verifying compatibility with the published base version.

## License

[MIT](LICENSE).
