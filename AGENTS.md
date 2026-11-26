# Agent contributor rules

- Implement the add-on in `src/mobilerun_agents/`. Base owns devices, extensions, macro playback and CLI dispatch; keep the base checkout read-only.
- Imports go down: CLI/public exports → `macro_agent` → `agent/operators` → `agent/workflows` → services (including `agent/planner` and `agent/tools`) → `agent/run.py` → `workflows/core` → `auth` → `config`/`errors`. Within `agent/tools`, order is `registry` → `atomic` → `helpers`; selectors may use helpers. The agent must not import macro replay.
- External plugins are base `mobilerun.core.operator.Operator` implementations, not workflows. Register a no-argument subclass or an instance in `mobilerun.agents`; `ExternalOperator` delegates directly to it.
- Use specific types. No `isinstance` dispatch, no legacy aliases/shims, and no bare `*` except where `execute` mirrors the base API. Keep prompts in `assets/prompts/` and tunables in typed config.
- Test installed wheels on Python 3.11–3.13. Use `python3 scripts/prepare_split_env.py --base-source ../mobilerun --env .venv --extras dev,langfuse` from the repo root. The harness recreates `.venv`; do not use editable installs or inject source via `PYTHONPATH`.
- Never print credentials. Unit tests use fakes, not real models, cloud services or devices. Get consent before live tests, clean up resources and report costs. See [CONTRIBUTING.md](CONTRIBUTING.md).
