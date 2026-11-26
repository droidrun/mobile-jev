# Agent contributor rules

- Implement the add-on in `src/mobilerun_agents/`. Base owns devices, extension registration, macro playback and CLI dispatch. Leave the base checkout read-only.
- Imports go down: CLI/public exports → `macro_agent/` → `agent/operators/` → `agent/workflows/` → services (`action_selector/`, `memory/`, `tools/`, `checkpoints/`, `analytics/`, `logs/` under `agent/`) → `agent/run.py` → `agent/workflows/core/` contracts → `agent/auth/` → `config/`, `errors/`. Inside `agent/tools/`, imports go from `registry` to `atomic` to `helpers`; selectors can also use `helpers`. The agent must not import macro replay.
- Use specific types. Do not dispatch via `isinstance`, `getattr`, `hasattr` or `cast`. Keep model-facing prompts in `assets/prompts/`, tunables in typed configuration, and device actions and run tools in one run-local registry.
- No legacy paths, aliases, shims, subscription logins or unused helpers. Update examples and fake-only tests alongside an API change.
- Test installed wheels on Python 3.11–3.13. From the repo root, with an external Python: `python3 scripts/prepare_split_env.py --base-source ../mobilerun --env .venv --extras dev,langfuse`. The harness recreates `.venv`; never use editable installs or inject the source tree via `PYTHONPATH`.
- Never print credentials. Unit tests must not use real models, cloud services or devices. Get consent before live tests, clean up resources and report costs. See [CONTRIBUTING.md](CONTRIBUTING.md) for checks.
