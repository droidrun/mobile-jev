# Contributing

Use Python 3.11–3.13. Add-on code belongs in `src/mobilerun_agents/`; keep the Mobilerun base checkout read-only. Test installed split wheels, never editable installs or `PYTHONPATH=src`.

## Development checks

```sh
uv lock --check
python3 scripts/prepare_split_env.py --base-source ../mobilerun --env .venv --extras dev,langfuse
.venv/bin/python -m pytest
.venv/bin/python -m black --check --diff src tests scripts
.venv/bin/python -m ruff check src tests scripts
```

Run the split harness with an external Python 3.11–3.13 interpreter; it recreates `.venv`. You may instead use `--base-wheel PATH` or `--base-version VERSION`. A published-base compatibility failure is a release blocker.

## Tests and safety

Tests use fakes and isolated configuration. Never commit secrets or call providers, cloud services or real devices in CI. Smoke tests are opt-in; inspect `tests/smoke/README.md` and get consent before live or paid runs. Clean up resources and report costs. Add fake tests for behavior changes.

## Layout and import direction

Imports go down: CLI/public exports → `macro_agent/` → `agent/operators/` → `agent/workflows/` → services (including `agent/planner/` and `agent/tools/`) → `agent/run.py` → `agent/workflows/core/` → `agent/auth/` → `config/` and `errors/`. Within `agent/tools/`, imports go from registry to atomic to helpers; selectors may use helpers. Replay may call the agent, never the reverse. Base owns device operations, extensions and macro playback.

## Adding a selector

LLM adapters live in `agent/action_selector/llm/`; decision selectors use `DecisionModel` and `DecisionSelector`. Keep model-facing text in `assets/prompts/` and tunables in typed configuration. Use explicit types and canonical import paths; do not add compatibility aliases or shims. Jev's registry choices are finite, while selectors choose their own `vision` mode and exclusions.
