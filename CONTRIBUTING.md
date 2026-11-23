# Contributing

Use Python 3.11–3.13. Add-on code belongs in `src/mobilerun/agents/`. The Mobilerun base checkout is read-only; do not build in or modify it. Test non-editable installed wheels, not `pip install -e` or `PYTHONPATH=src`.

## Development loop

Install `uv` with `pipx install uv` or the [official installer](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```sh
uv lock --check
python3 scripts/prepare_split_env.py --base-source ../mobilerun --env .venv --extras dev,anthropic,langfuse
.venv/bin/python -m pytest
.venv/bin/python -m black --check --diff src tests scripts
.venv/bin/python -m ruff check src tests scripts
```

Replace the base source path with your own read-only checkout, or use `--base-wheel PATH` or `--base-version 0.7.4`. Run the harness with Python 3.11–3.13 outside the target environment, never `.venv/bin/python`; it **recreates** the selected environment. If `python3` points inside `.venv` or is an unsupported version, choose another Python, or use `uv run --no-project --python 3.13 python scripts/prepare_split_env.py ...`. The harness builds the add-on wheel, strips bundled agents from the test base install, installs the add-on, and records provenance in `.build/split-report.json`. Use a disposable path. Re-run it after code changes so tests use the new wheel. A published-base compatibility failure is a release blocker; do not silently replace the base with a local snapshot.

## Tests and security

Normal tests use fakes and isolated configuration and credentials. Never commit secrets or call providers, OAuth, cloud services, or real devices in CI. Smoke tests are opt-in with `--smoke`; inspect their prerequisites first. Get explicit consent before live or paid tests, clean up any created resources, and report costs and resource IDs.

Add tests for behavior changes. Keep public APIs and compatibility paths covered. Update module READMEs and docstrings when behavior changes.
