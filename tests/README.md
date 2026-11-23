# Tests

Build and install split wheels with `python3 scripts/prepare_split_env.py --base-source ../mobilerun --env .venv --extras dev,anthropic,langfuse`, using Python 3.11–3.13 outside `.venv`; see [CONTRIBUTING.md](../CONTRIBUTING.md). Do not use editable installs or the base checkout on `PYTHONPATH`. Run `.venv/bin/python -m pytest tests/packaging tests/compat tests/migration tests/unit -q`. The normal suite isolates user credentials and device discovery. `tests/unit/eval_coordinate_grounding.py` is an opt-in evaluation script, not a default test.

Real-device/LLM smoke tests are in `tests/smoke` and require `--smoke`, explicit configuration and approval for provider token costs. See `tests/smoke/README.md` before running them.
