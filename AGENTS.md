# Agent contributor rules

Mobilerun Agents is the optional LLM-backed add-on for Mobilerun. Use Python 3.11–3.13. Implementation lives in `src/mobilerun/agents/`; base owns devices, shared APIs, and CLI dispatch. Keep the base checkout read-only.

## Development and tests

Use installed wheels in a fresh split environment, never editable installs or a checkout added to `PYTHONPATH`. Follow [CONTRIBUTING.md](CONTRIBUTING.md) for the harness and checks. Run the harness using a Python outside the target environment. It removes the base wheel's bundled agents before installing the add-on and recreates the selected environment; do not point it at an environment with data to keep.

Keep normal CI and unit tests fake-only: no provider calls, OAuth logins, cloud resources, or real devices. Isolate credentials and configuration; never print or commit secrets. Smoke tests are opt-in. Get explicit consent before live or paid tests, review costs, clean up created resources, and report costs and resource IDs.

Use type hints, small focused modules, and tests for behavior changes. Preserve public imports and legacy shims. Update relevant documentation when interfaces change.
