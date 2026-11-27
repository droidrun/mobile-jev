# Local smoke (opt-in)

This test controls a real local Android device and spends LLM tokens. Do not run it without explicit approval. No cloud resources are created. The device UI is changed and the test attempts to return it home and close the connection in `finally`. Configure `MOBILERUN_SMOKE_SERIAL`, `MOBILERUN_SMOKE_LLM_PROVIDER`, and `MOBILERUN_SMOKE_LLM_MODEL`, plus the provider's credentials. The harness restores secrets only in memory for smoke tests.

Free checks: `.venv/bin/python -m pytest --collect-only tests/smoke -q` and `.venv/bin/python -m pytest tests/smoke -q` (skipped by default).

Approved paid run: `.venv/bin/python -m pytest --smoke -m smoke_llm tests/smoke/test_local_agent.py -q -s`
