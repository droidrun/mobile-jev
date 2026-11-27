# Console logging

**Purpose:** Render workflow progress with standard logging.

**What belongs here / what doesn't:** Console formatting and CLI log routing. The workflow emits raw events and does not format console messages.

**Contents:**
- `__init__.py` — package entry.
- `console.py` — console messages for actions, repairs, subgoals and outcomes.
- `memory.py` — debug-only memory details from selector inputs.
- `setup.py` — CLI log routing and transport-level filtering.

**Examples:** `configure_cli_logging(debug=True)` shows additional details on stderr; the final answer is on stdout. Set `MOBILERUN_LOG_LEVEL=debug` to include transport diagnostics.

Debug shows a short memory summary per selection. With checkpoints enabled it also saves the full unmasked detail under `memory/NNNN.txt` in the run folder. Retries and handoffs on the same step share that step's file.

If an agent or subgoal completes without a message, the consumer renders a display-only default from the configured completion prompt. The workflow state and checkpoint still contain `null`.

**Depends on / used by:** The operator routes events to the console. The base's `secret_scope` masks sensitive values in log records during sensitive calls; run data is stored unmasked.
