# Console logging

**Purpose:** Render workflow progress without exposing tracked sensitive values.

**What belongs here / what doesn't:** Console formatting, output redaction and CLI log routing. The workflow emits raw events and does not format console messages.

**Contents:**
- `__init__.py` — package entry.
- `console.py` — console messages for actions, repairs, subgoals and outcomes.
- `memory.py` — debug-only memory details from selector inputs.
- `redaction.py` — run-local redactor for output consumers; observes base vault input only in the calling task.
- `setup.py` — CLI log routing and transport-level filtering.

**Examples:** `configure_cli_logging(debug=True)` shows additional details on stderr; the final answer is on stdout. Set `MOBILERUN_LOG_LEVEL=debug` to include transport diagnostics.

Debug shows a short memory summary per selection. With checkpoints enabled it also saves the full masked detail under `memory/NNNN.txt` in the run folder. Retries and handoffs on the same step share that step's file.

If an agent or subgoal completes without a message, the consumer renders a display-only default from the configured completion prompt. The workflow state and checkpoint still contain `null`.

**Depends on / used by:** The operator routes events to the console and supplies the same redactor to the checkpoint and macro consumers.
