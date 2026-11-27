# Console logging

**Purpose:** Render workflow progress with standard logging.

**What belongs here / what doesn't:** Console formatting and CLI log routing. The workflow emits raw events and does not format console messages.

**Contents:**
- `__init__.py` — package entry.
- `console.py` — console messages for actions, repairs, subgoals and outcomes.
- `memory.py` — debug-only memory details from selector inputs.
- `setup.py` — CLI log routing and transport-level filtering.

**Examples:** `configure_cli_logging(debug=True)` shows additional details on stderr; the final answer is on stdout. Set `MOBILERUN_LOG_LEVEL=debug` to include transport diagnostics.

Debug shows a short memory summary per selection, even without recording. The console does not write memory files; recorded memory details live in `agent.json`.

If an agent or subgoal completes without a message, the consumer renders a display-only default from the configured completion prompt. The workflow state still contains `null`.

**Depends on / used by:** The operator routes events to `ConsoleLog(redact)` and supplies the redactor used for console messages.
