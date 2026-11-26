# Console logging

**Purpose:** Show human-readable run progress without mixing display logic into workflow steps.

**What belongs here / what doesn't:** Format log-safe messages and filter library noise here. Product analytics and checkpoint files have separate consumers.

**Contents:**

- `__init__.py` — package entry.
- `console.py` — `ConsoleLog` callbacks for repairs, steps and results.
- `diagnostics.py` — run-local debug events and sensitive-value redaction.
- `redaction.py` — masks tracked sensitive values at run output boundaries.
- `setup.py` — configure library log filtering.

**Examples:** `configure_cli_logging(debug=True)` enables agent diagnostics. `ConsoleLog().on_repair()` reports an invalid selection. CLI progress goes to stderr; the final result goes to stdout. `--debug` keeps tracebacks; set `MOBILERUN_LOG_LEVEL=debug` to include transport diagnostics.

Jev debug output summarizes which kinds of choices the model received. With checkpoints enabled, `decisions/NNNN.json` holds the masked option list for each Jev call in run order, including retries. Debug output links to its relative path; the sidecar is also recorded without `--debug`.

**Depends on / used by:** Uses logging and core state/result types. The operator calls the display callbacks as it consumes the workflow stream.
