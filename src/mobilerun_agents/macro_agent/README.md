# Agent-assisted macro replay

**Purpose:** Replay a recorded macro. If its next pre-action screen does not match, stop before that action and hand the remaining work to an agent on the same device.

**What belongs here / what doesn't:** Own the handoff and shared step/time budgets here. Base owns deterministic playback; ordinary replay errors do not trigger agent recovery.

**Contents:**

- `__init__.py` — public `MacroAgentOperator` export.
- `operator.py` — guarded replay, divergence recovery and optional output extraction.
- `paths.py` — resolves run IDs and explicit macro paths for the CLI and operator.

**Examples:** `await MacroAgentOperator().execute(device, "recording/macro.json")` replays the file; `await MacroAgentOperator("recording/macro.json").execute(device, goal)` supplies a recovery goal. Passing `output=Model` can invoke an agent even after successful replay.

Use `mobilerun macro list` to find a saved run ID, then `mobilerun macro-agent replay <run_id>`. A run ID points to a folder inside the configured macro directory. You can instead pass an explicit macro folder (such as `./recording`) or its `macro.json` file. The checkpoint folder is not a replay target.

**Depends on / used by:** Uses the base macro player and `AgentOperator` for recovery. Python callers and `mobilerun macro-agent replay` use this operator; `agent/` never imports it.
