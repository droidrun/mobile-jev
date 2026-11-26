# Shared run contracts

**Purpose:** Represent serializable run state and the final typed workflow result.

**What belongs here / what doesn't:** Keep plain state models here. Live device handles, clients and step orchestration belong higher up.

**Contents:**

- `images.py` — run-scoped screenshot files and cleanup.
- `screen.py` — observation fingerprinting.
- `snapshot.py` — the timestamped UI observation shared by actions and memory blocks.

- `__init__.py` — package entry.
- `state.py` — raw `AgentState` (goal, notes, action history and guards). Each executed `ActionRecord` has macro-compatible arguments, observation references, joined selection thoughts, and a nullable `screen_changed` flag.
- `result.py` — generic `AgentResult` and `FailureSummary` with counted `ScreenVisit` entries.

**Examples:** `from mobilerun_agents.agent.workflows.core.state import AgentState`; `state = AgentState(goal="Read the battery level")` starts with no recorded actions.

**Depends on / used by:** Uses Pydantic and base JSON types. Workflows, run consumers and memory can share these contracts without importing workflow implementations.
