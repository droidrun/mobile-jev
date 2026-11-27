# Shared run contracts

**Purpose:** Represent serializable run state and the final typed workflow result.

**What belongs here / what doesn't:** Keep plain state models here. Live device handles, clients and step orchestration belong higher up.

**Contents:**

- `__init__.py` — package entry.
- `snapshot.py` — timestamped UI observation, path-only checkpoint screenshot and cached screen fingerprint. `snapshot.compare(other)` reports changed, unchanged, or incomparable.
- `state.py` — raw `AgentState` and `PlanningState`, with `ActionRecord`, `Outcome` and `SubgoalResult`. Each executed action stores its native tool selection, pre-action observation, joined selection thoughts and a nullable `screen_changed` flag.
- `targets.py` — resolves an action's target against its pre-action UI. Labels omit editable values. Target repeat detection uses class, resource ID, label and native bounds for indexed or hit-tested elements; it groups coordinate misses into 48-pixel cells. Transient indices and screen fingerprints do not participate in target repeat detection. Shared progress counters count unchanged steps and screen visits for workflow guards and history warnings.

**Examples:** `from mobilerun_agents.agent.workflows.core.state import AgentState`; `state = AgentState(goal="Read the battery level")` starts with no recorded actions.

**Depends on / used by:** Uses Pydantic and base JSON types. Workflows, run consumers and memory can share these contracts without importing workflow implementations.
