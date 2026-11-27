# Shared run contracts

**Purpose:** Represent live workflow state and the final typed result.

**What belongs here / what doesn't:** Keep plain state models here. Live device handles, clients and step orchestration belong higher up.

**Contents:**

- `__init__.py` — package entry.
- `privacy.py` — apply core secret masking to an individual workflow value.
- `snapshot.py` — adapts sanitized core `DeviceSnapshot` text to workflow state. A screenshot is a live `ImageBlock` and is excluded from serialization. `snapshot.compare(other)` detects changes in visible or focused text before comparing structural fingerprints. It returns `None` when the text is unchanged and either fingerprint is unavailable. Macro fingerprints remain structural.
- `state.py` — live `AgentState` and `PlanningState`, with `ActionRecord`, `Outcome` and `SubgoalResult`. `ActionRecord` stores the masked tool selection, pre-action observation, selection thoughts and `screen_changed` for memory and guards. It is not used for macro export.
- `targets.py` — resolves an action's target against its pre-action UI. Labels omit editable values. Target repeat detection uses class, resource ID, label and native bounds for indexed or hit-tested elements; it groups coordinate misses into 48-pixel cells. Transient indices and screen fingerprints do not participate in target repeat detection. Shared progress counters count unchanged steps and screen visits for workflow guards and history warnings.

**Examples:** `from mobilerun_agents.agent.workflows.core.state import AgentState`; `state = AgentState(goal="Read the battery level")` starts with an empty action history.

**Depends on / used by:** Uses Pydantic and base JSON types. Workflows, run consumers and memory can share these contracts without importing workflow implementations.
