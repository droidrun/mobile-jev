# Planning

**Purpose:** Decide the next subgoals or finish the overall run.

**What belongs here / what doesn't:** Planners return a decision. They do not execute actions or write workflow state. `LlamaIndexPlanner` sees text only and must call `set_plan` or `complete`.

**Contents:**

- `__init__.py` — planner interfaces and exports.
- `base.py` — `Planner`, its text-only `PlanningInput`, subgoal results and decisions.
- `llama_index.py` — LlamaIndex tool calls, validation and repair.

**Examples:** `PlanningOperator(planner=LlamaIndexPlanner(llm), executor=selector)`.

**Depends on / used by:** `PlanningWorkflow` calls the planner and applies its decision. The LlamaIndex implementation uses the shared chat and tool-parsing helpers.
