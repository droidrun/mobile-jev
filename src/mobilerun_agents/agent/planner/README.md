# Planning

**Purpose:** Decide the next subgoals or finish the overall run.

**What belongs here / what doesn't:** Planners read `Context[PlanningState]` and return a decision. They do not execute actions or write workflow state. `LlamaIndexPlanner` requires a `set_plan` or `complete` tool call. When a child has no message, the planner sees current screen evidence; vision-capable models also see the screenshot if available. Failed children include recent thoughts, action results, screen visits, and observed blockers.

**Contents:**

- `__init__.py` — planner interfaces and exports.
- `base.py` — `Planner`, `Plan`, and `Finish`.
- `llama_index.py` — LlamaIndex tool calls, validation and repair.

**Examples:** `PlanningOperator(planner=LlamaIndexPlanner(llm), executor=selector)`.

**Depends on / used by:** `PlanningWorkflow` calls the planner and applies its decision. The LlamaIndex implementation uses the shared chat and tool-parsing helpers.
