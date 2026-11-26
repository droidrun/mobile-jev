# Planning

**Purpose:** Decide the next subgoals or finish the overall run.

**What belongs here / what doesn't:** Planners read `Context[PlanningState]` and return a decision. They do not execute actions or write workflow state. `LlamaIndexPlanner` requires a `set_plan` or `complete` tool call. The planner sees the executor setup and full current UI, even when the child reported a result. Vision-capable models also get the current screenshot if available. Failed or unfinished children include recent thoughts, action results, screen visits, and observed blockers. When Jev hands back an unfinished subgoal, the planner also sees the raw choice and confidence details.

Planner turns and executor actions each use one step from the same budget. If only one step remains when the planner is called, it must call `complete` and report any blocker instead of setting another plan.

`set_plan` can include findings from the current screen, such as a shop's price. The workflow keeps them alongside child notes for later planner turns. It does not pass them to the executor.

**Contents:**

- `__init__.py` — planner interfaces and exports.
- `base.py` — `Planner`, `Plan`, and `Finish`.
- `llama_index.py` — LlamaIndex tool calls, validation and repair.

**Examples:** `PlanningOperator(planner=LlamaIndexPlanner(llm), executor=selector)`.

**Depends on / used by:** `PlanningWorkflow` calls the planner and applies its decision. The LlamaIndex implementation uses the shared chat and tool-parsing helpers.
