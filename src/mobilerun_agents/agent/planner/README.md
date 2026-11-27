# Planning

**Purpose:** Decide the next subgoals or finish the overall run.

**What belongs here / what doesn't:** Planners receive `Context[PlanningState]` and run-scoped LlamaIndex `Memory` and return a decision. They do not execute actions or write workflow state. `LlamaIndexPlanner` requires a `set_plan` or `complete` tool call. It sends system instructions, memory, then a step message with the full current UI and device context. Vision-capable models also get the current screenshot if available.

`memory.planner.blocks` selects and orders setup, goal, findings, the last subgoal, and older subgoal history. The latest subgoal includes the before and after app/activity, failure evidence, and any raw Jev abstention data. Older entries are capped by `memory.planner.history` (default 5) and dropped oldest first under token pressure. The planner stores no chat history. When restoring a checkpoint, rebuild memory using the restored state's executor kind; each read renders from persisted state.

Planner turns and executor actions each use one step from the same budget. If only one step remains when the planner is called, it must call `complete` and report any blocker instead of setting another plan.

`set_plan` can include findings from the current screen, such as a shop's price. The workflow keeps them alongside child notes for later planner turns. It does not pass them to the executor.

**Contents:**

- `__init__.py` — planner interfaces and exports.
- `base.py` — `Planner`, `Plan`, and `Finish`.
- `llama_index.py` — LlamaIndex tool calls, validation and repair.

**Examples:** `PlanningOperator(planner=LlamaIndexPlanner(llm), executor=selector)`.

**Depends on / used by:** `PlanningWorkflow` calls the planner and applies its decision. The LlamaIndex implementation uses the shared chat and tool-parsing helpers.
