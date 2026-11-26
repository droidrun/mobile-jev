# Choice models

**Purpose:** Ask a decision model to pick from available actions rather than generate free-form calls.

**What belongs here / what doesn't:** Candidate preparation, decision confidence and Jev transport belong here. An LLM handoff is a selector composition policy.

**Contents:**

- `__init__.py` — model and selector exports.
- `base.py` — `DecisionModel` contract and `DecisionSelector`.
- `images.py` — prepare Jev input images and retain the visible bounds.
- `typesafe.py` — TypeSafe Jev requests and response decoding.

**Examples:** `DecisionSelector(TypeSafeDecisionModel(), vision="on", exclude={"press_button"}, min_confidence=0.7)` selects with Jev without offering system buttons. It maps atomic registry tools to options, then returns a call such as `tap_element(index=3)`. With `vision="on"` the screenshot is context for these tree choices. Only `vision="only"` offers the located tap choice. All vision modes offer fixed scroll directions when supported: `scroll down` reveals content below by moving the finger up. If no choice can continue the subgoal, Jev can choose Unfinished. A pick below the confidence cutoff also ends unfinished. Neither case retries selection or calls a delegate. In a planning run, the planner gets the subgoal back and can re-plan.

If no config is passed, the selector reads `agent.decision` from `agents.yaml` or the active `ConfigLoader.override(...)` when selecting. Explicit `timeout` and `min_confidence` arguments take precedence. Use an LLM selector or wrap Jev in `DelegatingSelector` for typed output.

**Depends on / used by:** Uses selector contracts, atomic registry choices, config and key resolution. The selector loader and direct Python callers construct decision selectors.
