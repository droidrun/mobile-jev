# Choice models

**Purpose:** Ask a decision model to pick from available actions rather than generate free-form calls.

**What belongs here / what doesn't:** Candidate preparation, decision confidence and Jev transport belong here. An LLM handoff is a selector composition policy.

**Contents:**

- `__init__.py` — model and selector exports.
- `base.py` — `DecisionModel` contract and `DecisionSelector`.
- `images.py` — prepare Jev input images and retain the visible bounds.
- `loader.py` — construct a model from a decision profile.
- `typesafe.py` — TypeSafe Jev requests and response decoding.
- `candidates/` — preserve accessibility metadata for atomic tool expansion.

**Examples:** `DecisionSelector(TypeSafeDecisionModel(), exclude={"press_button"})` selects with Jev without offering system buttons. It maps atomic registry tools to options, then returns a call such as `tap_element(index=3)`. With `vision="on"` the screenshot is context for these tree choices. Only `vision="only"` asks for a box to produce a `tap(x, y)` call.

**Depends on / used by:** Uses selector contracts, atomic registry choices, config and key resolution. The selector loader and direct Python callers construct decision selectors.
