# Run analytics and tracing

**Purpose:** Record allowlisted run metrics and optional LLM traces without making them part of the action loop.

**What belongs here / what doesn't:** Collect observations and final counts here. Device commands and local diagnostic artifacts belong elsewhere.

**Contents:**

- `__init__.py` — package entry.
- `consumer.py` — `RunAnalytics` and product events.
- `tracing.py` — run-scoped OpenInference tracing and exporters.

**Examples:** `analytics = RunAnalytics(config.analytics, run_id)`; `analytics.on_observe(package, activity, step)` tracks package and activity changes when enabled. End the run with `await analytics.finish(success, steps)`.

**Depends on / used by:** Uses agent and base telemetry settings. The operator forwards package, activity and step data and closes telemetry at the end of a run.
