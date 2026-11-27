# Agent-assisted macro replay

**Purpose:** Replay a macro using the base `MacroPlayer`. On a mismatch, an agent takes
over on the same recording device to reach only that step's recorded after
screen. Replay checks the screen independently, then resumes at the next
original step. Replay and the repair agent share any supplied step limit and
deadline; a failed action that may have dispatched also uses a step.

**What belongs here / what doesn't:** This add-on owns recovery and splicing;
base owns deterministic playback, the run format, and version publication.

**Contents:**

| File | Purpose |
| --- | --- |
| `__init__.py` | Public `MacroAgentOperator` export |
| `operator.py` | Replay, guarded repair, and publication |
| `paths.py` | Extract the publication name and version from the selected file |

**Examples:**

```python
result = await MacroAgentOperator().execute(device, "inbox", max_steps=30, timeout=90)
```

`mobilerun macro-agent replay inbox DEVICE` selects the latest numeric version.
Use a folder for its latest version or `./macros/inbox/v3.jsonl` to pin a file.
Use `--dry-run` to inspect the selected actions without a device or model.
Live repair can consume LLM tokens; known SMS actions require confirmation
before dispatch in the CLI. Agent repair and output extraction cannot dispatch
SMS sends or extension actions marked for billing retry, even through nested
extension actions. Assisted replay checks all selected rows for credential and
argument safety before running any action, keeps credential masks across rows,
and refuses agent handoff after potentially sensitive input.

After a full successful replay with a safe repair, the builder publishes a new
version under the configured macro directory. It keeps unaffected source steps,
replaces the failed step with successful repair actions, and never overwrites a
version. Publication requires guarded repair rows and, for post-action recovery,
evidence that the replacement starts from the original precondition. Otherwise
replay can recover without publishing. A partial replay never publishes.
Actions-only macros replay without screen guards, but repair is refused without
a recorded after fingerprint. Repair refuses if recording is disabled by agent
config, `MOBILERUN_AGENT_RECORDING_ENABLED`, `record=False`, or stealth. It also
refuses if commerce may already have dispatched.

An operator-owned replay recording uses full capture (including unredacted
screenshots) if any selected step has an after fingerprint, actions-only capture
otherwise. A caller-owned recorder is reused at its original capture level and
is not finalized. The replay recording retains the original attempt and, on
failure, the resolved source and original failed step index.

Set `macro_file` at construction to pass a separate goal to `execute` for
optional structured output extraction after replay. That goal is not used for
step repair.

**Depends on / used by:** Python callers and the CLI use this operator. See
[agent operators](../agent/operators/README.md) for execution and recording policy.
