# Run checkpoints and macro export

**Purpose:** Write inspectable per-step evidence and a macro the base player can replay.

**What belongs here / what doesn't:** Store observations before publishing their checkpoint. Record executed primitives, not model intent. A checkpoint is diagnostic state, not a resumable workflow.

**Contents:**

- `__init__.py` — package entry.
- `store.py` — `RunRecorder`, numbered checkpoints and observation files.
- `macro_export.py` — export successful primitives with pre-action UI state and vault references.

**Examples:** `RunRecorder(run_folder, macro_path=macro_file)` writes checkpoints, screenshots and UI states to the run folder. After streaming observations and steps, `await recorder.finish(state)` exports `macro.json` to the separate macro path. The operator defaults to `<Mobilerun config dir>/runs/<run_id>/` and `<Mobilerun macro dir>/<run_id>/macro.json`. An indexed type operation may record a tap followed by text input.

Each primitive keeps its own screen guard: the tap uses the screen before focus, and text entry uses the focused screen. Numbered checkpoints retain step and action metadata, not goals, summaries, or argument values. Free-form run notes survive workflow context serialization, but numbered diagnostic checkpoints omit them because notes can quote typed secrets. Typed text is marked `[REDACTED]`, even when entry fails. The operator returns no trajectory path when a workflow records no steps. Run-only tool calls are not replayable device operations and are omitted from macros. Exported device extension calls keep their qualified names and redact sensitive arguments; the player refuses to replay a redacted argument.

**Depends on / used by:** The operator feeds the recorder snapshots and core state. Base macro replay consumes the exported file. Protect the folder: replay literals, UI trees and screenshots may contain private data.
