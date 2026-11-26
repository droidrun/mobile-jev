# Run checkpoints and macro export

**Purpose:** Write inspectable per-step evidence and a macro the base player can replay.

**What belongs here / what doesn't:** Store observations before publishing their checkpoint. Record executed primitives, not model intent. A checkpoint is diagnostic state, not a resumable workflow.

**Contents:**

- `__init__.py` — package entry.
- `store.py` — `RunRecorder`, numbered checkpoints and observation files.
- `macro_export.py` — export successful primitives with pre-action UI state and vault references.

**Examples:** `RunRecorder(run_folder, macro_path=macro_file)` writes checkpoints beneath the run's `<run_id>/` directory and references screenshots saved separately as `<images>/<run_id>/NNNN.png`. `await recorder.finish(state)` exports the macro. The roots are `MOBILERUN_AGENT_RUNS_DIR` and `MOBILERUN_AGENT_IMAGES_DIR`, defaulting to `runs` and `images` under the base config directory; empty overrides use those defaults. Typed text is not secret by default; vault values and declared-sensitive parameters are masked in outputs, not in model inputs.

Each action's `ActionRecord.before` points to its `StepSnapshot`, the sole owner of that observation's UI state and screenshot reference. The snapshot includes its timestamp, UI state, optional `ImageBlock(path=...)`, device context and one fingerprint. Image bytes are retained only for the current turn. Macro export reads the final records directly; it drops actions whose identifiers or arguments would need masking and omits guards that would need masking. Numbered checkpoints retain structural metadata (including `screen_changed`), not goals, thoughts, summaries, notes or arguments. Plain text entry is replayable; credential entry records the vault reference, not the expanded value. The operator returns no trajectory path when a workflow records no steps. Run-only tool calls and device calls with sensitive arguments are omitted from macros. Exported device extension calls keep their qualified names.

**Depends on / used by:** The operator feeds the recorder snapshots and core state. Base macro replay consumes the exported file. Protect the folder: replay literals, UI trees and screenshots may contain private data.
