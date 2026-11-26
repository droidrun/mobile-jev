# Run a device demo

**These are live examples.** They operate the selected phone, can change its screen, and can send screen contents to model providers. Cloud device time and model tokens may cost money. Use a test phone without private data. Importing the script does not run it.

1. Install the agents package and base device dependencies. For cloud, configure Mobilerun credentials. For models, use provider environment variables or `mobilerun agent setup`; never paste keys into the script.
2. Open the **edit me** block in [`demo.py`](demo.py). Set `DEVICE` to an existing cloud ID (`CLOUD = True`), or an Android serial or visual-remote URL (`CLOUD = False`). Set `RUN` to the names you want, e.g. `["basic"]` for one run. `["all"]` runs every registered demo, including comparisons; it is not a smoke test.
3. From the repo root, run `python tests/demo/demo.py` in the installed environment.

## Selector and observation matrix

| Selector | Tree (`off`) | Tree + screenshot (`on`) | Screenshot only (`only`) |
| --- | --- | --- | --- |
| Jev | `jev_no_vision` | `jev_vision`, `jev_only` (selector default) | `jev_vision_only` |
| LLM | `no_vision` | `vision` | `vision_only` |
| Jev + LLM | Jev primary in `jev_delegate` and `jev_executor` | LLM delegate in those same setups | Not used by the delegation demos |

`jev_exclude` demonstrates hiding a tool from Jev: it uses `DecisionSelector(model, exclude={"press_button"})`, so Jev cannot choose a system button.

- `jev_only` means *no LLM fallback*, not screenshot-only. Jev uses `TYPESAFE_API_KEY` or the key saved by `mobilerun agent setup`. In `jev_vision_only`, Jev locates the tap itself and returns `tap(x, y)`, not `tap_element(index)`.
- `no_vision`, `vision`, `vision_only`, and the delegate LLM use `VISION_MODEL`. `VISION` controls the LLM selector in examples that do not specify a mode explicitly.
- `jev_delegate` gives each child its own observation mode: Jev sees only the tree; the LLM sees the tree and screenshot when it takes over. `JEV_MIN_CONFIDENCE` sets the delegation threshold for a low-confidence Jev choice.
- The LLM image adapter automatically draws a ten-division grid on screenshots for pixel-coordinate models and omits it for models using normalized coordinates. There is no public grid toggle and no separate `llm_grid` demo.

## Planning, comparisons, and other demos

| `RUN` name | What it does |
| --- | --- |
| `basic` | Runs `GOAL` with `AgentOperator` and the `DEFAULT` provider from `MODELS`. |
| `compare_models` | Runs `GOAL` once per provider in `COMPARE`; reports model, success, steps, elapsed seconds, and answer. Keyed providers without an environment or saved key show `skipped (no key)`. |
| `compare_setups` | Runs `GOAL` once per name in `SETUPS` (from `SETUP_DEMOS`, or `["all"]` for all setups); reports setup, success, steps, seconds, tokens, and answer. The public run result does not expose token usage, so the column shows `-` rather than an estimate. Failed setups stay in the table without stopping the comparison. |
| `planning` | Uses separate `PLANNER` and `EXECUTOR` provider/model pairs. |
| `jev_exclude` | Runs Jev with `press_button` excluded from its choices. |
| `jev_executor` | Uses `PLANNER` as the LLM planner and the `jev_delegate` setup as the executor. |
| `structured` | Returns battery `level` (0–100) and `charging` as validated JSON. Use a battery-related `GOAL`. |
| `custom_tool` | Gives this run a `note` tool and asks it to save an observation in run memory. |
| `macro` | `MACRO = None` picks the newest `*/macro.json` in the Mobilerun macro directory (`MOBILERUN_MACRO_DIRECTORY` overrides it). If there is no recording, the demo skips. To replay a specific macro, set `MACRO` to its folder or file. Set `RESET = False` and start on the recorded screen for replay without handoff. |
| `device` | Reads screenshot size, UI element count and app count. No model calls; cloud device time still applies. |

Both planning demos wrap the configured LLM in `LlamaIndexPlanner`. It sees text only and writes subgoals with `set_plan`; Jev can execute those subgoals.

`DEBUG = False` prints the normal one-line progress for each step. Set `DEBUG = True` for the CLI's `--debug` details, including the screen, selector, image use, model calls, and results.

`MAX_STEPS` and `TIMEOUT` apply per operator run, including each comparison row. Set `TIMEOUT = None` for no whole-run deadline. Demos share one connection. With `RESET = True`, the runner presses Home and waits briefly before each demo and each comparison run. This returns to the launcher but does not reset the apps or their data. Set `RESET = False` to keep the current screen or use a device that does not support Home.

[`demo_extension.py`](demo_extension.py) remains separate: it shows how to bind and unbind a device extension, rather than register a tool on one agent run.
