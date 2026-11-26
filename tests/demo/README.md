# Run a device demo

**These are live examples.** They operate the selected phone, can change its screen, and can send screen contents to model providers. Cloud device time and model tokens may cost money. Use a test phone without private data. Importing the script does not run it.

1. Install the agents package and base device dependencies. For cloud, configure Mobilerun credentials. For models, use provider environment variables or `mobilerun agent providers add PROVIDER`; never paste keys into the script.
2. Open the **edit me** block in [`demo.py`](demo.py). Set `DEVICE` to an existing cloud ID (`CLOUD = True`), or an Android serial or visual-remote URL (`CLOUD = False`). Set `RUN` to the names you want, e.g. `["basic"]` for one run. `["all"]` runs every registered demo, including comparisons; it is not a smoke test.
3. From the repo root, run `python tests/demo/demo.py` in the installed environment.

## Selector and observation matrix

| Selector | Tree (`off`) | Tree + screenshot (`on`) | Screenshot only (`only`) |
| --- | --- | --- | --- |
| Jev | `jev_no_vision` | `jev_vision`, `jev_only` (selector default) | `jev_vision_only` |
| LLM | `no_vision` | `vision` | `vision_only` |
| Jev + LLM | `JEV_VISION`/`DELEGATE_VISION` in `jev_delegate` and `jev_executor` | (same) | (same) |

`jev_exclude` demonstrates hiding a tool from Jev: it uses `DecisionSelector(model, exclude={"press_button"})`, so Jev cannot choose a system button.

- `jev_only` means *no LLM fallback*, not screenshot-only. Jev uses `TYPESAFE_API_KEY` or the key saved by `mobilerun agent providers add typesafe`. In `jev_vision_only`, Jev locates the tap itself and returns `tap(x, y)`, not `tap_element(index)`.
- `no_vision`, `vision`, `vision_only`, and the delegate LLM use `VISION_MODEL`. `VISION` controls the LLM selector in examples that do not specify a mode explicitly.
- `jev_delegate` gives each child its own observation mode: `JEV_VISION` for Jev (default `"only"`: screenshot only, Jev locates the tap point) and `DELEGATE_VISION` for the LLM when it takes over (default `"on"`). `JEV_MIN_CONFIDENCE` sets the delegation threshold for a low-confidence Jev choice.
- The LLM image adapter automatically draws a ten-division grid on screenshots for pixel-coordinate models and omits it for models using normalized coordinates. There is no public grid toggle and no separate `llm_grid` demo.

## Planning, comparisons, and other demos

| `RUN` name | What it does |
| --- | --- |
| `easy_jev_executor` | Runs the short `EASY_GOAL` with the LLM planner and Jev-first, LLM-fallback executor. Configure its vision, confidence, and step limit in the easy demo section. |
| `easy_llm_executor` | Runs the same `EASY_GOAL` with the LLM planner and an LLM executor using vision. |
| `tiktok` | Opens TikTok on a separate device from `TIKTOK_DEVICE`, then reports the creator and caption of the first For You video. It closes that connection after the run. |
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
| `credentials` | Binds a vault to the device and asks the agent to type a saved credential into a test app. |
| `phone_number` | Temporarily binds an existing cloud number to the device and asks the agent to read its inbound SMS. |

To run any one of the new examples, set `RUN` to `['easy_jev_executor']`, `['easy_llm_executor']`, or `['tiktok']` in the edit-me block, then use the standard command above. `tiktok` needs `TIKTOK_DEVICE` set to a second device and uses `TIKTOK_CLOUD` to choose cloud versus local/remote connection handling; it never resets or presses Home on the main device.

Both planning demos wrap the configured LLM in `LlamaIndexPlanner`. It sees text only and writes subgoals with `set_plan`; Jev can execute those subgoals.

`DEBUG = False` prints the normal one-line progress for each step. Set `DEBUG = True` for the CLI's `--debug` details, including the screen, selector, image use, model calls, and results.

`MAX_STEPS` and `TIMEOUT` apply per operator run, including each comparison row. Set `TIMEOUT = None` for no whole-run deadline. Demos share one connection. With `RESET = True`, the runner presses Home and waits briefly before each demo and each comparison run. This returns to the launcher but does not reset the apps or their data. Set `RESET = False` to keep the current screen or use a device that does not support Home.

## Device extension demos

`credentials` and `phone_number` bind extensions on the device, like [`demo_extension.py`](demo_extension.py). They are not custom run tools or comparison setups. They use the default LLM provider from `MODELS`: these actions take arguments that Jev's fixed-choice tool list cannot represent. Each demo unbinds its extension even if the operator fails.

- **`credentials`:** Binds local fake or existing cloud credentials and asks the agent to paste only the saved password into Chrome's search bar with `vault.type_field`; it must not press Enter. The local entry uses `CREDENTIAL_PACKAGE` as its vault lookup key and contains only `password`.
- Base's `vault.type_field` requires a freshly reported focused, editable field. Tap Chrome's home-screen “Search or type URL” box once to focus the real address bar before calling it.
- The demo reports action evidence, whether `vault.type_field` was called, and whether the known local password appears in run artifacts. It never prints the value.
- **`phone_number`:** Set `PHONE_NUMBER_ID` to an existing, active, unbound cloud number ID and use a cloud test phone with telephony support and no existing number binding. With no ID, the demo skips without opening a device if it is the only selected demo. `PHONE_NUMBER_GOAL` defaults to checking the latest inbound SMS without asking the agent to repeat the code. `sms.send` is excluded. The demo neither buys a number nor sends a message. Use a number with **synthetic SMS only**: the SMS API returns message text to the agent, and tool results can appear in debug output. Optionally set `PHONE_NUMBER_TEST_CODE` to a known synthetic value for the artifact check. Cloud device time and model tokens can still cost money for both demos.

The report lists recorded action names in the last numbered checkpoint for this run. If the checkpoint only names an `extension_call` primitive, the report uses this run's exported macro to identify the successful replayable extension call. `credential tool called: yes` means it found `vault.type_field`, not just `vault.list`. `phone number tool called: yes` means it found `sms.messages` or `sms.wait_for_code`. A recorded action does not prove the app logged in or a code was found; `no` means the target call was not found in the available evidence.

The secret check searches file bytes throughout that run directory and in its separately exported macro. `secret visible in run folder: no` means it found none of the known fake values in those bytes. It is not an OCR or encoding check. The demo won't print matched values. It reports `not checked` when there's no known test value (including cloud credentials) or recording evidence is incomplete and no match was found. `DEBUG = True` also shows step-by-step console output.
