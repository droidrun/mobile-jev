# Run a device demo

These examples can change the phone's screen and send screen contents to model providers. Use a test device with no private data; cloud time and model calls may cost money. For offline verification, run `python -m pytest tests/demo` instead.

1. Install the agents package and base device dependencies. Configure provider credentials with environment variables or `mobilerun agent providers add PROVIDER`. Jev uses the `typesafe` provider.
2. In [`demo.py`](demo.py), set `DEVICE` to an existing cloud ID (`CLOUD = True`), or an Android serial or remote URL (`CLOUD = False`). Choose one or more names in `RUN`. `['all']` includes the comparisons and replay; it is not a smoke test.
3. Run `python tests/demo/demo.py` from the repo root. `DEBUG` shows CLI per-step detail. `MAX_STEPS` and `TIMEOUT` apply to each operator run. `RESET` stops the foreground app when supported and presses Home before each run; disable it to keep the current screen.

| `RUN` name | Example |
| --- | --- |
| `llm` | Use a plain LLM with the UI tree, without screenshots. |
| `llm_vision` | Run the same goal with tree + screenshot, then screenshot only. |
| `jev` | Run Jev alone; choices below `JEV_MIN_CONFIDENCE` abstain. |
| `jev_exclude` | Remove `press_button` from Jev's choices. |
| `delegation` | Hand a step to the LLM only when Jev explicitly chooses `delegate`. |
| `planner_llm` | Use an LLM planner with a separate LLM executor. |
| `planner_jev` | Use Jev alone to execute subgoals. A failed subgoal returns to the planner. |
| `planner_delegation` | Use the delegating executor on the harder `GOAL`. |
| `vault_login` | Bind the local fake or an existing cloud vault and type a password reference into Chrome. |
| `phone_number` | Temporarily bind an existing, unbound cloud number to a telephony-enabled cloud phone and read inbound SMS. Set `PHONE_NUMBER_ID`; the demo disables `sms.send`. |
| `mailbox` | Bind an existing cloud mailbox (`MAILBOX_ID`) and wait for a verification email. Send the fixture during the wait; the base API searches for new mail by default. |
| `structured` | Validate battery level and charging status with a Pydantic model. |
| `custom_tool` | Register `note`, whose workflow context is injected by type. |
| `macro` | Replay the automatic macro export for the latest checkpoint using base `MacroPlayer`. |
| `inspect_run` | Read the newest `context.json` and print action targets and whether each action succeeded or changed the screen. |
| `compare_models` | Run `COMPARE_GOAL` once per provider in `COMPARE`. |
| `compare_setups` | Run `COMPARE_GOAL` with LLM, Jev, delegation, and planner + delegation. |

The vault example requires the backend to report an editable focused field after Chrome's address bar is tapped. If the phone reports the keyboard instead, base can reject the action. Use fake credentials and **synthetic SMS/email only**. The SMS and mailbox demos report only whether a code was found and the step count, even with `DEBUG` enabled. Extension setup does not buy resources. See [`demo_extension.py`](demo_extension.py) for standalone extension wiring.

The demo follows your checkpoint settings. To generate data for `macro` and `inspect_run`, enable checkpoints in your config and run an agent example first. Checkpoints use `MOBILERUN_AGENT_RUNS_DIR` (default: `runs` under the Mobilerun user config directory). Macro exports use `MOBILERUN_MACRO_DIRECTORY`, keyed by the same run ID. Replay requires the recording's starting screen; set `RESET = False` and prepare that screen yourself if Home is not the right starting point. Macros don't recreate temporary extension bindings; use a prior plain device-action run for replay.
