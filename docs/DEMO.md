# Choosing a useful speed demo

The first supported demo is **“Turn on dark theme in Android Settings.”** It has a visible before/after result, requires no account or third-party content load, and can be checked against an actual accessibility switch state. It is a small, reproducible utility task—not a demonstration of arbitrary travel booking.

## Run and verify

Use an English-language Android device with a Settings app exposing a labeled Dark theme switch. Configure your keys and device as described in the main README.

```sh
# One goal from the current state:
pnpm demo dark-theme

# Ask Jev to establish an off baseline, then measure enabling it:
pnpm demo dark-theme --reset

# Repeat, preserving every successful and failed attempt:
pnpm demo dark-theme --reset --repeat 3
```

The reset is also a Jev goal. It has no prepared tap sequence and must independently verify the switch is off. Without `--reset`, a verified final state does not prove a setting changed; the report marks `stateChangeVerified: false`. Repeats require a verified reset so no-op completions cannot be presented as speed wins.

The timer starts before readiness, UI observation, and installed-app discovery. It includes model calls, state reads, device actions, transition waits, stale retries, and trace writes, ending at the accepted model outcome. A separate fresh observation verifies the switch. Reports contain execution time, verification time, and their combined total. Reset time is reported separately; after reset, the model connection may already be warm and Settings may already be on the relevant screen. Do not compare that timing with a cold start from another app without stating the difference.

All requests use the Mobilerun API, with Jev at TypeSafe selecting the operations and targets. There is no ADB shortcut. `OPEN_APP` uses the installed-app registry and the public app-start endpoint; it is recorded as a selected action, not hidden setup.

## Development observations

One exploratory dark-theme run took **8.685 seconds**, including three executed actions, five Jev calls, and one stale decision rejected before execution. The recorded final Settings observation contained `Dark theme`, `checkable: true`, and `checked: true`. That trial predates the packaged demo runner's additional fresh verification request and does not establish a repeatability or speed guarantee. No multi-run success rate is claimed.

Earlier exploration showed why independent verification is necessary:

- A five-minute timer task was reported DONE after the agent entered five **seconds**. It was not a verified success.
- A world-clock task stalled at a search field. It was not a verified success.
- One inference attempt failed at the transport layer before any device action.

Clock scenarios are not included as demos. Raw exploratory traces remain local and are excluded from the repository because they contain device data. The application and offline tests can be reproduced without those traces.

## What to show

For a live walkthrough, display the phone alongside the actual executed-action timeline and task clock. Show the returned model name and the verified result. Quote full task time as well as model latency; do not omit API waits or speed up playback.

For a more substantial follow-up, destination + date selection in a travel app is a candidate, but app loading, login state, UI changes, and prices dominate reproducibility. It needs its own independent verifier and repeated successful measurements before advertising a speed figure. No purchase or reservation is part of the included demo.
