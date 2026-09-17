# Working on Jev Mobile Agent

- Keep this repository standalone. Do not add imports or paths to the original Droidrun monorepo.
- Device IDs and credentials come from environment variables. Never commit `.env` files, raw device traces, screenshots with personal data, or API keys.
- Jev selects operations and targets; code owns candidate discovery, validation, coordinate resolution, and execution. Do not insert task-specific tap scripts into the agent policy.
- Read current TypeSafe API/primitive docs before changing inference contracts: https://docs.typesafe.ai/llms.txt.
- Read current Mobilerun endpoint contracts before changing device operations: https://docs.mobilerun.ai/llms.txt.
- Keep navigation actions sequential. Retry only stale decisions rejected before input; never replay an uncertain mutation automatically.
- Report model-declared completion separately from independently verified outcomes. Record unsuccessful benchmark attempts as well as successful ones.
- Run `pnpm check` before publishing changes. CI must remain offline; live benchmarks need explicit device configuration.
