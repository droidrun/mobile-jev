# Contributing to Mobile Jev

Mobile Jev is a standalone project by [Droidrun](https://droidrun.ai), using [mobilerun](https://mobilerun.ai) for device control and [TypeSafe's Jev](https://docs.typesafe.ai/) for decisions. Read the [project overview](README.md) and [AGENTS.md](AGENTS.md) before changing the agent loop.

## Local development

Use Node.js 24 and the pinned pnpm version from `package.json`:

```sh
corepack enable
pnpm install --frozen-lockfile
pnpm check
```

`pnpm check` runs the offline tests, lint, typecheck, formatting check, and production build. No API keys or device are needed. For the studio with a live phone, follow the [setup guide](README.md#run-it). Keep credentials in `.env.local`, which is ignored by git.

## Pull requests

Create a branch, keep the change focused, and explain the resulting behavior and validation. Run `pnpm check` before opening your PR. For a visual change, check desktop and mobile layouts and include a screenshot without device or account data.

Keep the repository independent of the Python framework and other Droidrun checkouts. Read the current provider docs before changing API contracts. Live demos and `--execute` commands control the configured phone and incur provider usage; the offline checks never need them.

## Bugs and questions

Use [GitHub issues](https://github.com/droidrun/mobile-jev/issues) for reproducible problems with this repository. Include:

- The commit, OS, Node.js and pnpm versions, and command or studio action.
- Expected and actual behavior, plus minimal reproduction steps.
- For device-specific failures: Android version, app version, locale, and the returned model name if available.
- Sanitized error output. Remove API keys, stream tokens, device identifiers, private task text, and personal screen content before sharing.

Do not attach raw traces, `.env` files, or screenshots containing personal data. Distinguish Jev's declared completion from an independently verified outcome.

For platform setup, use the [mobilerun docs](https://docs.mobilerun.ai/); for service availability, check [mobilerun status](https://status.mobilerun.ai). For model questions, use the [TypeSafe docs](https://docs.typesafe.ai/).
