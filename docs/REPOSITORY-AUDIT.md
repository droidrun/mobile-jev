# Repository audit and follow-up plan

Reviewed on 2026-09-17 against the initial `main` commit, the repository settings, and the public pages linked below. This is a focused onboarding and discoverability audit, not a complete audit of every page or the agent runtime.

## Product relationship

- **Droidrun** is the company and GitHub organization. [droidrun.ai](https://droidrun.ai) presents mobilerun Cloud and the framework as its products.
- **mobilerun Cloud** provides device infrastructure, APIs, and streaming. [mobilerun.ai](https://mobilerun.ai) is the product site; [cloud.mobilerun.ai](https://cloud.mobilerun.ai) is the dashboard; [docs.mobilerun.ai](https://docs.mobilerun.ai/) is the documentation.
- **mobilerun Framework** is the general-purpose Python agent. The former `droidrun/droidrun` repository redirects to [droidrun/mobilerun](https://github.com/droidrun/mobilerun).
- **Mobile Jev** is this standalone Node.js integration: its own local agent loop, TypeSafe inference, mobilerun device operations, and a React studio using `@mobilerun/react`. It does not depend on the Python framework or Mobile Harness.
- **TypeSafe** supplies Jev, the decision model. It is a separate service. This integration needs both provider keys because it calls Jev directly rather than using the hosted mobilerun agent.

The code already includes a CLI, studio, live streaming, execution traces, latency measurements, offline tests, CI, an MIT license, environment examples, a recorded Uber walkthrough, and a dark-theme demo with independent outcome verification. The onboarding should make those assets and the relationships above easy to find.

## Plan and status

| Priority | Finding                                                                           | Action                                                                                  | Delivery                                             |
| -------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| P1       | GitHub About has no description, website, or topics                               | Apply the exact settings below                                                          | Maintainer action; cannot be changed by merging a PR |
| P1       | README does not explain Droidrun, the renamed framework, or how this repo differs | Add the product map, canonical repository links, Android scope, and two-key explanation | Included in this PR                                  |
| P1       | The initial README leads with a large GIF before explaining the project           | Put the project summary and quick links before the existing demo                        | Included in this PR                                  |
| P1       | Studio title and navigation are generic, with no project/help links               | Add Mobile Jev branding and working Setup, About, and GitHub links; name both providers | Included in this PR                                  |
| P2       | No contribution guide or PR template                                              | Document offline validation, bug-report details, and sanitized evidence                 | Included in this PR                                  |
| P2       | Demo guide has no direct route back to setup                                      | Add relative README and setup links                                                     | Included in this PR                                  |
| P2       | No Mobile Jev reference on the checked product pages or documentation index       | Add a discoverable integration entry and reciprocal links                               | Website/docs follow-up described below               |
| P3       | No release/tag is shown on the initial repository                                 | Publish a first release after the initial review, with setup and demo links             | Maintainer follow-up; no release created here        |

The PR changes documentation and studio presentation only. It does not change model contracts, device operations, or the agent loop.

## GitHub About: ready to apply

GitHub stores About in repository settings, outside git. The account used for this audit has read access, without push or admin permission on `droidrun/mobile-jev`. A maintainer with permission to edit repository metadata needs to apply this after reviewing the wording.

**Description**

> Android agent by Droidrun: TypeSafe's Jev makes typed decisions, mobilerun executes them. Live React studio, CLI, and traces.

**Website**: <https://mobilerun.ai/integrations/>

Use the existing integrations page until a dedicated Mobile Jev page exists. Update this field to the dedicated page when it is published; do not point users to an uncreated URL or the localhost studio.

**Topics**: `android`, `android-automation`, `mobile-automation`, `ai-agents`, `mobilerun`, `droidrun`, `jev`, `typesafe`, `nextjs`, `typescript`

Run with an authorized maintainer account:

```sh
gh repo edit droidrun/mobile-jev \
  --description "Android agent by Droidrun: TypeSafe's Jev makes typed decisions, mobilerun executes them. Live React studio, CLI, and traces." \
  --homepage "https://mobilerun.ai/integrations/" \
  --add-topic android,android-automation,mobile-automation,ai-agents,mobilerun,droidrun,jev,typesafe,nextjs,typescript

gh repo view droidrun/mobile-jev \
  --json description,homepageUrl,repositoryTopics
```

## Website and documentation follow-up

The fetched HTML/text of [droidrun.ai](https://droidrun.ai), the [mobilerun homepage](https://mobilerun.ai), [framework page](https://mobilerun.ai/framework/), [integrations page](https://mobilerun.ai/integrations/), [website llms.txt](https://mobilerun.ai/llms.txt), and [documentation index](https://docs.mobilerun.ai/llms.txt) contained no `mobile-jev`, `Jev`, or `TypeSafe` references on the review date. This checks those entry points, not every page or client-rendered state.

1. **Product website:** add a Mobile Jev card on the integrations page, linking directly to [this repo](https://github.com/droidrun/mobile-jev) and its [setup section](https://github.com/droidrun/mobile-jev#run-it). Add the same entry to the website's `llms.txt` resource list. Suggested copy: “Run a Jev-driven Android agent on mobilerun. Includes a live studio, CLI, and execution traces. Bring mobilerun and TypeSafe API keys.”
2. **Documentation:** add a runnable integration entry next to the examples repository. Explain that Mobile Jev calls the device API and TypeSafe directly. The [general quickstart](https://docs.mobilerun.ai/quickstart) says the hosted Cloud agent needs only a mobilerun key; link to Mobile Jev's two-key setup so users do not apply that hosted-agent instruction here.
3. **Company website:** retain Droidrun as the company and mobilerun as the product. Add a small open-source/demo link for Mobile Jev when the integration entry is ready; it does not need to become a third top-level product.
4. **Related repositories:** add a Mobile Jev link to the framework's examples/resources area and the examples repository. Describe Mobile Harness as tools for an existing coding agent, and Mobile Jev as a complete Jev-driven agent.

Acceptance: users can reach the repo from the integrations page and docs, the repo links back to both providers, and any new landing page is live before its URL is used in About. Preserve the Android-only scope, separate provider usage, and the recorded demo's actual outcome: it reaches payment selection, without demonstrating a completed booking. Implement these changes in the repositories that own those surfaces.

## Link verification

Checked with HTTPS GET requests following redirects. Local README/media paths were also checked in the checkout. HTTP reachability does not verify behavior after login.

| Target                                                                                 | Result on review date                                         | Interpretation                                                                                                  |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| Droidrun, mobilerun, TypeSafe sites; framework, integrations, and Mobile Harness pages | HTTP 200                                                      | Reachable public pages                                                                                          |
| mobilerun and TypeSafe documentation roots                                             | HTTP 200 after redirects to `/quickstart` and `/introduction` | Valid entry points                                                                                              |
| mobilerun API-key page; TypeSafe console                                               | HTTP 200 after redirects to sign-in/login                     | Expected authentication, not broken links                                                                       |
| Mobile Jev repo, issues, and demo-video page; framework, harness, and examples repos   | HTTP 200                                                      | Canonical repository links are reachable                                                                        |
| mobilerun status page and both checked `llms.txt` indexes                              | HTTP 200                                                      | Reachable                                                                                                       |
| npm package web page for `@mobilerun/react`                                            | HTTP 403 to automated retrieval                               | Inconclusive web-page check; the npm registry confirms version `0.1.0` and the frozen-lockfile install succeeds |
| `https://api.mobilerun.ai/v1`                                                          | HTTP 404 on the bare base path                                | API prefix, not a documentation page or broken navigation link; do not change the configured endpoint           |

The main navigation gap is missing cross-links and context. The checked public website/docs links are reachable; authenticated setup was not exercised and no live device task was run for this audit.

## Validation of this PR

`pnpm check` passes with Node.js 24 and pnpm 10.30.1: 55 offline tests, ESLint, TypeScript, Prettier, and the production build. All 14 relative Markdown file/section links resolve. The production studio was visually checked at 1280 × 720 and 390 × 844 without provider credentials; the new resource links are visible at both sizes, and the mobile page has no horizontal overflow. Live phone control and authenticated dashboard flows were not exercised.
