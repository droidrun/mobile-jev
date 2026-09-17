import { parseArgs } from 'node:util';
import { mkdir, open, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { loadEnvironment } from './env.mjs';
import { MobilerunDevice } from './mobile-agent/device.mjs';
import { TypeSafePolicy, runAgent } from './mobile-agent/agent.mjs';
import { pooledRequest, decodeJson } from './mobile-agent/http.mjs';
import { measuredRequest, summarizeMetrics } from './mobile-agent/metrics.mjs';
import { darkThemeState } from './demo-verifiers.mjs';

loadEnvironment();
const { values, positionals } = parseArgs({
  allowPositionals: true,
  options: {
    reset: { type: 'boolean', default: false },
    repeat: { type: 'string', default: '1' },
    out: { type: 'string', default: 'artifacts/demos' },
    help: { type: 'boolean' },
  },
});

async function main() {
  if (values.help || !positionals.length) {
    console.log(
      'Usage: pnpm demo dark-theme [--reset] [--repeat 3] [--out artifacts/demos]\n--reset asks Jev to disable dark theme before each measured run, then verifies it is off.',
    );
    return;
  }
  if (positionals.length !== 1 || positionals[0] !== 'dark-theme')
    throw new Error('Available demo: dark-theme.');
  const count = Number(values.repeat);
  if (!Number.isInteger(count) || count < 1 || count > 10)
    throw new Error('--repeat must be between 1 and 10.');
  if (count > 1 && !values.reset)
    throw new Error(
      'Repeated runs require --reset so already-completed tasks are not timed as speed wins.',
    );
  const directory = resolve(values.out);
  await mkdir(directory, { recursive: true, mode: 0o700 });
  const reports = [];
  for (let attempt = 1; attempt <= count; attempt++) {
    const id = `${Date.now()}-${randomUUID().slice(0, 8)}`;
    const file = await open(resolve(directory, `${id}.jsonl`), 'wx', 0o600);
    const record = async (event) => {
      await file.write(JSON.stringify({ at: new Date().toISOString(), ...event }) + '\n');
    };
    const metrics = [],
      decisions = [];
    const device = new MobilerunDevice({
      request: measuredRequest({
        service: 'mobilerun',
        metrics,
        onMetric: (metric) => record({ event: 'request_timing', ...metric }),
      }),
    });
    const modelRequest = measuredRequest({
      service: 'typesafe',
      metrics,
      request: pooledRequest,
      onMetric: (metric) => record({ event: 'request_timing', ...metric }),
    });
    const policy = new TypeSafePolicy({
      request: async (input) => {
        await record({ event: 'model_request', body: input.body });
        const bytes = await modelRequest(input);
        await record({ event: 'model_response', body: decodeJson(bytes) });
        return bytes;
      },
    });
    const execute = (goal, phase) =>
      runAgent({
        device,
        policy,
        goal,
        execute: true,
        maxSteps: 16,
        onStep: async (decision) => {
          decisions.push(decision);
          await record({ event: 'decision', phase, ...decision });
          console.log(`${phase}: ${decision.operation || decision.status} ${decision.label || ''}`);
        },
        onAction: (action) => record({ event: 'action_executed', phase, ...action }),
        onObservation: (observation) => record({ event: 'observation', phase, ...observation }),
      });
    let started,
      setupMs = 0;
    try {
      let baselineVerifiedOff = false;
      if (values.reset) {
        const setupStarted = performance.now();
        const preparation = await execute(
          'Turn off dark theme in Android Settings. Stop with the dark theme switch visible and off.',
          'setup',
        );
        baselineVerifiedOff =
          preparation.status === 'done' && darkThemeState(await device.observe()) === false;
        setupMs = performance.now() - setupStarted;
        if (!baselineVerifiedOff)
          throw new Error('Setup did not verify dark theme off; no measured run was started.');
      }
      metrics.length = 0;
      decisions.length = 0;
      const goal =
        'Turn on dark theme in Android Settings. Stop with the dark theme switch visible and on.';
      await record({ event: 'measurement_start', goal, baselineVerifiedOff });
      started = performance.now();
      const result = await execute(goal, 'measured');
      const executionMs = performance.now() - started;
      const executionMetrics = [...metrics];
      const verifyStarted = performance.now();
      const finalState = await device.observe();
      const verified = result.status === 'done' && darkThemeState(finalState) === true;
      const report = {
        attempt,
        trace: `${id}.jsonl`,
        claimedStatus: result.status,
        verified,
        baselineVerifiedOff,
        stateChangeVerified: baselineVerifiedOff && verified,
        steps: result.steps,
        modelCalls: result.timings.modelCalls,
        returnedModels: [...new Set(decisions.map((d) => d.responseModel).filter(Boolean))],
        setupMs: Math.round(setupMs),
        executionMs: Math.round(executionMs),
        verificationMs: Math.round(performance.now() - verifyStarted),
        totalMeasuredMs: Math.round(performance.now() - started),
        requests: summarizeMetrics(executionMetrics),
      };
      reports.push(report);
      await record({ event: 'verification', verified, darkTheme: darkThemeState(finalState) });
      await record({ event: 'report', ...report });
      console.log(JSON.stringify(report, null, 2));
    } catch (error) {
      const report = {
        attempt,
        trace: `${id}.jsonl`,
        verified: false,
        error: error.message,
        elapsedMs: started === undefined ? null : Math.round(performance.now() - started),
      };
      reports.push(report);
      await record({ event: 'failure', ...report });
      console.error(JSON.stringify(report));
    } finally {
      await file.close();
    }
  }
  const path = resolve(directory, `summary-${Date.now()}.json`);
  await writeFile(path, JSON.stringify({ demo: 'dark-theme', reports }, null, 2) + '\n', {
    flag: 'wx',
    mode: 0o600,
  });
  console.log(`Saved all ${reports.length} attempts to ${path}`);
  if (reports.some((report) => !report.verified)) process.exitCode = 2;
}

main().catch((error) => {
  console.error(error.message);
  process.exitCode = 1;
});
