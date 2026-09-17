import { parseArgs } from 'node:util';
import { mkdir, open, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';
import { MobilerunDevice } from './device.mjs';
import { TypeSafePolicy, runAgent } from './agent.mjs';
import { pooledRequest, decodeJson } from './http.mjs';
import { measuredRequest, summarizeMetrics } from './metrics.mjs';
import { confirmInput } from './input-verification.mjs';

const help = `Mobilerun device harness (Node 22+ and curl)

pnpm agent COMMAND [ARGS] [OPTIONS]

Commands:
  devices                        List device IDs, names, and states
  observe                        Print compact UI state and element IDs
  profile                        Time readiness and five UI-state reads without changing the device
  screenshot                     Save a PNG (--out PATH)
  tap X Y                        Tap coordinates
  tap-element ID                  Tap from a saved observation (--snapshot PATH)
  swipe X1 Y1 X2 Y2 [MS]          Swipe in screen pixels
  type TEXT                      Type into the focused field (--clear to replace)
  clear                          Clear the focused field
  key enter|tab|delete|forward_delete|back
  home | back | recent           System navigation
  run GOAL                       Preview a TypeSafe decision; --execute runs the loop

Options:
  --device ID                    Or MOBILERUN_DEVICE_ID
  --out PATH                     Save observe JSON or screenshot PNG
  --snapshot PATH                Original observe JSON for tap-element
  --text VALUE                   Repeatable exact text candidates for TypeSafe
  --steps N                      Maximum actions (default 10, maximum 100)
  --confidence N                 Optional operation/target cutoff (default 0; disabled)
  --settle-ms N                  Extra fixed wait after actions (default 0)
  --wait-timeout-ms N            Consecutive loading wait budget (default 15000)
  --text-completion MODE         accepted (verify locally) or committed (server waits)
  --execute                      Execute TypeSafe-selected actions
  --trace PATH                   Save model requests, responses, actions, and final state (JSONL)

Keys: MOBILERUN_API_KEY (or MOBILERUN_CLOUD_API_KEY); TYPESAFE_API_KEY for run.
Direct commands execute immediately. Run sends goal and UI text to TypeSafe.
`;

async function save(path, data) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, data, { flag: 'wx', mode: 0o600 });
}

async function main() {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      help: { type: 'boolean', short: 'h' },
      device: { type: 'string' },
      out: { type: 'string' },
      snapshot: { type: 'string' },
      clear: { type: 'boolean' },
      execute: { type: 'boolean' },
      trace: { type: 'string' },
      steps: { type: 'string', default: '10' },
      confidence: { type: 'string', default: '0' },
      'settle-ms': { type: 'string', default: '0' },
      'wait-timeout-ms': { type: 'string', default: '15000' },
      'text-completion': { type: 'string' },
      text: { type: 'string', multiple: true, default: [] },
    },
  });
  const [command, ...args] = positionals;
  if (!command || values.help) {
    console.log(help);
    return;
  }
  const print = (value) => console.log(JSON.stringify(value, null, 2));
  const arity = (min, max = min) => {
    if (args.length < min || args.length > max)
      throw new Error(`Invalid arguments for ${command}. Use --help.`);
  };
  const device = new MobilerunDevice({
    deviceId: values.device || process.env.MOBILERUN_DEVICE_ID,
    textCompletionMode: values['text-completion'],
  });
  if (command === 'devices') {
    arity(0);
    print(await device.listDevices());
    return;
  }
  if (command === 'profile') {
    arity(0);
    const metrics = [];
    device.request = measuredRequest({ service: 'mobilerun', metrics });
    await device.assertReady();
    for (let i = 0; i < 5; i++) await device.observe();
    const report = {
      baseUrl: device.baseUrl,
      deviceId: device.deviceId,
      requests: summarizeMetrics(metrics),
    };
    if (values.out) await save(values.out, JSON.stringify(report, null, 2) + '\n');
    print(report);
    return;
  }
  if (command === 'observe') {
    arity(0);
    const observation = await device.observe();
    if (values.out) {
      await save(values.out, JSON.stringify(observation, null, 2) + '\n');
      print({ saved: values.out });
    } else print(observation);
    return;
  }
  if (command === 'screenshot') {
    arity(0);
    const path = values.out || `artifacts/screen-${Date.now()}.png`;
    await save(path, await device.screenshot());
    print({ saved: path });
    return;
  }
  if (command === 'run') {
    arity(1);
    let trace;
    if (values.trace) {
      await mkdir(dirname(values.trace), { recursive: true });
      trace = await open(values.trace, 'wx', 0o600);
    }
    const record = async (event) => {
      if (trace)
        await trace.write(JSON.stringify({ at: new Date().toISOString(), ...event }) + '\n');
    };
    const metrics = [];
    const onMetric = async (metric) => record({ event: 'request_timing', ...metric });
    device.request = measuredRequest({ service: 'mobilerun', metrics, onMetric });
    const modelRequest = measuredRequest({
      service: 'typesafe',
      metrics,
      onMetric,
      request: pooledRequest,
    });
    try {
      await record({
        event: 'run',
        deviceId: device.deviceId,
        baseUrl: device.baseUrl,
        goal: args[0],
        execute: values.execute ?? false,
        confidenceThreshold: Number(values.confidence),
        maxSteps: Number(values.steps),
        settleMs: Number(values['settle-ms']),
        waitTimeoutMs: Number(values['wait-timeout-ms']),
      });
      const result = await runAgent({
        device,
        policy: new TypeSafePolicy({
          threshold: Number(values.confidence),
          request: async (request) => {
            await record({ event: 'model_request', url: request.url, body: request.body });
            const bytes = await modelRequest(request);
            await record({ event: 'model_response', body: decodeJson(bytes) });
            return bytes;
          },
        }),
        goal: args[0],
        texts: values.text,
        execute: values.execute,
        maxSteps: Number(values.steps),
        settleMs: Number(values['settle-ms']),
        waitTimeoutMs: Number(values['wait-timeout-ms']),
        onStep: async (step) => {
          await record({ event: 'decision', ...step });
          const summary = Object.fromEntries(
            Object.entries(step).filter(([key]) => !key.toLowerCase().includes('probabilities')),
          );
          print({ event: 'decision', ...summary });
        },
        onAction: async (event) => record({ event: 'action_executed', ...event }),
        onObservation: async (event) => record({ event: 'observation', ...event }),
      });
      await record({ event: 'result', ...result });
      const performanceReport = { timings: result.timings, requests: summarizeMetrics(metrics) };
      await record({ event: 'performance', ...performanceReport });
      print({
        status: result.status,
        steps: result.steps,
        timings: result.timings,
        ...(values.trace ? { trace: values.trace } : {}),
      });
      if (!['done', 'preview'].includes(result.status)) process.exitCode = 2;
    } catch (error) {
      await record({ event: 'error', message: error.message });
      throw error;
    } finally {
      await trace?.close();
    }
    return;
  }
  let action, expected;
  switch (command) {
    case 'tap':
      arity(2);
      action = { type: 'tap', x: Number(args[0]), y: Number(args[1]) };
      break;
    case 'tap-element':
      arity(1);
      if (!values.snapshot) throw new Error('tap-element requires --snapshot from observe --out.');
      expected = JSON.parse(await readFile(values.snapshot, 'utf8'));
      action = { type: 'tap-element', elementId: args[0] };
      break;
    case 'swipe':
      arity(4, 5);
      action = {
        type: 'swipe',
        startX: Number(args[0]),
        startY: Number(args[1]),
        endX: Number(args[2]),
        endY: Number(args[3]),
        duration: Number(args[4] ?? 300),
      };
      break;
    case 'type':
      arity(1);
      action = { type: 'type', text: args[0], clear: values.clear ?? false };
      break;
    case 'clear':
      arity(0);
      action = { type: 'clear' };
      break;
    case 'key':
      arity(1);
      action = { type: 'key', key: args[0] };
      break;
    case 'home':
    case 'back':
    case 'recent':
      arity(0);
      action = { type: 'global', name: command };
      break;
    default:
      throw new Error('Unknown command. Use --help.');
  }
  const receipt = await device.act(action, { expected });
  let observation = await device.observe();
  if (receipt?.inputVerification) {
    const confirmation = await confirmInput({
      initial: observation,
      verification: receipt.inputVerification,
      observe: () => device.observe(),
    });
    observation = confirmation.observation;
    if (!confirmation.verified) {
      print({ status: 'input_unverified', observation });
      process.exitCode = 2;
      return;
    }
  }
  print({ status: 'executed', observation });
}

main().catch((error) => {
  // Mask keys even if a local parse/argument error happens to contain one.
  let message = error.message;
  for (const key of [
    process.env.MOBILERUN_API_KEY,
    process.env.MOBILERUN_CLOUD_API_KEY,
    process.env.TYPESAFE_API_KEY,
  ]) {
    if (key) message = message.split(key).join('[redacted]');
  }
  console.error(message);
  process.exitCode = 1;
});
