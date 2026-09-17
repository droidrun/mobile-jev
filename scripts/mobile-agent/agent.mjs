import { setTimeout as delay } from 'node:timers/promises';
import { StaleObservationError } from './device.mjs';
import { describeAction } from './actions.mjs';
import { confirmInput } from './input-verification.mjs';
export { candidatesFor } from './actions.mjs';
export { TypeSafePolicy } from './policy.mjs';

// A policy selects one operation and (where needed) its target in one model request.
export async function runAgent({
  device,
  policy,
  goal,
  texts = [],
  execute = false,
  maxSteps = 10,
  settleMs = 0,
  settleTimeoutMs = 400,
  waitTimeoutMs = 15_000,
  inputTimeoutMs = 2500,
  onStep = () => {},
  onAction = () => {},
  onObservation = () => {},
}) {
  if (!Number.isSafeInteger(maxSteps) || maxSteps < 1 || maxSteps > 100)
    throw new Error('maxSteps must be 1–100.');
  for (const [key, value] of Object.entries({ settleMs, settleTimeoutMs, inputTimeoutMs })) {
    if (!Number.isFinite(value) || value < 0 || value > 10_000)
      throw new Error(`${key} must be 0–10000.`);
  }
  if (!Number.isFinite(waitTimeoutMs) || waitTimeoutMs < 0 || waitTimeoutMs > 60_000)
    throw new Error('waitTimeoutMs must be 0–60000.');
  const started = performance.now();
  const timings = {
    modelMs: 0,
    actionMs: 0,
    observationMs: 0,
    waitMs: 0,
    staleRetries: 0,
    modelCalls: 0,
  };
  const history = [],
    repeated = new Set();
  const observe = async () => {
    const start = performance.now();
    try {
      return await device.observe();
    } finally {
      timings.observationMs += performance.now() - start;
    }
  };
  const wait = async (ms) => {
    const start = performance.now();
    await delay(ms);
    timings.waitMs += performance.now() - start;
  };
  await device.assertReady();
  const [initialObservation, installedApps] = await Promise.all([
    observe(),
    typeof device.listApps === 'function' ? device.listApps() : Promise.resolve([]),
  ]);
  let observation = initialObservation;
  const finish = (status, decision) => ({
    status,
    steps: history.length,
    decision,
    observation,
    timings: {
      ...Object.fromEntries(Object.entries(timings).map(([k, v]) => [k, Math.round(v * 10) / 10])),
      wallMs: Math.round((performance.now() - started) * 10) / 10,
    },
  });
  let consecutiveWaits = 0,
    consecutiveStale = 0;
  let waitingSince;
  // Stale decisions never dispatch input, but still consume a separate model-call budget.
  for (let attempt = 0; attempt < maxSteps * 2 + 4; attempt++) {
    const modelStart = performance.now();
    const decision = await policy.decide({
      goal,
      observation,
      history,
      texts,
      apps: installedApps,
    });
    timings.modelMs += performance.now() - modelStart;
    timings.modelCalls++;
    const step = history.length;
    await onStep({ step, attempt, ...decision });
    if (decision.status !== 'action') return finish(decision.status, decision);
    if (!execute) return finish('preview', decision);
    if (step >= maxSteps) return finish('step_limit', decision);
    const isWait = decision.action.type === 'wait';
    const label =
      decision.label ||
      (isWait ? 'Wait for screen update' : describeAction(decision.action, observation));
    const signature = `${observation.fingerprint}:${JSON.stringify(decision.action)}`;
    if (!isWait && repeated.has(signature)) return finish('stuck', decision);
    if (isWait) {
      waitingSince ??= performance.now();
      if (performance.now() - waitingSince >= waitTimeoutMs)
        return finish('loading_timeout', decision);
    } else waitingSince = undefined;
    const actionStart = performance.now();
    let stale = false;
    let receipt;
    try {
      if (isWait)
        await wait(
          Math.min(
            100 * 2 ** Math.min(consecutiveWaits, 4),
            1000,
            Math.max(0, waitTimeoutMs - (performance.now() - waitingSince)),
          ),
        );
      else receipt = await device.act(decision.action, { expected: observation });
    } catch (error) {
      if (!(error instanceof StaleObservationError)) throw error; // Never retry uncertain mutations.
      timings.staleRetries++;
      stale = true;
    } finally {
      if (!isWait) timings.actionMs += performance.now() - actionStart;
    }
    if (stale) {
      if (++consecutiveStale >= 3) return finish('unstable_screen', decision);
      observation = await observe();
      continue;
    }
    consecutiveStale = 0;
    repeated.add(signature);
    consecutiveWaits = isWait ? consecutiveWaits + 1 : 0;
    const entry = {
      operation: decision.operation || decision.action.type,
      label,
      action: decision.action,
      ...(decision.action.type === 'type' ? { text: decision.action.text } : {}),
      before: observation.fingerprint,
      screenChanged: null,
    };
    history.push(entry);
    // Log the mutation before observing: a failed read must not erase an executed action.
    await onAction({
      step,
      action: decision.action,
      operation: entry.operation,
      label,
      executedMs: Math.round(performance.now() - started),
    });
    if (settleMs) await wait(settleMs);
    let after = await observe();
    if (receipt?.inputVerification) {
      const confirmation = await confirmInput({
        initial: after,
        verification: receipt.inputVerification,
        observe,
        timeoutMs: inputTimeoutMs,
        sleep: wait,
      });
      after = confirmation.observation;
      if (!confirmation.verified) {
        observation = after;
        await onObservation({
          step,
          observation: after,
          screenChanged: entry.before !== after.fingerprint,
        });
        return finish('input_unverified', {
          ...decision,
          reason:
            'Text was sent, but its complete value could not be confirmed in the input field. Inspect the screen before retrying.',
        });
      }
    }
    const waitDeadline = performance.now() + settleTimeoutMs;
    // Skip a transient system-bar-only snapshot (no foreground app) before asking Jev again.
    while (
      !isWait &&
      !receipt?.inputVerification &&
      (after.fingerprint === observation.fingerprint || !after.phone.packageName) &&
      performance.now() + 60 < waitDeadline
    ) {
      await wait(60);
      after = await observe();
    }
    entry.after = after.fingerprint;
    entry.screenChanged = observation.fingerprint !== after.fingerprint;
    await onObservation({ step, observation: after, screenChanged: entry.screenChanged });
    observation = after;
  }
  return finish('decision_limit');
}
