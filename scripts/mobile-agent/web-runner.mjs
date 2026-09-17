import { MobilerunDevice } from './device.mjs';
import { TypeSafePolicy, runAgent } from './agent.mjs';

const emit = (type, data) =>
  process.stdout.write(JSON.stringify({ type, at: Date.now(), ...data }) + '\n');
try {
  let input = '';
  for await (const chunk of process.stdin) {
    input += chunk;
    if (input.length > 10_000) throw new Error('Task input is too long.');
  }
  const { goal, maxSteps } = JSON.parse(input);
  const result = await runAgent({
    device: new MobilerunDevice(),
    policy: new TypeSafePolicy(),
    goal,
    maxSteps,
    execute: true,
    onStep: ({
      step,
      attempt,
      status,
      operation,
      label,
      confidence,
      targetConfidence,
      latencyMs,
      responseModel,
      reason,
    }) =>
      emit('decision', {
        step,
        attempt,
        status,
        operation,
        label,
        confidence,
        targetConfidence,
        latencyMs,
        model: responseModel,
        reason,
      }),
    onAction: ({ step, operation, label, executedMs }) =>
      emit('action', { step, operation, label, executedMs }),
  });
  emit('result', {
    outcome: result.status,
    steps: result.steps,
    timings: result.timings,
    reason: result.decision?.reason,
  });
} catch (error) {
  let message = error.message;
  for (const key of [
    process.env.MOBILERUN_API_KEY,
    process.env.MOBILERUN_CLOUD_API_KEY,
    process.env.TYPESAFE_API_KEY,
  ]) {
    if (key) message = message.split(key).join('[redacted]');
  }
  emit('error', { message });
  process.exitCode = 1;
}
