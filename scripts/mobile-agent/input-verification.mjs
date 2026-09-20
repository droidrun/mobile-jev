import { setTimeout as delay } from 'node:timers/promises';

export function prepareInputVerification(observation, action) {
  // Replacing a readable field has an exact expected value. Appends may insert at an
  // unknown cursor position, so retain the server completion mode for those.
  if (action.type !== 'type' || action.clear !== true) return null;
  const target = observation.elements.find((e) => e.id === observation.phone.inputElementId);
  if (!target?.editable || !target.enabled || target.password) return null;
  return {
    deviceId: observation.deviceId,
    packageName: observation.phone.packageName,
    target: {
      id: target.id,
      resourceId: target.resourceId,
      hint: target.hint,
    },
    text: action.text,
  };
}

export function inputMatches(observation, verification) {
  if (
    observation.deviceId !== verification.deviceId ||
    observation.phone.packageName !== verification.packageName
  )
    return false;
  const inputs = observation.elements.filter((e) => e.editable && e.enabled && !e.password);
  const { target } = verification;
  let candidates;
  if (target.resourceId) {
    candidates = inputs.filter((e) => e.resourceId === target.resourceId);
    if (candidates.length > 1) candidates = candidates.filter((e) => e.id === target.id);
  } else {
    // Layout may move or resize after typing (for example, a Send button appears).
    // Keep tree identity and hint, then require unambiguous input/focus evidence.
    candidates = inputs.filter(
      (e) => e.id === target.id && e.hint === target.hint && !e.resourceId,
    );
    if (candidates.length !== 1) return false;
    const editable = observation.elements.filter((e) => e.editable);
    const focused = editable.filter((e) => e.focused);
    if (observation.phone.inputElementId && observation.phone.inputElementId !== target.id)
      return false;
    if (focused.length > 0) {
      if (focused.length !== 1 || focused[0] !== candidates[0]) return false;
    } else if (editable.length !== 1) return false;
  }
  return candidates.length === 1 && candidates[0].text === verification.text;
}

export async function confirmInput({
  initial,
  verification,
  observe,
  timeoutMs = 2500,
  pollMs = 60,
  sleep = delay,
}) {
  if (!Number.isFinite(timeoutMs) || timeoutMs < 0 || !Number.isFinite(pollMs) || pollMs < 1)
    throw new Error('Invalid input verification timing.');
  const deadline = performance.now() + timeoutMs;
  let observation = initial;
  while (!inputMatches(observation, verification)) {
    const remaining = deadline - performance.now();
    if (remaining <= 0) return { verified: false, observation };
    await sleep(Math.min(pollMs, remaining));
    observation = await observe();
  }
  return { verified: true, observation };
}
