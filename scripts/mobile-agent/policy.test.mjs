import { test } from 'node:test';
import assert from 'node:assert/strict';
import { buildQuestions, TypeSafePolicy, validateChoice } from './policy.mjs';
import { assertFresh, StaleObservationError } from './device.mjs';
import { runAgent } from './agent.mjs';

const node = (id, text, extra = {}) => ({
  id,
  text,
  label: '',
  resourceId: id,
  bounds: { left: 0, top: 0, right: 200, bottom: 200 },
  enabled: true,
  clickable: true,
  editable: false,
  scrollable: false,
  ...extra,
});
const observe = () => ({
  deviceId: 'phone',
  observedAt: Date.now(),
  fingerprint: 'screen',
  screen: { width: 200, height: 400 },
  phone: { packageName: 'settings', isEditable: true },
  elements: [
    node('ui.0', 'Search', { editable: true }),
    node('ui.1', 'About phone'),
    node('ui.2', 'Settings list', { clickable: false, scrollable: true }),
  ],
});
const answer = (criteria, choice = Object.keys(criteria)[0], confidence = 1) => ({
  type: 'choice',
  choice,
  confidence,
  probabilities: Object.fromEntries(
    Object.keys(criteria).map((key) => [key, key === choice ? 1 : 0]),
  ),
});

test('operation and compatible targets are asked in one call; only selected target is consumed', async () => {
  let calls = 0;
  const policy = new TypeSafePolicy({
    request: async ({ body }) => {
      calls++;
      assert.deepEqual(Object.keys(body.questions), [
        'operation',
        'tap_target',
        'scroll_target',
        'text_value',
      ]);
      assert.equal(Object.keys(body.questions.tap_target.criteria).length, 2);
      assert.equal(Object.keys(body.questions.scroll_target.criteria).length, 1);
      return Buffer.from(
        JSON.stringify({
          model: 'jev-test',
          answers: {
            operation: answer(body.questions.operation.criteria, 'TAP'),
            tap_target: answer(body.questions.tap_target.criteria, '2'),
            scroll_target: { malicious: 'invalid unused answer' },
            text_value: null,
          },
        }),
      );
    },
  });
  const decision = await policy.decide({
    goal: 'Open About phone',
    observation: observe(),
    texts: ['hello'],
  });
  assert.equal(calls, 1);
  assert.equal(decision.operation, 'TAP');
  assert.equal(decision.action.elementId, 'ui.1');
  assert.equal(decision.responseModel, 'jev-test');
});

test('scroll choice consumes its region only; single-candidate target is supported', async () => {
  const policy = new TypeSafePolicy({
    request: async ({ body }) =>
      Buffer.from(
        JSON.stringify({
          answers: {
            operation: answer(body.questions.operation.criteria, 'SCROLL_DOWN'),
            scroll_target: answer(body.questions.scroll_target.criteria),
            tap_target: { choice: 'not-offered' },
          },
        }),
      ),
  });
  const decision = await policy.decide({ goal: 'Find version', observation: observe() });
  assert.equal(decision.action.type, 'swipe');
  assert.equal(decision.action.regionId, 'ui.2');
  assert.ok(decision.action.startY > decision.action.endY);
});

test('typing uses exactly one supplied value and never an unused tap answer', async () => {
  const policy = new TypeSafePolicy({
    request: async ({ body }) =>
      Buffer.from(
        JSON.stringify({
          answers: {
            operation: answer(body.questions.operation.criteria, 'TYPE_TEXT'),
            text_value: answer(body.questions.text_value.criteria, '2'),
          },
        }),
      ),
  });
  const result = await policy.decide({
    goal: 'Type Berlin',
    observation: observe(),
    texts: ['London', 'Berlin'],
  });
  assert.deepEqual(result.action, { type: 'type', text: 'Berlin', clear: true });
});

test('invalid selected distributions stop; unused questions are not required for DONE', async () => {
  let operation = 'TAP';
  const policy = new TypeSafePolicy({
    request: async ({ body }) =>
      Buffer.from(
        JSON.stringify({
          answers: {
            operation: answer(body.questions.operation.criteria, operation),
            tap_target: {
              type: 'choice',
              choice: 'shell-command',
              confidence: 1,
              probabilities: {},
            },
          },
        }),
      ),
  });
  await assert.rejects(policy.decide({ goal: 'Go', observation: observe() }), /invalid choice/);
  operation = 'DONE';
  assert.equal((await policy.decide({ goal: 'Go', observation: observe() })).status, 'done');
  for (const invalid of [
    { choice: 'x' },
    { confidence: NaN },
    { probabilities: { a: 0.9 } },
    { probabilities: { a: 0.2, b: 0.8 } },
    { probabilities: { a: 1, b: Infinity } },
  ])
    assert.throws(
      () => validateChoice({ ...answer({ a: '', b: '' }, 'a'), ...invalid }, { a: '', b: '' }),
      /invalid/,
    );
});

test('optional cutoff considers only operation and selected target', async () => {
  const policy = new TypeSafePolicy({
    threshold: 0.8,
    request: async ({ body }) =>
      Buffer.from(
        JSON.stringify({
          answers: {
            operation: answer(body.questions.operation.criteria, 'TAP'),
            tap_target: answer(body.questions.tap_target.criteria, '1', 0.4),
          },
        }),
      ),
  });
  assert.equal((await policy.decide({ goal: 'Go', observation: observe() })).status, 'uncertain');
  policy.threshold = 0;
  assert.equal((await policy.decide({ goal: 'Go', observation: observe() })).status, 'action');
});

test('unavailable operations are omitted', () => {
  const observation = observe();
  observation.phone.isEditable = false;
  observation.elements = [];
  const { questions } = buildQuestions(observation, ['text']);
  assert.deepEqual(Object.keys(questions), ['operation']);
  assert.equal(questions.operation.criteria.TAP, undefined);
  assert.equal(questions.operation.criteria.TYPE_TEXT, undefined);
  assert.equal(questions.operation.criteria.SCROLL_DOWN, undefined);
});

test('Jev selects a discovered app; exact goal names narrow the inventory', async () => {
  const policy = new TypeSafePolicy({
    request: async ({ body }) => {
      assert.equal(Object.keys(body.questions.app_target.criteria).length, 1);
      assert.equal(body.state.availableApps[0].label, 'Notes');
      return Buffer.from(
        JSON.stringify({
          answers: {
            operation: answer(body.questions.operation.criteria, 'OPEN_APP'),
            app_target: answer(body.questions.app_target.criteria, '1'),
          },
        }),
      );
    },
  });
  const result = await policy.decide({
    goal: 'Open Notes',
    observation: observe(),
    apps: [
      { packageName: 'com.example.notes', label: 'Notes' },
      { packageName: 'com.example.music', label: 'Music' },
    ],
  });
  assert.equal(result.action.packageName, 'com.example.notes');
  assert.equal(result.operation, 'OPEN_APP');
});

test('target guards allow unrelated text updates and movement but reject changed identity or checked state', () => {
  const before = observe(),
    after = structuredClone(before);
  after.fingerprint = 'new';
  after.elements.push(node('clock', '12:34', { clickable: false }));
  after.elements[1].bounds.top = 20;
  assert.doesNotThrow(() => assertFresh(after, before, { type: 'tap-element', elementId: 'ui.1' }));
  after.elements[1].text = 'Purchase';
  assert.throws(
    () => assertFresh(after, before, { type: 'tap-element', elementId: 'ui.1' }),
    StaleObservationError,
  );
  after.elements[1].text = 'About phone';
  after.elements[1].checked = true;
  assert.throws(
    () => assertFresh(after, before, { type: 'tap-element', elementId: 'ui.1' }),
    StaleObservationError,
  );
  after.phone.packageName = 'other';
  assert.throws(
    () => assertFresh(after, before, { type: 'global', name: 'home' }),
    StaleObservationError,
  );
});

test('Back tolerates live text counters but rejects changed navigation controls', () => {
  const before = observe(),
    after = structuredClone(before);
  before.elements.push(node('uptime', '1:00:00', { clickable: false }));
  after.elements.push(node('uptime', '1:00:01', { clickable: false }));
  after.fingerprint = 'changed-clock';
  assert.doesNotThrow(() => assertFresh(after, before, { type: 'global', name: 'back' }));
  after.elements[1].text = 'Different screen';
  assert.throws(
    () => assertFresh(after, before, { type: 'global', name: 'back' }),
    StaleObservationError,
  );
});

test('three consecutive stale decisions stop without executing or consuming the action budget', async () => {
  let decisions = 0;
  const device = {
    assertReady: async () => {},
    observe: async () => observe(),
    act: async () => {
      throw new StaleObservationError('changed');
    },
  };
  const policy = {
    decide: async () => {
      decisions++;
      return { status: 'action', action: { type: 'global', name: 'home' } };
    },
  };
  const result = await runAgent({ device, policy, goal: 'Home', execute: true });
  assert.equal(result.status, 'unstable_screen');
  assert.equal(result.steps, 0);
  assert.equal(decisions, 3);
});

test('changed screens proceed without any fixed post-action wait', async () => {
  let reads = 0,
    decisions = 0;
  const device = {
    assertReady: async () => {},
    observe: async () => ({ ...observe(), fingerprint: String(reads++) }),
    act: async () => {},
  };
  const policy = {
    decide: async () =>
      decisions++
        ? { status: 'done' }
        : { status: 'action', action: { type: 'global', name: 'home' } },
  };
  const result = await runAgent({ device, policy, goal: 'Home', execute: true });
  assert.equal(result.timings.waitMs, 0);
  assert.equal(reads, 2);
});

test('transitional snapshots without a foreground app are polled before another model call', async () => {
  let reads = 0,
    decisions = 0;
  const device = {
    assertReady: async () => {},
    observe: async () => {
      const state = observe();
      reads++;
      state.fingerprint = String(reads);
      if (reads === 2) state.phone.packageName = '';
      return state;
    },
    act: async () => {},
  };
  const policy = {
    decide: async ({ observation }) => {
      assert.ok(observation.phone.packageName);
      return decisions++
        ? { status: 'done' }
        : { status: 'action', action: { type: 'global', name: 'home' } };
    },
  };
  const result = await runAgent({ device, policy, goal: 'Home', execute: true });
  assert.equal(result.status, 'done');
  assert.equal(reads, 3);
  assert.equal(decisions, 2);
});

test('execution is logged before a failing post-action observation', async () => {
  let reads = 0,
    logged = false,
    acted = false;
  const device = {
    assertReady: async () => {},
    observe: async () => {
      if (reads++) throw new Error('Read failed');
      return observe();
    },
    act: async () => {
      acted = true;
    },
  };
  const policy = {
    decide: async () => ({ status: 'action', action: { type: 'global', name: 'home' } }),
  };
  await assert.rejects(
    runAgent({
      device,
      policy,
      goal: 'Home',
      execute: true,
      onAction: async () => {
        assert.equal(acted, true);
        logged = true;
      },
    }),
    /Read failed/,
  );
  assert.equal(logged, true);
});

test('stale decisions re-observe and re-predict; uncertain mutations never retry', async () => {
  for (const stale of [true, false]) {
    let acts = 0,
      decisions = 0;
    const device = {
      assertReady: async () => {},
      observe: async () => observe(),
      act: async () => {
        acts++;
        if (acts === 1)
          throw stale ? new StaleObservationError('stale') : new Error('transport failed');
      },
    };
    const policy = {
      decide: async () => {
        decisions++;
        return decisions >= 3
          ? { status: 'done' }
          : { status: 'action', action: { type: 'global', name: 'home' } };
      },
    };
    const run = runAgent({ device, policy, goal: 'Home', execute: true, settleTimeoutMs: 0 });
    if (stale) {
      const result = await run;
      assert.equal(result.steps, 1);
      assert.equal(result.timings.staleRetries, 1);
      assert.equal(decisions, 3);
    } else {
      await assert.rejects(run, /transport failed/);
      assert.equal(acts, 1);
      assert.equal(decisions, 1);
    }
  }
});
