import { test } from 'node:test';
import assert from 'node:assert/strict';
import { prepareInputVerification, inputMatches, confirmInput } from './input-verification.mjs';
import { MobilerunDevice } from './device.mjs';
import { runAgent } from './agent.mjs';

const action = { type: 'type', text: 'Berlin', clear: true };
function screen(text = '') {
  return {
    deviceId: 'phone',
    fingerprint: text,
    phone: { packageName: 'test.app', isEditable: true, inputElementId: 'input' },
    elements: [
      {
        id: 'input',
        resourceId: 'search',
        hint: 'Search',
        bounds: { left: 0, top: 0, right: 100, bottom: 50 },
        editable: true,
        enabled: true,
        password: false,
        text,
      },
    ],
  };
}

test('fast keyboard mode requires an exactly verifiable replacement; other inputs retain committed mode', async () => {
  for (const [mode, change, input, expected] of [
    ['accepted', () => {}, action, 'accepted'],
    ['committed', () => {}, action, 'committed'],
    ['accepted', () => {}, { ...action, clear: false }, 'committed'],
    [
      'accepted',
      (o) => {
        o.elements[0].password = true;
      },
      action,
      'committed',
    ],
    [
      'accepted',
      (o) => {
        o.phone.inputElementId = null;
      },
      action,
      'committed',
    ],
  ]) {
    const current = screen();
    change(current);
    const device = new MobilerunDevice({ deviceId: 'phone', textCompletionMode: mode });
    device.assertReady = async () => {};
    device.observe = async () => current;
    let body;
    device.api = async (_path, _method, value) => {
      body = value;
    };
    const receipt = await device.act(input);
    assert.equal(body.completionMode, expected);
    assert.equal(Boolean(receipt?.inputVerification), expected === 'accepted');
  }
});

test('readback requires exact text on the same device, app and identified field', () => {
  const verification = prepareInputVerification(screen(), action);
  assert.equal(inputMatches(screen('Berlin'), verification), true);
  for (const change of [
    (o) => {
      o.elements[0].text = 'Ber';
    },
    (o) => {
      o.deviceId = 'other';
    },
    (o) => {
      o.phone.packageName = 'other';
    },
    (o) => {
      o.elements[0].resourceId = 'other';
    },
    (o) => {
      o.elements[0].enabled = false;
    },
    (o) => {
      o.elements[0].password = true;
    },
    (o) => {
      o.elements.push({ ...o.elements[0] });
    },
  ]) {
    const current = screen('Berlin');
    change(current);
    assert.equal(inputMatches(current, verification), false);
  }
});

test('verification reuses a complete initial read and polls partial text without mutation', async () => {
  const verification = prepareInputVerification(screen(), action);
  let reads = 0;
  const observe = async () => {
    reads++;
    return screen('Berlin');
  };
  assert.equal(
    (await confirmInput({ initial: screen('Berlin'), verification, observe })).verified,
    true,
  );
  assert.equal(reads, 0);
  assert.equal(
    (await confirmInput({ initial: screen('Ber'), verification, observe, pollMs: 1 })).verified,
    true,
  );
  assert.equal(reads, 1);
  assert.equal(
    (await confirmInput({ initial: screen('Ber'), verification, observe, timeoutMs: 0 })).verified,
    false,
  );
  assert.equal(reads, 1);
});

test('agent confirms text before another decision and never repeats an unverified mutation', async () => {
  for (const complete of [true, false]) {
    let mutations = 0,
      decisions = 0,
      reads = 0,
      logged = 0;
    const device = {
      assertReady: async () => {},
      observe: async () => screen(reads++ === 0 ? '' : complete ? 'Berlin' : 'Ber'),
      act: async () => {
        mutations++;
        return { inputVerification: prepareInputVerification(screen(), action) };
      },
    };
    const policy = {
      decide: async ({ observation }) => {
        if (decisions++ === 0) return { status: 'action', action };
        assert.equal(observation.elements[0].text, 'Berlin');
        return { status: 'done' };
      },
    };
    const result = await runAgent({
      device,
      policy,
      goal: 'Berlin',
      execute: true,
      inputTimeoutMs: 0,
      onAction: () => {
        logged++;
      },
    });
    assert.equal(result.status, complete ? 'done' : 'input_unverified');
    assert.equal(mutations, 1);
    assert.equal(logged, 1);
    assert.equal(decisions, complete ? 2 : 1);
  }
});

function anonymousScreen(text = '') {
  const current = screen(text);
  current.elements[0].resourceId = '';
  current.elements[0].focused = true;
  return current;
}

test('anonymous input readback survives resizing and keyboard layout movement', () => {
  const verification = prepareInputVerification(anonymousScreen(), action);
  for (const focused of [true, false]) {
    const current = anonymousScreen('Berlin');
    current.elements[0].bounds = { left: 0, top: 200, right: 80, bottom: 250 };
    current.elements[0].focused = focused;
    if (!focused) delete current.phone.inputElementId;
    assert.equal(inputMatches(current, verification), true);
    current.elements[0].text = 'Ber';
    assert.equal(inputMatches(current, verification), false);
  }
});

test('anonymous input identity cannot be replaced by matching text or conflicting focus', () => {
  const verification = prepareInputVerification(anonymousScreen(), action);
  const other = { ...anonymousScreen('Berlin').elements[0], id: 'other', focused: false };
  for (const change of [
    (o) => {
      o.elements[0].id = 'other';
    },
    (o) => {
      o.elements[0].hint = 'Another field';
    },
    (o) => {
      o.elements[0].resourceId = 'another';
    },
    (o) => {
      o.elements[0].text = '';
      o.elements.push({ ...other });
    },
    (o) => {
      o.elements.push({ ...other, focused: true });
    },
    (o) => {
      o.elements[0].focused = false;
      o.elements.push({ ...other, focused: true });
    },
    (o) => {
      o.elements[0].focused = false;
      o.elements.push({ ...other });
    },
    (o) => {
      o.phone.inputElementId = 'other';
    },
    (o) => {
      o.elements.push({ ...o.elements[0] });
    },
  ]) {
    const current = anonymousScreen('Berlin');
    change(current);
    assert.equal(inputMatches(current, verification), false);
  }
  const current = anonymousScreen('Berlin');
  current.elements.push(other);
  assert.equal(inputMatches(current, verification), true);
});

test('agent proceeds after anonymous input resizes without repeating the input', async () => {
  const before = anonymousScreen();
  const after = anonymousScreen('Berlin');
  after.elements[0].bounds.right = 80;
  let reads = 0,
    mutations = 0,
    decisions = 0;
  const result = await runAgent({
    device: {
      assertReady: async () => {},
      observe: async () => (reads++ === 0 ? before : after),
      act: async () => {
        mutations++;
        return { inputVerification: prepareInputVerification(before, action) };
      },
    },
    policy: {
      decide: async ({ observation }) => {
        if (decisions++ === 0) return { status: 'action', action };
        assert.equal(observation, after);
        return { status: 'done' };
      },
    },
    goal: 'Berlin',
    execute: true,
    inputTimeoutMs: 0,
  });
  assert.equal(result.status, 'done');
  assert.equal(mutations, 1);
  assert.equal(decisions, 2);
});
