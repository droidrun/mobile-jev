import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { once } from 'node:events';
import { MobilerunDevice, summarizeState } from './device.mjs';
import { candidatesFor, runAgent } from './agent.mjs';
import { curlRequest, decodeJson } from './http.mjs';

const json = (value) => Buffer.from(JSON.stringify(value));
test('staging base URL is used for every device request', async () => {
  let url;
  const device = new MobilerunDevice({
    deviceId: 'staging-phone',
    baseUrl: 'https://staging-api.mobilerun.ai/v1/',
    request: async (request) => {
      url = request.url;
      return json({ state: 'ready' });
    },
  });
  await device.assertReady();
  assert.equal(url, 'https://staging-api.mobilerun.ai/v1/devices/staging-phone');
});

test('nested identical scroll regions produce one candidate per direction', () => {
  const observation = summarizeState(state(), 'phone');
  const scroll = {
    enabled: true,
    scrollable: true,
    bounds: { left: 0, top: 100, right: 720, bottom: 1600 },
  };
  observation.elements.push({ ...scroll, id: 'scroll' }, { ...scroll, id: 'scroll.child' });
  const candidates = candidatesFor(observation);
  assert.equal(Object.keys(candidates).filter((key) => key.startsWith('scroll_')).length, 4);
});
function state({ label = 'Settings', editable = false } = {}) {
  return {
    device_context: { screen_bounds: { width: 720, height: 1600 } },
    phone_state: { currentApp: 'Home', packageName: 'home', isEditable: editable },
    a11y_tree: {
      children: [
        {
          text: label,
          isClickable: true,
          isEnabled: true,
          boundsInScreen: { left: 10, top: 100, right: 110, bottom: 200 },
        },
        {
          text: 'hidden',
          isVisibleToUser: false,
          isClickable: true,
          boundsInScreen: { left: 0, top: 0, right: 100, bottom: 100 },
        },
      ],
    },
  };
}

function fakeDevice(initial = state()) {
  const calls = [];
  const fixture = { raw: initial, ready: true, calls, apps: [] };
  fixture.device = new MobilerunDevice({
    apiKey: 'test-key',
    deviceId: 'test/device',
    request: async (request) => {
      calls.push(request);
      if (request.url.endsWith('/ui-state?filter=false')) return json(fixture.raw);
      if (request.url.endsWith('/apps?includeSystemApps=true')) return json(fixture.apps);
      if (request.method === 'GET')
        return json({
          id: 'test/device',
          name: 'Test',
          state: fixture.ready ? 'ready' : 'disconnected',
        });
      return Buffer.alloc(0);
    },
  });
  return fixture;
}

test('summarizes UI without hidden nodes or password values; fingerprint excludes time', () => {
  const raw = state();
  raw.a11y_tree.children.push({
    isPassword: true,
    text: 'secret',
    contentDescription: 'secret',
    isEditable: true,
    boundsInScreen: { left: -5, top: 200, right: 800, bottom: 250 },
  });
  const a = summarizeState(raw, 'phone');
  assert.equal(a.elements.length, 2);
  assert.equal(JSON.stringify(a).includes('secret'), false);
  assert.deepEqual(a.elements[1].bounds, { left: 0, top: 200, right: 720, bottom: 250 });
  assert.equal(a.fingerprint, summarizeState(raw, 'phone').fingerprint);
  assert.throws(() => summarizeState({}, 'phone'), /UI state/);
});

test('element action uses center of original bounds and encoded device ID', async () => {
  const { device, calls } = fakeDevice();
  const expected = await device.observe();
  await device.act({ type: 'tap-element', elementId: 'ui.0' }, { expected });
  assert.deepEqual(calls.at(-1).body, { x: 60, y: 150 });
  assert.equal(calls.at(-1).url, 'https://api.mobilerun.ai/v1/devices/test%2Fdevice/tap');
});

test('changed, expired, and wrong-device snapshots never dispatch an action', async () => {
  const fixture = fakeDevice();
  const expected = await fixture.device.observe();
  fixture.raw = state({ label: 'Buy now' });
  await assert.rejects(
    fixture.device.act({ type: 'tap-element', elementId: 'ui.0' }, { expected }),
    /Screen changed/,
  );
  fixture.raw = state();
  await assert.rejects(
    fixture.device.act(
      { type: 'global', name: 'home' },
      { expected: { ...expected, observedAt: 0 } },
    ),
    /expired/,
  );
  await assert.rejects(
    fixture.device.act(
      { type: 'global', name: 'home' },
      { expected: { ...expected, deviceId: 'other' } },
    ),
    /Screen changed/,
  );
  assert.equal(
    fixture.calls.some((call) => call.method !== 'GET'),
    false,
  );
});

test('rejects invalid actions, out-of-screen coordinates and typing without focus', async () => {
  const { device, calls } = fakeDevice();
  for (const action of [
    { type: 'tap', x: 720, y: 0 },
    { type: 'tap', x: NaN, y: 0 },
    { type: 'tap', x: 1.5, y: 0 },
    { type: 'type', text: 'hello' },
    { type: 'swipe', startX: 1, startY: 1, endX: 5, endY: 5, duration: 0 },
    { type: 'global', name: 'toString' },
    { type: 'shell', command: 'anything' },
    { type: 'tap-element', elementId: 'ui.0' },
  ])
    await assert.rejects(device.act(action));
  assert.equal(
    calls.some((call) => call.method !== 'GET'),
    false,
  );
});

test('disconnected devices stop before observing or acting', async () => {
  const fixture = fakeDevice();
  fixture.ready = false;
  await assert.rejects(fixture.device.act({ type: 'global', name: 'home' }), /disconnected/);
  assert.equal(fixture.calls.length, 1);
});

test('app launch targets must come from discovery and do not need a redundant UI read', async () => {
  const fixture = fakeDevice();
  fixture.apps = [{ packageName: 'com.example.notes', label: 'Notes' }];
  await fixture.device.assertReady();
  await fixture.device.listApps();
  fixture.calls.length = 0;
  await fixture.device.act({ type: 'open-app', packageName: 'com.example.notes' });
  assert.equal(fixture.calls.length, 1);
  assert.equal(fixture.calls[0].method, 'PUT');
  assert.match(fixture.calls[0].url, /\/apps\/com.example.notes$/);
  await assert.rejects(
    fixture.device.act({ type: 'open-app', packageName: 'com.unobserved.app' }),
    /not observed/,
  );
  assert.equal(fixture.calls.length, 1);
});

test('readiness is cached briefly but each action still observes before dispatch', async () => {
  const { device, calls } = fakeDevice();
  await device.assertReady();
  await device.act({ type: 'global', name: 'home' });
  await device.act({ type: 'global', name: 'home' });
  assert.equal(
    calls.filter((call) => call.method === 'GET' && !call.url.includes('ui-state')).length,
    1,
  );
  assert.equal(calls.filter((call) => call.url.includes('ui-state')).length, 2);
  device.readyAt -= 30_001;
  await device.act({ type: 'global', name: 'home' });
  assert.equal(
    calls.filter((call) => call.method === 'GET' && !call.url.includes('ui-state')).length,
    2,
  );
});

test('scroll gestures resolve current bounds when a toolbar changes region geometry', async () => {
  const fixture = fakeDevice();
  fixture.raw.a11y_tree.children[0].isScrollable = true;
  const expected = await fixture.device.observe();
  fixture.raw.a11y_tree.children[0].boundsInScreen = {
    left: 10,
    top: 200,
    right: 210,
    bottom: 600,
  };
  await fixture.device.act(
    {
      type: 'swipe',
      regionId: 'ui.0',
      startX: 60,
      startY: 180,
      endX: 60,
      endY: 120,
      duration: 300,
    },
    { expected },
  );
  assert.deepEqual(fixture.calls.at(-1).body, {
    startX: 110,
    startY: 520,
    endX: 110,
    endY: 280,
    duration: 300,
  });
});

test('keyboard, swipe, and navigation use documented methods and payloads', async () => {
  const { device, calls } = fakeDevice(state({ editable: true }));
  await device.act({ type: 'type', text: 'こんにちは\n"test"', clear: true });
  assert.deepEqual(calls.at(-1).body, {
    text: 'こんにちは\n"test"',
    clear: true,
    completionMode: 'committed',
  });
  await device.act({ type: 'key', key: 'enter' });
  assert.equal(calls.at(-1).method, 'PUT');
  assert.deepEqual(calls.at(-1).body, { key: 66 });
  await device.act({ type: 'clear' });
  assert.equal(calls.at(-1).method, 'DELETE');
  await device.act({ type: 'global', name: 'back' });
  assert.deepEqual(calls.at(-1).body, { action: 1 });
  await device.act({ type: 'swipe', startX: 100, startY: 500, endX: 100, endY: 200 });
  assert.equal(calls.at(-1).body.duration, 300);
});

test('device listings paginate and omit stream credentials', async () => {
  let calls = 0;
  const device = new MobilerunDevice({
    request: async () => {
      calls++;
      return json({
        items: Array.from({ length: calls === 1 ? 100 : 1 }, (_, i) => ({
          id: `${calls}-${i}`,
          name: 'Phone',
          state: 'ready',
          streamToken: 'secret',
        })),
      });
    },
  });
  const devices = await device.listDevices();
  assert.equal(devices.length, 101);
  assert.equal(calls, 2);
  assert.equal(JSON.stringify(devices).includes('secret'), false);
});

test('screenshots reject JSON payloads masquerading as successful images', async () => {
  const { device } = fakeDevice();
  await assert.rejects(device.screenshot(), /not a PNG/);
});

test('disabled elements and unfocused text entry are excluded from policy actions', () => {
  const raw = state();
  raw.a11y_tree.children[0].isEnabled = false;
  const candidates = candidatesFor(summarizeState(raw, 'phone'), ['text']);
  assert.equal(candidates['tap_ui.0'], undefined);
  assert.equal(candidates.text_0, undefined);
});

test('preview does not mutate a device; unchanged repeated action stops the loop', async () => {
  const { device, calls } = fakeDevice();
  const policy = {
    decide: async () => ({ status: 'action', action: { type: 'tap-element', elementId: 'ui.0' } }),
  };
  const preview = await runAgent({ device, policy, goal: 'Settings' });
  assert.equal(preview.status, 'preview');
  assert.equal(
    calls.some((call) => call.method === 'POST'),
    false,
  );
  const result = await runAgent({
    device,
    policy,
    goal: 'Settings',
    execute: true,
    settleMs: 0,
    settleTimeoutMs: 0,
  });
  assert.equal(result.status, 'stuck');
  assert.equal(result.steps, 1);
  assert.equal(calls.filter((call) => call.method === 'POST').length, 1);
});

test('agent checks completion after the final allowed action and never exceeds budget', async () => {
  for (const complete of [true, false]) {
    const fixture = fakeDevice();
    let decisions = 0;
    const policy = {
      decide: async () => {
        decisions++;
        return complete && decisions > 1
          ? { status: 'done' }
          : { status: 'action', action: { type: 'global', name: 'home' } };
      },
    };
    const result = await runAgent({
      device: fixture.device,
      policy,
      goal: 'Home',
      execute: true,
      maxSteps: 1,
      settleMs: 0,
      settleTimeoutMs: 0,
    });
    assert.equal(result.status, complete ? 'done' : 'step_limit');
    assert.equal(result.steps, 1);
    assert.equal(fixture.calls.filter((call) => call.method === 'POST').length, 1);
  }
});

test('curl transport preserves Unicode, quotes and shell characters; handles binary, errors and timeouts without retries', async (t) => {
  const received = [];
  const png = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10, 0, 255, 10]);
  const server = createServer(async (req, res) => {
    let body = '';
    for await (const part of req) body += part;
    received.push({ url: req.url, body, authorization: req.headers.authorization });
    if (req.url === '/slow') return;
    if (req.url === '/error') {
      res.writeHead(401);
      res.end('secret-key');
      return;
    }
    if (req.url === '/png') {
      res.end(png);
      return;
    }
    if (req.url === '/empty') {
      res.writeHead(204);
      res.end();
      return;
    }
    res.end(JSON.stringify({ ok: true }));
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => {
    server.closeAllConnections();
    server.close();
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const body = { text: '你好\n"quoted" \\ `echo hi` $(echo hello)', clear: true };
  assert.deepEqual(
    decodeJson(await curlRequest({ url: base, apiKey: 'secret-key', method: 'POST', body })),
    { ok: true },
  );
  assert.deepEqual(JSON.parse(received[0].body), body);
  assert.equal(received[0].authorization, 'Bearer secret-key');
  assert.deepEqual(await curlRequest({ url: base + '/png', apiKey: 'secret-key' }), png);
  assert.equal(decodeJson(await curlRequest({ url: base + '/empty', apiKey: 'secret-key' })), null);
  await assert.rejects(
    curlRequest({ url: base + '/error', apiKey: 'secret-key' }),
    (error) => /401/.test(error.message) && !error.message.includes('secret-key'),
  );
  await assert.rejects(
    curlRequest({ url: base + '/slow', apiKey: 'secret-key', timeoutMs: 100 }),
    /not retried/,
  );
  assert.equal(received.filter((req) => req.url === '/slow').length, 1);
});
