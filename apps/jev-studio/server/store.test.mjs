import { test } from 'node:test';
import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { PassThrough } from 'node:stream';
import { createStore, checkRequest } from './store.mjs';

function fixture() {
  let time = 1000;
  const child = new EventEmitter();
  child.stdout = new PassThrough();
  child.stderr = new PassThrough();
  child.stdin = new PassThrough();
  child.kill = () => {};
  const store = createStore({ spawnTask: () => child, now: () => time });
  return {
    child,
    store,
    setTime: (value) => {
      time = value;
    },
  };
}

test('timer begins on acceptance and ends only when the process exits; split events are preserved', () => {
  const { child, store, setTime } = fixture();
  const run = store.start({ goal: 'Open Settings', maxSteps: 10 });
  assert.equal(run.startedAt, 1000);
  assert.equal(run.endedAt, null);
  child.stdout.write('{"type":"act');
  child.stdout.write('ion","label":"Tap Settings"}\n');
  assert.equal(store.get(run.id).events[0].label, 'Tap Settings');
  child.stdout.write('{"type":"result","outcome":"done"}\n');
  assert.equal(store.get(run.id).endedAt, null);
  setTime(4125);
  child.emit('close', 0);
  assert.equal(store.get(run.id).status, 'succeeded');
  assert.equal(store.get(run.id).endedAt, 4125);
  setTime(9999);
  child.emit('close', 0);
  assert.equal(store.get(run.id).endedAt, 4125);
});

test('one run owns the device; stop freezes time only after process exit', () => {
  const { child, store, setTime } = fixture();
  const run = store.start({ goal: 'First' });
  assert.throws(() => store.start({ goal: 'Second' }), /already running/);
  store.stop(run.id);
  assert.equal(store.get(run.id).status, 'stopping');
  assert.equal(store.get(run.id).endedAt, null);
  setTime(2000);
  child.emit('close', null);
  assert.equal(store.get(run.id).status, 'stopped');
  assert.equal(store.get(run.id).endedAt, 2000);
});

test('blocked and failed tasks do not become successful', () => {
  for (const outcome of ['blocked', 'needs_input', null]) {
    const { child, store } = fixture();
    const run = store.start({ goal: 'A goal' });
    if (outcome) child.stdout.write(JSON.stringify({ type: 'result', outcome }) + '\n');
    child.emit('close', 0);
    assert.equal(store.get(run.id).status, outcome ? 'blocked' : 'failed');
  }
});

test('validation and origin checks reject malformed tasks and cross-origin control', () => {
  const { store } = fixture();
  for (const input of [
    { goal: '' },
    { goal: 'x', maxSteps: 0 },
    { goal: 'x', maxSteps: 100 },
    { goal: 'a'.repeat(4001) },
  ])
    assert.throws(() => store.start(input));
  assert.doesNotThrow(() =>
    checkRequest(
      new Request('http://127.0.0.1:3040/api/studio/runs', {
        method: 'POST',
        headers: { origin: 'http://127.0.0.1:3040' },
      }),
    ),
  );
  assert.throws(
    () =>
      checkRequest(
        new Request('http://127.0.0.1:3040/api/studio/runs', {
          method: 'POST',
          headers: { origin: 'https://other.test' },
        }),
      ),
    /Origin/,
  );
  assert.throws(
    () => checkRequest(new Request('http://127.0.0.1:3040/api/studio/runs', { method: 'POST' })),
    /same-origin/,
  );
  assert.throws(
    () => checkRequest(new Request('http://attacker.test:3040/api/studio/device')),
    /localhost/,
  );
});

test('live subscribers receive updates and can detach', () => {
  const { child, store } = fixture();
  const run = store.start({ goal: 'A goal' });
  const updates = [];
  const unsubscribe = store.subscribe(run.id, (snapshot) => updates.push(snapshot));
  child.stdout.write('{"type":"action","label":"Tap"}\n');
  unsubscribe();
  child.stdout.write('{"type":"result","outcome":"done"}\n');
  child.emit('close', 0);
  assert.equal(updates.length, 1);
  assert.equal(updates[0].events.length, 1);
});

test('clear removes ended runs from both list and detail, including their timing', () => {
  const { child, store, setTime } = fixture();
  const run = store.start({ goal: 'A goal' });
  child.stdout.write('{"type":"result","outcome":"done"}\n');
  setTime(4000);
  child.emit('close', 0);
  assert.equal(store.get(run.id).endedAt, 4000);
  assert.deepEqual(store.clear(), { cleared: 1 });
  assert.deepEqual(store.list(), []);
  assert.equal(store.get(run.id), null);
  assert.deepEqual(store.clear(), { cleared: 0 });
});

test('clear cannot orphan a running or stopping task', () => {
  const { store } = fixture();
  const run = store.start({ goal: 'A goal' });
  assert.throws(
    () => store.clear(),
    (error) => error.status === 409,
  );
  assert.equal(store.get(run.id).status, 'running');
  store.stop(run.id);
  assert.throws(
    () => store.clear(),
    (error) => error.status === 409,
  );
  assert.equal(store.get(run.id).status, 'stopping');
});
