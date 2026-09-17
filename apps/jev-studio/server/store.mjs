import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';

export class StudioError extends Error {
  constructor(message, status = 400) {
    super(message);
    this.status = status;
  }
}

export function checkRequest(request) {
  const host = request.headers.get('host') || new URL(request.url).host;
  const name = host.split(':')[0];
  if (!['localhost', '127.0.0.1'].includes(name))
    throw new StudioError('This studio is available on localhost only.', 403);
  const origin = request.headers.get('origin');
  if (
    (origin && new URL(origin).host !== host) ||
    request.headers.get('sec-fetch-site') === 'cross-site'
  )
    throw new StudioError('Origin not allowed.', 403);
  if (request.method === 'POST' && !origin)
    throw new StudioError('A same-origin request is required.', 403);
}

export function createStore({
  spawnTask = spawn,
  now = Date.now,
  runner = resolve(process.cwd(), '../../scripts/mobile-agent/web-runner.mjs'),
} = {}) {
  const runs = new Map(),
    listeners = new Map(),
    processes = new Map();
  const snapshot = (id) => (runs.get(id) ? structuredClone(runs.get(id)) : null);
  const publish = (id) => {
    for (const callback of listeners.get(id) || []) {
      try {
        callback(snapshot(id));
      } catch {
        listeners.get(id)?.delete(callback);
      }
    }
  };
  const active = () => [...runs.values()].find((r) => ['running', 'stopping'].includes(r.status));
  return {
    list: () => [...runs.values()].reverse().map((r) => snapshot(r.id)),
    get: snapshot,
    clear() {
      if (active()) throw new StudioError('Stop the active task before clearing recent runs.', 409);
      const cleared = runs.size;
      runs.clear();
      listeners.clear();
      return { cleared };
    },
    subscribe(id, callback) {
      if (!runs.has(id)) throw new StudioError('Run not found.', 404);
      if (!listeners.has(id)) listeners.set(id, new Set());
      listeners.get(id).add(callback);
      return () => {
        listeners.get(id)?.delete(callback);
      };
    },
    start({ goal, maxSteps = 30 }) {
      if (typeof goal !== 'string' || !goal.trim() || goal.length > 4000)
        throw new StudioError('Enter a goal between 1 and 4,000 characters.');
      if (!Number.isInteger(maxSteps) || maxSteps < 1 || maxSteps > 50)
        throw new StudioError('The step limit must be between 1 and 50.');
      if (active()) throw new StudioError('A task is already running on this device.', 409);
      while (runs.size >= 30) {
        const id = runs.keys().next().value;
        runs.delete(id);
        listeners.delete(id);
      }
      const id = randomUUID();
      const run = {
        id,
        goal: goal.trim(),
        maxSteps,
        status: 'running',
        startedAt: now(),
        endedAt: null,
        events: [],
        outcome: null,
        error: null,
      };
      runs.set(id, run);
      const finish = (status, message) => {
        if (run.endedAt !== null) return;
        run.status = status;
        run.endedAt = now();
        if (message) run.error = message;
        processes.delete(id);
        publish(id);
      };
      let child;
      try {
        child = spawnTask(process.execPath, [runner], {
          stdio: ['pipe', 'pipe', 'pipe'],
          env: process.env,
          detached: process.platform !== 'win32',
        });
      } catch {
        finish('failed', 'Unable to start the task process.');
        return snapshot(id);
      }
      processes.set(id, child);
      let buffer = '',
        outcome;
      child.stdin.on('error', () => {});
      child.stderr.resume(); // Never relay process environment or raw stderr to the browser.
      child.stdout.on('data', (chunk) => {
        buffer += chunk.toString();
        if (buffer.length > 1_000_000) {
          child.kill('SIGTERM');
          finish('failed', 'Task output exceeded its limit.');
          return;
        }
        let index;
        while ((index = buffer.indexOf('\n')) >= 0) {
          const line = buffer.slice(0, index);
          buffer = buffer.slice(index + 1);
          if (!line.trim()) continue;
          try {
            const event = JSON.parse(line);
            if (!['decision', 'action', 'result', 'error'].includes(event.type)) continue;
            run.events.push({ ...event, sequence: run.events.length });
            if (event.type === 'result') {
              outcome = event.outcome;
              run.outcome = outcome;
            }
            if (event.type === 'error') run.error = event.message;
            publish(id);
          } catch {
            /* Ignore non-protocol output; a result is still required for success. */
          }
        }
      });
      child.once('error', () => finish('failed', 'Unable to start the task process.'));
      child.once('close', (code) => {
        if (run.status === 'stopping') finish('stopped');
        else if (code !== 0 || run.error || !outcome)
          finish('failed', run.error || 'The task process ended unexpectedly.');
        else finish(outcome === 'done' ? 'succeeded' : 'blocked');
      });
      child.stdin.end(JSON.stringify({ goal: run.goal, maxSteps }));
      return snapshot(id);
    },
    stop(id) {
      const run = runs.get(id),
        child = processes.get(id);
      if (!run) throw new StudioError('Run not found.', 404);
      if (!child || run.endedAt !== null) return snapshot(id);
      run.status = 'stopping';
      publish(id);
      const kill = (signal) => {
        try {
          if (process.platform !== 'win32' && child.pid) process.kill(-child.pid, signal);
          else child.kill(signal);
        } catch {
          /* Process already exited. */
        }
      };
      kill('SIGTERM');
      const timer = setTimeout(() => {
        if (processes.has(id)) kill('SIGKILL');
      }, 3000);
      timer.unref();
      return snapshot(id);
    },
  };
}

const key = Symbol.for('mobilerun.jev.studio.store');
if (!globalThis[key]) {
  const store = createStore();
  globalThis[key] = store;
  process.once('exit', () => {
    for (const run of store.list()) {
      if (['running', 'stopping'].includes(run.status)) store.stop(run.id);
    }
  });
}
export const studio = globalThis[key];
