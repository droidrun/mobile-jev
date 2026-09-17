import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { root } from './env.mjs';

test('CLI help works from outside the repository without credentials', () => {
  const help = execFileSync(process.execPath, [join(root, 'scripts/run.mjs'), '--help'], {
    cwd: '/tmp',
    env: { PATH: process.env.PATH },
    encoding: 'utf8',
  });
  assert.match(help, /pnpm agent COMMAND/);
  assert.match(help, /--execute/);
});

test('configuration template has no embedded account keys or device ID', () => {
  const template = readFileSync(join(root, '.env.example'), 'utf8');
  for (const key of ['MOBILERUN_API_KEY', 'MOBILERUN_DEVICE_ID', 'TYPESAFE_API_KEY'])
    assert.match(template, new RegExp(`^${key}=$`, 'm'));
  assert.match(template, /^MOBILERUN_BASE_URL=https:\/\/api.mobilerun.ai\/v1$/m);
});
