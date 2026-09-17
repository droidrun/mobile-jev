import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';

import { loadEnvironment } from '../../scripts/env.mjs';
loadEnvironment();
const child = spawn(
  process.execPath,
  [
    fileURLToPath(new URL('./node_modules/next/dist/bin/next', import.meta.url)),
    process.argv[2] || 'dev',
    '--hostname',
    '127.0.0.1',
    '--port',
    process.env.STUDIO_PORT || '3040',
  ],
  { stdio: 'inherit', env: process.env, cwd: fileURLToPath(new URL('.', import.meta.url)) },
);
child.on('exit', (code) => {
  process.exitCode = code || 0;
});
process.on('SIGTERM', () => child.kill('SIGTERM'));
process.on('SIGINT', () => child.kill('SIGINT'));
