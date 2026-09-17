import { loadEnvironment } from './env.mjs';
loadEnvironment();
await import('./mobile-agent/cli.mjs');
