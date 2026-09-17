import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export const root = fileURLToPath(new URL('../', import.meta.url));
export function loadEnvironment() {
  // Existing exported variables win; .env.local overrides .env. No key is printed.
  for (const name of ['.env.local', '.env']) {
    const path = new URL(`../${name}`, import.meta.url);
    if (existsSync(path)) process.loadEnvFile(path);
  }
}
