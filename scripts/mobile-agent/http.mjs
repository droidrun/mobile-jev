import { spawn } from 'node:child_process';
import https from 'node:https';
import http from 'node:http';

const modelAgent = new https.Agent({ keepAlive: true, maxSockets: 2 });
const loopbackAgent = new http.Agent({ keepAlive: true, maxSockets: 2 });

// Keep the TypeSafe TLS connection warm across decisions. Mobilerun continues to use curl.
export function pooledRequest({ url, apiKey, method = 'GET', body, timeoutMs = 30_000 }) {
  const target = new URL(url);
  const loopback = target.protocol === 'http:' && target.hostname === '127.0.0.1';
  if ((target.protocol !== 'https:' && !loopback) || target.username || target.password)
    throw new Error('Model API requires HTTPS.');
  if (!apiKey || /[\r\n]/.test(apiKey)) throw new Error('A valid API key is required.');
  const payload = body === undefined ? undefined : Buffer.from(JSON.stringify(body));
  const started = performance.now();
  return new Promise((resolve, reject) => {
    let dns = 0,
      connected = 0,
      secure = 0,
      ready = 0;
    const elapsed = () => performance.now() - started;
    const req = (loopback ? http : https).request(
      target,
      {
        agent: loopback ? loopbackAgent : modelAgent,
        method,
        headers: {
          Authorization: `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
          ...(payload ? { 'Content-Length': payload.length } : {}),
        },
      },
      (res) => {
        const firstByte = elapsed();
        const chunks = [];
        let length = 0;
        res.on('data', (chunk) => {
          length += chunk.length;
          if (length > 20 * 1024 * 1024) req.destroy(new Error('Response too large'));
          else chunks.push(chunk);
        });
        res.on('error', () => {
          clearTimeout(timer);
          reject(new Error('Model response interrupted; no action executed.'));
        });
        res.on('end', () => {
          clearTimeout(timer);
          if (res.statusCode < 200 || res.statusCode >= 300) {
            reject(new Error(`Model API returned HTTP ${res.statusCode}.`));
            return;
          }
          const wall = elapsed(),
            round = (n) => Math.round(n * 10) / 10;
          const bytes = Buffer.concat(chunks);
          Object.defineProperty(bytes, 'timing', {
            value: {
              wallMs: round(wall),
              dnsMs: round(dns),
              tcpMs: round(connected - dns),
              tlsMs: round(secure - connected),
              responseWaitMs: round(Math.max(0, firstByte - ready)),
              downloadMs: round(wall - firstByte),
              totalMs: round(wall),
              reusedConnection: req.reusedSocket,
              httpVersion: res.httpVersion,
              status: res.statusCode,
            },
          });
          resolve(bytes);
        });
      },
    );
    const timer = setTimeout(() => req.destroy(new Error('timeout')), timeoutMs);
    req.on('socket', (socket) => {
      if (!socket.connecting) {
        ready = elapsed();
        return;
      }
      socket.once('lookup', () => {
        dns = elapsed();
      });
      socket.once('connect', () => {
        connected = elapsed();
        if (loopback) {
          secure = connected;
          ready = connected;
        }
      });
      socket.once('secureConnect', () => {
        secure = elapsed();
        ready = secure;
      });
    });
    req.on('error', () => {
      clearTimeout(timer);
      reject(new Error('Model API transport failed; no action executed.'));
    });
    req.end(payload);
  });
}

// Pass credentials and payload through stdin, never through shell interpolation or argv.
export function curlRequest({ url, apiKey, method = 'GET', body, timeoutMs = 30_000 }) {
  if (!apiKey || /[\r\n]/.test(apiKey)) throw new Error('A valid API key is required.');
  const target = new URL(url);
  if (
    target.protocol !== 'https:' &&
    !(target.protocol === 'http:' && target.hostname === '127.0.0.1')
  ) {
    throw new Error('API requests require HTTPS (except loopback tests).');
  }
  const quote = (value) => JSON.stringify(String(value));
  const config = [
    `url = ${quote(target.href)}`,
    `request = ${quote(method)}`,
    `header = ${quote(`Authorization: Bearer ${apiKey}`)}`,
    'header = "Content-Type: application/json"',
    ...(body === undefined ? [] : [`data = ${quote(JSON.stringify(body))}`]),
  ].join('\n');
  const started = performance.now();
  return new Promise((resolve, reject) => {
    const child = spawn(
      'curl',
      [
        '--disable',
        '--silent',
        '--show-error',
        '--config',
        '-',
        '--connect-timeout',
        '10',
        '--max-time',
        String(timeoutMs / 1000),
        '--write-out',
        '\n%{json}',
      ],
      { stdio: ['pipe', 'pipe', 'pipe'], timeout: timeoutMs + 1000 },
    );
    const chunks = [];
    let bytes = 0;
    child.on('error', () => reject(new Error('Could not run curl. Install curl and retry.')));
    child.stdin.on('error', () => {});
    child.stderr.resume(); // Do not echo remote responses, credentials, or request bodies in errors.
    child.stdout.on('data', (chunk) => {
      bytes += chunk.length;
      if (bytes > 20 * 1024 * 1024) {
        child.kill();
        reject(new Error('API response exceeds 20 MiB.'));
      } else chunks.push(chunk);
    });
    child.on('close', (code) => {
      if (code !== 0) {
        reject(
          new Error(
            `API transport failed (curl exit ${code}). Actions are not retried; observe before trying again.`,
          ),
        );
        return;
      }
      const output = Buffer.concat(chunks);
      const split = output.lastIndexOf(10);
      let metadata;
      try {
        metadata = JSON.parse(output.subarray(split + 1).toString());
      } catch {
        reject(new Error('curl returned invalid response metadata.'));
        return;
      }
      const status = metadata.http_code;
      const ms = (seconds) => Math.round(seconds * 1000 * 10) / 10;
      const timing = {
        wallMs: Math.round((performance.now() - started) * 10) / 10,
        dnsMs: ms(metadata.time_namelookup),
        tcpMs: ms(Math.max(0, metadata.time_connect - metadata.time_namelookup)),
        tlsMs: ms(Math.max(0, metadata.time_appconnect - metadata.time_connect)),
        responseWaitMs: ms(Math.max(0, metadata.time_starttransfer - metadata.time_pretransfer)),
        downloadMs: ms(Math.max(0, metadata.time_total - metadata.time_starttransfer)),
        totalMs: ms(metadata.time_total),
        httpVersion: metadata.http_version,
        status,
      };
      if (!Number.isInteger(status) || status < 200 || status >= 300) {
        reject(
          new Error(
            `API returned HTTP ${status}: ${status === 401 ? 'check your API key' : 'request failed'}.`,
          ),
        );
        return;
      }
      const bytes = output.subarray(0, split);
      Object.defineProperty(bytes, 'timing', { value: timing });
      resolve(bytes);
    });
    child.stdin.end(config + '\n');
  });
}

export function decodeJson(bytes) {
  if (!bytes.length) return null;
  try {
    return JSON.parse(bytes.toString('utf8'));
  } catch {
    throw new Error('API returned invalid JSON.');
  }
}
