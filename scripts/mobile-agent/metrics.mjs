import { curlRequest } from './http.mjs';

export function measuredRequest({ service, metrics, onMetric = () => {}, request = curlRequest }) {
  return async (input) => {
    const started = performance.now();
    const url = new URL(input.url);
    const endpoint = `${input.method || 'GET'} ${url.pathname.replace(/\/devices\/[^/]+/, '/devices/:id')}${url.search}`;
    let bytes, error;
    try {
      bytes = await request(input);
      return bytes;
    } catch (failure) {
      error = failure;
      throw failure;
    } finally {
      const metric = {
        service,
        endpoint,
        ...bytes?.timing,
        wallMs: Math.round((performance.now() - started) * 10) / 10,
        ok: !error,
      };
      metrics.push(metric);
      await onMetric(metric);
    }
  };
}

function stats(values) {
  const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
  const round = (n) => Math.round(n * 10) / 10;
  return {
    count: sorted.length,
    total: round(sorted.reduce((sum, n) => sum + n, 0)),
    median: sorted.length
      ? round(
          (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2,
        )
      : null,
    p95: sorted.length ? sorted[Math.ceil(sorted.length * 0.95) - 1] : null,
  };
}

export function summarizeMetrics(metrics) {
  const groups = new Map();
  for (const metric of metrics) {
    const key = `${metric.service} ${metric.endpoint}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(metric);
  }
  return [...groups].map(([endpoint, group]) => ({
    endpoint,
    failures: group.filter((m) => !m.ok).length,
    ...Object.fromEntries(
      ['wallMs', 'dnsMs', 'tcpMs', 'tlsMs', 'responseWaitMs', 'downloadMs'].map((key) => [
        key,
        stats(group.map((m) => m[key])),
      ]),
    ),
  }));
}
