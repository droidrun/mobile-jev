import { studio, checkRequest, StudioError } from '../../../../server/store.mjs';
import { MobilerunDevice } from '../../../../../../scripts/mobile-agent/device.mjs';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
const json = (body: unknown, status = 200) =>
  Response.json(body, { status, headers: { 'Cache-Control': 'no-store' } });

async function handle(request: Request, context: { params: Promise<{ path: string[] }> }) {
  try {
    checkRequest(request);
    const { path } = await context.params;
    if (request.method === 'GET' && path.join('/') === 'device') {
      const client = new MobilerunDevice();
      const d = await client.api(client.path());
      return json({
        id: d.id,
        name: d.name,
        state: d.state,
        streamUrl: d.streamUrl || null,
        streamToken: d.streamToken || null,
        environment: new URL(client.baseUrl).hostname.includes('staging') ? 'Staging' : 'Cloud',
      });
    }
    if (request.method === 'GET' && path.join('/') === 'runs') return json(studio.list());
    if (request.method === 'POST' && path.join('/') === 'runs/clear') return json(studio.clear());
    if (request.method === 'POST' && path.join('/') === 'runs') {
      if (!process.env.MOBILERUN_API_KEY && !process.env.MOBILERUN_CLOUD_API_KEY)
        throw new StudioError('Mobilerun credentials are missing on the server.', 503);
      if (!process.env.TYPESAFE_API_KEY)
        throw new StudioError('TypeSafe credentials are missing on the server.', 503);
      const text = await request.text();
      if (text.length > 8000) throw new StudioError('Task input is too long.');
      return json(studio.start(JSON.parse(text)), 201);
    }
    if (path[0] === 'runs' && path[1]) {
      if (request.method === 'POST' && path[2] === 'stop') return json(studio.stop(path[1]));
      const run = studio.get(path[1]);
      if (!run) throw new StudioError('Run not found.', 404);
      if (request.method === 'GET' && path[2] === 'events') {
        const encoder = new TextEncoder();
        let cleanup = () => {};
        const body = new ReadableStream({
          start(controller) {
            let closed = false;
            const send = (state: unknown) => {
              if (!closed) controller.enqueue(encoder.encode(`data: ${JSON.stringify(state)}\n\n`));
            };
            const unsubscribe = studio.subscribe(path[1], send);
            const heartbeat = setInterval(() => {
              if (!closed) controller.enqueue(encoder.encode(': heartbeat\n\n'));
            }, 15_000);
            cleanup = () => {
              closed = true;
              clearInterval(heartbeat);
              unsubscribe();
            };
            request.signal.addEventListener('abort', cleanup, { once: true });
            send(run);
          },
          cancel() {
            cleanup();
          },
        });
        return new Response(body, {
          headers: {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache, no-transform',
            Connection: 'keep-alive',
          },
        });
      }
      if (request.method === 'GET' && path.length === 2) return json(run);
    }
    return json({ error: 'Not found.' }, 404);
  } catch (error) {
    return json(
      {
        error:
          error instanceof StudioError
            ? error.message
            : error instanceof SyntaxError
              ? 'Invalid request.'
              : 'Mobilerun is unavailable. Check your server configuration with pnpm doctor.',
      },
      error instanceof StudioError ? error.status : error instanceof SyntaxError ? 400 : 502,
    );
  }
}
export const GET = handle;
export const POST = handle;
