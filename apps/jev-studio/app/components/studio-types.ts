import { Smartphone, ArrowUpRight, Command } from 'lucide-react';

export type Device = {
  id: string;
  name: string;
  state: string;
  streamUrl: string | null;
  streamToken: string | null;
  environment: string;
};
export type Event = {
  type: 'decision' | 'action' | 'result' | 'error';
  at: number;
  step?: number;
  operation?: string;
  label?: string;
  confidence?: number;
  targetConfidence?: number;
  latencyMs?: number;
  model?: string;
  outcome?: string;
  reason?: string;
  sequence: number;
};
export type Run = {
  id: string;
  goal: string;
  maxSteps: number;
  status: string;
  startedAt: number;
  endedAt: number | null;
  events: Event[];
  outcome: string | null;
  error: string | null;
};
export const isActive = (run: Run | null) =>
  Boolean(run && ['running', 'stopping'].includes(run.status));
export const statusLabels: Record<string, string> = {
  running: 'In progress',
  stopping: 'Stopping',
  succeeded: 'Completed',
  stopped: 'Stopped',
  blocked: 'Needs attention',
  failed: 'Run failed',
};
export const examples = [
  {
    label: 'Find Android version',
    goal: 'Show the Android version of this device in Android Settings and leave it visible.',
    icon: Smartphone,
  },
  {
    label: 'Enable dark theme',
    goal: 'Turn on dark theme in Android Settings. Stop with the dark theme switch visible and on.',
    icon: ArrowUpRight,
  },
  {
    label: 'Open Chrome',
    goal: 'Open Chrome and leave the browser visible on screen.',
    icon: Command,
  },
];

export function clock(ms: number) {
  const seconds = Math.floor(ms / 1000);
  return {
    main: `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`,
    fraction: String(Math.floor((ms % 1000) / 10)).padStart(2, '0'),
  };
}

export function actionDetail(event: Event) {
  const label = event.label || event.operation || 'Action';
  if (event.operation === 'TAP') return label.replace(/^Tap /, '').replace(/\.$/, '');
  if (event.operation?.startsWith('SCROLL')) return 'Move through the current screen';
  if (event.operation === 'TYPE_TEXT') {
    try {
      return `“${JSON.parse(label).text}”`;
    } catch {
      return label;
    }
  }
  return (
    {
      HOME: 'Return to the home screen',
      BACK: 'Go back one screen',
      WAIT: 'Give the app a moment to load',
      ENTER: 'Submit the current input',
    }[event.operation || ''] || label
  );
}

export function operationName(operation?: string) {
  return (
    (
      {
        TAP: 'Tap',
        OPEN_APP: 'Open app',
        TYPE_TEXT: 'Type text',
        SCROLL_DOWN: 'Scroll down',
        SCROLL_UP: 'Scroll up',
        SCROLL_LEFT: 'Scroll left',
        SCROLL_RIGHT: 'Scroll right',
        HOME: 'Home',
        BACK: 'Back',
        WAIT: 'Wait',
        ENTER: 'Enter',
      } as Record<string, string>
    )[operation || ''] ||
    operation ||
    'Action'
  );
}
