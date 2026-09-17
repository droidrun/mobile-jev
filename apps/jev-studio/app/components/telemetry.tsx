import { Clock3 } from 'lucide-react';
import type { StudioState } from './use-studio';
export function Telemetry({ state }: { state: StudioState }) {
  const { busy, timer, actions, run, modelLatency } = state;
  return (
    <div className={`telemetry ${busy ? 'running' : ''}`}>
      <div className="clock-block">
        <div className="metric-label">
          <Clock3 size={12} />
          TASK TIME{busy && <span className="clock-live" />}
        </div>
        <div className="task-clock" aria-label="Task elapsed time">
          <span>{timer.main}</span>
          <span className="clock-fraction">.{timer.fraction}</span>
        </div>
      </div>
      <div className="small-metric">
        <div className="metric-label">ACTIONS</div>
        <strong>{String(actions.length).padStart(2, '0')}</strong>
        <span>{run ? `of ${run.maxSteps} allowed` : 'ready to execute'}</span>
      </div>
      <div className="small-metric">
        <div className="metric-label">MODEL LATENCY</div>
        <strong>
          {modelLatency ?? '—'}
          {modelLatency !== null && <small>ms</small>}
        </strong>
        <span>average decision</span>
      </div>
    </div>
  );
}
