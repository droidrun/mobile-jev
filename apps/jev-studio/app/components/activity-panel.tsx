'use client';
import {
  ArrowDown,
  ArrowRight,
  ArrowUpRight,
  Check,
  Clock3,
  CornerDownLeft,
  History,
  Loader2,
  MousePointer2,
  Play,
  RotateCcw,
  Square,
  Terminal,
} from 'lucide-react';
import { actionDetail, operationName, statusLabels } from './studio-types';
import type { StudioState } from './use-studio';
export function ActivityPanel({ state }: { state: StudioState }) {
  const {
    panel,
    setPanel,
    actions,
    run,
    busy,
    feed,
    decisions,
    eventConnected,
    runs,
    setRun,
    setGoal,
    clear,
    clearing,
    submitting,
    activeAnywhere,
  } = state;
  return (
    <div className="activity panel">
      <div className="activity-heading">
        <div className="tabs">
          <button
            onClick={() => setPanel('activity')}
            className={panel === 'activity' ? 'active' : ''}
          >
            <Terminal size={15} />
            Live activity{actions.length > 0 && <span>{actions.length}</span>}
          </button>
          <button
            onClick={() => setPanel('history')}
            className={panel === 'history' ? 'active' : ''}
          >
            <History size={15} />
            Recent runs
          </button>
        </div>
        <div className="activity-controls">
          <span className={`task-status ${run?.status || ''}`}>
            <i />
            {run ? statusLabels[run.status] : 'Standby'}
          </span>
          <button
            className="clear-button"
            onClick={() => void clear()}
            disabled={activeAnywhere || submitting || clearing || (!runs.length && !run)}
            title={
              activeAnywhere
                ? 'Stop the task before clearing'
                : 'Clear recent runs and reset the timer'
            }
          >
            <RotateCcw size={12} className={clearing ? 'spin' : undefined} />
            {clearing ? 'Clearing…' : 'Clear'}
          </button>
        </div>
      </div>
      {panel === 'activity' ? (
        <div className="activity-body" ref={feed} aria-live="polite">
          {!run ? (
            <div className="empty-state">
              <div className="empty-orbit">
                <MousePointer2 size={23} strokeWidth={1.3} />
                <span />
                <span />
              </div>
              <h3>A little intent goes a long way.</h3>
              <p>
                Your agent&apos;s actions will appear here,
                <br />
                one decision at a time.
              </p>
              <div className="empty-steps">
                <span>Observe</span>
                <ArrowRight size={11} />
                <span>Decide</span>
                <ArrowRight size={11} />
                <span>Act</span>
              </div>
            </div>
          ) : (
            <>
              <div className="run-start">
                <div className="timeline-dot">
                  <Play size={9} fill="currentColor" />
                </div>
                <div>
                  <strong>Task started</strong>
                  <span>
                    {new Date(run.startedAt).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                      second: '2-digit',
                    })}
                  </span>
                </div>
              </div>
              {actions.map((action, index) => {
                const decision = decisions.findLast(
                  (entry) => entry.step === action.step && entry.at <= action.at,
                );
                const Icon =
                  action.operation === 'WAIT'
                    ? Clock3
                    : action.operation?.startsWith('SCROLL')
                      ? ArrowDown
                      : action.operation === 'TYPE_TEXT'
                        ? CornerDownLeft
                        : MousePointer2;
                return (
                  <div className="action-row" key={action.sequence}>
                    <span className="action-number">{String(index + 1).padStart(2, '0')}</span>
                    <div className="action-icon">
                      <Icon size={16} />
                    </div>
                    <div className="action-copy">
                      <strong>{operationName(action.operation)}</strong>
                      <p>{actionDetail(action)}</p>
                    </div>
                    <div className="action-meta">
                      <span>
                        {decision?.latencyMs ? `${Math.round(decision.latencyMs)} ms` : '—'}
                      </span>
                      <small>
                        <Check size={10} />
                        Executed
                      </small>
                    </div>
                  </div>
                );
              })}
              {busy && (
                <div className="thinking-row">
                  <Loader2 size={15} className="spin" />
                  <span>
                    {run.status === 'stopping'
                      ? 'Finishing the current action…'
                      : !eventConnected
                        ? 'Reconnecting to live activity…'
                        : 'Jev is observing and deciding…'}
                  </span>
                  <span className="thinking-dots">···</span>
                </div>
              )}
              {!busy && (
                <div className={`result-row ${run.status}`}>
                  <div>
                    {run.status === 'succeeded' ? <Check size={18} /> : <Square size={15} />}
                  </div>
                  <span>
                    <strong>{statusLabels[run.status]}</strong>
                    <p>
                      {run.error ||
                        run.events.findLast((e) => e.type === 'result')?.reason ||
                        (run.status === 'succeeded'
                          ? 'Jev reports the goal is complete. The device is yours.'
                          : run.status === 'stopped'
                            ? 'The task has stopped. You can start a new goal.'
                            : `The agent stopped: ${(run.outcome || 'unknown').replaceAll('_', ' ')}.`)}
                    </p>
                  </span>
                </div>
              )}
            </>
          )}
        </div>
      ) : (
        <div className="history-list">
          {runs.length ? (
            runs.map((item) => (
              <button
                className="history-item"
                key={item.id}
                disabled={busy && item.id !== run?.id}
                onClick={() => {
                  setRun(item);
                  setGoal(item.goal);
                  setPanel('activity');
                }}
              >
                <div>
                  <span className={`history-dot ${item.status}`} />
                  <strong>{item.goal}</strong>
                  <ArrowUpRight size={14} />
                </div>
                <p>
                  {statusLabels[item.status]}
                  <span>
                    {new Date(item.startedAt).toLocaleTimeString([], {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                    {item.endedAt
                      ? ` · ${((item.endedAt - item.startedAt) / 1000).toFixed(2)}s`
                      : ''}
                  </span>
                </p>
              </button>
            ))
          ) : (
            <div className="empty-history">
              <History size={25} />
              <h3>A fresh start.</h3>
              <p>Your recent runs will live here.</p>
            </div>
          )}
        </div>
      )}
      <div className="activity-footer">
        <span>
          <span className={`tiny-dot ${busy ? 'orange' : ''}`} />
          {busy ? 'Streaming agent events' : 'All actions selected by Jev'}
        </span>
        <span>{run ? run.id.slice(0, 8) : 'SESSION READY'}</span>
      </div>
    </div>
  );
}
