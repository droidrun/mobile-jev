'use client';
import {
  ArrowUpRight,
  ChevronDown,
  Command,
  History,
  Loader2,
  Play,
  Square,
  X,
  Zap,
} from 'lucide-react';
import { examples } from './components/studio-types';
import { useStudio } from './components/use-studio';
import { ActivityPanel } from './components/activity-panel';
import { DevicePanel } from './components/device-panel';
import { Telemetry } from './components/telemetry';
export default function Studio() {
  const state = useStudio();
  const {
    device,
    panel,
    setPanel,
    model,
    goal,
    goalInput,
    busy,
    setGoal,
    maxSteps,
    setMaxSteps,
    stop,
    run,
    submitting,
    clearing,
    start,
    activeAnywhere,
    error,
    setError,
  } = state;
  return (
    <div className="studio-shell">
      <aside className="rail" aria-label="Studio navigation">
        <a className="brand-mark" href="#workspace" aria-label="Workspace home">
          <span />
          <span />
          <span />
        </a>
        <div className="rail-items">
          <button
            className={panel === 'activity' ? 'rail-button selected' : 'rail-button'}
            onClick={() => setPanel('activity')}
            title="Workspace"
            aria-label="Workspace"
          >
            <Command size={21} />
          </button>
          <button
            className={panel === 'history' ? 'rail-button selected' : 'rail-button'}
            onClick={() => setPanel('history')}
            title="Run history"
            aria-label="Run history"
          >
            <History size={21} />
          </button>
        </div>
        <div className="rail-bottom">
          <span className="rail-label">MOBILE AGENT</span>
          <span className="avatar">J</span>
        </div>
      </aside>

      <div className="workspace" id="workspace">
        <main>
          <div className="work-grid">
            <section className="control-column">
              <div className="composer panel">
                <div className="section-heading">
                  <span className="section-index">01</span>
                  <h2>What&apos;s the mission?</h2>
                  <span className="live-model">
                    <Zap size={12} fill="currentColor" />
                    {model}
                  </span>
                </div>
                <label className="sr-only" htmlFor="goal">
                  Task goal
                </label>
                <textarea
                  id="goal"
                  ref={goalInput}
                  placeholder="Ask Jev to do something on your phone…"
                  value={goal}
                  maxLength={4000}
                  disabled={busy}
                  onChange={(event) => setGoal(event.target.value)}
                  onKeyDown={(event) => {
                    if ((event.metaKey || event.ctrlKey) && event.key === 'Enter') {
                      event.preventDefault();
                      void start();
                    }
                  }}
                />
                <div className="composer-footer">
                  <label className="step-select">
                    <span>Step limit</span>
                    <select
                      aria-label="Step limit"
                      value={maxSteps}
                      disabled={busy}
                      onChange={(event) => setMaxSteps(Number(event.target.value))}
                    >
                      <option value={10}>10 steps</option>
                      <option value={20}>20 steps</option>
                      <option value={30}>30 steps</option>
                      <option value={50}>50 steps</option>
                    </select>
                    <ChevronDown size={12} />
                  </label>
                  {busy ? (
                    <button
                      className="stop-button"
                      onClick={() => void stop()}
                      disabled={run?.status === 'stopping'}
                    >
                      <Square size={13} fill="currentColor" />
                      {run?.status === 'stopping' ? 'Stopping…' : 'Stop task'}
                    </button>
                  ) : (
                    <button
                      className="run-button"
                      onClick={() => void start()}
                      disabled={
                        !goal.trim() ||
                        submitting ||
                        clearing ||
                        activeAnywhere ||
                        device?.state !== 'ready'
                      }
                    >
                      {submitting ? (
                        <Loader2 size={15} className="spin" />
                      ) : (
                        <Play size={13} fill="currentColor" />
                      )}
                      Run task<span className="shortcut">⌘ ↵</span>
                    </button>
                  )}
                </div>
                {error && (
                  <div className="inline-error" role="alert">
                    {error}
                    <button aria-label="Dismiss error" onClick={() => setError('')}>
                      <X size={14} />
                    </button>
                  </div>
                )}
              </div>

              <div className="suggestions">
                <span>A place to start</span>
                <div>
                  {examples.map((example) => (
                    <button
                      key={example.label}
                      disabled={busy}
                      onClick={() => {
                        setGoal(example.goal);
                        goalInput.current?.focus();
                      }}
                    >
                      <example.icon size={13} />
                      {example.label}
                      <ArrowUpRight size={11} />
                    </button>
                  ))}
                </div>
              </div>

              <Telemetry state={state} />
              <ActivityPanel state={state} />
            </section>

            <DevicePanel state={state} />
          </div>
        </main>
      </div>
    </div>
  );
}
