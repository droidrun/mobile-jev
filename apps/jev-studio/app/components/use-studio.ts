'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import type { RemoteControlHandle } from '@mobilerun/react';
import { type Device, type Run, isActive, clock } from './studio-types';
export function useStudio() {
  const [device, setDevice] = useState<Device | null>(null);
  const [deviceError, setDeviceError] = useState('');
  const [connected, setConnected] = useState(false);
  const [goal, setGoal] = useState('');
  const [maxSteps, setMaxSteps] = useState(30);
  const [run, setRun] = useState<Run | null>(null);
  const [runs, setRuns] = useState<Run[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [clearing, setClearing] = useState(false);
  const clearedRunIds = useRef(new Set<string>());
  const [error, setError] = useState('');
  const [panel, setPanel] = useState<'activity' | 'history'>('activity');
  const [now, setNow] = useState(0);
  const [eventConnected, setEventConnected] = useState(true);
  const controls = useRef<RemoteControlHandle>(null);
  const phoneStage = useRef<HTMLDivElement>(null);
  const goalInput = useRef<HTMLTextAreaElement>(null);
  const feed = useRef<HTMLDivElement>(null);
  const busy = isActive(run);
  const activeAnywhere = runs.some((item) => isActive(item));

  const loadDevice = useCallback(async () => {
    try {
      const response = await fetch('/api/studio/device', { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setDevice(data);
      setDeviceError('');
    } catch (failure) {
      setDeviceError(
        failure instanceof Error ? failure.message : 'Could not connect to the device.',
      );
    }
  }, []);

  const updateRun = useCallback((next: Run) => {
    if (clearedRunIds.current.has(next.id)) return;
    setRun(next);
    setRuns((previous) => [next, ...previous.filter((item) => item.id !== next.id)].slice(0, 30));
  }, []);

  useEffect(() => {
    const initialLoad = setTimeout(() => void loadDevice(), 0);
    void fetch('/api/studio/runs')
      .then((response) => response.json())
      .then((data: Run[]) => {
        if (Array.isArray(data)) {
          setRuns(data);
          if (data[0]) {
            setRun(data[0]);
            setGoal(data[0].goal);
          }
        }
      })
      .catch(() => setError('Could not restore the recent runs.'));
    return () => clearTimeout(initialLoad);
  }, [loadDevice]);

  useEffect(() => {
    if (!device || device.streamUrl || deviceError) return;
    const timer = setTimeout(() => void loadDevice(), 3000);
    return () => clearTimeout(timer);
  }, [device, deviceError, loadDevice]);

  const runId = run?.id;
  useEffect(() => {
    if (!runId || !busy) return;
    const stream = new EventSource(`/api/studio/runs/${runId}/events`);
    stream.onopen = () => setEventConnected(true);
    stream.onmessage = (message) => {
      setEventConnected(true);
      updateRun(JSON.parse(message.data));
    };
    stream.onerror = () => setEventConnected(false);
    return () => stream.close();
  }, [runId, busy, updateRun]);

  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => setNow(Date.now()), 40);
    return () => clearInterval(timer);
  }, [busy]);

  const actions = run?.events.filter((event) => event.type === 'action') || [];
  const decisions = run?.events.filter((event) => event.type === 'decision') || [];
  const latest = decisions.at(-1);
  const elapsed = run ? Math.max(0, (run.endedAt ?? now) - run.startedAt) : 0;
  const timer = clock(elapsed);
  const modelLatency = decisions.length
    ? Math.round(
        decisions.reduce((total, event) => total + (event.latencyMs || 0), 0) / decisions.length,
      )
    : null;
  const model = latest?.model || 'jev-latest';

  useEffect(() => {
    if (feed.current) feed.current.scrollTop = feed.current.scrollHeight;
  }, [actions.length, latest?.sequence]);

  const start = async () => {
    if (!goal.trim() || submitting || clearing || activeAnywhere) return;
    setSubmitting(true);
    setError('');
    setPanel('activity');
    try {
      const response = await fetch('/api/studio/runs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ goal, maxSteps }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      setNow(Date.now());
      updateRun(data);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Unable to start the task.');
    } finally {
      setSubmitting(false);
    }
  };

  const stop = async () => {
    if (!run || run.status === 'stopping') return;
    try {
      const response = await fetch(`/api/studio/runs/${run.id}/stop`, { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      updateRun(data);
    } catch {
      setError('Could not stop the task. Please try again.');
    }
  };

  const clear = async () => {
    if (activeAnywhere || submitting || clearing) return;
    setClearing(true);
    try {
      const response = await fetch('/api/studio/runs/clear', { method: 'POST' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error);
      for (const item of runs) clearedRunIds.current.add(item.id);
      if (run) clearedRunIds.current.add(run.id);
      setRun(null);
      setRuns([]);
      setNow(0);
      setError('');
      setEventConnected(true);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : 'Could not clear recent runs.');
    } finally {
      setClearing(false);
    }
  };

  return {
    device,
    deviceError,
    connected,
    setConnected,
    goal,
    setGoal,
    maxSteps,
    setMaxSteps,
    run,
    setRun,
    runs,
    submitting,
    clearing,
    clear,
    error,
    setError,
    panel,
    setPanel,
    controls,
    phoneStage,
    goalInput,
    feed,
    busy,
    activeAnywhere,
    loadDevice,
    actions,
    decisions,
    latest,
    timer,
    modelLatency,
    model,
    eventConnected,
    start,
    stop,
  };
}
export type StudioState = ReturnType<typeof useStudio>;
