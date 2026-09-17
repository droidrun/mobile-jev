'use client';
import dynamic from 'next/dynamic';
import { ArrowLeft, Circle, Expand, Radio, RefreshCw, Smartphone, Square } from 'lucide-react';
import type { StudioState } from './use-studio';
const DeviceStream = dynamic(
  () => import('@mobilerun/react').then((module) => module.DeviceStream),
  { ssr: false },
);
export function DevicePanel({ state }: { state: StudioState }) {
  const { connected, device, phoneStage, deviceError, controls, busy, loadDevice, setConnected } =
    state;
  return (
    <section className="device-column">
      <div className="device-heading">
        <div>
          <span className="section-index">02</span>
          <h2>Your live device</h2>
        </div>
        <span className={`connection-badge ${connected ? 'connected' : ''}`}>
          <Radio size={12} />
          {connected ? 'LIVE' : deviceError ? 'OFFLINE' : 'CONNECTING'}
        </span>
      </div>
      <div className="device-stage" ref={phoneStage}>
        <div className="stage-label">
          <Smartphone size={13} />
          <span>{device?.name || 'Your phone'}</span>
        </div>
        <button
          className="expand-button"
          title="Expand device view"
          aria-label="Expand device view"
          onClick={() => {
            if (document.fullscreenElement) void document.exitFullscreen();
            else void phoneStage.current?.requestFullscreen();
          }}
        >
          <Expand size={15} />
        </button>
        <div className="stage-grid" />
        <div className="phone-frame">
          <div className="phone-screen">
            {deviceError ? (
              <div className="stream-error">
                <Smartphone size={28} />
                <h3>Connection interrupted</h3>
                <p>{deviceError}</p>
                <button onClick={() => void loadDevice()}>
                  <RefreshCw size={13} />
                  Reconnect
                </button>
              </div>
            ) : (
              <DeviceStream
                ref={controls}
                streamUrl={device?.streamUrl || undefined}
                streamToken={device?.streamToken || undefined}
                hasControl={!busy}
                muted
                placeholderLabel={device?.state || 'Connecting to your phone'}
                onStreamHealed={loadDevice}
                onConnectionStateChange={setConnected}
              />
            )}
          </div>
        </div>
        <div className="device-navigation">
          <button
            aria-label="Android back"
            title="Back"
            disabled={!connected || busy}
            onClick={() => controls.current?.sendSystemKey('BACK')}
          >
            <ArrowLeft size={17} />
          </button>
          <button
            aria-label="Android home"
            title="Home"
            disabled={!connected || busy}
            onClick={() => controls.current?.sendSystemKey('HOME')}
          >
            <Circle size={15} />
          </button>
          <button
            aria-label="Android recent apps"
            title="Recent apps"
            disabled={!connected || busy}
            onClick={() => controls.current?.sendSystemKey('RECENT')}
          >
            <Square size={13} />
          </button>
          <span />
          <button aria-label="Reconnect stream" title="Reconnect" onClick={() => void loadDevice()}>
            <RefreshCw size={14} />
          </button>
        </div>
        <div className="stage-caption">
          <i />
          {busy ? 'Agent in control' : 'Touch & keyboard enabled'}
          <span>ANDROID · MOBILERUN</span>
        </div>
      </div>
    </section>
  );
}
