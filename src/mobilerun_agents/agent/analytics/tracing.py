"""OpenInference owns LlamaIndex spans; a private provider limits export to opted-in runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING

from mobilerun.config.env import secret_value
from mobilerun.config.user_config import TracingConfig
from opentelemetry.context import Context, attach, detach
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import (
    Event,
    ReadableSpan,
    SpanProcessor,
    SynchronousMultiSpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_OFF, ALWAYS_ON, Sampler, SamplingResult
from opentelemetry.trace import Link, Span, SpanKind, Status, TraceState, get_current_span
from opentelemetry.util.types import Attributes

from mobilerun_agents.config.settings import (
    DEFAULT_LANGFUSE_BASE_URL,
    get_agent_secrets,
    get_agent_settings,
)

if TYPE_CHECKING:
    from langfuse import Langfuse

_active = ContextVar("agent_tracing_enabled", default=False)
_lock = Lock()
_log = logging.getLogger(__name__)


class _PrivateSpanProcessor(SynchronousMultiSpanProcessor):
    """Strip exception text before either backend's processors receive the finished span."""

    def on_end(self, span: ReadableSpan) -> None:
        if span.status.description or any(event.name == "exception" for event in span.events):
            events = [
                (
                    Event(
                        event.name,
                        {
                            k: v
                            for k, v in (event.attributes or {}).items()
                            if k == "exception.type"
                        },
                        event.timestamp,
                    )
                    if event.name == "exception"
                    else event
                )
                for event in span.events
            ]
            span = ReadableSpan(
                name=span.name,
                context=span.get_span_context(),
                parent=span.parent,
                resource=span.resource,
                attributes=span.attributes,
                events=events,
                links=span.links,
                kind=span.kind,
                status=Status(span.status.status_code),
                start_time=span.start_time,
                end_time=span.end_time,
                instrumentation_scope=span.instrumentation_scope,
            )
        super().on_end(span)


class _RunSampler(Sampler):
    def should_sample(
        self,
        parent_context: Context | None,
        trace_id: int,
        name: str,
        kind: SpanKind | None = None,
        attributes: Attributes = None,
        links: Sequence[Link] | None = None,
        trace_state: TraceState | None = None,
    ) -> SamplingResult:
        sampler = ALWAYS_ON if _active.get() else ALWAYS_OFF
        return sampler.should_sample(
            parent_context, trace_id, name, kind, attributes, links, trace_state
        )

    def get_description(self) -> str:
        return "Mobilerun opted-in run"


@dataclass
class _Pipeline:
    destination: tuple[str, ...]
    provider: TracerProvider
    client: Langfuse | None

    async def flush(self) -> None:
        timeout = get_agent_settings().MOBILERUN_FINALIZE_TELEMETRY_SECONDS
        try:
            if self.client is not None:
                await asyncio.wait_for(asyncio.to_thread(self.client.flush), timeout)
            else:
                await asyncio.wait_for(
                    asyncio.to_thread(self.provider.force_flush, int(timeout * 1000)), timeout
                )
        except Exception:
            _log.warning("Unable to flush traces before deadline")


_pipeline: _Pipeline | None = None


def _phoenix_processor(endpoint: str) -> SpanProcessor:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    return BatchSpanProcessor(OTLPSpanExporter(endpoint.rstrip("/") + "/v1/traces"))


def _setup(config: TracingConfig) -> _Pipeline | None:
    from openinference.instrumentation import TraceConfig
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    global _pipeline
    settings = get_agent_settings()
    backend = config.provider.lower()
    resource = Resource({})
    if backend == "phoenix":
        destination = (backend, settings.PHOENIX_URL, settings.PHOENIX_PROJECT_NAME)
        if settings.PHOENIX_PROJECT_NAME:
            resource = Resource({"openinference.project.name": settings.PHOENIX_PROJECT_NAME})
    elif backend == "langfuse":
        secrets = get_agent_secrets()
        key = config.langfuse_public_key or secret_value(secrets.LANGFUSE_PUBLIC_KEY)
        secret = config.langfuse_secret_key or secret_value(secrets.LANGFUSE_SECRET_KEY)
        if not key or not secret:
            _log.warning("Langfuse credentials missing; skipping trace")
            return None
        url = config.langfuse_host or settings.langfuse_url or DEFAULT_LANGFUSE_BASE_URL
        destination = (backend, url, key)
    else:
        _log.warning("Unknown tracing provider; skipping trace")
        return None
    with _lock:
        if _pipeline is not None and _pipeline.destination == destination:
            return _pipeline
        instrumentor = LlamaIndexInstrumentor()
        if instrumentor.is_instrumented_by_opentelemetry or _pipeline is not None:
            _log.warning(
                "LlamaIndex already instrumented for a different destination; skipping trace"
            )
            return None
        provider = TracerProvider(
            sampler=_RunSampler(),
            resource=resource,
            # The OTel superclass constructor has no return annotation.
            active_span_processor=_PrivateSpanProcessor(),  # type: ignore[no-untyped-call]
        )
        client = None
        if backend == "langfuse":
            from langfuse import Langfuse

            client = Langfuse(
                public_key=key, secret_key=secret, base_url=url, tracer_provider=provider
            )
        else:
            provider.add_span_processor(_phoenix_processor(settings.PHOENIX_URL))
        instrumentor.instrument(
            tracer_provider=provider,
            config=TraceConfig(
                hide_inputs=True, hide_outputs=True, hide_llm_invocation_parameters=True
            ),
        )
        _pipeline = _Pipeline(destination, provider, client)
        return _pipeline


@dataclass
class RunTrace:
    _root: Span | None
    _client: Langfuse | None
    _screenshots: bool

    def screenshot(self, png: bytes) -> None:
        if self._client is not None and self._screenshots and png:
            from langfuse.api import MediaContentType
            from langfuse.media import LangfuseMedia

            try:
                self._client.create_event(
                    name="screenshot",
                    input=LangfuseMedia(
                        content_bytes=png, content_type=MediaContentType("image/png")
                    ),
                )
            except Exception:
                _log.warning("Unable to record screenshot")

    def finish(self, success: bool, steps: int) -> None:
        if self._root is not None:
            self._root.set_attributes({"agent.success": success, "agent.steps": steps})


@asynccontextmanager
async def trace_run(config: TracingConfig, run_id: str) -> AsyncIterator[RunTrace]:
    """Enter before scheduling workflow.run(); exit after draining and awaiting its handler."""
    async with AsyncExitStack() as scope:
        token = _active.set(False)
        scope.callback(_active.reset, token)
        run = RunTrace(None, None, False)
        if config.enabled:
            try:
                pipeline = _setup(config)
                if pipeline is not None:
                    from openinference.instrumentation import using_attributes

                    scope.push_async_callback(pipeline.flush)
                    _active.set(True)
                    scope.callback(detach, attach(Context()))
                    session = config.langfuse_session_id or run_id
                    user = config.langfuse_user_id
                    scope.enter_context(using_attributes(session_id=session, user_id=user))
                    if pipeline.client is not None:
                        from langfuse import propagate_attributes

                        scope.enter_context(
                            pipeline.client.start_as_current_observation(
                                name="agent.run", as_type="agent"
                            )
                        )
                        root = get_current_span()
                        scope.enter_context(propagate_attributes(session_id=session, user_id=user))
                    else:
                        root = scope.enter_context(
                            pipeline.provider.get_tracer("mobilerun.agent").start_as_current_span(
                                "agent.run",
                                attributes={
                                    "openinference.span.kind": "AGENT",
                                    "session.id": session,
                                    "user.id": user,
                                },
                                record_exception=False,
                                set_status_on_exception=False,
                            )
                        )
                    run = RunTrace(root, pipeline.client, config.langfuse_screenshots)
            except Exception as error:
                _active.set(False)
                _log.warning("Unable to initialize tracing (%s)", type(error).__name__)
        yield run
