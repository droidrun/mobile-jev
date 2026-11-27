import asyncio
from threading import Event
from unittest.mock import Mock, patch

import pytest
from mobilerun.config.user_config import TracingConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from mobilerun_agents.agent.analytics import tracing


@pytest.mark.parametrize("interruption", [asyncio.CancelledError, KeyboardInterrupt])
def test_interrupted_run_does_not_start_a_slow_trace_flush(monkeypatch, interruption):
    started = Event()
    release = Event()

    class SlowExporter(InMemorySpanExporter):
        def export(self, spans):
            started.set()
            release.wait(0.25)
            return super().export(spans)

    processor = BatchSpanProcessor(SlowExporter(), schedule_delay_millis=60_000)
    provider = TracerProvider(shutdown_on_exit=False, active_span_processor=processor)
    monkeypatch.setattr(tracing, "_setup", lambda _: tracing._Pipeline(("fake",), provider, None))

    async def exercise():
        async with tracing.trace_run(TracingConfig(enabled=True), "interrupted"):
            if interruption is asyncio.CancelledError:
                asyncio.current_task().cancel()
                await asyncio.sleep(0)
            raise interruption

    try:
        with pytest.raises(interruption):
            asyncio.run(exercise())
        assert not started.is_set()
    finally:
        release.set()
        provider.shutdown()


def test_phoenix_only_records_scoped_llamaindex_spans(monkeypatch):
    from llama_index.core.instrumentation import get_dispatcher
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    exported = InMemorySpanExporter()
    monkeypatch.setattr(
        tracing, "_phoenix_processor", lambda endpoint: SimpleSpanProcessor(exported)
    )
    monkeypatch.setattr(tracing, "_pipeline", None)
    dispatcher = get_dispatcher(__name__)

    @dispatcher.span
    def select_action(prompt: str) -> str:
        return prompt

    async def exercise():
        async with tracing.trace_run(TracingConfig(enabled=True), "first") as run:
            select_action("secret")
            run.finish(True, 2)
        select_action("outside")
        async with tracing.trace_run(TracingConfig(enabled=False), "disabled"):
            select_action("disabled")

    try:
        asyncio.run(exercise())
        spans = exported.get_finished_spans()
        child, root = spans
        assert child.parent.span_id == root.context.span_id
        assert child.attributes["session.id"] == "first"
        assert root.attributes["agent.success"] is True and root.attributes["agent.steps"] == 2
        assert "secret" not in str(child.attributes)
    finally:
        LlamaIndexInstrumentor().uninstrument()


@pytest.mark.parametrize("backend", ["phoenix", "langfuse"])
def test_failed_span_exports_type_without_private_exception_text(monkeypatch, backend):
    from llama_index.core.instrumentation import get_dispatcher
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor

    exported = InMemorySpanExporter()
    monkeypatch.setattr(tracing, "_pipeline", None)
    clients = []
    if backend == "phoenix":
        monkeypatch.setattr(tracing, "_phoenix_processor", lambda _: SimpleSpanProcessor(exported))
    else:
        langfuse = pytest.importorskip("langfuse")
        sdk = langfuse.Langfuse

        def local_client(**kwargs):
            client = sdk(**kwargs, span_exporter=exported)
            clients.append(client)
            return client

        monkeypatch.setattr(langfuse, "Langfuse", local_client)

    sentinel = "private-exception-sentinel"

    @get_dispatcher(__name__).span
    def fail():
        raise ValueError(sentinel)

    async def exercise():
        config = TracingConfig(
            enabled=True,
            provider=backend,
            langfuse_public_key="pk-local-fail-test",
            langfuse_secret_key="sk-local-fail-test",
        )
        async with tracing.trace_run(config, "failed-run"):
            fail()

    try:
        with pytest.raises(ValueError, match=sentinel):
            asyncio.run(exercise())
        spans = exported.get_finished_spans()
        child, root = spans
        assert child.parent.span_id == root.context.span_id
        assert child.status.status_code is StatusCode.ERROR
        assert any(
            e.name == "exception" and e.attributes.get("exception.type") == "ValueError"
            for e in child.events
        )
        for span in spans:
            assert sentinel not in span.to_json()
            for event in span.events:
                if event.name == "exception":
                    assert "exception.message" not in event.attributes
                    assert "exception.stacktrace" not in event.attributes
    finally:
        LlamaIndexInstrumentor().uninstrument()
        for client in clients:
            client.shutdown()


def test_screenshot_only_sent_when_explicitly_enabled():
    pytest.importorskip("langfuse")
    from langfuse.api import MediaContentType
    from langfuse.media import LangfuseMedia

    client = Mock()
    with patch("langfuse.media.LangfuseMedia", wraps=LangfuseMedia) as media:
        for enabled in [False, True]:
            tracing.RunTrace(None, client, enabled).screenshot(b"png")
        media.assert_called_once_with(content_bytes=b"png", content_type=MediaContentType.IMAGE_PNG)
    client.create_event.assert_called_once()
    assert type(client.create_event.call_args.kwargs["input"]) is LangfuseMedia


def test_langfuse_exports_root_and_llm_without_replacing_host_provider(monkeypatch):
    langfuse = pytest.importorskip("langfuse")
    from llama_index.core.llms import CompletionResponse, MockLLM
    from llama_index.core.llms.callbacks import llm_completion_callback
    from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    exporter = InMemorySpanExporter()
    sdk = langfuse.Langfuse
    clients = []

    class LocalLLM(MockLLM):
        @llm_completion_callback()
        def complete(self, prompt, **kwargs):
            return CompletionResponse(
                text=prompt, raw={"usage": {"prompt_tokens": 5, "completion_tokens": 2}}
            )

    def local_client(**kwargs):
        client = sdk(**kwargs, span_exporter=exporter)
        clients.append(client)
        return client

    monkeypatch.setattr(langfuse, "Langfuse", local_client)
    monkeypatch.setattr(tracing, "_pipeline", None)
    config = TracingConfig(
        enabled=True,
        provider="langfuse",
        langfuse_public_key="pk-local-analytics-test",
        langfuse_secret_key="sk-local-test",
    )
    host = TracerProvider()
    original = trace.get_tracer_provider()

    async def execute(run_id):
        async with tracing.trace_run(config, run_id) as run:
            await asyncio.sleep(0)
            LocalLLM().complete("sentinel secret")
            run.finish(True, 1)

    async def both():
        await asyncio.gather(execute("first"), execute("second"))

    try:
        with host.get_tracer("host").start_as_current_span("host") as outer:
            asyncio.run(both())
            assert trace.get_current_span() is outer
        spans = exporter.get_finished_spans()
        roots = [span for span in spans if span.name == "agent.run"]
        assert len(roots) == 2
        assert len({s.context.trace_id for s in roots}) == 2
        assert {s.attributes["session.id"] for s in roots} == {"first", "second"}
        assert all(s.parent is None for s in roots)
        for root in roots:
            assert any(
                s.parent
                and s.parent.span_id == root.context.span_id
                and s.attributes.get("openinference.span.kind") == "LLM"
                and s.attributes.get("llm.token_count.prompt") == 5
                and s.attributes.get("llm.token_count.completion") == 2
                for s in spans
            )
        assert "sentinel secret" not in str([s.attributes for s in spans])
        assert trace.get_tracer_provider() is original
    finally:
        LlamaIndexInstrumentor().uninstrument()
        for client in clients:
            client.shutdown()
        host.shutdown()
