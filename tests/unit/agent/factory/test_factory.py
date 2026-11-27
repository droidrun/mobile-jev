import json
import logging
import re
import socket
import traceback
from importlib.metadata import version

import google.genai
import httpx
import httpx2
import pytest
from click.testing import CliRunner
from llama_index.core.llms import ChatMessage
from llama_index.llms.anthropic import Anthropic
from llama_index.llms.openai.responses import OpenAIResponses
from mobilerun.cli.app import cli
from openai import RateLimitError

from mobilerun_agents.agent.action_selector import LlamaIndexLLMSelector
from mobilerun_agents.agent.action_selector.llm.calls import achat
from mobilerun_agents.agent.action_selector.llm.factory import build_llm, capabilities
from mobilerun_agents.agent.action_selector.loader import default_setup
from mobilerun_agents.agent.auth import store
from mobilerun_agents.config.loader import ConfigLoader
from mobilerun_agents.config.user_config import LLMProfile, MobileConfig, SetupConfig
from mobilerun_agents.errors import UnknownModel
from tests.unit.agent.action_selector.fixtures import Turn
from tests.unit.device_fixtures import MemoryDevice


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture(autouse=True)
def block_network(monkeypatch):
    def blocked(*args, **kwargs):
        pytest.fail("Factory unit tests must not open a connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)


@pytest.mark.parametrize(
    "provider,model,fmt,side,normalized",
    [
        ("deepseek", "deepseek-flash", "chat_completions", None, False),
        ("anthropic", "claude-opus-5", None, 2576, False),
        ("anthropic", "claude-sonnet-5", None, 2576, False),
        ("anthropic", "unknown", None, 1568, False),
        ("openai", "gpt-5", "responses", None, False),
        ("grok", "grok-4", "responses", None, False),
        ("gemini", "Gemini-3.8-flash-low", None, None, True),
        ("ollama", "Gemma:latest", None, None, True),
    ],
)
def test_capabilities(provider, model, fmt, side, normalized):
    caps = capabilities(provider, model)
    assert (caps.tool_call_format, caps.max_image_side, caps.normalized_coords, caps.vision) == (
        fmt,
        side,
        normalized,
        provider in {"openai", "anthropic", "gemini"},
    )


def test_sonnet_grounding_budget_is_distinct_from_input_resolution():
    caps = capabilities("anthropic", "claude-sonnet-5")
    assert caps.grounding_max_side == 1568
    assert caps.image_token_limit == 4784


@pytest.mark.anyio
async def test_key_order_checks_saved_and_environment(monkeypatch):
    with pytest.raises(ValueError) as failure:
        await default_setup()
    assert str(failure.value) == "No model configured. Run: mobilerun agent profiles add --help"
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-router")
    assert (await default_setup()).selector.llm.model == "openai/gpt-5-mini"
    monkeypatch.setenv("GEMINI_API_KEY", "unit-google")
    assert (await default_setup()).selector.llm.model == "gemini-3.8-flash"
    store.save_key("anthropic", "unit-anthropic")
    assert (await default_setup()).selector.llm.model == "claude-sonnet-5"
    monkeypatch.setenv("OPENAI_API_KEY", "unit-openai")
    assert (await default_setup()).selector.llm.model == "gpt-5-mini"


@pytest.mark.anyio
@pytest.mark.parametrize(
    "role,model,override,expected",
    [
        ("planner", "gpt-5-mini", None, "high"),
        ("agent", "gpt-5-mini", None, "medium"),
        ("executor", "gpt-5-mini", None, "low"),
        ("planner", "gpt-5-mini", "low", "low"),
        ("planner", "gpt-4o", None, None),
    ],
)
async def test_profile_effort_on_wire(monkeypatch, role, model, override, expected):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}

    install_transport(monkeypatch, "openai", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    profile = LLMProfile(provider="openai", model=model, reasoning=override, vision="off")
    setup = (
        SetupConfig(selector="chosen")
        if role == "agent"
        else SetupConfig(planner="chosen", executor="chosen")
    )
    ConfigLoader.save(
        MobileConfig(
            profiles={"chosen": profile}, setups={"current": setup}, default_setup="current"
        )
    )
    built = await default_setup()
    llm = {
        "planner": lambda: built.planner.llm,
        "agent": lambda: built.selector.llm,
        "executor": lambda: built.executor.llm,
    }[role]()
    await achat(llm, [ChatMessage(content="test")])
    assert captured[0].get("reasoning") == ({"effort": expected} if expected else None)


@pytest.mark.anyio
async def test_anthropic_without_offline_thinking_capability(monkeypatch):
    captured = []

    def respond(request):
        captured.append(json.loads(request.content))
        return {"content": [{"type": "text", "text": "ok"}]}

    install_transport(monkeypatch, "anthropic", respond)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    ConfigLoader.save(
        MobileConfig(
            profiles={"main": LLMProfile(provider="anthropic", model="claude-sonnet-5")},
            setups={"planned": SetupConfig(planner="main", executor="main")},
            default_setup="planned",
        )
    )
    await achat((await default_setup()).planner.llm, [ChatMessage(content="test")])
    assert "thinking" not in captured[0]


@pytest.mark.anyio
async def test_openai_planner_requires_one_native_tool_on_the_wire(monkeypatch):
    from workflows import Context

    from mobilerun_agents.agent.memory.memory import build_planner_memory
    from mobilerun_agents.agent.planner import Plan
    from mobilerun_agents.agent.workflows.core.state import PlanningState
    from mobilerun_agents.config.user_config import AgentConfig
    from tests.unit.agent.planner.test_llama_index import PlanningProbe

    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return {
            "output": [
                {
                    "type": "function_call",
                    "name": "set_plan",
                    "call_id": "call_1",
                    "arguments": '{"subgoals":["Check status"]}',
                }
            ]
        }

    install_transport(monkeypatch, "openai", respond)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    ctx = Context[PlanningState](PlanningProbe())
    state = PlanningState(goal="Check status")
    await ctx.store.set_state(state)
    config = MobileConfig(
        agent=AgentConfig(streaming=False),
        profiles={"main": LLMProfile(provider="openai", model="gpt-5-mini")},
        setups={"planned": SetupConfig(planner="main", executor="main")},
        default_setup="planned",
    )
    ConfigLoader.save(config)
    memory = build_planner_memory(config, state, "android")
    try:
        assert await (await default_setup()).planner.plan(ctx, memory) == Plan(("Check status",))
    finally:
        await memory.aclose()
    [request] = requests
    assert [tool["name"] for tool in request["tools"]] == ["set_plan", "complete"]
    assert request["tool_choice"] == "required" and request["parallel_tool_calls"] is False


@pytest.mark.parametrize("provider", ["openai", "anthropic", "gemini", "grok"])
def test_subscription_models_are_not_registered(provider):
    with pytest.raises(KeyError):
        capabilities(f"{provider}-subscription", "model")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider", ["openai", "openrouter", "deepseek", "minimax", "zai", "grok", "openai_like"]
)
async def test_openai_transports_have_one_retry_owner(provider):
    llm = await build_llm(provider, "gpt-4o", "fake", max_retries=4, timeout=12)
    assert llm.max_retries == 4
    assert llm._get_credential_kwargs()["max_retries"] == 0
    assert llm._get_credential_kwargs(is_async=True)["timeout"] == 12
    assert llm.api_key == "fake"


@pytest.mark.anyio
async def test_keyless_local_ollama():
    llm = await build_llm("ollama", "gemma", timeout=7)
    assert llm.request_timeout == 7


@pytest.mark.anyio
async def test_google_key_timeout_and_retry_owner(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    llm = await build_llm(
        "gemini",
        "gemini-3.8-flash",
        timeout=1.5,
        max_retries=3,
        context_window=2048,
        max_tokens=256,
    )
    http = llm._client._api_client._http_options
    assert http.timeout == 1500
    assert http.retry_options.attempts == 1
    assert llm.max_retries == 3


@pytest.mark.anyio
async def test_anthropic_sdk_is_the_only_retry_layer():
    llm = await build_llm("anthropic", "claude-sonnet-4-6", "fake", timeout=10, max_retries=2)
    assert llm._client.max_retries == 2
    assert llm._client.timeout == 10


@pytest.mark.anyio
async def test_responses_attempt_budget_is_not_multiplied():
    attempts = []

    def limited(request):
        attempts.append(request.url.path)
        return httpx.Response(429, json={"error": {"message": "busy"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(limited)) as client:
        llm = await build_llm("openai", "gpt-4o", "fake", max_retries=1, async_http_client=client)
        with pytest.raises(RateLimitError):
            await llm.achat([ChatMessage(content="test")])
    assert attempts == ["/v1/responses"]


@pytest.mark.anyio
async def test_metadata_comes_from_installed_responses_adapter():
    llm = await build_llm("openai", "gpt-6-luna", "fake")
    assert llm.metadata.context_window == 1_050_000 and llm.metadata.is_function_calling_model


@pytest.fixture
def stale_openai_metadata(monkeypatch):
    def metadata(llm):
        raise ValueError(f"Unknown model {llm.model!r}. Private adapter details.")

    monkeypatch.setattr(OpenAIResponses, "metadata", property(metadata))


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider,adapter,model,error",
    [
        ("openai", OpenAIResponses, "gpt-6-sol", "Unknown model 'gpt-6-sol'."),
        ("anthropic", Anthropic, "claude-test", "Unknown model: claude-test."),
    ],
)
async def test_unknown_model_fails_at_build_time(monkeypatch, provider, adapter, model, error):
    def metadata(llm):
        raise ValueError(f"{error} Private adapter details.")

    monkeypatch.setattr(adapter, "metadata", property(metadata))
    with pytest.raises(UnknownModel) as failure:
        await build_llm(provider, model, "fake")
    message = str(failure.value)
    distribution = f"llama-index-llms-{provider}"
    assert f"{provider} adapter ({distribution} {version(distribution)})" in message
    assert repr(model) in message and "scripts/reinstall.sh" in message
    assert ("openai_like" in message) == (provider == "openai")
    assert "Private adapter details" not in message


def test_cli_reports_unknown_model_before_device_target(stale_openai_metadata, monkeypatch):
    def device_target(**options):
        pytest.fail("The device target must not be resolved for an unknown model")

    monkeypatch.setattr("mobilerun_agents.cli.agent.run._device_target", device_target)
    monkeypatch.setenv("OPENAI_API_KEY", "fake")
    ConfigLoader.save(
        MobileConfig(
            profiles={"default": LLMProfile(provider="openai", model="gpt-6-sol")},
        )
    )
    result = CliRunner().invoke(cli, ["agent", "run", "read settings", "emulator-5554"])
    assert result.exit_code == 1
    line = result.stderr.splitlines()[-1]
    assert "openai adapter" in line and "doesn't know model 'gpt-6-sol'" in line
    assert "scripts/reinstall.sh" in line
    assert "Private adapter details" not in result.output


@pytest.mark.anyio
async def test_provider_value_error_keeps_private_details_hidden(monkeypatch, caplog):
    async def rejected(llm, messages):
        raise ValueError("Unknown model: private body")

    caplog.set_level(logging.DEBUG, logger="mobilerun")
    llm = await build_llm("openai", "gpt-4o", "fake")
    monkeypatch.setattr(OpenAIResponses, "achat", rejected)
    with pytest.raises(RuntimeError) as failure:
        await achat(llm, [ChatMessage(content="test")])
    assert str(failure.value) == "LLM request failed (ValueError)"
    assert "private body" not in "".join(traceback.format_exception(failure.value)) + caplog.text


def install_transport(monkeypatch, provider, payload):
    async def send(_, request):
        return await transport.handle_async_request(request)

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload(request)))

    async def send2(_, request):
        return httpx2.Response(200, json=payload(request))

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", send)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", send2)
    if provider == "gemini":
        client = google.genai.Client

        def offline_client(*args, http_options, **kwargs):
            transport = httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload(request))
            )
            http_options.httpx_client = httpx.Client(transport=transport)
            http_options.httpx_async_client = httpx.AsyncClient(transport=transport)
            return client(*args, http_options=http_options, **kwargs)

        monkeypatch.setattr(google.genai, "Client", offline_client)


@pytest.mark.anyio
@pytest.mark.parametrize(
    "provider,model,api",
    [
        ("openai", "gpt-6-luna", "responses"),
        ("deepseek", "deepseek-flash", "chat_completions"),
        ("anthropic", "claude-sonnet-5", "anthropic"),
        ("gemini", "gemini-3.8-flash", "gemini"),
    ],
)
async def test_native_registry_roundtrip(monkeypatch, provider, model, api):
    captured = []

    def payload(request):
        captured.append(json.loads(request.content))
        if api == "responses":
            return {
                "output": [
                    {"type": "function_call", "call_id": "c", "name": "wait", "arguments": "{}"}
                ]
            }
        if api == "chat_completions":
            return {
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": "c",
                                    "function": {"name": "wait", "arguments": "{}"},
                                }
                            ],
                        },
                    }
                ]
            }
        if api == "anthropic":
            return {"content": [{"type": "tool_use", "id": "c", "name": "wait", "input": {}}]}
        return {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "wait", "args": {}}}],
                    },
                    "finishReason": "STOP",
                }
            ]
        }

    install_transport(monkeypatch, provider, payload)
    options = {"max_tokens": 1024, "context_window": 8192} if provider == "gemini" else {}
    llm = await build_llm(provider, model, "fake", temperature=0.2, **options)
    selector = LlamaIndexLLMSelector(llm, native_tool_api=api)
    turn = Turn(
        goal="open battery and check status",
        device=MemoryDevice(
            supported_operations={"tap", "input_text", "swipe", "start_app", "press_button"}
        ),
    )
    result = await turn.select(selector, parallel_tools=False)
    [call] = result.calls
    assert call.tool_name == "wait"
    assert turn.registry.by_name[call.tool_name].action.name == "wait"
    (body,) = captured
    specs = body["tools"]
    if api == "responses":
        assert not {"temperature", "top_p", "max_tokens"} & body.keys()
        assert body.get("max_output_tokens") is None or body["max_output_tokens"] >= 16
        assert body["tool_choice"] == "required" and body["parallel_tool_calls"] is False
        assert all(
            m["role"] in {"developer", "system", "user"}
            and (
                isinstance(m["content"], str)
                or all(b["type"] == "input_text" for b in m["content"])
            )
            for m in body["input"]
        )
        assert all(not s.get("strict") and s["type"] == "function" for s in specs)
    elif api == "chat_completions":
        assert all(s["type"] == "function" for s in specs)
        specs = [s["function"] for s in specs]
    elif api == "gemini":
        specs = [f for group in specs for f in group["functionDeclarations"]]
    names = {s["name"] for s in specs}
    assert (
        {"start_app", "press_button", "wait", "complete"} <= names <= turn.registry.by_name.keys()
    )
    assert all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", n) for n in names)
    for spec in specs:
        schema = spec.get("parametersJsonSchema") or spec.get("parameters") or spec["input_schema"]
        assert schema["type"].lower() == "object"


@pytest.mark.anyio
@pytest.mark.parametrize("log_response", [True, False])
@pytest.mark.parametrize("param", ["model", "temperature", "passphrase"])
async def test_openai_400_parameter_and_debug_note(log_response, param):
    import traceback

    from mobilerun.cli.base.errors import safe_error_message

    key = "unit-only-credential"

    def rejected(request):
        prompt = json.loads(request.content)["input"]
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "echo: " + prompt + key,
                    "param": prompt if param == "passphrase" else param,
                    "code": prompt if param == "passphrase" else "unsupported_parameter",
                    "type": "invalid_request_error",
                }
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(rejected)) as client:
        llm = await build_llm("openai", "gpt-6-luna", key, async_http_client=client)
        with pytest.raises(RuntimeError) as failure:
            await achat(llm, [ChatMessage(content="passphrase")], log_response=log_response)
    normal = safe_error_message(failure.value)
    debug = "".join(
        traceback.TracebackException.from_exception(failure.value).format_exception_only()
    )
    assert normal == (
        "LLM request failed (400, param: model)" if param == "model" else "LLM request failed (400)"
    )
    assert all(text not in normal + debug for text in ("passphrase", "echo:", key))
    if not log_response:
        assert not getattr(failure.value, "__notes__", [])
        return
    assert "invalid_request_error" in debug
    assert ("unsupported_parameter" in debug) == (param != "passphrase")


@pytest.mark.anyio
@pytest.mark.parametrize("profile_options", [{}, {"temperature": 0.0}])
async def test_gemini_profile_temperature_on_wire(monkeypatch, profile_options):
    from mobilerun_agents.config.user_config import LLMProfile, MobileConfig

    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "ok"}]}, "finishReason": "STOP"}
            ]
        }

    install_transport(monkeypatch, "gemini", respond)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    config = MobileConfig()
    config.profiles["default"] = LLMProfile(
        provider="gemini",
        model="gemini-3.8-flash",
        **profile_options,
    )
    with ConfigLoader.override(config):
        selector = (await default_setup()).selector
    await achat(selector.llm, [ChatMessage(content="test")])
    generation = bodies[0]["generationConfig"]
    assert {k: generation[k] for k in generation if k == "temperature"} == profile_options


@pytest.mark.anyio
@pytest.mark.parametrize(
    "model,sampling", [("gpt-6-astra", False), ("gpt-5.4-mini", False), ("gpt-4o", True)]
)
async def test_sampling_policy_on_wire(monkeypatch, model, sampling):
    payloads = []

    def respond(request):
        payloads.append(json.loads(request.content))
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]}

    install_transport(monkeypatch, "openai", respond)
    llm = await build_llm("openai", model, "fake", temperature=0.2)
    await achat(llm, [ChatMessage(content="test")])
    (body,) = payloads
    keys = {"temperature", "top_p"}
    assert body.keys() & keys == (keys if sampling else set())
