import importlib
import json
from pathlib import Path

import pytest

MAP = json.loads((Path(__file__).parents[2] / "scripts/module_map.json").read_text())


@pytest.mark.parametrize("module", MAP)
def test_old_modules_import(module):
    assert importlib.import_module(module)


@pytest.mark.parametrize(
    ("module", "names"),
    [
        ("mobilerun.agents", ["MobileAgent"]),
        ("mobilerun.agents.operator", ["AgentOperator"]),
        ("mobilerun.agents.cli.agent.group", ["agent"]),
        ("mobilerun.agents.cli.agent.run", ["run"]),
        ("mobilerun.agents.cli.agent.llm.configure", ["configure"]),
        ("mobilerun.agents.agent_macro", ["AgentMacroOperator"]),
        ("mobilerun.agents.config.loader", ["ConfigLoader", "OutdatedConfigError"]),
        ("mobilerun.agents.config.user_config", ["LLMProfile", "MobileConfig"]),
        ("mobilerun.agents.utils.paths", ["PathResolver"]),
        ("mobilerun.agents.clients.oauth.profiles", []),
        ("mobilerun.agents.agent.utils.errors", ["HtmlErrorPageFilter"]),
        (
            "mobilerun.agents.agent",
            ["ResultEvent", "WorkflowOptions", "WorkflowOptionsWithoutTimeout"],
        ),
        ("mobilerun.agents.agent.droid", ["MobileAgent", "MobileAgentState"]),
        ("mobilerun.agents.agent.droid.events", ["ResultEvent"]),
        ("mobilerun.agents.agent.utils.llm_picker", ["load_llm"]),
    ],
)
def test_external_paths(module, names):
    obj = importlib.import_module(module)
    for name in names:
        assert getattr(obj, name) is not None


def test_legacy_utility_exports():
    from mobilerun.agents.agent.utils import (
        PromptResolver,
        Trajectory,
        build_tool_registry,
        filter_empty_messages,
        has_content,
        limit_history,
        to_chat_messages,
        tracing_setup,
    )

    assert all(
        (
            tracing_setup,
            to_chat_messages,
            has_content,
            filter_empty_messages,
            limit_history,
            PromptResolver,
            build_tool_registry,
            Trajectory,
        )
    )
