import ast
import re
from pathlib import Path

import pytest
from jinja2 import StrictUndefined, UndefinedError
from jinja2.sandbox import ImmutableSandboxedEnvironment

import mobilerun_agents
from mobilerun_agents.agent.action_selector import DecisionSelector
from mobilerun_agents.agent.action_selector.decision_model.typesafe import TypeSafeDecisionModel
from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig


def test_decision_defaults():
    assert DecisionConfig().timeout == 120
    assert DecisionConfig().max_options == 252


def test_config_override_changes_threshold():
    config = DecisionConfig.model_validate({"min_confidence": 0.25})
    assert DecisionSelector(TypeSafeDecisionModel(), config=config).min_confidence == 0.25


def test_prompt_loader_rejects_missing_override_variables(tmp_path):
    prompt = tmp_path / "custom.jinja2"
    prompt.write_text("{{ missing_variable }}")
    with pytest.raises(UndefinedError, match="missing_variable"):
        render_prompt(str(prompt))


TEMPLATES = {
    "delegate/invalid.jinja2": {"kind": "parameters"},
    "delegate/option.jinja2": {},
    "decision/system.jinja2": {
        "available_tools": set(),
        "can_type": False,
        "can_complete": True,
    },
    "decision/locate.jinja2": {"kind": "instructions"},
    "decision/criteria.jinja2": {"kind": "abstain"},
    "decision/completion.jinja2": {"success": True},
    "decision/candidate.jinja2": {
        "kind": "element",
        "index": 7,
        "element_kind": "button",
        "label": "Battery",
        "notes": [],
        "unnamed_class": "Button",
    },
    "llm/selection_request.jinja2": {"state": ""},
    "llm/invalid_arguments.jinja2": {},
}


@pytest.mark.parametrize("name,variables", TEMPLATES.items())
def test_templates_ship_in_installed_wheel_and_render_strict(name, variables):
    path = Path(mobilerun_agents.__file__).parent / "assets/prompts" / name
    assert "site-packages" in str(path), "Check the installed wheel, not the checkout"
    text = path.read_text()
    template = ImmutableSandboxedEnvironment(undefined=StrictUndefined).from_string(text)
    assert template.render(**variables).strip()


@pytest.mark.parametrize("kind", ["tap", "located", "schema", "app", "element", "scroll"])
def test_candidate_template_branches_render_strict(kind):
    assert render_prompt(
        DecisionConfig().candidate_prompt,
        kind=kind,
        x=10,
        y=20,
        name="scroll",
        arguments=['"up"'],
        label="Battery",
        identifier="app.test",
        index=7,
        direction="down",
        element_kind="switch",
        unnamed_class="Switch",
        notes=["state_true", "state_false", "top", "bottom", "selected", "focused", "password"],
    )


def test_no_model_facing_prose_in_selector_code():
    """Long literals are prose unless docstrings or explicit exception messages.

    Raise statements are recognized by their AST parent. The sole explicit
    diagnostic below is a terminal run error, never sent back to the model.
    Short labels are checked separately through the candidate template tests.
    """
    root = Path(__file__).resolve().parents[4] / "src/mobilerun_agents"
    diagnostics = {
        ("agent/action_selector/llm/base.py", "LLM response is None. This is a critical error.")
    }
    violations = []
    for path in (root / "agent/action_selector").rglob("*.py"):
        tree = ast.parse(path.read_text())
        exempt = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.body and isinstance(node.body[0], ast.Expr):
                    exempt.update(ast.walk(node.body[0]))
            if isinstance(node, ast.Raise):
                exempt.update(ast.walk(node))
            # Logs are operational diagnostics, not model-facing prompts.
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"debug", "info", "warning", "error", "log"}
            ):
                exempt.update(ast.walk(node))
        for node in ast.walk(tree):
            if (
                node not in exempt
                and isinstance(node, ast.Constant)
                and isinstance(node.value, str)
            ):
                if (str(path.relative_to(root)), node.value) in diagnostics:
                    continue
                if len(node.value) > 80 or len(re.findall(r"[.!?](?:\s|$)", node.value)) > 1:
                    violations.append(f"{path.name}:{node.lineno}: {node.value}")
    assert not violations, "\n".join(violations)


def test_fast_agent_prompt_reports_partial_results_and_visible_blockers():
    path = Path(mobilerun_agents.__file__).parent / "assets/prompts/fast_agent/system.jinja2"
    assert "site-packages" in str(path), "Check the installed wheel, not the checkout"
    prompt = path.read_text()
    assert "including partial values. Don't keep searching the same places." in prompt
    assert (
        "If the screen can't answer the goal (for example there is nothing to compare), complete with what you see instead of repeating actions."
        in prompt
    )
    assert "Name any visible blocker" in prompt
    assert (
        '"blocked: price hidden because item can\'t ship to selected location (Germany)"' in prompt
    )
