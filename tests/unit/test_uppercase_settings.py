"""Every environment-backed field is its own documented, exact environment name."""

import ast
import inspect
import json
from enum import Enum
from pathlib import Path
from typing import get_args

import pytest
from mobilerun.config.env import Secrets
from mobilerun.config.settings import Settings
from pydantic import SecretStr

from mobilerun_agents.config.settings import AgentSecrets, AgentSettings

MODELS = (AgentSettings, AgentSecrets)


@pytest.mark.parametrize(
    "model,name",
    [(model, name) for model in MODELS for name in model.model_fields],
    ids=lambda value: value.__name__ if isinstance(value, type) else value,
)
def test_exact_documented_environment_field(monkeypatch, model, name):
    field = model.model_fields[name]
    assert name == name.upper()
    assert field.validation_alias is None
    assert field.description
    annotation = field.annotation
    types = get_args(annotation) or (annotation,)
    default = field.get_default(call_default_factory=True)
    if SecretStr in types:
        raw = "dummy-secret-for-test"
        expected = SecretStr(raw)
    elif Path in types:
        raw = "/dummy/settings-test"
        expected = Path(raw)
    elif isinstance(default, Enum):
        raw = default.value
        expected = default
    elif bool in types or isinstance(default, bool):
        raw, expected = "false", False
    elif isinstance(default, int):
        raw, expected = "4321", 4321
    elif isinstance(default, float):
        raw, expected = "12.5", 12.5
    elif isinstance(default, (list, tuple)):
        raw = json.dumps([0.1, 0.2])
        expected = type(default)([0.1, 0.2])
    else:
        raw = "dummy-value"
        expected = raw
    monkeypatch.setenv(name, raw)
    assert getattr(model(), name) == expected


def test_base_models_do_not_own_agent_fields():
    assert set(Secrets.model_fields) == {"MOBILERUN_API_KEY", "MOBILERUN_DEVICE_TOKEN"}
    assert not (set(Settings.model_fields) & {"PHOENIX_URL", "PHOENIX_PROJECT_NAME"})
    assert not (set(Secrets.model_fields) & set(AgentSecrets.model_fields))


@pytest.mark.parametrize("model", MODELS)
def test_fields_are_single_line_with_attribute_docstrings(model):
    cls = ast.parse(inspect.getsource(model)).body[0]
    for index, node in enumerate(cls.body):
        if not isinstance(node, ast.AnnAssign):
            continue
        assert node.lineno == node.end_lineno, node.target.id
        description = cls.body[index + 1]
        assert isinstance(description, ast.Expr)
        assert isinstance(description.value, ast.Constant)
        assert isinstance(description.value.value, str)
        assert description.lineno == description.end_lineno
        assert description.lineno == node.end_lineno + 1
        if isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "Field":
            assert {keyword.arg for keyword in node.value.keywords} <= {
                "gt",
                "ge",
                "le",
                "min_length",
                "default_factory",
            }
            assert node.value.keywords
