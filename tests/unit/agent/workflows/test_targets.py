import pytest
from llama_index.core.tools import ToolSelection
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.workflows.core.snapshot import StepSnapshot
from mobilerun_agents.agent.workflows.core.state import ActionRecord, AgentState
from mobilerun_agents.agent.workflows.core.targets import element_at, element_label, target
from mobilerun_agents.agent.workflows.guards import Stop, check
from mobilerun_agents.config.user_config import AgentConfig


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        (
            {"type": "XCUIElementTypeSecureTextField", "contentDescription": "private"},
            "SecureTextField",
        ),
        (
            {
                "className": "EditText",
                "password": True,
                "contentDescription": "private",
                "resourceId": "pwd",
            },
            'EditText "pwd"',
        ),
        (
            {
                "className": "EditText",
                "details": {"password": True, "description": "private", "hint": "Password"},
            },
            'EditText "Password"',
        ),
        ({"className": "EditText", "details": {"hint": "ZIP"}}, 'EditText "ZIP"'),
        ({"type": "XCUIElementTypeTextField", "placeholder": "Email"}, 'TextField "Email"'),
        (
            {"className": "View", "isEditable": True, "label": "private", "resourceId": "field"},
            'View "field"',
        ),
    ],
)
def test_editable_and_password_labels_never_use_current_values(node, expected):
    assert element_label({"text": "private", "value": "private", **node}) == expected


def test_focus_metadata_protects_a_tree_node_with_no_editable_flag():
    node = {"index": 1, "className": "View", "resourceId": "field", "text": "private"}
    ui = UIState(
        [node], "", "", {"isEditable": True, "focusedElement": {"resourceId": "field"}}, 100, 100
    )
    assert target(record(ui, 1, "tap_element", index=1)) == 'View "field"'


@pytest.mark.parametrize("mode", ["jitter", "empty_tree", "distinct_indexed", "distinct_points"])
def test_stuck_uses_hits_or_grid_instead_of_labels_or_exact_pixels(mode):
    nodes = (
        []
        if mode == "empty_tree"
        else [
            {"index": 1, "className": "Button", "text": "Continue", "bounds": "0,0,48,50"},
            {"index": 2, "className": "Button", "text": "Continue", "bounds": "48,0,100,50"},
        ]
    )
    ui = UIState(nodes, "", "", {}, 100, 100)
    actions = []
    for i in range(5):
        if mode == "distinct_indexed":
            action = record(ui, i, "tap_element", index=1 + i % 2)
        else:
            x = 11 + i if mode != "distinct_points" else 20 + 50 * (i % 2)
            action = record(ui, i, "tap", x=x, y=20)
        actions.append(action)
    assert check(AgentState(actions=actions), AgentConfig()) == (
        Stop("stuck") if mode in {"jitter", "empty_tree"} else None
    )


@pytest.mark.parametrize("blocker", [False, True])
def test_point_targets_unique_unblocked_control(blocker):
    small = {"index": 1, "bounds": "10,10,20,20"}
    foreground = {"index": 2, "bounds": "0,0,100,100"}
    if blocker:
        small["tapBlockers"] = [foreground["bounds"]]
        elements = [foreground, small]
    else:
        elements = [small, foreground]
    ui = UIState(elements, "", "", {}, 100, 100)
    assert element_at(ui, (15, 15)) is (foreground if blocker else None)


def record(ui, step, name, **args):
    return ActionRecord(
        step=step,
        call=ToolSelection(tool_id=str(step), tool_name=name, tool_kwargs=args),
        before=StepSnapshot(ui=ui),
        ok=True,
        summary="",
        screen_changed=True,
    )


def test_ios_projected_siblings_without_draw_order_use_grid_but_respect_blockers():
    button = {
        "index": 0,
        "type": "Button",
        "className": "Button",
        "text": "Continue",
        "bounds": "15,15,50,50",
        "rect": "15,15,35,35",
    }
    container = {
        "index": 1,
        "type": "Other",
        "className": "Other",
        "bounds": "0,0,100,100",
        "rect": "0,0,100,100",
    }
    ui = UIState([button, container], "", "", {}, 100, 100)
    assert element_at(ui, (30, 30)) is None
    assert target(record(ui, 1, "tap", x=30, y=30)) == "(30, 30)"
    calls = [record(ui, step, "tap", x=x, y=30) for step, x in enumerate(range(30, 35))]
    assert check(AgentState(actions=calls), AgentConfig()) == Stop("stuck")
    container["tapBlockers"] = [button["bounds"]]
    assert element_at(ui, (30, 30)) is button
    container.pop("tapBlockers")
    container["children"] = [button]
    ui.elements = [container]
    assert element_at(ui, (30, 30)) is button
