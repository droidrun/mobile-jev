from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.action_selector.decision_model.candidates.formatter import (
    DecisionFormatter,
)
from mobilerun_agents.agent.tools.helpers import element_targets


def state(elements):
    return UIState(elements, "screen", "", {}, 1080, 2400)


def test_candidate_keeps_typed_parameters_and_skips_disabled_offscreen():
    ui = state(
        [
            {"index": 1, "text": "Battery", "className": "Button", "bounds": "0,20,200,80"},
            {"index": 2, "text": "Disabled", "isEnabled": False, "bounds": "0,80,200,100"},
            {"index": 3, "text": "Offscreen", "bounds": "0,2500,200,2600"},
        ]
    )
    assert element_targets(ui) == ((1, 'tap_element [1] "Battery" (button) (top of screen)'),)


def test_candidate_borrows_row_label():
    ui = state(
        [
            {
                "index": 1,
                "className": "ViewGroup",
                "bounds": "0,0,500,100",
                "descendants": 1,
                "isLongClickable": True,
            },
            {"index": 2, "text": "Battery", "bounds": "10,10,400,80", "isClickable": False},
        ]
    )
    assert [index for index, label in element_targets(ui) if "Battery" in label] == [1]


def test_decision_formatter_retains_metadata_through_public_format():
    formatter = DecisionFormatter()
    _, _, elements, _ = formatter.format(
        {
            "className": "android.widget.EditText",
            "text": "",
            "hint": "Email",
            "isPassword": True,
            "isEnabled": False,
            "isVisibleToUser": False,
            "boundsInScreen": {"left": 0, "top": 0, "right": 100, "bottom": 50},
            "children": [],
        },
        {},
    )
    assert elements[0]["details"] == {"hint": "Email", "password": True}
    assert elements[0]["descendants"] == 0
    assert elements[0]["isEnabled"] is False


def test_decision_formatter_does_not_repeat_masked_secret_details():
    _, _, elements, _ = DecisionFormatter().format(
        {
            "text": "[secret:password]",
            "hint": "raw secret",
            "contentDescription": "raw secret",
        },
        {},
    )
    assert not elements[0].get("details")


def test_named_ios_row_and_its_toggle_are_both_offered():
    ui = state(
        [
            {
                "index": 1,
                "className": "XCUIElementTypeSwitch",
                "text": "Wi-Fi",
                "bounds": "0,100,900,200",
                "descendants": 1,
            },
            {
                "index": 2,
                "className": "XCUIElementTypeSwitch",
                "value": "1",
                "bounds": "700,120,800,180",
                "descendants": 0,
            },
        ]
    )
    options = element_targets(ui)
    assert [index for index, _ in options] == [1, 2]
    assert options[1][1] == 'tap_element [2] "Wi-Fi" (switch) (on, top of screen)'


def test_clickable_keyboard_keys_are_choices():
    ui = state(
        [
            {
                "index": 1,
                "className": "XCUIElementTypeWindow",
                "bounds": "0,0,1080,2400",
                "descendants": 3,
                "isClickable": False,
            },
            {
                "index": 2,
                "className": "XCUIElementTypeKeyboard",
                "bounds": "0,1500,1080,2400",
                "descendants": 2,
                "isClickable": False,
            },
            {
                "index": 3,
                "className": "XCUIElementTypeKey",
                "text": "A",
                "bounds": "0,1600,100,1700",
                "descendants": 0,
            },
            {
                "index": 4,
                "className": "XCUIElementTypeKey",
                "text": "Search",
                "bounds": "800,2100,1080,2200",
                "descendants": 0,
            },
        ]
    )
    assert [index for index, _ in element_targets(ui)] == [3, 4]
