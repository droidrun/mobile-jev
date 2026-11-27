from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.tools.helpers import element_targets


def test_widget_and_borrowed_label_use_smallest_container_and_toggle_allows_slack():
    elements = [
        {"index": 1, "bounds": "0,20,90,70", "descendants": 5},
        {"index": 2, "bounds": "5,25,70,65", "descendants": 4},
        {"index": 3, "bounds": "10,32,68,58", "text": "Dark mode", "isClickable": False},
        {
            "index": 4,
            "bounds": "45,33,65,57",
            "className": "Switch",
            "checkedState": "isChecked=True",
            "isClickable": False,
        },
        {"index": 5, "bounds": "45,30,73,60", "className": "Switch", "descendants": 0},
    ]
    labels = dict(element_targets(UIState(elements, "", "", {}, 100, 100)))
    assert "Dark mode" not in labels[1]
    assert '"Dark mode" (switch) (on)' in labels[2]
    assert '"Dark mode" (switch)' in labels[5]
