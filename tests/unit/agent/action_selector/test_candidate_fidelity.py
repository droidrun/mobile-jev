"""Adapted from origin/develop:tests/test_device_action_candidates.py.

Construct public UIState records directly rather than invoking private parsers.
The test names retain the corresponding develop cases where applicable.
"""

import pytest
from mobilerun.core.state.formatters import IndexedFormatter
from mobilerun.core.state.ui_state import UIState

from mobilerun_agents.agent.tools.helpers import element_targets


def node(index, cls, bounds, name="", descendants=None, **extra):
    return dict(
        index=index,
        className=f"XCUIElementType{cls}",
        bounds=bounds,
        text=name,
        **({"descendants": descendants} if descendants is not None else {}),
        isClickable=cls not in {"Window", "Other", "NavigationBar", "Table", "ScrollView"},
        **extra,
    )


def labels(elements):
    # Include leaf counts when exercising hierarchical observations.
    if any("descendants" in e for e in elements):
        elements = [{"descendants": 0, **e} for e in elements]
    return [
        _semantic_label(label)
        for _, label in element_targets(UIState(elements, "", "", {}, 1080, 2400))
    ]


def _semantic_label(label):
    import re

    _, name, rest = re.fullmatch(r'tap_element \[(\d+)\] "(.*?)"(.*)', label).groups()
    if m := re.match(
        r" \((button|text field|search field|switch|checkbox|slider|keyboard key)\)", rest
    ):
        return f'Click {m[1]} "{name}"{rest[m.end():]}'
    return f"Click {name}{rest}"


def test_half_height_sheet_does_not_filter_reported_clickable_nodes():
    elements = [
        node(1, "Window", "0,0,1080,2400", descendants=8),
        node(2, "Other", "0,0,1080,2400", descendants=2),
        node(3, "Button", "0,600,440,700", "Directions"),
        node(4, "Button", "0,1700,440,1800", "Hidden row"),
        node(5, "Other", "0,0,1080,2400", descendants=4),
        node(6, "Other", "0,1300,1080,2400", descendants=3),
        node(7, "NavigationBar", "0,1320,1080,1440", descendants=1),
        node(8, "Button", "0,1340,200,1420", "Close"),
    ]
    result = labels(elements)
    assert 'Click button "Directions"' in result
    assert 'Click button "Hidden row"' in result
    assert 'Click button "Close"' in result


def test_empty_field_takes_the_label_beside_it():
    assert labels(
        [
            dict(
                index=1,
                className="android.widget.TextView",
                text="Name",
                bounds="40,610,280,690",
                isClickable=False,
            ),
            dict(index=2, className="android.widget.EditText", text="", bounds="300,600,1000,700"),
        ]
    ) == ['Click text field "Name"']


def test_empty_formatted_field_does_not_report_its_class_as_contents():
    tree = {
        "isClickable": False,
        "children": [
            {
                "className": "android.widget.TextView",
                "text": "Name",
                "isClickable": False,
                "boundsInScreen": {"left": 40, "top": 610, "right": 280, "bottom": 690},
            },
            {
                "className": "android.widget.EditText",
                "text": "",
                "isEditable": True,
                "boundsInScreen": {"left": 300, "top": 600, "right": 1000, "bottom": 700},
            },
        ],
    }
    choices = labels(IndexedFormatter().format(tree, {})[2])
    assert [label for label in choices if label.startswith("Click text field")] == [
        'Click text field "Name"'
    ]


def test_field_keeps_its_label_over_a_cell_with_the_same_bounds():
    assert labels(
        [
            node(1, "Cell", "0,600,400,700", descendants=1),
            node(2, "TextField", "0,600,400,700", "Gym", details={"description": "Title"}),
        ]
    ) == ["Click unnamed Cell", 'Click text field "Title" (contains "Gym")']


def test_unlabelled_clickable_container_is_offered_with_its_clickable_child():
    elements = [
        node(1, "Cell", "0,600,400,800", descendants=1),
        node(2, "Button", "0,650,400,700", "Edit", descendants=0),
    ]
    options = element_targets(UIState(elements, "", "", {}, 1080, 2400))
    assert [(index, _semantic_label(label)) for index, label in options] == [
        (1, "Click unnamed Cell"),
        (2, 'Click button "Edit"'),
    ]


def test_legacy_ios_field_is_named_by_its_placeholder_not_its_identifier():
    assert labels(
        [
            dict(
                index=1,
                type="TextField",
                className="TextField",
                text="email-input",
                resourceId="email-input",
                placeholder="Email",
                bounds="16,600,386,634",
            )
        ]
    ) == ['Click text field "Email"']


@pytest.mark.parametrize("value,expected", [("Your name", ""), ("Bob", ' (contains "Bob")')])
def test_legacy_ios_placeholder_is_not_reported_as_contents(value, expected):
    assert labels(
        [
            dict(
                index=1,
                type="TextField",
                placeholder="Your name",
                value=value,
                bounds="16,600,386,634",
            )
        ]
    ) == ['Click text field "Your name"' + expected]


def test_unlabelled_ios_switch_takes_the_label_beside_it():
    assert labels(
        [
            node(1, "Other", "0,600,402,640", descendants=2),
            node(2, "StaticText", "16,605,200,635", "Dark mode"),
            node(3, "Switch", "330,604,381,635", "0"),
        ]
    ) == ["Click Dark mode", 'Click switch "Dark mode" (off)']


def test_bars_and_scrolled_content_with_same_bounds_keep_their_options():
    result = labels(
        [
            node(1, "Window", "0,0,1080,2400", descendants=5),
            node(2, "Other", "0,0,1080,2400", descendants=4),
            node(3, "NavigationBar", "0,500,1080,650", descendants=1),
            node(4, "Button", "0,520,200,620", "Back"),
            node(5, "Table", "0,500,1080,2200", descendants=1),
            node(6, "Button", "0,520,200,620", "Scrolled row"),
        ]
    )
    assert 'Click button "Back"' in result
    assert 'Click button "Scrolled row"' in result


def test_later_separate_list_does_not_hide_overlapping_options():
    result = labels(
        [
            node(1, "Window", "0,0,1080,2400", descendants=6),
            node(2, "Table", "0,400,1080,2200", descendants=2),
            node(3, "Button", "0,600,300,700", "Hidden"),
            node(4, "Button", "0,1800,300,1900", "Visible"),
            node(5, "Table", "0,400,1080,1200", descendants=1),
            node(6, "Button", "0,600,300,700", "Suggestion"),
        ]
    )
    assert any('"Hidden"' in s for s in result)
    assert any('"Visible"' in s for s in result)
    assert any('"Suggestion"' in s for s in result)


def test_later_separate_option_does_not_hide_earlier_option():
    result = labels(
        [
            node(1, "Window", "0,0,1080,2400", descendants=2),
            node(2, "Button", "0,600,300,800", "Hidden"),
            node(3, "Button", "100,650,400,850", "Overlay"),
        ]
    )
    assert result == ['Click button "Hidden"', 'Click button "Overlay"']


def test_ios_value_keeps_commas_and_skips_the_placeholder():
    # Public data can preserve the value; the base legacy parser cannot yet.
    assert labels(
        [
            dict(
                index=1,
                type="TextField",
                label="Address",
                value="Paris, France",
                placeholder="City",
                bounds="0,600,400,700",
            )
        ]
    ) == ['Click text field "Address" (contains "Paris, France")']


def test_real_iphone_fields_show_name_and_contents():
    assert labels(
        [
            node(1, "TextField", "0,600,402,660", "Gym", details={"description": "Title"}),
            node(
                2,
                "TextField",
                "0,700,402,760",
                "Title",
                details={"description": "Title", "hint": "Enter a title"},
            ),
            node(3, "SearchField", "0,800,402,860", "Bold Text", details={"description": "Search"}),
            node(4, "Cell", "0,900,402,960", "XCUIElementTypeCell"),
        ]
    ) == [
        'Click text field "Title" (contains "Gym")',
        'Click text field "Title"',
        'Click search field "Search" (contains "Bold Text")',
        "Click unnamed Cell",
    ]


@pytest.mark.parametrize(
    "knob", [{"isEnabled": False}, {"isClickable": False}, {"isVisibleToUser": False}]
)
def test_row_switch_stays_when_its_knob_cannot_be_offered(knob):
    small = node(2, "Switch", "305,606,368,654", "0")
    small.update(knob)
    assert labels(
        [node(1, "Switch", "0,600,402,660", "0", details={"description": "Bold Text"}), small]
    ) == ['Click switch "Bold Text" (off)']


def test_real_iphone_switch_keeps_named_options_for_row_and_knob():
    assert labels(
        [
            node(1, "Switch", "0,600,402,660", "0", details={"description": "Bold Text"}),
            node(2, "Switch", "305,606,368,654", "0"),
        ]
    ) == ['Click switch "Bold Text" (off)', 'Click switch "Bold Text" (off)']


def test_on_screen_knob_of_an_off_screen_ios_row_is_still_offered():
    assert labels(
        [
            node(
                1,
                "Switch",
                "0,0,0,0",
                "0",
                descendants=1,
                details={"description": "Off screen"},
            ),
            node(2, "Switch", "0,616,63,644", "0"),
        ]
    ) == ['Click switch "unnamed Switch" (off)']


def test_platform_state_words_do_not_repeat_the_state():
    assert labels(
        [
            dict(
                index=1,
                className="android.widget.CheckBox",
                text="Sunday",
                bounds="0,600,1080,700",
                checkedState="isChecked=False",
                details={"stateDescription": "Not selected"},
            ),
            dict(
                index=2,
                className="android.widget.SeekBar",
                text="Brightness",
                bounds="0,800,1080,900",
                details={"stateDescription": "83%"},
            ),
        ]
    ) == ['Click checkbox "Sunday" (unchecked)', 'Click slider "Brightness" (83%)']


def test_keyboard_submit_keeps_its_label_without_filtering_other_clickable_nodes():
    result = labels(
        [
            node(1, "Window", "0,0,1080,2400", descendants=5),
            node(2, "Button", "0,1800,300,1900", "Hidden"),
            {**node(3, "Keyboard", "0,1500,1080,2400", descendants=2), "isClickable": False},
            node(4, "Button", "0,1800,100,1900", "A"),
            node(5, "Button", "700,1800,1080,1900", "Search"),
        ]
    )
    assert result == ['Click button "Hidden"', 'Click button "A"', 'Click keyboard key "Search"']


@pytest.mark.parametrize("focused_text,expected", [("", []), ("2", ["2"])])
def test_focus_is_marked_only_on_the_one_focused_field(focused_text, expected):
    ui = UIState(
        [
            dict(
                index=i,
                className="EditText",
                text=str(i),
                resourceId="shop:id/qty",
                bounds=f"0,{300*i},1080,{300*i+200}",
            )
            for i in (1, 2, 3)
        ],
        "",
        "",
        {"isEditable": True, "focusedElement": {"resourceId": "shop:id/qty", "text": focused_text}},
        1080,
        2400,
    )
    assert [
        label.split('"')[1] for _, label in element_targets(ui) if "focused" in label
    ] == expected


def test_invalid_focus_metadata_does_not_discard_valid_options():
    ui = UIState(
        [dict(index=1, text="Save", bounds="0,600,200,700")],
        "",
        "",
        {"focusedElement": "invalid"},
        1080,
        2400,
    )
    assert [_semantic_label(label) for _, label in element_targets(ui)] == ["Click Save"]
