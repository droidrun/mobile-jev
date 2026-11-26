import json
from hashlib import sha256

from llama_index.core.base.llms.types import ImageBlock
from mobilerun.core.state.ui_state import UIState
from mobilerun.models.context import JSONValue

_ELEMENT_FIELDS = (
    ("text", "label", "value"),
    ("contentDescription", "content_desc", "content_description", "description"),
    ("resourceId", "resource_id", "viewIdResourceName", "id"),
    ("className", "class_name", "class", "type"),
    ("bounds", "boundsInScreen", "bounds_in_screen"),
    ("checkedState", "isChecked", "checked"),
    ("isSelected", "selected"),
    ("focused", "isFocused"),
    ("enabled", "isEnabled"),
)


def _element_structure(element: dict[str, JSONValue]) -> list[JSONValue]:
    details = element.get("details")
    properties = {**details, **element} if isinstance(details, dict) else element
    fields = [
        next((properties[k] for k in aliases if k in properties), None)
        for aliases in _ELEMENT_FIELDS
    ]
    nodes = element.get("children")
    fields.append(
        [_element_structure(child) for child in nodes if isinstance(child, dict)]
        if isinstance(nodes, list)
        else []
    )
    return fields


def screen_fingerprint(ui: UIState, screenshot: ImageBlock | None) -> tuple[str, str]:
    location = sha256(
        json.dumps([ui.phone_state.get("packageName"), ui.phone_state.get("activityName")]).encode()
    ).hexdigest()
    if not ui.elements and screenshot is None:
        return location, ""
    evidence: JSONValue
    if ui.elements:
        evidence = [_element_structure(element) for element in ui.elements]
    else:
        assert screenshot is not None
        evidence = sha256(screenshot.resolve_image().read()).hexdigest()
    return (
        location,
        sha256(
            json.dumps(
                [evidence, ui.focused_text, ui.phone_state.get("focusedElement")], sort_keys=True
            ).encode()
        ).hexdigest(),
    )
