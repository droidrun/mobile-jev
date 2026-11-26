"""Choice labels and reachable element targets from public UIState data."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mobilerun.core.state.ui_state import UIState
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from mobilerun_agents.config.prompts import render_prompt
from mobilerun_agents.config.user_config import DecisionConfig


class Element(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")
    index: int
    text: str = ""
    class_name: str = Field(default="", alias="className")
    type: str = ""
    bounds: str = ""
    resource: str = Field(default="", alias="resourceId")
    value: str = ""
    placeholder: str = ""
    label: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    descendants: int | None = None
    clickable: bool = Field(default=True, alias="isClickable")
    long_clickable: bool = Field(default=False, alias="isLongClickable")
    enabled: bool = Field(default=True, alias="isEnabled")
    visible: bool = Field(default=True, alias="isVisibleToUser")
    checked: str = Field(default="", alias="checkedState")

    @property
    def offerable(self) -> bool:
        left, top, right, bottom = self.box
        return (
            self.enabled
            and self.visible
            and right > left
            and bottom > top
            and (self.clickable or self.long_clickable or self.kind == "text_field")
        )

    @property
    def box(self) -> tuple[int, int, int, int]:
        values = TypeAdapter(tuple[int, int, int, int]).validate_python(self.bounds.split(","))
        return values

    @property
    def short_class(self) -> str:
        return (self.class_name or self.type).rsplit(".", 1)[-1].removeprefix("XCUIElementType")

    @property
    def ios(self) -> bool:
        return bool(self.type or self.class_name.startswith("XCUIElementType"))

    @property
    def kind(self) -> str:
        short = self.short_class.lower()
        if "switch" in short or short in {"toggle", "togglebutton"}:
            return "switch"
        if "checkbox" in short:
            return "checkbox"
        if "radiobutton" in short:
            return "radio_button"
        if "edittext" in short or short in {
            "textfield",
            "securetextfield",
            "searchfield",
            "searchview",
            "autocompletetextview",
        }:
            return "text_field"
        if short == "textview" and (self.type or self.class_name.startswith("XCUI")):
            return "text_field"
        if "button" in short:
            return "button"
        return {
            "tab": "tab",
            "slider": "slider",
            "seekbar": "slider",
            "spinner": "dropdown",
            "checkedtextview": "option",
        }.get(short, "")

    @property
    def own_text(self) -> str:
        text = self.text.strip()
        short = self.class_name.rsplit(".", 1)[-1]
        if (
            text in {self.resource, self.class_name}
            or text.startswith("XCUIElementType")
            or (short and text.endswith(f".{short}"))
        ):
            text = ""
        description = str(self.details.get("description") or "").strip()
        if description.startswith("XCUIElementType"):
            description = ""
        if (
            self.class_name.startswith("XCUIElementType")
            and text in {"0", "1"}
            and (
                self.kind in {"switch", "checkbox", "radio_button", "option"}
                or self.short_class == "Cell"
                or (description and description != text)
            )
        ):
            text = ""
        return text

    @property
    def name(self) -> str:
        text = self.own_text
        description = str(self.details.get("description") or "").strip()
        if description.startswith("XCUIElementType"):
            description = ""
        if self.kind == "text_field":
            return (
                description
                or str(self.details.get("hint") or "")
                or self.placeholder
                or ("" if self.details.get("password") else text)
            )
        return ", ".join(
            s for s in (text, description if description.lower() not in text.lower() else "") if s
        )

    @property
    def state_note(self) -> str:
        if self.checked.startswith("isChecked="):
            on = self.checked == "isChecked=True"
        elif self.kind in {"switch", "checkbox", "radio_button", "option"} and (
            self.value or self.text
        ) in {"0", "1"}:
            on = (self.value or self.text) == "1"
        else:
            return ""
        return "state_true" if on else "state_false"

    def field_label_and_contents(self, name: str, beside: str) -> tuple[str, str]:
        description = str(self.details.get("description") or "").strip() or beside
        hint = str(self.details.get("hint") or self.placeholder).strip()
        if self.details.get("password"):
            return description or hint or name, ""
        if self.kind == "text_field" and self.type:
            field_name = beside or (self.label if self.label != hint else "") or hint or name
            return field_name, self.value if self.value not in {hint, field_name} else ""
        if self.kind == "text_field":
            field_name = description or hint
            text = self.own_text
            if self.details.get("showingHint") or text in {description, hint}:
                return field_name or name, ""
            if field_name and text:
                return field_name, text
            name = field_name or name
        value = self.value if self.value not in {name, self.placeholder} else ""
        return name, value if self.kind == "text_field" else ""


def _flatten(elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        item for element in elements for item in (element, *_flatten(element.get("children") or []))
    ]


def element_targets(
    state: UIState, config: DecisionConfig | None = None
) -> tuple[tuple[int, str], ...]:
    config = config or DecisionConfig()
    elements = []
    for raw in _flatten(state.elements):
        try:
            element = Element.model_validate(raw)
            left, top, right, bottom = element.box
        except ValidationError:
            continue
        if element.visible:
            elements.append(element)
    options: list[tuple[int, str]] = []
    screen_area = (state.screen_width or 0) * (state.screen_height or 0)
    borrowed = borrowed_names(elements, screen_area, config)
    toggles = toggle_names(elements, borrowed, screen_area, config)
    submit = keyboard_submit_indices(elements)
    field_labels = {
        e.index: label_beside(e, elements, config.knob_slack)
        for e in elements
        if e.kind == "text_field" and not e.details.get("description") and not e.details.get("hint")
    }
    try:
        focused = FocusState.model_validate(state.phone_state or {})
    except ValidationError:
        focused = FocusState()
    matches = [
        e
        for e in elements
        if focused.isEditable
        and focused.focusedElement
        and e.resource == focused.focusedElement.resourceId
        and e.resource
    ]
    if len(matches) > 1 and focused.focusedElement and focused.focusedElement.text:
        matches = [e for e in matches if e.text == focused.focusedElement.text]
    focus_index = matches[0].index if len(matches) == 1 else None
    widgets: dict[int, Element] = {}
    for widget in elements:
        if widget.clickable or widget.long_clickable or not widget.state_note:
            continue
        owner = _smallest_container(
            widget,
            (c for c in elements if c.clickable or c.long_clickable),
            screen_area * config.max_owner_screen_share,
        )
        if owner is not None:
            widgets.setdefault(owner.index, widget)
    seen = set()
    for e in elements:
        if e.index in seen or not e.offerable:
            continue
        seen.add(e.index)
        try:
            x, y = state.get_element_coords(e.index)
        except (KeyError, ValueError):
            continue
        if (state.screen_width and not 0 <= x < state.screen_width) or (
            state.screen_height and not 0 <= y < state.screen_height
        ):
            continue
        name = e.name or toggles.get(e.index, "")
        resource = e.resource.rsplit("/", 1)[-1]
        if not name:
            name = ", ".join(borrowed.get(e.index, []))
            if name and resource and not any(char.isalpha() for char in name):
                name = f"{resource}: {name}"
        name = name or resource
        name, contents = e.field_label_and_contents(name, field_labels.get(e.index, ""))
        nested_widget = widgets.get(e.index)
        element_kind = e.kind or (nested_widget.kind if nested_widget else "")
        if e.short_class in {"SearchField", "SearchView"}:
            element_kind = "search_field"
        if not element_kind and e.details.get("editable"):
            element_kind = "text_field"
        state_note = e.state_note or (nested_widget.state_note if nested_widget else "")
        notes = [state_note] if state_note else []
        extra = str(e.details.get("stateDescription") or "")
        if state_note or (extra and extra in name):
            extra = ""
        if e.kind in {"slider", "dropdown"} and e.value not in {name, e.placeholder}:
            extra = extra or e.value
        notes.extend(k for k in ("selected", "focused", "password") if e.details.get(k))
        if e.index == focus_index and "focused" not in notes:
            notes.append("focused")
        if y < state.screen_height * config.edge_share:
            notes.append("top")
        elif y > state.screen_height * (1 - config.edge_share):
            notes.append("bottom")
        options.append(
            (
                e.index,
                render_prompt(
                    config.candidate_prompt,
                    kind="element",
                    index=e.index,
                    element_kind="keyboard_key" if e.index in submit else element_kind,
                    label=name,
                    contents=contents,
                    extra=extra,
                    notes=notes,
                    unnamed_class=e.short_class,
                ),
            )
        )
    return tuple(options)


class FocusedElement(BaseModel):
    resourceId: str = ""
    text: str = ""


class FocusState(BaseModel):
    isEditable: bool = False
    focusedElement: FocusedElement | None = None


def contains(outer: Element, inner: Element) -> bool:
    return outer.descendants is None or outer.index < inner.index <= outer.index + outer.descendants


def inside(inner: Element, outer: Element, slack: int = 0) -> bool:
    a, b = inner.box, outer.box
    return (
        b[0] - slack <= a[0]
        and b[1] - slack <= a[1]
        and a[2] <= b[2] + slack
        and a[3] <= b[3] + slack
    )


def area(element: Element) -> int:
    left, top, right, bottom = element.box
    return (right - left) * (bottom - top)


def _smallest_container(
    element: Element, candidates: Iterable[Element], max_area: float, slack: int = 0
) -> Element | None:
    containers = (
        c
        for c in candidates
        if c is not element
        and inside(element, c, slack)
        and contains(c, element)
        and (not max_area or area(c) <= max_area)
    )
    return min(containers, key=area, default=None)


def label_beside(element: Element, elements: list[Element], slack: int) -> str:
    y = (element.box[1] + element.box[3]) // 2
    labels = [
        other
        for other in elements
        if other is not element
        and other.visible
        and not other.kind
        and other.name
        and other.box[1] <= y <= other.box[3]
        and other.box[2] <= element.box[0] + slack
    ]
    nearest = max(labels, key=lambda other: other.box[2], default=None)
    return nearest.name if nearest else ""


def borrowed_names(
    elements: list[Element], screen_area: int, config: DecisionConfig
) -> dict[int, list[str]]:
    borrowed: dict[int, list[str]] = {}
    containers = [e for e in elements if e.clickable or e.long_clickable]
    for e in elements:
        if e.clickable or e.long_clickable or not e.name or e.kind == "text_field":
            continue
        owner = _smallest_container(e, containers, screen_area * config.max_owner_screen_share)
        if owner is not None:
            texts = borrowed.setdefault(owner.index, [])
            if e.name not in texts and len(texts) < config.borrowed_text_limit:
                texts.append(e.name)
    return borrowed


def toggle_names(
    elements: list[Element],
    borrowed: dict[int, list[str]],
    screen_area: int,
    config: DecisionConfig,
) -> dict[int, str]:
    names: dict[int, str] = {}
    for e in elements:
        if (
            e.kind not in {"switch", "checkbox", "radio_button", "option"}
            or e.name
            or not e.offerable
        ):
            continue
        owner = _smallest_container(
            e,
            (c for c in elements if c.name or borrowed.get(c.index)),
            screen_area * config.max_owner_screen_share,
            config.knob_slack,
        )
        if owner is not None:
            names[e.index] = owner.name or ", ".join(borrowed.get(owner.index, []))
        else:
            names[e.index] = label_beside(e, elements, config.knob_slack)
    return names


def keyboard_submit_indices(elements: list[Element]) -> set[int]:
    verbs = {"search", "go", "done", "return", "send", "next", "join", "continue", "route"}
    return {
        e.index
        for keyboard in elements
        if keyboard.ios and keyboard.short_class == "Keyboard" and keyboard.descendants
        for e in elements
        if contains(keyboard, e) and {e.name.lower(), e.resource.lower()} & verbs
    }
