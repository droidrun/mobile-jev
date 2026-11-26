"""Preserve choice-model metadata using the public formatter interface only."""

from typing import Any

from mobilerun.core.state.formatters import IndexedFormatter


def _nodes(node: dict[str, Any]) -> list[tuple[dict[str, Any], int]]:
    children = [item for child in node.get("children") or [] for item in _nodes(child)]
    return [(node, len(children)), *children]


def _node_details(node: dict[str, Any]) -> dict[str, Any]:
    texts = {
        "description": "contentDescription",
        "hint": "hint",
        "stateDescription": "stateDescription",
    }
    details: dict[str, Any] = (
        {}
        if "[secret:" in str(node.get("text") or "")
        else {name: str(node[source]).strip() for name, source in texts.items() if node.get(source)}
    )
    for name, source in {
        "selected": "isSelected",
        "focused": "isFocused",
        "editable": "isEditable",
        "showingHint": "isShowingHintText",
        "password": "isPassword",
    }.items():
        if node.get(source) is True:
            details[name] = True
    return details


class DecisionFormatter(IndexedFormatter):
    def format(
        self, filtered_tree: dict[str, Any] | None, phone_state: dict[str, Any]
    ) -> tuple[str, str, list[dict[str, Any]], dict[str, Any]]:
        text, focused, elements, phone = super().format(filtered_tree, phone_state)
        raw = _nodes(filtered_tree) if filtered_tree is not None else []
        for element, (node, descendants) in zip(elements, raw, strict=True):
            element["descendants"] = descendants
            if details := _node_details(node):
                element["details"] = details
            for key in ("isClickable", "isLongClickable", "isEnabled", "isVisibleToUser"):
                if key in node:
                    element[key] = node[key]
        return text, focused, elements, phone
