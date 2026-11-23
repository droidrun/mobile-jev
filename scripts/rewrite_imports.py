"""Rewrite migration imports using AST locations, leaving other source text intact."""

import argparse
import ast
import io
import json
import re
import tokenize
from pathlib import Path


def mapped(name, mapping):
    for old in sorted(mapping, key=len, reverse=True):
        if name == old or name.startswith(old + "."):
            return mapping[old] + name[len(old) :]
    return name


def rewrite(text, module, is_package, mapping):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    edits = []
    tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))

    def position(line, bytecol):
        return offsets[line - 1] + len(lines[line - 1].encode("utf-8")[:bytecol].decode("utf-8"))

    def span(node):
        return position(node.lineno, node.col_offset), position(
            node.end_lineno, node.end_col_offset
        )

    def tokenpos(pos):
        return offsets[pos[0] - 1] + pos[1]

    def edit(start, end, replacement):
        if text[start:end] != replacement:
            edits.append((start, end, replacement))

    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                new = mapped(alias.name, mapping)
                if new != alias.name:
                    a, _ = span(alias)
                    edit(a, a + len(alias.name), new)
                    if not alias.asname:
                        imported[alias.name] = new
        elif isinstance(node, ast.ImportFrom):
            old = node.module or ""
            if node.level:
                package = module if is_package else module.rpartition(".")[0]
                parts = package.split(".")
                old = ".".join(parts[: len(parts) - node.level + 1] + ([old] if old else []))
            targets = []
            for alias in node.names:
                child = old + "." + alias.name
                if child in mapping:
                    parent, _, name = mapping[child].rpartition(".")
                else:
                    parent, name = mapped(old, mapping), alias.name
                targets.append((parent, name, alias))
            parent = targets[0][0]
            start, end = span(node)
            prefix = text[start : span(node.names[0])[0]]
            match = re.match(r"from\s+([.\w]+)\s+import\b", prefix)
            if not match:
                raise ValueError(f"unsupported from import at line {node.lineno}")
            edit(start + match.start(1), start + match.end(1), parent)
            moved = [(i, target) for i, target in enumerate(targets) if target[0] != parent]
            retained = [(i, target) for i, target in enumerate(targets) if target[0] == parent]
            if moved:
                if any(
                    token.type == tokenize.COMMENT
                    and (
                        start <= tokenpos(token.start) < end
                        or token.start[0] == node.end_lineno
                        and tokenpos(token.start) >= end
                    )
                    for token in tokens
                ):
                    raise ValueError(
                        f"line {node.lineno}: split import with comments; rewrite manually"
                    )
                inserts = []
                for i, (target_parent, name, alias) in moved:
                    local = alias.asname or alias.name
                    inserts.append(
                        f"from {target_parent} import {name}"
                        + (f" as {local}" if local != name else "")
                    )
                    a, b = span(alias)
                    edit(a, b, "")
                    next_alias = span(node.names[i + 1])[0] if i + 1 < len(node.names) else end
                    commas = [
                        t for t in tokens if t.string == "," and b <= tokenpos(t.start) < next_alias
                    ]
                    if not commas or not any(index > i for index, _ in retained):
                        previous_end = span(node.names[i - 1])[1]
                        commas = [
                            t
                            for t in tokens
                            if t.string == "," and previous_end <= tokenpos(t.start) < a
                        ]
                    if not commas:
                        raise ValueError(f"cannot split import at line {node.lineno}")
                    t = commas[0]
                    edit(tokenpos(t.start), tokenpos(t.end), "")
                edits.append((start, start, "; ".join(inserts) + "; "))
            for _, (_, name, alias) in retained:
                if name != alias.name:
                    a, _ = span(alias)
                    edit(
                        a,
                        a + len(alias.name),
                        name + (" as " + alias.name if not alias.asname else ""),
                    )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if re.fullmatch(r"mobilerun(?:\.[A-Za-z_]\w*)+", node.value):
                changed = mapped(node.value, mapping)
                if changed != node.value:
                    a, b = span(node)
                    literal = text[a:b]
                    idx = literal.find(node.value)
                    if idx >= 0:
                        edit(a + idx, a + idx + len(node.value), changed)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or isinstance(parents.get(node), ast.Attribute):
            continue
        a, b = span(node)
        source = text[a:b]
        parts = []
        cursor = node
        while isinstance(cursor, ast.Attribute):
            parts.append(cursor.attr)
            cursor = cursor.value
        if not isinstance(cursor, ast.Name):
            continue
        dotted = ".".join([cursor.id, *reversed(parts)])
        for old, new in sorted(imported.items(), key=lambda item: len(item[0]), reverse=True):
            if dotted == old or dotted.startswith(old + "."):
                if not isinstance(node.ctx, ast.Load):
                    raise ValueError(f"unsupported qualified assignment at line {node.lineno}")
                if not (source == old or source.startswith(old + ".")):
                    raise ValueError(
                        f"unsupported qualified reference layout at line {node.lineno}"
                    )
                edit(a, a + len(old), new)
                break
    for start, end, value in sorted(edits, reverse=True):
        text = text[:start] + value + text[end:]
    return text


def legacy_imports(text, module, is_package, mapping):
    """Find references to moved old modules without treating prose as code."""
    moved = {old for old, new in mapping.items() if old != new}
    found = []

    def inspect(name, line):
        for old in sorted(moved, key=len, reverse=True):
            if name == old or name.startswith(old + "."):
                if mapped(name, mapping) != name:
                    found.append((line, name))
                break

    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                inspect(alias.name, node.lineno)
        elif isinstance(node, ast.ImportFrom):
            old = node.module or ""
            if node.level:
                package = module if is_package else module.rpartition(".")[0]
                parts = package.split(".")
                old = ".".join(parts[: len(parts) - node.level + 1] + ([old] if old else []))
            inspect(old, node.lineno)
            for alias in node.names:
                inspect(old + "." + alias.name, node.lineno)
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and re.fullmatch(r"mobilerun(?:\.[A-Za-z_]\w*)+", node.value)
        ):
            inspect(node.value, node.lineno)
    return sorted(set(found))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--map", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    mapping = json.loads(args.map.read_text())
    changed = []
    for path in args.root.rglob("*.py"):
        agent_root = next(
            (
                parent
                for parent in path.parents
                if parent.name == "agents" and parent.parent.name == "mobilerun"
            ),
            args.root,
        )
        relative = path.relative_to(agent_root)
        if args.check and (
            relative.parts[0] in {"agent", "agent_macro"}
            or relative.as_posix() in {"operator.py", "logging_setup.py", "errors/timeout.py"}
        ):
            continue
        parts = list(relative.with_suffix("").parts)
        package = parts[-1] == "__init__"
        if package:
            parts.pop()
        module = "mobilerun.agents" + ("." + ".".join(parts) if parts else "")
        original = path.read_text()
        if args.check:
            changed.extend(
                f"{path}:{line}: {name}"
                for line, name in legacy_imports(original, module, package, mapping)
            )
            continue
        inverse = {new: old for old, new in mapping.items() if old != new}
        old_module = inverse.get(module, module)
        try:
            result = rewrite(original, old_module, package, mapping)
        except ValueError as exc:
            raise SystemExit(f"{path}: {exc}") from exc
        if result != original:
            path.write_text(result)
    if args.check and changed:
        print("\n".join(changed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
