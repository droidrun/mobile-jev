import ast


def canonical_signature(signature):
    """Normalize version-specific empty AST fields and expression contexts."""
    tree = ast.parse(signature, mode="eval")
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            node.keywords = [
                keyword
                for keyword in node.keywords
                if not (isinstance(keyword.value, ast.List) and not keyword.value.elts)
                and keyword.arg != "ctx"
            ]
    return ast.dump(tree, include_attributes=False)
