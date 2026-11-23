import pytest
from signature_helper import canonical_signature


def test_empty_fields_from_different_python_versions_match():
    assert canonical_signature(
        "arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[])"
    ) == canonical_signature("arguments()")


@pytest.mark.parametrize(
    "changed",
    [
        "arguments(args=[arg(arg='y')])",
        "arguments(args=[arg(arg='x')], defaults=[Constant(value=1)])",
        "arguments(args=[arg(arg='x', annotation=Name(id='int'))])",
        "arguments(args=[arg(arg='x')], kwonlyargs=[arg(arg='y')])",
        "arguments(args=[arg(arg='x')], vararg=arg(arg='rest'))",
        "arguments(args=[arg(arg='x')], kwarg=arg(arg='rest'))",
        "arguments(args=[arg(arg='x')], defaults=[List()])",
    ],
)
def test_meaningful_signature_changes_remain_distinct(changed):
    assert canonical_signature("arguments(args=[arg(arg='x')])") != canonical_signature(changed)


def test_expression_context_is_ignored():
    assert canonical_signature(
        "arguments(args=[arg(arg='x', annotation=Name(id='int', ctx=Load()))])"
    ) == canonical_signature("arguments(args=[arg(arg='x', annotation=Name(id='int'))])")


def test_required_keyword_only_differs_from_none_default():
    required = "arguments(kwonlyargs=[arg(arg='y')], kw_defaults=[None])"
    defaulted = "arguments(kwonlyargs=[arg(arg='y')], kw_defaults=[Constant(value=None)])"
    assert canonical_signature(required) != canonical_signature(defaulted)
