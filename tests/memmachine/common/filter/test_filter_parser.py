import pytest

from memmachine.common.filter.filter_parser import And, Comparison, Filter, Or, parse_filter


def _flatten_and(expr: And) -> list[Comparison]:
    result: list[Comparison] = []

    def _walk(node):
        if isinstance(node, And):
            _walk(node.left)
            _walk(node.right)
        else:
            assert isinstance(node, Comparison)
            result.append(node)

    _walk(expr)
    return result


def test_parse_filter_empty_string() -> None:
    assert parse_filter("") == Filter(expr=None)
    assert parse_filter(None) == Filter(expr=None)


def test_parse_filter_simple_equality() -> None:
    filt = parse_filter("owner = alice")
    assert filt.expr == Comparison(field="owner", op="=", value="alice")


def test_parse_filter_in_clause() -> None:
    filt = parse_filter("priority in (HIGH,LOW)")
    assert filt.expr == Comparison(
        field="priority",
        op="in",
        value=["HIGH", "LOW"],
    )


def test_parse_filter_boolean_and_numeric_values() -> None:
    filt = parse_filter("count = 10 AND pi = 3.14 AND done = true AND flag = FALSE")
    assert isinstance(filt.expr, And)
    children = _flatten_and(filt.expr)
    assert children[0] == Comparison(field="count", op="=", value=10)
    assert children[1] == Comparison(field="pi", op="=", value=3.14)
    assert children[2] == Comparison(field="done", op="=", value=True)
    assert children[3] == Comparison(field="flag", op="=", value=False)


def test_parse_filter_and_or_precedence() -> None:
    filt = parse_filter("owner = alice OR priority = HIGH AND status = OPEN")
    assert isinstance(filt.expr, Or)
    left = filt.expr.left
    right = filt.expr.right
    assert left == Comparison(field="owner", op="=", value="alice")
    assert isinstance(right, And)
    assert _flatten_and(right) == [
        Comparison(field="priority", op="=", value="HIGH"),
        Comparison(field="status", op="=", value="OPEN"),
    ]


def test_parse_filter_grouping_changes_precedence() -> None:
    filt = parse_filter("(owner = alice OR priority = HIGH) AND status = OPEN")
    assert isinstance(filt.expr, And)
    left = filt.expr.left
    right = filt.expr.right
    assert isinstance(left, Or)
    assert left.left == Comparison(field="owner", op="=", value="alice")
    assert left.right == Comparison(field="priority", op="=", value="HIGH")
    assert right == Comparison(field="status", op="=", value="OPEN")


def test_keywords_case_insensitive() -> None:
    filt = parse_filter("Owner In (Alice, Bob) or PRIORITY = high")
    assert isinstance(filt.expr, Or)
    assert filt.expr.left == Comparison(
        field="Owner", op="in", value=["Alice", "Bob"]
    )
    assert filt.expr.right == Comparison(
        field="PRIORITY", op="=", value="high"
    )


def test_legacy_mapping_generation() -> None:
    filt = parse_filter("owner = alice AND project = memmachine")
    assert filt.episodic_filter == {
        "owner": "alice",
        "project": "memmachine",
    }


def test_legacy_mapping_rejects_or_and_in() -> None:
    with pytest.raises(ValueError):
        parse_filter("owner = alice OR owner = bob").episodic_filter
    with pytest.raises(ValueError):
        parse_filter("owner IN (alice, bob)").episodic_filter
