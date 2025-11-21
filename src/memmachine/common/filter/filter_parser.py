"""Module for parsing filter strings into dictionaries."""

import re
from dataclasses import dataclass
from typing import NamedTuple, Protocol

from memmachine.common.data_types import FilterablePropertyValue


class FilterExpr(Protocol):
    """Marker protocol for filter expression nodes."""



@dataclass(frozen=True)
class Comparison(FilterExpr):
    field: str
    op: str  # "=", "in"
    value: FilterablePropertyValue | list[FilterablePropertyValue]


@dataclass(frozen=True)
class And(FilterExpr):
    left: FilterExpr
    right: FilterExpr


@dataclass(frozen=True)
class Or(FilterExpr):
    left: FilterExpr
    right: FilterExpr


@dataclass(frozen=True)
class Filter:
    """Container for a parsed filter expression."""

    expr: FilterExpr | None

    @property
    def session_data_filter(self) -> dict[str, FilterablePropertyValue] | None:
        """Return a mapping accepted by existing session store filters."""

        return self._as_simple_equality_mapping()

    @property
    def episodic_filter(self) -> dict[str, FilterablePropertyValue] | None:
        """Return a mapping accepted by episodic memory filters."""

        return self._as_simple_equality_mapping()

    def _as_simple_equality_mapping(
        self,
    ) -> dict[str, FilterablePropertyValue] | None:
        if self.expr is None:
            return None

        comparisons = _flatten_conjunction(self.expr)
        if not comparisons:
            return None

        legacy: dict[str, FilterablePropertyValue] = {}
        for comp in comparisons:
            if comp.op != "=":
                raise ValueError(
                    "Legacy property filters only support '=' comparisons",
                )
            value = comp.value
            if isinstance(value, list):
                raise ValueError(
                    "Legacy property filters do not support 'IN' values",
                )
            legacy[comp.field] = value
        return legacy


class Token(NamedTuple):
    type: str
    value: str


_TOKEN_SPEC = [
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("EQ", r"="),
    ("IDENT", r"[A-Za-z0-9_\.]+"),
    ("WS", r"\s+"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC)
)


def _tokenize(s: str) -> list[Token]:
    tokens: list[Token] = []
    for m in _TOKEN_RE.finditer(s):
        kind = m.lastgroup
        value = m.group()
        if kind == "WS":
            continue
        if kind == "IDENT":
            upper = value.upper()
            if upper in ("AND", "OR", "IN"):
                tokens.append(Token(upper, upper))
            else:
                tokens.append(Token("IDENT", value))
        else:
            tokens.append(Token(kind, value))
    return tokens


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> Token | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _accept(self, *types: str) -> Token | None:
        tok = self._peek()
        if tok and tok.type in types:
            self.pos += 1
            return tok
        return None

    def _expect(self, *types: str) -> Token:
        tok = self._peek()
        if not tok or tok.type not in types:
            expected = " or ".join(types)
            actual = tok.type if tok else "EOF"
            raise ValueError(f"Expected {expected}, got {actual}")
        self.pos += 1
        return tok

    def parse(self) -> FilterExpr | None:
        if not self.tokens:
            return None
        expr = self._parse_or()
        if self._peek() is not None:
            raise ValueError(f"Unexpected token: {self._peek()}")
        return expr

    def _parse_or(self) -> FilterExpr:
        expr = self._parse_and()
        while self._accept("OR"):
            right = self._parse_and()
            expr = Or(left=expr, right=right)
        return expr

    def _parse_and(self) -> FilterExpr:
        expr = self._parse_primary()
        while self._accept("AND"):
            right = self._parse_primary()
            expr = And(left=expr, right=right)
        return expr

    def _parse_primary(self) -> FilterExpr:
        if self._accept("LPAREN"):
            expr = self._parse_or()
            self._expect("RPAREN")
            return expr
        return self._parse_comparison()

    def _parse_comparison(self) -> FilterExpr:
        field_tok = self._expect("IDENT")
        field = field_tok.value

        if self._accept("EQ"):
            # field = value
            value = self._parse_value()
            return Comparison(field=field, op="=", value=value)

        if self._accept("IN"):
            self._expect("LPAREN")
            values: list[FilterablePropertyValue] = []
            values.append(self._parse_value())
            while self._accept("COMMA"):
                values.append(self._parse_value())
            self._expect("RPAREN")
            return Comparison(field=field, op="in", value=values)

        raise ValueError(f"Expected '=' or IN after field {field}")

    def _parse_value(self) -> FilterablePropertyValue:
        tok = self._expect("IDENT")
        raw = tok.value
        upper = raw.upper()
        if upper == "TRUE":
            return True
        if upper == "FALSE":
            return False
        if raw.isdigit():
            return int(raw)
        if _looks_like_float(raw):
            return float(raw)
        return raw


def _looks_like_float(value: str) -> bool:
    if value.count(".") != 1:
        return False
    left, right = value.split(".")
    return bool(left) and bool(right) and left.isdigit() and right.isdigit()


def parse_filter(spec: str | None) -> Filter:
    """Parse the given textual filter specification."""

    if spec is None:
        return Filter(expr=None)
    spec = spec.strip()
    if not spec:
        return Filter(expr=None)
    tokens = _tokenize(spec)
    expr = _Parser(tokens).parse()
    return Filter(expr=expr)


def _flatten_conjunction(expr: FilterExpr) -> list[Comparison]:
    if isinstance(expr, Comparison):
        return [expr]
    if isinstance(expr, And):
        flattened: list[Comparison] = []
        flattened.extend(_flatten_conjunction(expr.left))
        flattened.extend(_flatten_conjunction(expr.right))
        return flattened
    raise ValueError(
        "Only AND expressions made of simple comparisons can be flattened",
    )
