# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Function-level attribute declaration — ``pl.func_attr``.

Written as the **first statement** of a function body, ``pl.func_attr``
declares metadata about the function as a whole, including attributes that
reference the function's own parameters (a decorator cannot: it evaluates
before the signature binds any name)::

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32],
               w: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        pl.func_attr({"stationary": w})
        ...

The dict lands in ``Function.attrs`` and emits no IR statement of its own.
Keys must be string literals; a bare name is always a parameter reference;
multiple calls (or a call plus a decorator ``attrs=``) merge, with a repeated
key rejected.

Under the tracing frontend the call is an **expression-statement marker**: it
returns a marker object whose trace hook validates and coerces the dict (the
value rules mirror the AST parser's ``_parse_func_attr_value``) and delegates
the prologue-placement check and the attrs merge to the tracing context's
named operation. Outside a trace the call is inert: the parser frontend
intercepts ``pl.func_attr`` before executing the body.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pypto._function_attrs import AUTO_SCOPE_ATTR, EXTERNAL_SOURCE_ATTR
from pypto.language.ambient import active_frame
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.facade import is_facade
from pypto.language.protocol import TraceContext
from pypto.pypto_core import ir

#: Keys the frontends consume before the body runs, so a body-position
#: declaration would arrive too late to take effect. Both spellings keep their
#: dedicated ``@pl.function(...)`` keyword (the parser rejects them in
#: ``pl.func_attr`` for the same reason).
_DECORATOR_ONLY_FUNC_ATTRS: frozenset[str] = frozenset({AUTO_SCOPE_ATTR, EXTERNAL_SOURCE_ATTR})

#: Sentinel: an attr value that must not be stored at all — ``SplitMode.NONE``
#: is a non-canonical spelling of "no split" that the printer filters, so
#: storing it would break the print -> parse round trip.
_OMIT_ATTR: object = object()


def func_attr(attrs: dict[str, Any]) -> Any:
    """Attach function-level attributes from inside the function body.

    Args:
        attrs: Attribute dict. Keys must be string literals; values may
            reference the function's parameters.

    Returns:
        An expression-statement marker under tracing; ``None`` outside a
        trace (the parser frontend intercepts the directive before the body
        executes).
    """
    if active_frame() is None:
        return None
    return _FuncAttrMarker(attrs)


class _FuncAttrMarker:
    """Expression-statement marker returned by ``func_attr`` inside a trace.

    ``annotation_state`` is true: the directive collects no IR of its own —
    the trace interpreter's first-statement fact does not count it, so a
    prologue may hold several directives."""

    __slots__ = ("attrs",)

    annotation_state = True

    def __init__(self, attrs: dict[str, Any]) -> None:
        self.attrs: dict[str, Any] = attrs

    def __trace_stmt__(self, ctx: TraceContext) -> None:
        ctx.collect_func_attr(_validated_attrs(self.attrs))


def _validated_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    """Validate and coerce one ``pl.func_attr`` dict to its stored form.

    The value rules mirror the AST parser's ``_parse_func_attr_value`` (which
    shares the syntax-inference contract with the printer): host scalars pass
    through, a staged value unwraps to its ``Var`` (a parameter reference —
    the prologue is the only place a binding can already exist), lists coerce
    elementwise, and ``split`` takes the ``pl.SplitMode`` enum spelled and
    printed by the printer, stored as its lowered integer.
    """
    validated: dict[str, Any] = {}
    for key, value in attrs.items():
        if type(key) is not str:
            raise ParserSyntaxError(
                f"pl.func_attr() key must be a string literal, got {key!r}",
                hint='Use string literal keys, e.g. pl.func_attr({"split": ...}).',
            )
        if key in _DECORATOR_ONLY_FUNC_ATTRS:
            raise ParserSyntaxError(
                f"'{key}' cannot be set with pl.func_attr()",
                hint=(
                    f"Pass it as the '{key}=' keyword of @pl.function(...) instead. The frontend "
                    "reads it before the body runs, so a body-position declaration comes "
                    "too late to take effect."
                ),
            )
        coerced = _coerce_attr_value(key, value)
        if coerced is _OMIT_ATTR:
            continue
        validated[key] = coerced
    return validated


def _coerce_attr_value(key: str, value: Any) -> Any:
    """Coerce one attr value from its runtime form to the stored form."""
    if key == "split":
        # ``split`` stores an int but is spelled (and printed) as the
        # ``pl.SplitMode.X`` enum — the parser's dedicated handling, including
        # dropping ``SplitMode.NONE`` (storing the 0 would be a non-canonical
        # spelling of "no split" that the printer filters).
        if not isinstance(value, ir.SplitMode):
            raise ParserSyntaxError(
                f"attrs['split'] on call to 'pl.func_attr': expected pl.SplitMode.X, got {value!r}",
                hint="Use the enum spelling, e.g. pl.func_attr({'split': pl.SplitMode.UP_DOWN}).",
            )
        if value == ir.SplitMode.NONE:
            return _OMIT_ATTR
        return value.value
    if isinstance(value, (bool, int, float, str)):
        return value
    if is_facade(value):
        return value.unwrap()
    if isinstance(value, (list, tuple)):
        return [_coerce_attr_value(key, element) for element in _as_elements(value)]
    return value


def _as_elements(value: Any) -> Sequence[Any]:
    """Re-type a runtime list/tuple attr value as its element sequence.

    A function boundary instead of a local annotation: narrowing ``Any`` with
    ``isinstance`` leaves the element type unknown to the strict overlay, and
    a parameter is the one place the element openness is stated outright."""
    return value
