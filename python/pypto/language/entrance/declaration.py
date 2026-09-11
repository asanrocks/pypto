# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The entrance foundation's declaration data — ``Param`` and ``Decl``.

``Param`` is one declared IR parameter; ``Decl`` is the complete immutable
specification for one function trace: the ordered parameters and return
contract, the stable symbol and display name, the function classification
facts, and the tracing controls that configure materialization. Neither owns
a body, a Python signature, a collector, a codec, cache identity, runtime
state, or a Python invocation recipe — an entrance combines those facts
itself.

Parameter identity — not structural equality or name lookup — resolves an
invocation placeholder: a recipe may reorder, repeat, or omit parameters by
passing the exact ``Param`` objects the declaration owns.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from pypto.pypto_core import ir

__all__ = [
    "Decl",
    "Infer",
    "Param",
    "fold_auto_scope",
]


class _InferSentinel:
    """The return-contract sentinel: derive the returns from the traced body."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "Infer"


Infer: Final = _InferSentinel()
"""The ``Decl.returns`` value meaning "no explicit return contract": the
materialized function keeps the return types the trace produced. A
declaration referenced before its body is materialized needs explicit
returns instead — a typed call cannot infer results from a future body."""


@dataclass(frozen=True, eq=False)
class Param:
    """One declared IR parameter: a name, an ``ir.Type``, and a direction.

    ``eq=False`` pins the equality domain to object identity: a placeholder
    in an invocation recipe resolves by ``is``, so two structurally equal
    parameters from different declarations can never be conflated silently.
    Name uniqueness within one declaration is a :class:`Decl` fact.
    """

    name: str
    type: ir.Type
    direction: ir.ParamDirection = ir.ParamDirection.In

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"Param name must be a non-empty str, got {self.name!r}")
        if not isinstance(self.type, ir.Type):
            raise TypeError(f"Param {self.name!r}: type must be an ir.Type, got {self.type!r}")
        if not isinstance(self.direction, ir.ParamDirection):
            raise TypeError(
                f"Param {self.name!r}: direction must be an ir.ParamDirection, got {self.direction!r}"
            )


@dataclass(frozen=True)
class Decl:
    """The complete immutable specification for one function trace.

    Fields beyond an ABI descriptor configure materialization: the source
    span, the automatic-scope policy, and the strict-SSA mode. ``name`` is
    the display name; ``symbol`` is the stable IR symbol and defaults to
    ``name``. ``returns`` is either an explicit sequence of ``ir.Type`` or
    :data:`Infer`.
    """

    name: str
    params: Sequence[Param] = ()
    returns: Sequence[ir.Type] | _InferSentinel = Infer
    symbol: str | None = None
    function_type: ir.FunctionType = ir.FunctionType.Opaque
    level: ir.Level | None = None
    role: ir.Role | None = None
    attrs: Mapping[str, Any] | None = None
    requires_runtime_binding: bool = False
    span: ir.Span | None = None
    auto_scope: bool = True
    strict_ssa: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError(f"Decl name must be a non-empty str, got {self.name!r}")
        if self.symbol is not None and not isinstance(self.symbol, str):
            raise TypeError(f"Decl {self.name!r}: symbol must be a str or None, got {self.symbol!r}")
        params = tuple(self.params)
        seen: set[str] = set()
        for param in params:
            if not isinstance(param, Param):
                raise TypeError(f"Decl {self.name!r}: params must be Param values, got {param!r}")
            if param.name in seen:
                raise ValueError(f"Decl {self.name!r}: duplicate parameter name {param.name!r}")
            seen.add(param.name)
        returns: object = self.returns
        if isinstance(returns, _InferSentinel):
            if returns is not Infer:
                raise TypeError(f"Decl {self.name!r}: returns must use the Infer singleton")
        elif isinstance(returns, Sequence) and not isinstance(returns, (str, bytes)):
            normalized = tuple(returns)
            for return_type in normalized:
                if not isinstance(return_type, ir.Type):
                    raise TypeError(
                        f"Decl {self.name!r}: returns must be ir.Type values or Infer, got {return_type!r}"
                    )
            returns = normalized
        else:
            raise TypeError(
                f"Decl {self.name!r}: returns must be a sequence of ir.Type or Infer, got {returns!r}"
            )
        # Frozen-dataclass normalization: accept any sequence at the boundary,
        # store the immutable copy (attrs is copied for the same reason).
        object.__setattr__(self, "params", params)
        object.__setattr__(self, "returns", returns)
        if self.attrs is not None:
            object.__setattr__(self, "attrs", dict(self.attrs))

    @property
    def symbol_name(self) -> str:
        """The stable IR symbol — the display name when no symbol was supplied."""
        return self.symbol if self.symbol is not None else self.name

    @property
    def explicit_returns(self) -> Sequence[ir.Type] | None:
        """The explicit return contract, or ``None`` when it is :data:`Infer`."""
        if isinstance(self.returns, _InferSentinel):
            return None
        return self.returns


def fold_auto_scope(attrs: Mapping[str, Any] | None, auto_scope: bool) -> dict[str, Any] | None:
    """Record the ``auto_scope`` policy into the function attrs.

    (the pass-side ``materialize_runtime_scopes`` reads ``attrs["auto_scope"]``
    and skips insertion when ``False``)

    Only the **non-default** value is recorded: ``auto_scope=True`` is the
    printer's "absent key ⇒ default True" convention (python_printer.cpp
    filters the key from the printed ``attrs={...}`` dict), so folding it
    into a non-empty attrs dict would break print→parse structural
    equality (Kwargs size mismatch 1 != 2)."""
    if auto_scope:
        return dict(attrs) if attrs else {}
    folded: dict[str, Any] = dict(attrs) if attrs else {}
    folded["auto_scope"] = False
    return folded
