# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``__trace_bind__`` structural-projection helpers (the Sema's
``_EmitCtx`` channel — a builder lent through the tracing context,
narrowed to a
value-side structural side-effect).

Shared by the collect-tier facades that yield a *structural* tuple
(``pl.submit``'s ``(result, tid)`` and ``pl.Tuple``'s ``MakeTuple``/multi-return
call): the facade is builder-blind (the collect tier never touches the
builder), so it returns projection facades carrying ``__trace_bind__`` + a
shared ``_EmitState``; the Sema dispatches the bind hook at the first bind, naming
the shared node once via ``ctx.let_temp`` (the AST parser's ``_tuple_tmp`` /
``_submit_tmp`` shape), and the subsequent projections reuse the temp Var.

The Var-wrapped case (a facade already bound to its ``ir.Var`` — e.g.
``ret = call`` then ``ret[0]``) projects directly: that ``ir.Var`` is the
temp, so no emit fires.
"""

from __future__ import annotations

from typing import Any

from pypto.language.typing.scalar import Scalar
from pypto.pypto_core import ir


class _EmitState:
    """The stdlib-internal shared state of a structural tuple — the raw IR
    node (built eagerly, never emitted) + the named temp Var (set by the
    first ``__trace_bind__``; reused by the subsequent projections).
    Op-internal dedup — the Sema never sees it (no read-view)."""

    def __init__(self, node: ir.Expr) -> None:
        self.node = node
        self.temp: ir.Var | None = None


def _structural_projection(facade: Any, state: _EmitState, index: int, hint: str) -> Any:
    """Attach ``__trace_bind__`` to a projection facade. The hook names
    the shared node once (via ``ctx.let_temp`` — the first projection;
    ``state.temp`` dedups), re-wraps the facade's expr over
    ``Var(temp)``, and returns the facade for the Sema's ``_default_bind``."""

    def emit(ctx: Any, name: str) -> Any:
        if state.temp is None:
            state.temp = ctx.let_temp(state.node, hint)
        assert state.temp is not None  # set above; pyright closure narrowing
        expr = ir.TupleGetItemExpr(state.temp, index, ir.Span.unknown())
        if isinstance(facade, Scalar):
            facade.expr = expr
            facade.dtype = None
        else:
            facade._expr = expr  # type: ignore[attr-defined]
        facade._annotation_only = False  # type: ignore[attr-defined]
        return facade

    facade.__trace_bind__ = emit  # type: ignore[attr-defined]
    return facade


__all__ = ["_EmitState", "_structural_projection"]
