# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The execution layer every tracing producer shares.

A traced ``ir.Function`` is produced by one lifecycle, whichever tier owns
the policy around it: the body's callees are adapted for one execution,
the Sema's frame is installed as the active trace frame, the instrumented
body runs inside it, the previous frame is restored on every exit path, and
the completed trace is checked against its declared returns before they are
pinned onto the signature. This module is that lifecycle, stated once — the
entrance foundation (:mod:`pypto.language.tracing.entry`), the decorator
tier (``function_trace`` and the jit envelope), and the authoring session
all call it instead of carrying private copies that drift.

The dependency direction is the layering: the policy tiers import this
module, and this module imports only the tracer core — the collector
driver, the language Sema, the return contract. The one lazy upward import
is the jit envelope inside the callee adaptation: its per-call variant lane
is policy *over* adaptation while ``jit_trace`` imports this module, so the
cycle closes only at execution time, never at import time.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.op.marker import Marker
from pypto.language.tracing.collector.driver import InstrumentedFunction
from pypto.language.tracing.return_contract import (
    check_declared_return_values,
    check_return_paths,
    rebuild_with_return_types,
    same_return_types,
)
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

__all__ = [
    "adapt_captured_callees",
    "adapt_global_callees",
    "pin_declared_returns",
    "run_instrumented",
]


def adapt_captured_callees(
    kernel: types.FunctionType, *, specialize: bool = False
) -> tuple[types.FunctionType, list[ir.Function]]:
    """Make captured standalone functions callable for one traced invocation.

    ``specialize`` — the jit envelope's lanes: a captured callee routes
    through the per-call variant lane (the same wrapper the module-global
    lane uses), so a call-site host knob bakes identically whether the
    callee was reached through a closure cell or a global name. The
    standalone ``pl.function`` build keeps the plain ``Marker`` — the
    parser-aligned verbatim lane the conformance oracle pins."""
    closure = kernel.__closure__
    if not closure:
        return kernel, []

    captured: list[ir.Function] = []
    cells: list[types.CellType] = []
    changed = False
    for cell in closure:
        value = cell.cell_contents
        from pypto.language.tracing.decorators import jit_trace as _jit  # noqa: PLC0415

        if specialize and isinstance(value, ir.Function):
            captured.append(value)
            cells.append(types.CellType(_jit._LazyGlobalFn(value, _jit._eager_trace_record(value))))
            changed = True
        elif not specialize and (frozen := _jit.parser_aligned_callee(value)) is not None:
            # An eagerly traced jit object captured in a closure contributes
            # its eager trace to the parser-aligned lane, which must see the
            # frozen ``ir.Function`` rather than the jit envelope.
            captured.append(frozen)
            cells.append(types.CellType(Marker._from_ir_function(frozen)))
            changed = True
        else:
            cells.append(cell)
    if not changed:
        return kernel, []

    adapted = types.FunctionType(
        kernel.__code__,
        kernel.__globals__,
        name=kernel.__name__,
        argdefs=kernel.__defaults__,
        closure=tuple(cells),
    )
    adapted.__kwdefaults__ = kernel.__kwdefaults__
    adapted.__annotations__ = kernel.__annotations__
    adapted.__qualname__ = kernel.__qualname__
    adapted.__module__ = kernel.__module__
    return adapted, captured


def adapt_global_callees(kernel: types.FunctionType, *, specialize: bool = False) -> types.FunctionType:
    """Make module-global standalone ``ir.Function`` callees callable for one
    traced execution.

    The parser frontend resolved a bare-name call through its closure-var
    registry; tracing has no ambient registry, so a module-global standalone
    function (an eager-traced ``ir.Function``) reaches the traced body
    uncallable. This rebinds the kernel over an execution copy of its globals
    with every ``ir.Function`` value wrapped callable; the module's real
    globals are never touched, and a global the body never calls keeps the
    cost at one dict copy.

    ``specialize`` — the jit envelope's lanes: every ``ir.Function`` global
    becomes the jit lazy wrapper, which routes a callee registered at eager
    decoration through the per-call variant lane (re-trace with the
    call-site argument types) and every other function through the
    verbatim+registering lane. ``specialize=False`` (the standalone
    function build) keeps the plain ``Marker`` — the parser frontend's
    bare-name call is verbatim, and the conformance oracle pins that. The
    jit dispatch lane pre-wraps globals
    with the same lazy wrapper, so those values are no longer
    ``ir.Function`` and pass through.
    """
    if specialize:
        from pypto.language.tracing.decorators import (  # noqa: PLC0415 — jit imports this module
            jit_trace as _jit,
        )

        def _wrap(value: ir.Function) -> Any:
            return _jit._LazyGlobalFn(value, _jit._eager_trace_record(value))

        def _callee(value: Any) -> ir.Function | None:
            return value if isinstance(value, ir.Function) else None

    else:
        from pypto.language.tracing.decorators import jit_trace as _jit  # noqa: PLC0415

        def _wrap(value: ir.Function) -> Any:
            # The parser-aligned lane's callee is the frozen function — an
            # eagerly traced jit object unwraps to its decoration trace
            # (no envelope policy: no world checks, no re-trace).
            return Marker._from_ir_function(value)

        _callee = _jit.parser_aligned_callee

    patched = False
    g = dict(kernel.__globals__)
    for name, value in g.items():
        frozen = _callee(value)
        if frozen is not None:
            g[name] = _wrap(frozen)
            patched = True
    if not patched:
        return kernel
    adapted = types.FunctionType(
        kernel.__code__,
        g,
        name=kernel.__name__,
        argdefs=kernel.__defaults__,
        closure=kernel.__closure__,
    )
    adapted.__kwdefaults__ = kernel.__kwdefaults__
    adapted.__annotations__ = kernel.__annotations__
    adapted.__qualname__ = kernel.__qualname__
    adapted.__module__ = kernel.__module__
    return adapted


def run_instrumented(
    runner: InstrumentedFunction[Any],
    sema: PtoSema,
    args: Sequence[object],
    kwargs: dict[str, object] | None = None,
) -> object:
    """Install the Sema's frame and run one instrumented body inside it.

    The run loop every producer shares: the active-trace ContextVar points
    at this Sema's frame while the body executes and is restored on every
    exit path, so a failed trace leaves no partial frame installed — the
    one-shot Sema's own abort discipline owns the rest of the cleanup. The
    frame must already exist: parameter binding (or the splice's explicit
    preparation) creates it, and running a body with no frame bound would
    silently emit into whatever frame was active before.
    """
    frame = sema.frame
    if frame is None:
        raise RuntimeError(
            "the trace frame was never prepared — bind the Sema's parameters before running its body"
        )
    token = set_active_frame(frame)
    try:
        return runner(*args, **(kwargs or {}))
    finally:
        reset_active_frame(token)


def pin_declared_returns(fn: ir.Function, declared: Sequence[ir.Type] | None) -> ir.Function:
    """Check a completed trace against its declared returns and pin them.

    The finalization half every producer shares. The traced return paths
    are verified first, declaration or not: a body whose alternative paths
    disagree is refused here, because an empty or Unknown signature would
    hide the disagreement from every later caller. A declared contract is
    then compared against the traced values — never stamped over them —
    and pinned onto the function when it agrees structurally. ``declared``
    empty or ``None`` verifies the paths and keeps the traced return types.
    """
    actual = check_return_paths(fn)
    if not declared:
        return fn
    check_declared_return_values(fn, declared, actual)
    if not same_return_types(fn.return_types, declared):
        fn = rebuild_with_return_types(fn, declared)
    return fn
