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
body runs inside it, and the previous frame is restored on every exit path.
This module is that lifecycle, stated once — the entrance foundation
(:mod:`pypto.language.tracing.entry`) and the decorator tier
(``function_trace`` and the jit envelope) call it instead of carrying
private copies that drift.

The dependency direction is the layering: the policy tiers import this
module, and this module imports only the tracer core — the collector
driver, the language Sema, the return contract.
"""

from __future__ import annotations

import types
from collections.abc import Sequence
from typing import Any

from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.op.marker import Marker
from pypto.language.tracing.collector.driver import InstrumentedFunction
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

__all__ = [
    "adapt_captured_callees",
    "adapt_global_callees",
    "run_instrumented",
]


def adapt_captured_callees(
    kernel: types.FunctionType,
) -> tuple[types.FunctionType, list[ir.Function]]:
    """Make captured standalone functions callable for one traced invocation.

    A closure cell holding a standalone ``ir.Function`` is rebound to the
    ``Marker`` that emits its detached call; the module's real cells are never
    touched. The per-call variant lane belongs to the jit envelope and
    rebinds these to its own wrapper instead.
    """
    closure = kernel.__closure__
    if not closure:
        return kernel, []

    captured: list[ir.Function] = []
    cells: list[types.CellType] = []
    changed = False
    for cell in closure:
        value = cell.cell_contents
        if isinstance(value, ir.Function):
            captured.append(value)
            cells.append(types.CellType(Marker._from_ir_function(value)))
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


def adapt_global_callees(kernel: types.FunctionType) -> types.FunctionType:
    """Make module-global standalone ``ir.Function`` callees callable.

    The parser frontend resolved a bare-name call through its closure-var
    registry; tracing has no ambient registry, so a module-global standalone
    function reaches the traced body uncallable. This rebinds the kernel over
    an execution copy of its globals with every ``ir.Function`` value wrapped
    callable; the module's real globals are never touched.
    """
    patched = False
    g = dict(kernel.__globals__)
    for name, value in g.items():
        if isinstance(value, ir.Function):
            g[name] = Marker._from_ir_function(value)
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
