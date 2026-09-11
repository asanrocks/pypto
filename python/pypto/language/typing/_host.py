# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The shared operand boundary host→constant coercion.

A host value that reaches a stdlib operator as a scalar operand is
lifted to a typed constant at that boundary — the *only* automatic
host-literal lifting in the dialect. The canonical dtypes
mirror the parser frontend's ``parse_constant`` (int→INDEX, float→
DEFAULT_CONST_FLOAT, bool→ConstBool) so both frontends build the same Const
node for the same source. ``bool`` is checked before ``int`` (a Python bool
is an int subclass). A raw ``ir.Expr`` passes through unchanged; any other
host type is a loud error.
"""

from __future__ import annotations

from typing import Any

from pypto.pypto_core import DataType, ir


def coerce_host_scalar(v: object, op_name: str, side: str) -> ir.Expr:
    """Reify a host scalar operand as a typed Const (operand
    boundary). Returns ``v`` unchanged when it is already an ``ir.Expr``;
    lifts bool/int/float to their canonical Const nodes; every other host
    type is a loud error naming the op and the operand side."""
    if isinstance(v, ir.Expr):
        return v
    span = ir.Span.unknown()
    if isinstance(v, bool):
        return ir.ConstBool(v, span)
    if isinstance(v, int):
        return ir.ConstInt(v, DataType.INDEX, span)
    if isinstance(v, float):
        return ir.ConstFloat(v, DataType.DEFAULT_CONST_FLOAT, span)
    raise TypeError(
        f"{op_name}: {side} is the host value {v!r} of an unsupported type "
        f"({type(v).__name__}); the operand boundary reifies bool/int/float "
        f"— reify it with pl.const(..., dtype) instead"
    )


def wrap_output_like(dest: Any, expr: ir.Expr) -> Any:
    """Reconstruct a facade of ``dest``'s class around ``expr``.

    An in-place/returning op's output shares the destination operand's class
    so downstream chaining type-checks (the ``X.__class__(expr=...)``
    pattern preserves distributed subclasses).

    The ``Any`` here is honest, not lazy: ``dest.__class__(expr=...)`` is
    polymorphic reconstruction whose "returns ``T``-of-``dest``" shape a
    TypeVar cannot express — ``plain_cls`` is the base fallback (``Tensor``),
    not the subtype (``DistributedTensor``) the call site's ``-> _TensorT``
    expects — so a TypeVar breaks the ``plain_cls=Tensor`` case. The two
    call sites (``tile_ops.store``/``mscatter``) return ``_TensorT``; the
    runtime preserves the subclass via ``dest.__class__``, the type can't."""
    return dest.__class__(expr=expr)


__all__ = ["coerce_host_scalar", "wrap_output_like"]
