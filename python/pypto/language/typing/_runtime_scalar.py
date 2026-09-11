# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The entry-call runtime-scalar marker: ``pl.runtime``.

A parameter's world is pinned by its declaration (types flow from
values): a plain ``int``/``float``/``bool``
(or unannotated) parameter is a **host knob** (baked at trace time, folded,
never IR); a ``pl.Scalar[...]`` parameter is a **runtime parameter** (always
in the signature, always a runtime argument). Because the host caller of a
jit entry holds Python scalars, the runtime-parameter world needs an explicit
spelling: ``kernel(x, pl.runtime(seq_len))`` binds ``seq_len`` as a runtime
scalar of the declared type — one artifact serves every value (the dispatch
cache keys on the dtype, never the value).
"""

from __future__ import annotations

from typing import Any

from pypto.pypto_core import DataType

from .scalar import Scalar

__all__ = ["RuntimeScalar", "runtime"]


def _infer_dtype(value: Any) -> DataType:
    """The host-scalar coercion defaults (the operand-boundary promotion)."""
    if isinstance(value, bool):
        return DataType.BOOL
    if isinstance(value, int):
        return DataType.INT32
    if isinstance(value, float):
        return DataType.FP32
    raise TypeError(
        f"pl.runtime accepts a host bool/int/float, got {type(value).__name__!r} — "
        "a runtime scalar must be a scalar value"
    )


class RuntimeScalar(Scalar):
    """An entry-call marker binding a host value as a **runtime scalar**.

    An annotation-mode ``Scalar`` facade (``dtype`` set, no expr) carrying
    the raw host value: the trace binds a runtime scalar parameter of the
    declared type, and the dispatch lane unwraps the marker back to the raw
    value for ``CompiledProgram.__call__``. Only the dtype participates in
    the dispatch cache key — never the value.
    """

    def __init__(self, value: Any, dtype: DataType | None = None) -> None:
        resolved = _infer_dtype(value) if dtype is None else dtype
        super().__init__(dtype=resolved, _annotation_only=True)
        self.value = value

    def __repr__(self) -> str:
        return f"pl.runtime({self.value!r}, {self.dtype})"


def runtime(value: Any, dtype: DataType | None = None) -> RuntimeScalar:
    """Mark a host scalar as a **runtime input** of a jit entry call.

    The counterpart of a plain host scalar (which binds a host-declared
    parameter and bakes into the trace): ``pl.runtime`` binds a
    ``pl.Scalar[...]``-declared parameter as a runtime argument, so one
    compiled artifact serves every value.

    Args:
        value: The host scalar (bool/int/float; ctypes scalars at dispatch).
        dtype: The scalar dtype; defaults to the host promotion rules
            (``int``→``INT32``, ``float``→``FP32``, ``bool``→``BOOL``).
    """
    return RuntimeScalar(value, dtype)  # type: ignore[return-value]  # metaclass __call__ typed as -> Scalar
