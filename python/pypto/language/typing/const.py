# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""The typed-constant constructor (``pl.const``).

This is a facade-value-constructor — it yields an IR ``Const*`` node
wrapped in a ``Scalar`` so that a bind stages a typed local and the
typed staged initialization requirement holds.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

if TYPE_CHECKING:
    # ``Scalar`` is the return type; a top-level runtime import would cycle
    # (``scalar`` imports ``type_utils``), so it is only needed for checking.
    from pypto.language.typing import Scalar


def const(value: bool | int | float, dtype: DataType) -> Scalar:
    """Create a typed constant with an explicit dtype.

    The parser frontend intercepts ``pl.const()`` at parse time and builds a
    ``ConstBool``, ``ConstInt``, or ``ConstFloat`` with the specified dtype.
    The executed path reifies it the same way — ``pl.const`` yields a
    ``Scalar`` wrapping the Const node, so a bind stages a typed local and
    the typed staged initialization requirement holds.

    Args:
        value: Boolean, integer, or floating-point value
        dtype: DataType for the constant; boolean values require ``BOOL``

    Returns:
        A ``Scalar`` wrapping the typed constant node.
    """
    from pypto.language.typing import Scalar  # noqa: PLC0415 (avoid import cycles)

    span = _ir.Span.unknown()
    if isinstance(value, bool):
        if dtype != DataType.BOOL:
            raise TypeError("boolean constants require dtype=pl.BOOL")
        expr = _ir.ConstBool(value, span)
    elif isinstance(value, float):
        expr = _ir.ConstFloat(value, dtype, span)
    else:
        expr = _ir.ConstInt(value, dtype, span)
    return Scalar(expr=expr)
