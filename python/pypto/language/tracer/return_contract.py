# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The traced-function return contract — comparison and pinning.

Shared tail of every root trace producer (the entrance foundation's
:class:`~pypto.language.entrance.TracedCallable.invoke`, the decorator
adapter, and the program/jit envelopes): after a successful trace, a
declared flat return contract is compared against the traced return types
and pinned onto the completed ``ir.Function`` when they match. The Sema
intentionally sees no structural type for a flat multi-return signature
(the trace keeps ``return a, b`` as two ReturnStmt values), so the pin is
what makes the flat signature visible to callers and the printer.
"""

from __future__ import annotations

from collections.abc import Sequence

from pypto.ir.utils import (
    derived_return_types,  # noqa: F401 — re-exported; defined in the shared IR layer
    return_value_types,  # noqa: F401 — moved to the shared IR layer
    signature_expressible,  # noqa: F401 — re-exported; defined in the shared IR layer
)
from pypto.pypto_core import ir

__all__ = [
    "derived_return_types",
    "pin_derived_return_types",
    "rebuild_with_return_types",
    "same_return_types",
    "return_value_types",
    "signature_expressible",
]


def same_return_types(left: Sequence[ir.Type], right: Sequence[ir.Type]) -> bool:
    """Structural return-contract comparison (auto-mapping absorbs per-trace Var identity)."""
    return len(left) == len(right) and all(
        ir.structural_equal(a, b, enable_auto_mapping=True) for a, b in zip(left, right)
    )


def rebuild_with_return_types(fn: ir.Function, return_types: Sequence[ir.Type]) -> ir.Function:
    """Rebuild ``fn`` with an explicit return contract pinned onto it."""
    params = list(zip(fn.params, fn.param_directions))
    return ir.Function(
        fn.name,
        params,
        list(return_types),
        fn.body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
    )


def pin_derived_return_types(fn: ir.Function) -> ir.Function:
    """Pin the derived annotationless return contract onto ``fn``.

    A no-op when ``fn`` already carries return types — a declared ``->``
    *is* the signature (it was checked against the trace at build time),
    and a derived pin must never overwrite a declaration. The jit
    envelope's contract: upstream's specializer synthesized the inferred
    ``->`` into its rewritten source for every jit kernel; the traced
    counterpart materializes the recovered types into the Function
    signature, so a later ``pl.spmd_submit`` callee sees them as declared
    facts."""
    if fn.return_types:
        return fn
    derived = derived_return_types(fn)
    if not derived:
        return fn
    return rebuild_with_return_types(fn, derived)
