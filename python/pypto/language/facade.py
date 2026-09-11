# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The closed facade vocabulary (the interpreter-owned staged-execution typing follow-up).

A **facade** is a stdlib value wrapper carrying the ``ir.Expr`` an op
produced; ops build detached expressions over facades. The set is closed —
seven base facades, normalization to one of them, subclasses included by
nominal subtyping (e.g. ``DistributedTensor`` is a ``Tensor``) — so the
static representation is a union, not a base class: every base facade
already carries a distinct annotation-machinery metaclass, an ABC would
force surgery on all of them, and nominal inheritance is open-world where
the set is closed by contract. The structural-protocol idiom is reserved
for *trace-protocol objects* (staging by trace-protocol hook membership — the
collector contract); reusing it for facades would erase the
facade/trace-protocol distinction that the
Frame's recognition queries depend on.

Every facade defines ``unwrap() -> Expr`` (see the canonical structural
Protocol ``_UnwrapsToExpr`` in ``language/typing/type_utils.py`` —
deliberately broader: it also matches internal expression wrappers).

The runtime tuple :data:`FACADE_TYPES` is derived from the static union —
one source of truth, no drift between the checker's set and the isinstance
set.
"""

from __future__ import annotations

from typing import TypeGuard, get_args

from pypto.language.distributed.typing import CommCtx
from pypto.language.typing import Array, Ptr, Scalar, Tensor, Tile, Tuple
from pypto.language.typing.prefetch_handle import AsyncEvent, AsyncSession, PrefetchAsyncContext

__all__ = ["FACADE_TYPES", "Facade", "is_facade"]

#: The closed union of stdlib facades — the staged value representation
#: ``Frame`` produces, the ``V`` the collector generic tower is instantiated with
#: on the language side.
Facade = (
    Tensor | Tile | Scalar | Array | Ptr | Tuple | CommCtx | PrefetchAsyncContext | AsyncEvent | AsyncSession
)

#: The runtime form of :data:`Facade` — the isinstance set behind the
#: Frame's recognition queries (derived, never hand-listed).
FACADE_TYPES: tuple[type[object], ...] = get_args(Facade)


def is_facade(v: object) -> TypeGuard[Facade]:
    """Narrow a value to the closed facade union.

    The bind-path half of the recognition query: facades are recognized
    nominally. A value staged *by trace protocol* (``__trace_iter__`` /
    ``__trace_enter__`` membership) is not a facade — see
    ``is_staged`` in ``language/protocol.py`` for the union
    query.
    """
    return isinstance(v, FACADE_TYPES)
