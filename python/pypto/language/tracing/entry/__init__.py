# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The entrance foundation — the public tracing-frontend entry tier.

An entrance such as ``pl.function``, ``pl.program``, or compatibility
``pl.jit`` is policy over this foundation, not a privileged frontend. The
foundation exposes exactly:

* :class:`Param` and :class:`Decl` — the immutable declaration data;
* :class:`TracedCallable` — reusable prepared instrumentation paired
  optionally with a declaration; ``invoke(args, kwargs)`` performs one
  complete root trace atomically.

The executable litmus contract: a user can implement equivalents of
``pl.function``, ``pl.program``, and compatibility ``pl.jit`` on top of this
module without importing private collector, Sema, Frame, or decorator
helpers.

The submodules above the foundation are framework libraries, not foundation
surface: :mod:`~pypto.language.tracing.entry.annotation` (the standard annotation
deriver) is the first. Explicit IR construction remains
:class:`pypto.ir.IRBuilder` and is not wrapped here.
"""

from .declaration import Decl, Infer, Param
from .traced import TracedCallable

__all__ = [
    "Decl",
    "Infer",
    "Param",
    "TracedCallable",
]
