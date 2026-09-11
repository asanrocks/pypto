# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Type wrappers for the distributed DSL.

Exposes:

* :class:`DistributedTensor` — annotation form of
  ``ir.DistributedTensorType``.
* :class:`CommCtx` — annotation + value wrapper for
  ``ir.CommCtxType`` (the handle returned by ``pld.system.get_comm_ctx``).

Mirrors :mod:`pypto.language.typing` for the single-device DSL.
"""

from pypto.language.typing.type_utils import register_facade
from pypto.pypto_core import ir as _ir

from .comm_ctx import CommCtx
from .distributed_tensor import DistributedTensor

# Register the distributed type→facade mappings in the funnel's MRO-sorted
# registry. ``DistributedTensorType`` is a pybind subclass of
# ``TensorType``; it must be matched first or the plain ``TensorType``
# branch swallows it — the silent-downgrade failure mode the live
# ``_call_facade`` chain had. Idempotent; runs at package import.
register_facade(_ir.DistributedTensorType, lambda e: DistributedTensor(expr=e))
register_facade(_ir.CommCtxType, lambda e: CommCtx(expr=e))

__all__ = ["CommCtx", "DistributedTensor"]
