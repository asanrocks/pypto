# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.<op>`` unified-dispatch bindings.

The unified elementwise wrappers (:mod:`pypto.language.op.unified_ops`)
dispatch on the operand's facade kind; the ``pl`` binding must select them,
not a kind-specific wrapper from ``tile_ops``/``tensor_ops``. #1495 added
the unified ``log`` but left ``pl.log`` bound to the tile-only wrapper —
``pl.log(tensor)`` reached the C++ ``tile.log`` deducer and failed its
precise ``TileType`` check (the deepseek v4-pro ``gate`` softplus shape).
These tests pin the binding and the dispatch so the class of bug — a
unified wrapper that exists but is never exported — cannot regress silently.
"""

import pypto.language as pl
from pypto.language.op import unified_ops


def test_pl_log_is_the_unified_dispatcher():
    """The binding itself: ``pl.log`` is :func:`unified_ops.log`, not a
    kind-specific wrapper (the #1495 half-landing this file exists for)."""
    assert pl.log is unified_ops.log


def test_pl_log_dispatches_tensor_operand():
    """``pl.log`` over a Tensor emits ``tensor.log`` (the corpus softplus
    shape: log(add(exp(neg(x)), 1.0)) over a matmul result)."""

    @pl.function
    def softplus(x: pl.Tensor[[4, 4], pl.FP32], y: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        v = pl.log(pl.add(pl.exp(pl.neg(x)), 1.0))
        y[0:4, 0:4] = v
        return y

    text = softplus.as_python()
    assert "pl.tensor.log" in text
    assert "pl.tile.log" not in text


def test_pl_log_dispatches_tile_operand():
    """``pl.log`` over a Tile still emits ``tile.log`` — the unified binding
    must not lose the tile lane while fixing the tensor one."""

    @pl.function
    def k(x: pl.Tensor[[4, 4], pl.FP32], y: pl.Out[pl.Tensor[[4, 4], pl.FP32]]):
        t = pl.load(x, [0, 0], [4, 4])
        v = pl.log(t)
        pl.store(v, [0, 0], y)
        return y

    text = k.as_python()
    assert "pl.tile.log" in text
