# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tracing-side contract of the split-transfer ops — ``pl.aiv_shard`` /
``pl.aic_gather``.

Two authoring forms, never mixed (the contract mirrors the AST parser's
``_parse_split_transfer_op``):

* **Region form** — no ``split=``: the mode is inherited from the enclosing
  ``for aiv_id in pl.split_aiv(mode=...)`` scope; a ``Tile`` or a high-level
  ``Tensor`` operand is valid.
* **Outlined form** — an explicit integer ``split=``: the printer's spelling
  for a function already lowered into the explicit form. Tile-only, rejected
  inside a ``split_aiv`` region, and ``lane_stride=`` rides only here (and
  only on the shard).
"""

import pypto.language as pl
import pytest
from pypto import ir
from pypto.language.diagnostics import InvalidOperationError
from pypto.language.tracer.decorators.function import function_trace

_INCORE = {"function_type": ir.FunctionType.InCore}


def _find_call(fn: ir.Function, op_name: str) -> ir.Call:
    found: list[ir.Call] = []

    class _Collector(ir.IRVisitor):
        def visit_call(self, expr: ir.Call) -> None:
            if expr.op.name == op_name:
                found.append(expr)
            super().visit_call(expr)

    _Collector().visit_stmt(fn.body)
    assert len(found) == 1, f"expected exactly one {op_name} call, got {len(found)}"
    return found[0]


def test_region_form_inherits_the_scope_mode_on_a_tile_operand():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            _ = pl.tile.aiv_shard(qk)

    call = _find_call(function_trace(kernel, **_INCORE), "tile.aiv_shard")
    assert call.kwargs["split"] == 1


def test_region_form_on_a_tensor_operand_lowers_to_the_tensor_op():
    def kernel(mm: pl.Tensor[[128, 128], pl.FP32]) -> None:
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            _ = pl.aiv_shard(mm)

    call = _find_call(function_trace(kernel, **_INCORE), "tensor.aiv_shard")
    assert call.kwargs["split"] == 1


def test_outlined_form_states_the_split_on_the_op():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aiv_shard(qk, split=1)

    call = _find_call(function_trace(kernel, **_INCORE), "tile.aiv_shard")
    assert call.kwargs["split"] == 1
    assert "lane_stride" not in call.kwargs


def test_outlined_form_carries_the_lane_stride():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aiv_shard(qk, split=1, lane_stride=7)

    call = _find_call(function_trace(kernel, **_INCORE), "tile.aiv_shard")
    assert call.kwargs["split"] == 1
    assert call.kwargs["lane_stride"] == 7


def test_outlined_gather_states_the_split():
    def kernel(half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aic_gather(half, split=1)

    call = _find_call(function_trace(kernel, **_INCORE), "tile.aic_gather")
    assert call.kwargs["split"] == 1


def test_outlined_form_rejects_a_tensor_operand():
    def kernel(mm: pl.Tensor[[128, 128], pl.FP32]) -> None:
        _ = pl.tile.aiv_shard(mm, split=1)

    with pytest.raises(InvalidOperationError, match="does not accept a high-level Tensor operand"):
        function_trace(kernel, **_INCORE)


def test_outlined_form_is_rejected_inside_a_region():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            _ = pl.tile.aiv_shard(qk, split=1)

    with pytest.raises(InvalidOperationError, match="does not take a split= argument inside"):
        function_trace(kernel, **_INCORE)


def test_region_form_requires_a_region():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aiv_shard(qk)

    with pytest.raises(RuntimeError, match="must be used inside a 'for ... in pl.split_aiv"):
        function_trace(kernel, **_INCORE)


def test_lane_stride_is_rejected_inside_a_region():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            _ = pl.tile.aiv_shard(qk, lane_stride=7)

    with pytest.raises(InvalidOperationError, match="does not take a lane_stride= argument inside"):
        function_trace(kernel, **_INCORE)


def test_lane_stride_is_rejected_on_the_gather():
    def kernel(half: pl.Tile[[64, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aic_gather(half, split=1, lane_stride=7)

    with pytest.raises(InvalidOperationError, match="pl.aic_gather\\(\\) does not take a lane_stride"):
        function_trace(kernel, **_INCORE)


def test_outlined_form_requires_an_integer_split():
    def kernel(qk: pl.Tile[[128, 128], pl.FP32, pl.Mem.Vec]) -> None:
        _ = pl.tile.aiv_shard(qk, split=pl.SplitMode.UP_DOWN)

    with pytest.raises(InvalidOperationError, match="requires an integer split"):
        function_trace(kernel, **_INCORE)
