# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for the producer-only ``pl.intrin.transmute`` escape hatch."""

from typing import cast

import pypto.language as pl
import pytest
from pypto import DataType, ir

# Validated at import so bare name comparisons can never drift from the
# registry (see .claude/rules/operator-identity-checks.md).
_TILE_LOAD = ir.get_op("tile.load").name
_TILE_STORE = ir.get_op("tile.store").name


def _flat_tile_type() -> pl.Tile:
    return pl.Tile[[6, 4], pl.FP32, pl.Mem.Vec]


def test_transmute_retypes_producer_without_adding_an_ir_node() -> None:
    source = ir.Var("x", ir.TensorType([2, 3, 4], DataType.FP32), ir.Span.unknown())
    before_facade = pl.Tensor(expr=source)
    before_facade = pl.tile.load(
        before_facade,
        [0, 0, 0],
        [2, 3, 4],
        target_memory=pl.Mem.Vec,
        attrs={"transmute_test": "kept"},
    )
    before = cast(ir.Call, before_facade.unwrap())
    flat_facade = pl.intrin.transmute(before_facade, _flat_tile_type())
    after = cast(ir.Call, flat_facade.unwrap())

    assert before is not after
    assert after.op.name == before.op.name == _TILE_LOAD
    assert len(after.args) == len(before.args)
    assert all(
        ir.structural_equal(lhs, rhs, enable_auto_mapping=True) for lhs, rhs in zip(after.args, before.args)
    )
    assert dict(after.kwargs) == dict(before.kwargs)
    assert dict(after.attrs) == dict(before.attrs)
    assert after.span.to_string() == before.span.to_string()
    assert isinstance(after.type, ir.TileType)
    assert [cast(ir.ConstInt, dim).value for dim in after.type.shape] == [6, 4]

    @pl.function
    def kernel(
        x: pl.Tensor[[2, 3, 4], pl.FP32],
        out: pl.Out[pl.Tensor[[2, 3, 4], pl.FP32]],
    ) -> pl.Tensor[[2, 3, 4], pl.FP32]:
        flat = pl.intrin.transmute(
            pl.tile.load(x, [0, 0, 0], [2, 3, 4], target_memory=pl.Mem.Vec),
            _flat_tile_type(),
        )
        result: pl.Tensor[[2, 3, 4], pl.FP32] = pl.tile.store(flat, [0, 0, 0], out, [2, 3, 4])
        return result

    assert isinstance(kernel, ir.Function)
    body = cast(ir.SeqStmts, kernel.body)
    calls = [stmt.value for stmt in body.stmts if isinstance(stmt, ir.AssignStmt)]
    assert [call.op.name for call in calls if isinstance(call, ir.Call)] == [_TILE_LOAD, _TILE_STORE]


def test_transmute_preserves_submit_binding_state() -> None:
    @pl.program
    class Program:
        @pl.function
        def worker(
            self,
            x: pl.Tensor[[8], pl.FP32],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
        ) -> pl.Tensor[[8], pl.FP32]:
            return pl.tile.store(pl.tile.load(x, [0], [8]), [0], out)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(
            self,
            x: pl.Tensor[[8], pl.FP32],
            out: pl.Out[pl.Tensor[[8], pl.FP32]],
        ) -> pl.Tensor[[8], pl.FP32]:
            with pl.manual_scope():
                result, _tid = pl.intrin.transmute(
                    pl.submit(self.worker, x, out),  # type: ignore[arg-type]
                    pl.Tuple[pl.Tensor[[8], pl.FP32], pl.Scalar[pl.TASK_ID]],
                )
            return result

    assert isinstance(Program, ir.Program)


@pytest.mark.parametrize("case", ["var", "constant", "expression"])
def test_transmute_rejects_non_producers(case: str) -> None:
    with pytest.raises(TypeError, match="detached Call or Submit"):

        @pl.function
        def invalid(x: pl.Scalar[pl.FP32]) -> pl.Scalar[pl.FP32]:
            if case == "var":
                value = x
            elif case == "constant":
                value = pl.const(1.0, pl.FP32)
            else:
                value = x + x
            return pl.intrin.transmute(value, pl.Scalar[pl.FP32])
