# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Codegen for ``pl.assemble`` whose source is a cube-produced (Acc) tile.

``pl.assemble(local_target, matmul_result, offset)`` — a *local* Vec target
(``pl.create_tensor`` or a loop-carried tile) fed by a cube result — is the
online-softmax accumulator-update shape of
``examples/models/03_flash_attention.py`` (``oiUpdate_90 = pl.assemble(oi_update,
oiUpdate_87, [0, 0])``). Assembling into an ``Out`` parameter works: that path
lowers to a plain ``pto.tstore`` of the Acc tile on the cube lane.

The local target is bridged by ``InferTileMemorySpace``: ``tile.assemble``
declares its source's legal space as the *target's* resolved space (a dynamic
matches-arg constraint, ``set_input_memory_matches_arg``), with the converting
Acc->Mat writeback (FIXPIPE) as the one exemption. A cube-produced (Acc) source
assembled into a Vec target therefore gets an explicit Acc->Vec ``tile.move``
queued ahead of the assemble, which ``ExpandMixedKernel`` folds into the
cross-core tpush/tpop boundary — the same shape the flash-attention
else-branch spelled by hand before the migration.
"""

import pypto.language as pl
from pypto import codegen, passes
from pypto.backend import BackendType, reset_for_testing, set_backend_type
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import ir

M = K = N = 32


def _lane_mlir(program, lane_suffix: str) -> str:
    """Run the default pipeline and emit PTO for one lane of the mixed kernel."""
    reset_for_testing()
    set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
        optimized = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)
    lanes = [f for f in optimized.functions.values() if f.name.endswith(lane_suffix)]
    assert len(lanes) == 1, f"expected one {lane_suffix!r} lane, got {[f.name for f in lanes]}"
    return codegen.PTOCodegen().generate(
        ir.Program([lanes[0]], lanes[0].name, optimized.span), emit_tile_addr=False
    )


@pl.program
class AssembleAccIntoLocalProgram:
    """A matmul result assembled into a ``pl.create_tensor`` local tile."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="acc_local"):
            prod = pl.matmul(a, b)
            tile = pl.create_tensor([M, N], dtype=pl.FP32)
            tile = pl.assemble(tile, prod, [0, 0])
            output = tile
        return output


@pl.program
class AssembleAccIntoCarryProgram:
    """The flash-attention shape: the local target is a loop-carried tile."""

    @pl.function(type=pl.FunctionType.Opaque)
    def main(
        self,
        a: pl.Tensor[[M, K], pl.FP32],
        b: pl.Tensor[[K, N], pl.FP32],
        output: pl.Out[pl.Tensor[[M, N], pl.FP32]],
    ) -> pl.Tensor[[M, N], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="acc_carry"):
            acc = pl.create_tensor([M, N], dtype=pl.FP32)
            for i in pl.range(2):
                if i == 0:
                    prod = pl.matmul(a, b)
                    acc = pl.assemble(acc, prod, [0, 0])
                else:
                    acc = pl.add(acc, acc)
            output = acc
        return output


def _assert_acc_source_reaches_the_vector_lane(program) -> None:
    """Both lanes emit, and the cube product crosses cores to its consumer."""
    aic = _lane_mlir(program, "_aic")
    aiv = _lane_mlir(program, "_aiv")
    assert "pto.tmatmul" in aic, f"the cube lane must still compute the product:\n{aic}"
    assert "pto.tpop_from_aic" in aiv, f"the Acc product must reach the vector lane that consumes it:\n{aiv}"


def test_assemble_acc_into_local_tile_compiles():
    """A cube result assembled into a create_tensor local must compile.

    The accumulator lives in Vec on the vector lane; the product is computed in
    Acc on the cube lane. The bridging Acc->Vec tile.move is the cross-core
    boundary: the AIC lane keeps the matmul (it feeds the tpush) and the AIV
    lane receives the product through a tpop.
    """
    _assert_acc_source_reaches_the_vector_lane(AssembleAccIntoLocalProgram)


def test_assemble_acc_into_loop_carried_tile_compiles():
    """The flash-attention carry shape: assemble into a loop-carried Vec tile."""
    _assert_acc_source_reaches_the_vector_lane(AssembleAccIntoCarryProgram)
