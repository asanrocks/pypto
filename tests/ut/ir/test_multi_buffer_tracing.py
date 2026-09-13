# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Runtime ``pl.MemRef(slots=N)`` slot subscripts under the tracing frontend.

The rotation index of every multi-buffer kernel — ``buf[i % 2]`` — is a staged
value. A bare local annotation never reaches the IR under tracing (it is
documentation, guide §4.4), so the slot is stated at the operation through
``pl.intrin.transmute`` (guide Appendix A.2), and the subscript inside the
transmute tile type must accept the staged index. This is the tracing sibling
of ``tests/ut/ir/parser/test_multi_buffer_codegen.py``; the two must reach
codegen in the same shape.
"""

import pypto.language as pl
import pytest
from pypto import DataType, backend, ir
from pypto.backend import BackendType
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.pypto_core import codegen, passes

ROTATING = pl.MemRef(slots=2)
CO_LIVE = pl.MemRef(slots=2)
SINGLE = pl.MemRef()


@pl.program
class RotatingSlot:
    """The ping-pong: one allocation, the slot alternating per iteration."""

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        seed = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Vec)
        (acc,) = pl.intrin.carry(seed)
        for i in pl.range(1, 4):
            t = pl.intrin.transmute(
                pl.load(a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec),
                pl.Tile[[64, 64], pl.FP32, ROTATING[i % 2], pl.Mem.Vec],
            )
            acc = pl.add(acc, t)
            pl.intrin.yield_(acc)
        (r,) = pl.intrin.leak()
        out = pl.store(r, [0, 0], output)
        return out


@pl.program
class CoLiveSlotsInLoop:
    """Two slots of one allocation live at the same time inside a loop.

    ptoas 0.54 derives the per-slot WAR guard only for the first
    ``multi_tile_get`` of an iteration, so this shape is refused rather than
    miscompiled — the refusal is part of the contract the tracing spelling must
    restore (a dropped slot view would compile without its region).
    """

    @pl.function(type=pl.FunctionType.InCore)
    def kernel(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        b: pl.Tensor[[256, 64], pl.FP32],
        output: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        for i in pl.range(4):
            lo = pl.intrin.transmute(
                pl.load(a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec),
                pl.Tile[[64, 64], pl.FP32, CO_LIVE[i % 2], pl.Mem.Vec],
            )
            hi = pl.intrin.transmute(
                pl.load(b, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec),
                pl.Tile[[64, 64], pl.FP32, CO_LIVE[(i + 1) % 2], pl.Mem.Vec],
            )
            s = pl.add(lo, hi)
            output = pl.store(s, [i * 64, 0], output)
        return output


def _codegen(program: ir.Program, planner: passes.MemoryPlanner) -> str:
    """Compile ``program``'s InCore kernel under ``planner`` and emit its PTO IR."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910B)
    with passes.PassContext([], memory_planner=planner):
        pm = PassManager.get_strategy(OptimizationStrategy.Default)
        optimized = pm.run_passes(program)
    func = next(f for f in optimized.functions.values() if f.name == "kernel")
    single = ir.Program([func], "kernel", optimized.span)
    emit_tile_addr = planner == passes.MemoryPlanner.PYPTO
    return codegen.PTOCodegen().generate(single, emit_tile_addr=emit_tile_addr)


def _lines(mlir: str, needle: str) -> list[str]:
    return [line.strip() for line in mlir.splitlines() if needle in line]


class TestSlotExpressionPath:
    """The DSL ``MemRef`` subscript accepts a staged index expression."""

    def test_staged_index_selects_a_slot(self):
        """A runtime index rides the declaration as its ``slot_index_``."""
        i = pl.Scalar(expr=ir.Var("i", ir.ScalarType(DataType.INDEX), ir.Span.unknown()))
        selected = ROTATING[i % 2]

        assert selected.is_pinned_
        assert selected.slot_count_ == 2
        assert ir.MemRef.same_allocation(selected, ROTATING), "a slot keeps the declaration's allocation"
        index = selected.slot_index_
        assert index is not None, "the staged index must be recorded, not dropped"
        assert isinstance(index, ir.Expr)
        assert not isinstance(index, ir.ConstInt), "a runtime index must not fold"
        slot_bytes = 64 * 64 * 4
        assert str(slot_bytes) not in repr(index), (
            f"the recorded index must be the selection, not a byte offset: {index!r}"
        )

    def test_constant_scalar_index_folds_to_the_constant_path(self):
        """A staged constant takes the same folded path as a literal int."""
        selected = ROTATING[pl.const(1, DataType.INDEX)]
        index = selected.slot_index_
        assert isinstance(index, ir.ConstInt)
        assert index.value == 1
        assert ROTATING[1].slot_index_ is not None

    def test_out_of_range_constant_is_loud(self):
        """A constant past the declared slot count is rejected either way."""
        with pytest.raises(ValueError, match="out of range"):
            ROTATING[2]
        with pytest.raises(ValueError, match="out of range"):
            ROTATING[pl.const(2, DataType.INDEX)]

    def test_single_slot_declaration_cannot_be_subscripted(self):
        """An unsubscripted declaration has no slots to select.

        The staged form is refused outright. A constant ``[0]`` keeps its
        pre-existing meaning — slot 0 of a one-slot declaration is the
        allocation itself — so only the runtime subscript is new here.
        """
        i = pl.Scalar(expr=ir.Var("i", ir.ScalarType(DataType.INDEX), ir.Span.unknown()))
        with pytest.raises(ValueError, match="single slot"):
            SINGLE[i % 2]


class TestPtoasPlannerEmitsMultiBuffer:
    """Under the ptoas planner a slotted declaration becomes one region."""

    def test_rotating_slot_becomes_one_region_read_by_slot(self):
        """N slots are one ``alloc_multi_tile``; the use selects its slot."""
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PTOAS)

        allocs = _lines(mlir, "pto.alloc_multi_tile")
        assert len(allocs) == 1, f"expected exactly one multi-buffer region:\n{mlir}"
        assert "count=2" in allocs[0], f"the region must carry the declared slot count:\n{allocs[0]}"
        assert "addr" not in allocs[0], f"the region must not carry an address:\n{allocs[0]}"

        gets = _lines(mlir, "pto.multi_tile_get")
        assert gets, f"expected the slot to be read from the region:\n{mlir}"
        region = allocs[0].split("=")[0].strip()
        assert all(region in get for get in gets), f"every slot must come from {region}:\n{mlir}"

    def test_rotating_slot_index_reaches_ptoas_as_an_index(self):
        """The slot operand is the index itself, not the byte offset.

        ptoas matches the slot's affine form (``iv % N``) to decide whether two
        accesses can touch the same slot, and that is what earns the per-slot
        event ids. Handing it a byte offset would defeat it.
        """
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PTOAS)
        get = _lines(mlir, "pto.multi_tile_get")[0]
        slot_ssa = get.split("[")[1].split("]")[0]

        definition = next((ln for ln in mlir.splitlines() if ln.strip().startswith(f"{slot_ssa} =")), None)
        assert definition is not None, f"slot operand {slot_ssa} has no definition:\n{mlir}"
        assert "remui" in definition or "remsi" in definition, (
            f"the slot operand must be the index expression, got: {definition}"
        )
        slot_bytes = 64 * 64 * 4
        assert str(slot_bytes) not in definition, (
            f"the slot operand must not be scaled to a byte offset: {definition}"
        )

    def test_pypto_planner_keeps_baked_addresses(self):
        """level3 gets no region: the ordinary baked-address path stays."""
        mlir = _codegen(RotatingSlot, passes.MemoryPlanner.PYPTO)
        assert not _lines(mlir, "pto.alloc_multi_tile"), f"level3 must not use a region:\n{mlir}"
        assert not _lines(mlir, "pto.multi_tile_get"), f"level3 must not use a region:\n{mlir}"
        assert _lines(mlir, "pto.alloc_tile"), f"expected the ordinary alloc path:\n{mlir}"

    def test_colive_slots_are_refused_under_ptoas(self):
        """Two co-live slots in one iteration are refused, not quietly planned.

        A dropped slot view would compile without its region and leave nothing
        to refuse — the refusal firing is itself the regression assertion.
        """
        with pytest.raises(ValueError, match="live at once inside a loop"):
            _codegen(CoLiveSlotsInLoop, passes.MemoryPlanner.PTOAS)

    def test_colive_slots_still_work_under_pypto(self):
        """The co-live shape is a legal program under the PyPTO planner."""
        mlir = _codegen(CoLiveSlotsInLoop, passes.MemoryPlanner.PYPTO)
        assert _lines(mlir, "pto.alloc_tile"), f"expected the ordinary alloc path:\n{mlir}"
