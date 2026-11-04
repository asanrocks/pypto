# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Storage is selected by producers, independently of local annotations."""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager
from pypto.language.diagnostics import InvalidOperationError, ParserSyntaxError

_TILE_LOAD = ir.get_op("tile.load").name
_TRANSPOSE = ir.get_op("tile.transpose").name


@pl.jit_trace
def slot_load(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    pool = pl.MemRef(slots=2)
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(2):
            t = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pool[i % 2])
            pl.store(t, [0, 0], out)
    return out


class Loads(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op):
        if op.op.name == _TILE_LOAD:
            self.calls.append(op)
        super().visit_call(op)


class Transposes(ir.IRVisitor):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op):
        if op.op.name == _TRANSPOSE:
            self.calls.append(op)
        super().visit_call(op)


def _memory(call: ir.Expr) -> ir.MemRef:
    assert isinstance(call.type, ir.TileType)
    assert call.type.memref is not None
    return call.type.memref


def _detached_tile(name: str, shape: list[int], dtype) -> pl.Tile:
    value_type = ir.TileType(shape, dtype, memory_space=pl.Mem.Vec)
    return pl.Tile(expr=ir.Var(name, value_type, ir.Span.unknown()))


def test_runtime_slot_is_part_of_produced_type():
    program = slot_load.specialize()
    visitor = Loads()
    visitor.visit_program(program)
    assert len(visitor.calls) == 1
    memory = _memory(visitor.calls[0])
    assert memory is not None and memory.is_pinned_
    assert memory.base_.name_hint == "pool"
    assert memory.slot_count_ == 2
    assert memory.slot_index_ is not None


def test_slot_substitution_preserves_allocation():
    pool = pl.MemRef("pool", slots=2)
    i = ir.Var("i", ir.ScalarType(pl.INDEX), ir.Span.unknown())
    j = ir.Var("j", ir.ScalarType(pl.INDEX), ir.Span.unknown())
    selected = pool[pl.Scalar(expr=i)]
    replaced = ir.substitute_expr(selected, [(i, j)])
    assert isinstance(replaced, ir.MemRef)
    assert replaced.slot_index_ is j
    assert ir.MemRef.same_allocation(pool, replaced)


@pytest.mark.parametrize("index", [-1, 2, True, 1.0])
def test_invalid_slot_is_rejected(index):
    with pytest.raises((ValueError, TypeError)):
        pl.MemRef("pool", slots=2)[index]


def test_arithmetic_result_can_select_storage():
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Vec), ir.Span.unknown())
    result = pl.bind_storage(pl.exp(pl.Tile(expr=x)), pl.MemRef("scratch"))
    call = result.unwrap()
    assert isinstance(call, ir.Call) and call.op.name == ir.get_op("tile.exp").name
    assert isinstance(call.type, ir.TileType) and call.type.memref is not None
    assert call.type.memref.base_.name_hint == "scratch"


def test_existing_value_cannot_be_relocated():
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Vec), ir.Span.unknown())
    with pytest.raises(TypeError, match="detached"):
        pl.bind_storage(pl.Tile(expr=x), pl.MemRef("scratch"))


def test_independent_traces_have_independent_allocations():
    left = pl.jit_trace(slot_load.kernel).specialize()
    right = pl.jit_trace(slot_load.kernel).specialize()
    a, b = Loads(), Loads()
    a.visit_program(left)
    b.visit_program(right)
    assert not ir.MemRef.same_allocation(_memory(a.calls[0]), _memory(b.calls[0]))


@pytest.mark.parametrize(
    "planner", [passes.MemoryPlanner.PYPTO, passes.MemoryPlanner.DSA_RP, passes.MemoryPlanner.PTOAS]
)
def test_declared_slots_survive_the_memory_pipeline(planner):
    with passes.PassContext([], memory_planner=planner):
        lowered = PassManager(OptimizationStrategy.Default).run_passes(slot_load.specialize())
    text = str(lowered)
    assert "pinned=True" in text
    assert "slots=2" in text
    assert "ir_stage: Buffer" not in text


def test_printed_program_retains_storage():
    original = slot_load.specialize()
    restored = pl.parse_program(str(original))
    visitor = Loads()
    visitor.visit_program(restored)
    assert _memory(visitor.calls[0]).slot_count_ == 2
    assert _memory(visitor.calls[0]).is_pinned_


def test_parser_accepts_the_producer_keyword():
    @pl.program
    class Program:
        @pl.function(type=pl.FunctionType.InCore)
        def main(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]):
            t = pl.load(
                x,
                [0, 0],
                [64, 64],
                target_memory=pl.Mem.Vec,
                memref=pl.MemRef("pool", slots=2)[0],
            )
            t = pl.bind_storage(pl.exp(t), pl.MemRef("scratch"))
            pl.store(t, [0, 0], out)

    visitor = Loads()
    visitor.visit_program(Program)
    assert _memory(visitor.calls[0]).slot_count_ == 2
    assert 'pl.MemRef("scratch")' in str(Program)


def test_inherited_storage_cannot_be_rebound():
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Vec), ir.Span.unknown())
    with pytest.raises(ValueError, match="inherits storage"):
        pl.bind_storage(pl.reshape(pl.Tile(expr=x), [64, 64]), pl.MemRef("scratch"))


def test_reused_input_storage_cannot_be_rebound():
    dst = _detached_tile("dst", [4, 8], pl.FP32)
    index = _detached_tile("index", [2, 2], pl.INT32)
    src = _detached_tile("src", [4, 8], pl.FP32)
    producer = pl.scatter_update(dst, -2, index, src)
    with pytest.raises(ValueError, match="writes its result into input argument 0"):
        pl.bind_storage(producer, pl.MemRef("scratch"))


def test_global_var_producer_is_rejected():
    # A GlobalVar carrying a registered operator's name must not inherit that
    # operator's storage contract (kind first, name second).
    pool = pl.MemRef("pool")
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memref=pool, memory_space=pl.Mem.Vec), ir.Span.unknown())
    call = ir.Call(
        ir.GlobalVar("tile.transpose"),
        [x],
        ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Vec),
        ir.Span.unknown(),
    )
    with pytest.raises(ValueError, match="GlobalVar"):
        ir.bind_result_storage(call, [pool])


def test_not_inplace_safe_producer_rejects_input_alias():
    pool = pl.MemRef("pool")
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memref=pool, memory_space=pl.Mem.Vec), ir.Span.unknown())
    with pytest.raises(ValueError, match="not in-place safe"):
        pl.bind_storage(pl.transpose(pl.Tile(expr=x), 0, 1), pool)


def test_not_inplace_safe_producer_accepts_distinct_storage():
    pool = pl.MemRef("pool")
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memref=pool, memory_space=pl.Mem.Vec), ir.Span.unknown())
    result = pl.bind_storage(pl.transpose(pl.Tile(expr=x), 0, 1), pl.MemRef("scratch"))
    assert _memory(result.unwrap()).base_.name_hint == "scratch"


def test_traced_parameter_storage_is_interned_for_alias_checking():
    # The parser interns signature annotations and body destinations into one
    # base per name; the tracing facade must intern parameter MemRefs the same
    # way, or an input Tile parameter and a destination naming one allocation
    # carry two base identities and the registry's alias check sees no overlap.
    pool = pl.MemRef("pool")

    # Pre-canonicalization: one shared base identity, the registry refuses.
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memref=pool, memory_space=pl.Mem.Vec), ir.Span.unknown())
    with pytest.raises(ValueError, match="must not alias input argument"):
        pl.bind_storage(pl.transpose(pl.Tile(expr=x), 0, 1), pool)

    # Post-canonicalization: the traced lane must refuse just as loudly.
    @pl.jit_trace
    def entry(t: pl.Tile[[64, 64], pl.FP32, pool, pl.Mem.Vec]):
        return pl.bind_storage(pl.transpose(t, 0, 1), pool)

    with pytest.raises(ValueError, match="must not alias input argument"):
        entry.specialize()


def test_traced_parameter_keeps_its_named_allocation():
    pool = pl.MemRef("pool")

    @pl.jit_trace
    def entry(t: pl.Tile[[64, 64], pl.FP32, pool, pl.Mem.Vec]):
        return pl.bind_storage(pl.transpose(t, 0, 1), pl.MemRef("scratch"))

    first = pl.jit_trace(entry.kernel).specialize()
    second = pl.jit_trace(entry.kernel).specialize()
    visitors = []
    for program in (first, second):
        visitor = Transposes()
        visitor.visit_program(program)
        assert len(visitor.calls) == 1
        visitors.append(visitor)
    for visitor in visitors:
        assert _memory(visitor.calls[0]).base_.name_hint == "scratch"
        param_memory = visitor.calls[0].args[0].type.memref
        assert param_memory is not None and param_memory.is_pinned_
        assert param_memory.base_.name_hint == "pool"
    # Each specialization interns its own bases: independent traces keep
    # independent allocations, parameter and destination alike.
    assert not ir.MemRef.same_allocation(_memory(visitors[0].calls[0]), _memory(visitors[1].calls[0]))
    assert not ir.MemRef.same_allocation(
        visitors[0].calls[0].args[0].type.memref, visitors[1].calls[0].args[0].type.memref
    )


def test_non_sequence_destination_is_rejected():
    # None is the tuple spelling's per-result "automatic placement" marker,
    # not a valid single-Tile destination.
    x = ir.Var("x", ir.TileType([64, 64], pl.FP32, memory_space=pl.Mem.Vec), ir.Span.unknown())
    with pytest.raises(TypeError, match="sequence of MemRef/None"):
        pl.bind_storage(pl.transpose(pl.Tile(expr=x), 0, 1), None)  # pyright: ignore[reportArgumentType] — intentional: None is the tuple spelling's marker


def test_forbidden_alias_operand_rejects_bound_storage():
    vec_pool = pl.MemRef("vec")
    target = _detached_tile("target", [4, 8], pl.FP32)
    row_vec = ir.Var(
        "row_vec", ir.TileType([4, 1], pl.FP32, memref=vec_pool, memory_space=pl.Mem.Vec), ir.Span.unknown()
    )
    with pytest.raises(ValueError, match="reads input argument 1"):
        pl.bind_storage(pl.tile.row_expand(target, pl.Tile(expr=row_vec)), vec_pool)


def test_prelu_destination_cannot_share_tmp_storage():
    # A5's TPRELU keeps its ABI-required tmp operand but never reads it; binding
    # is deliberately target-independent, so a destination sharing tmp's
    # allocation is refused for every target (A2/A3's TPRELU reads tmp while
    # writing dst). The target-aware planners exempt A5 tmp when they pack.
    pool = pl.MemRef("pool")
    src = ir.Var("src", ir.TileType([16, 64], pl.FP16, memory_space=pl.Mem.Vec), ir.Span.unknown())
    slope = ir.Var("slope", ir.TileType([16, 64], pl.FP16, memory_space=pl.Mem.Vec), ir.Span.unknown())
    tmp = ir.Var(
        "tmp", ir.TileType([17, 8], pl.UINT8, memref=pool, memory_space=pl.Mem.Vec), ir.Span.unknown()
    )
    with pytest.raises(ValueError, match="not in-place safe"):
        pl.bind_storage(pl.tile.prelu(pl.Tile(expr=src), pl.Tile(expr=slope), pl.Tile(expr=tmp)), pool)


def test_memoryless_producer_has_no_bindable_storage():
    # tile.get_block_idx opts out of the memory system: its spec is
    # engaged-but-empty, which must read as "no contract", not as one. The
    # None passthrough must not silently return a rebound clone.
    call = ir.Call(ir.get_op("tile.get_block_idx"), [], ir.ScalarType(pl.INDEX), ir.Span.unknown())
    with pytest.raises(ValueError, match="no output memory contract"):
        ir.bind_result_storage(call, [None])
    with pytest.raises(ValueError, match="no output memory contract"):
        ir.bind_result_storage(call, [pl.MemRef("scratch")])


def test_multi_result_destinations_are_explicit():
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    result = pl.bind_storage(producer, [pl.MemRef("indices"), pl.MemRef("counts")])
    result_type = result.unwrap().type
    assert isinstance(result_type, ir.TupleType)
    names = []
    for element in result_type.types:
        assert isinstance(element, ir.TileType) and element.memref is not None
        names.append(element.memref.base_.name_hint)
    assert names == ["indices", "counts"]
    with pytest.raises(ValueError, match="one destination per result"):
        pl.bind_storage(producer, [pl.MemRef("indices")])


def test_bound_declaration_alias_is_rejected():
    @pl.jit_trace
    def bad(x: pl.Tensor[[64, 64], pl.FP32]):
        pool = pl.MemRef(slots=2)
        alias = pool
        return alias

    with pytest.raises(ValueError, match="cannot also be bound"):
        bad.specialize()


def test_binary_roundtrip_retains_the_slot_expression():
    original = slot_load.specialize()
    restored = ir.deserialize(ir.serialize(original))
    assert isinstance(restored, ir.Program)
    a, b = Loads(), Loads()
    a.visit_program(original)
    b.visit_program(restored)
    # Compare in the loop's binding scope: its induction variable is free when
    # comparing either MemRef in isolation, and deserialization creates fresh Vars.
    assert ir.structural_equal(original, restored)
    assert _memory(b.calls[0]).slot_count_ == 2


def test_reduction_and_cube_producers_use_the_same_binding_contract():
    def tile(name, dtype, memory):
        value_type = ir.TileType([64, 64], dtype, memory_space=memory)
        return pl.Tile(expr=ir.Var(name, value_type, ir.Span.unknown()))

    source = tile("source", pl.FP32, pl.Mem.Vec)
    scratch = tile("tmp", pl.FP32, pl.Mem.Vec)
    reduced = pl.bind_storage(pl.tile.row_sum(source, scratch), pl.MemRef("reduction"))
    assert _memory(reduced.unwrap()).base_.name_hint == "reduction"
    left = tile("lhs", pl.FP16, pl.Mem.Left)
    right = tile("rhs", pl.FP16, pl.Mem.Right)
    product = pl.bind_storage(pl.tile.matmul(left, right), pl.MemRef("accumulator"))
    assert _memory(product.unwrap()).base_.name_hint == "accumulator"


def test_creation_and_move_accept_storage_keywords():
    created = pl.create_tile([64, 64], pl.FP16, target_memory=pl.Mem.Mat, memref=pl.MemRef("l1"))
    moved = pl.move(created, target_memory=pl.Mem.Left, memref=pl.MemRef("l0a"))
    assert _memory(created.unwrap()).base_.name_hint == "l1"
    assert _memory(moved.unwrap()).base_.name_hint == "l0a"


def test_independent_values_cannot_occupy_one_live_slot():
    @pl.jit_trace
    def bad(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        pool = pl.MemRef(slots=2)
        with pl.at(level=pl.Level.CORE_GROUP):
            a = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pool[0])
            b = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pool[0])
            pl.store(pl.add(a, b), [0, 0], out)
        return out

    with pytest.raises(ValueError, match="live"):
        PassManager(OptimizationStrategy.Default).run_passes(bad.specialize())


def test_named_inline_references_cannot_disagree_on_slot_count():
    @pl.jit_trace
    def bad(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            a = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pl.MemRef("pool", slots=2)[0])
            b = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pl.MemRef("pool", slots=3)[1])
            pl.store(pl.add(a, b), [0, 0], out)
        return out

    with pytest.raises(InvalidOperationError, match="conflicting slot counts"):
        bad.specialize()


def test_inline_producer_preserves_its_declared_storage():
    @pl.jit_trace.inline
    def helper(x: pl.Tensor[[64, 64], pl.FP32]):
        pool = pl.MemRef(slots=2)
        return pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec, memref=pool[0])

    @pl.jit_trace
    def entry(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        with pl.at(level=pl.Level.CORE_GROUP):
            pl.store(helper(x), [0, 0], out)
        return out

    lowered = PassManager(OptimizationStrategy.Default).run_passes(entry.specialize())
    assert "pinned=True" in str(lowered)
    assert "slots=2" in str(lowered)


def test_same_destination_for_two_results_is_rejected():
    # The registry checks each destination against producer inputs; two results
    # of one producer must also be checked against each other, or the second
    # write overwrites the first.
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    pool = pl.MemRef("pool")
    with pytest.raises(ValueError, match="must not alias each other"):
        pl.bind_storage(producer, [pool, pool])


def test_overlapping_slots_for_two_results_are_rejected():
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    pool = pl.MemRef("pool", slots=2)
    rotation = ir.Var("i", ir.ScalarType(pl.INDEX), ir.Span.unknown())
    # Same constant slot: the second result overwrites the first.
    with pytest.raises(ValueError, match="must not alias each other"):
        pl.bind_storage(producer, [pool[0], pool[0]])
    # A runtime index may resolve to slot 0, so it is conservatively an alias.
    with pytest.raises(ValueError, match="must not alias each other"):
        pl.bind_storage(producer, [pool[0], pool[rotation]])


def test_distinct_constant_slots_for_two_results_are_accepted():
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    pool = pl.MemRef("pool", slots=2)
    result = pl.bind_storage(producer, [pool[0], pool[1]])
    result_type = result.unwrap().type
    assert isinstance(result_type, ir.TupleType)
    slots = []
    for element in result_type.types:
        assert isinstance(element, ir.TileType) and element.memref is not None
        assert element.memref.base_.name_hint == "pool"
        assert element.memref.slot_index_ is not None
        slots.append(element.memref.slot_index_)
    assert slots[0] is not slots[1]


def test_none_destination_is_exempt_from_pairwise_checking():
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    result = pl.bind_storage(producer, [pl.MemRef("pool"), None])
    result_type = result.unwrap().type
    assert isinstance(result_type, ir.TupleType)
    first, second = result_type.types
    assert isinstance(first, ir.TileType) and isinstance(second, ir.TileType)
    assert first.memref is not None
    assert second.memref is None


def test_rebinding_an_already_placed_producer_is_rejected():
    # A producer with any placed result is already placed: a second request,
    # partial or not, leaves the placed positions out of the pairwise alias
    # check (which only sees destinations supplied in that request), so a
    # partial rebind could hand two results the same allocation.
    src = pl.Tile(expr=ir.Var("src", ir.TileType([16, 64], pl.FP32), ir.Span.unknown()))
    tmp = pl.Tile(expr=ir.Var("tmp", ir.TileType([16, 64], pl.UINT8), ir.Span.unknown()))
    producer = pl.tile.gather_compare(src, pl.const(0.0, pl.FP32), tmp, out_cols=64)
    placed = pl.bind_storage(producer, [pl.MemRef("pool"), None])
    with pytest.raises(ValueError, match="already has an explicit storage binding"):
        pl.bind_storage(placed, [None, pl.MemRef("pool")])
    with pytest.raises(ValueError, match="already has an explicit storage binding"):
        pl.bind_storage(placed, [pl.MemRef("other"), None])
    with pytest.raises(ValueError, match="already has an explicit storage binding"):
        pl.bind_storage(placed, [None, None])


def test_unpinned_destination_is_refused_at_the_facade():
    base = ir.Var("base", ir.PtrType(), ir.Span.unknown())
    with pytest.raises(TypeError, match="destination 0.*pinned allocation"):
        pl.bind_storage(pl.exp(_detached_tile("x", [64, 64], pl.FP32)), pl.MemRef(base, 0, 128))
    # A slot selection on an unpinned MemRef stays unpinned: the printed
    # ``pl.MemRef(base, offset, size, slots=N)[k]`` form re-parses, but it is
    # not a binding destination.
    slotted = pl.MemRef(base, 0, 256, slots=2)
    with pytest.raises(TypeError, match="destination 0.*pinned allocation"):
        pl.bind_storage(pl.exp(_detached_tile("x", [64, 64], pl.FP32)), slotted[0])


def test_multi_slot_destination_requires_slot_selection_at_the_facade():
    pool = pl.MemRef("pool", slots=2)
    with pytest.raises(TypeError, match="destination 0.*select a slot"):
        pl.bind_storage(pl.exp(_detached_tile("x", [64, 64], pl.FP32)), pool)


def test_explicit_storage_requires_target_memory_at_the_facade():
    x = pl.Tensor(expr=ir.Var("x", ir.TensorType([64, 64], pl.FP32), ir.Span.unknown()))
    with pytest.raises(ValueError, match="pass target_memory"):
        pl.load(x, [0, 0], [64, 64], memref=pl.MemRef("scratch"))


def test_parser_rejects_same_destination_for_two_results():
    with pytest.raises(ParserSyntaxError, match="must not alias each other"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[16, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
                bound = pl.bind_storage(
                    pl.tile.gather_compare(
                        pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Vec),
                        pl.const(0.0, pl.FP32),
                        pl.load(x, [0, 0], [16, 64], target_memory=pl.Mem.Vec),
                        out_cols=64,
                    ),
                    [pl.MemRef("pool", slots=2)[0], pl.MemRef("pool", slots=2)[0]],
                )
                pl.store(bound[0], [0, 0], out)


def test_parser_rejects_unpinned_destination():
    with pytest.raises(ParserSyntaxError, match="destination 0.*pinned allocation"):

        @pl.program
        class Program:
            @pl.function(type=pl.FunctionType.InCore)
            def main(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
                t = pl.bind_storage(
                    pl.exp(pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Vec)),
                    pl.MemRef("base", 0, 256),
                )
                pl.store(t, [0, 0], out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
