# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests for LowerPipelineToSlots — ``pl.pipeline`` via MemRef slots, not body copies.

The pass keeps ONE loop body and rebinds its top-level loads onto
``pl.MemRef(name, slots=F)[iv % F]``, then demotes the loop to Sequential. It is
self-gated on ``memory_planner=PTOAS`` and falls back to leaving a loop untouched
(for ``LowerPipelineLoops`` to replicate) whenever it cannot guarantee codegen
will accept the region.

Groups:
  * ``TestGating``   — the planner gate, and that a declined loop is left intact
  * ``TestBinding``  — the slot geometry actually written onto the tile
  * ``TestFallback`` — one case per expressible eligibility gate, each asserting NO
    slot binding
  * ``TestChaining`` — LowerPipelineLoops still replicates what this pass declined
"""

import pypto.language as pl
import pytest
from pypto import ir, passes
from pypto.ir.pass_manager import OptimizationStrategy, PassManager

_PASS_NAME = "LowerPipelineToSlots"


def _run_to_slots(program: ir.Program, planner: passes.MemoryPlanner) -> ir.Program:
    """Run the Default strategy up to and including LowerPipelineToSlots.

    The pass needs tile-level IR (memory spaces inferred, structure normalized),
    so it cannot run standalone on a freshly parsed program. The PassManager is
    built inside the context because its construction reads the planner.
    """
    with passes.PassContext([], memory_planner=planner):
        manager = PassManager(OptimizationStrategy.Default)
        names = manager.pass_names
        stop = names.index(_PASS_NAME)
        for pass_obj in manager.passes[: stop + 1]:
            pipeline = passes.PassPipeline()
            pipeline.add_pass(pass_obj)
            program = pipeline.run(program)
    return program


def _walk_stmts(program: ir.Program):
    """Yield every statement in every function body, depth-first."""

    def walk(stmt):
        if stmt is None:
            return
        yield stmt
        for attr in ("body", "then_body", "else_body", "stmts"):
            sub = getattr(stmt, attr, None)
            if sub is None:
                continue
            for child in sub if isinstance(sub, (list, tuple)) else [sub]:
                yield from walk(child)

    for func in program.functions.values():
        yield from walk(func.body)


def _source_name(name: str) -> str:
    """Strip the suffix ConvertToSSA appends, so tests can name the DSL variable."""
    return name.split("__ssa")[0]


def _slotted_memrefs(program: ir.Program) -> dict[str, ir.MemRef]:
    """Every tile assignment bound to a multi-slot allocation, by source var name."""
    found: dict[str, ir.MemRef] = {}
    for stmt in _walk_stmts(program):
        if isinstance(stmt, ir.AssignStmt) and isinstance(stmt.var.type, ir.TileType):
            memref = stmt.var.type.memref
            if memref is not None and memref.slot_count_ > 1:
                found[_source_name(stmt.var.name_hint)] = memref
    return found


def _pipeline_loops(program: ir.Program) -> list[ir.ForStmt]:
    """Every ForStmt still carrying the Pipeline kind."""
    return [
        stmt
        for stmt in _walk_stmts(program)
        if isinstance(stmt, ir.ForStmt) and stmt.kind == ir.ForKind.Pipeline
    ]


def _load_count(program: ir.Program) -> int:
    """How many tile.load calls the program contains (replication detector)."""
    text = program.as_python()
    return text.count("pl.load(") + text.count("tile.load(")


@pl.program
class SingleLoad:
    """The canonical shape: one i-dependent load per pipeline iteration."""

    @pl.function
    def main(
        self,
        a: pl.Tensor[[256, 64], pl.FP32],
        out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
    ) -> pl.Tensor[[256, 64], pl.FP32]:
        (acc,) = pl.intrin.carry(out)
        for i in pl.pipeline(0, 4, 1, stage=2):
            t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
            )
            e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
            nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
            pl.intrin.yield_(nxt)
        (y,) = pl.intrin.leak()
        return y


class TestGating:
    """The planner gate decides whether the pass does anything at all."""

    def test_pypto_planner_leaves_the_loop_for_replication(self):
        """Under the default planner no region is emitted, so nothing is bound."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PYPTO)
        assert _slotted_memrefs(after) == {}
        assert len(_pipeline_loops(after)) == 1, "the loop must stay Pipeline for LowerPipelineLoops"

    def test_ptoas_planner_slots_the_load_and_demotes_the_loop(self):
        """Under PTOAS the load takes a slot and the loop stops being a Pipeline."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"t"}
        assert _pipeline_loops(after) == [], "a slotted loop carries its ping-pong in the slots"

    def test_body_is_not_replicated(self):
        """One body, not F copies — the whole point of the transform."""
        before_loads = _load_count(SingleLoad)
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _load_count(after) == before_loads


class TestBinding:
    """The slot geometry written onto the tile's TileType."""

    def test_slot_count_matches_the_stage_count(self):
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after)["t"].slot_count_ == 2

    def test_slot_index_is_the_induction_variable_modulo_the_stage_count(self):
        """ptoas matches the index's affine form, so it must be literally ``i % 2``."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        index = _slotted_memrefs(after)["t"].slot_index_
        assert index is not None
        assert isinstance(index, ir.FloorMod)
        dividend = index.left
        assert isinstance(dividend, ir.Var), f"the dividend must be the loop var, got {dividend}"
        loop_vars = [stmt.loop_var.name_hint for stmt in _walk_stmts(after) if isinstance(stmt, ir.ForStmt)]
        assert loop_vars == [dividend.name_hint], (
            f"the dividend must be the loop's own induction variable, got {dividend.name_hint} "
            f"against loop vars {loop_vars}"
        )
        divisor = index.right
        assert isinstance(divisor, ir.ConstInt), f"the modulus must be a literal, got {divisor}"
        assert divisor.value == 2

    def test_declaration_is_pinned_so_init_memref_treats_it_as_the_authors(self):
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after)["t"].is_pinned_

    def test_compute_tiles_do_not_take_slots(self):
        """Only load buffers need per-stage privacy; slotting everything overflows."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        assert "e" not in _slotted_memrefs(after)

    def test_load_addressed_through_a_loop_carried_iter_arg_takes_a_slot(self):
        """``off`` is a loop-carried IterArg, so this load reads different data every
        iteration without ever naming the induction variable. Treating "does not read
        the loop var" as proof of loop-invariance stranded it: skipped here, and
        skipped by ``LowerPipelineLoops`` too once the sibling candidate demoted the
        loop to Sequential. Every unbound top-level load is a candidate now."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc, off) = pl.intrin.carry(out, pl.const(0, pl.INDEX))
                for i in pl.pipeline(0, 4, 1, stage=2):
                    carried_load: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [off, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    good: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    s: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(carried_load, good)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt, off + 64)
                (y, y_off) = pl.intrin.leak()
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"carried_load", "good"}

    def test_survives_print_parse_roundtrip(self):
        """A slotted dump must reparse as the same slot."""
        after = _run_to_slots(SingleLoad, passes.MemoryPlanner.PTOAS)
        reparsed = pl.parse_program(after.as_python())
        ir.assert_structural_equal(reparsed, after)


class TestFallback:
    """One case per gate that the DSL can express. Every one must decline silently,
    never raise. The memory-space and runtime-valid-shape gates have no case: a
    ``tile.load`` result always lands in Vec/Mat/Acc after ``InferTileMemorySpace``,
    and the tile shapes reaching this pass are static."""

    def _assert_declined(self, program: ir.Program):
        after = _run_to_slots(program, passes.MemoryPlanner.PTOAS)
        assert _slotted_memrefs(after) == {}
        assert len(_pipeline_loops(after)) >= 1, "a declined loop stays Pipeline for replication"

    def test_slots_that_overflow_the_memory_space(self):
        """The declared slots are pinned, so ptoas may not reuse any of them. Two
        128 KB slots exceed the Vec budget, and ptoas answers a region it cannot
        place with a hard ``overflow`` error rather than degrading — so the pass
        must decline and let the replication path's capacity gate shrink the depth
        instead."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[512, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 256], pl.FP32]],
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 128, 0], [128, 256], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[128, 256], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[512, 256], pl.FP32] = pl.store(e, [i * 128, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_load_reaching_a_phi_through_a_bare_alias(self):
        """``InitMemRef`` shares one MemRef across a bare ``v = t`` tile copy, so
        yielding the *alias* carries the candidate's slot into the phi just as
        yielding ``t`` would. Recording only the names that literally appear in the
        ``YieldStmt`` misses it, and the region reaches codegen anyway."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                flag: pl.Scalar[pl.INT32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                (carried,) = pl.intrin.carry(seed)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    if flag > 0:
                        v: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = t
                        pl.intrin.yield_(v)
                    else:
                        v2: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(carried)
                        pl.intrin.yield_(v2)
                    (yv,) = pl.intrin.leak()
                    pl.intrin.yield_(yv)
                (y,) = pl.intrin.leak()
                nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(y, [0, 0], out)
                return nxt

        self._assert_declined(Before)

    def test_author_region_counts_against_the_capacity_budget(self):
        """A declared allocation is pinned too, so ptoas cannot reuse it either.
        Budgeting only the regions this pass synthesizes lets a 128 KB region in
        alongside the author's 128 KB one, and the pair overflows the space."""

        # Parser-owned MemRef slot authoring, as in test_author_declared_
        # allocation_is_left_alone above.
        @pl.program_parser
        class Before:
            @pl.function_parser
            def main(
                self,
                a: pl.Tensor[[512, 256], pl.FP32],
                b: pl.Tensor[[512, 256], pl.FP32],
                out: pl.Out[pl.Tensor[[512, 256], pl.FP32]],
            ) -> pl.Tensor[[512, 256], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    mine: pl.Tile[[64, 256], pl.FP32, pl.MemRef("mine", slots=2)[i % 2], pl.Mem.Vec] = (
                        pl.load(a, [i * 64, 0], [64, 256], target_memory=pl.Mem.Vec)
                    )
                    t: pl.Tile[[64, 256], pl.FP32, pl.Mem.Vec] = pl.load(
                        b, [i * 64, 0], [64, 256], target_memory=pl.Mem.Vec
                    )
                    s: pl.Tile[[64, 256], pl.FP32, pl.Mem.Vec] = pl.add(mine, t)
                    nxt: pl.Tensor[[512, 256], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        assert set(_slotted_memrefs(after)) == {"mine"}, "only the author's region may survive"
        assert len(_pipeline_loops(after)) >= 1, "the loop must stay Pipeline for replication"

    def test_load_carried_into_a_nested_loops_iter_arg(self):
        """A nested loop's ``init_values`` reach the phi through ``IterArg.initValue_``
        rather than a ``YieldStmt``, but bind the same way. Slotting the initializer
        used to leave the inner init pointing at the pre-substitution Var, which fails
        SSA verification; the loop must be declined instead."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    (carried,) = pl.intrin.carry(t)
                    for _k in pl.range(2):
                        stepped: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(carried)
                        pl.intrin.yield_(stepped)
                    (yk,) = pl.intrin.leak()
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(yk, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_non_unit_step(self):
        """``((iv - start) / step) % F`` is not an affine form ptoas matches."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 256, 64, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_load_carried_out_as_a_phi(self):
        """A yielded tile makes the return var share its MemRef — codegen calls that
        'one of its slots is carried out of an if or a loop as a phi'."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                seed: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                    a, [0, 0], [64, 64], target_memory=pl.Mem.Vec
                )
                (carried,) = pl.intrin.carry(seed)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    pl.intrin.yield_(t)
                (y,) = pl.intrin.leak()
                nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(y, [0, 0], out)
                return nxt

        self._assert_declined(Before)

    def test_load_consumed_by_a_view_op(self):
        """A reshape's result IS its source's buffer, so it would land on the same
        allocation with a different tile_buf type — a codegen blocker."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    r: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(t, [32, 128])
                    e: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.exp(r)
                    f: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(e, [64, 64])
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(f, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_stage_count_above_the_ptoas_maximum(self):
        """ptoas describes 2..16 slots; 32 has no region form."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[2048, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[2048, 64], pl.FP32]],
            ) -> pl.Tensor[[2048, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 32, 1, stage=32):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[2048, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_start_not_a_multiple_of_the_stage_count(self):
        """``iv % F`` only walks slot 0 first when ``start`` is a multiple of ``F``."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[320, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[320, 64], pl.FP32]],
            ) -> pl.Tensor[[320, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(1, 5, 1, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[320, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_loop_nested_under_a_declined_pipeline_loop(self):
        """The outer step-64 loop is replicated, and its F clones would each select one
        slot of the same allocation inside one body — a shape codegen rejects."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (outer,) = pl.intrin.carry(out)
                for _o in pl.pipeline(0, 256, 64, stage=2):
                    (inner,) = pl.intrin.carry(outer)
                    for i in pl.pipeline(0, 4, 1, stage=2):
                        t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                            a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                        )
                        e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                        nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], inner)
                        pl.intrin.yield_(nxt)
                    (y_in,) = pl.intrin.leak()
                    pl.intrin.yield_(y_in)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_one_blocked_load_declines_the_whole_loop(self):
        """Dropping only the blocked load would still demote the loop to Sequential,
        so that load would reach neither these slots nor LowerPipelineLoops' copies."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                b: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 4, 1, stage=2):
                    ok: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    # Consumed by a view op, so it can never become a slot.
                    blocked: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        b, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    r: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(blocked, [32, 128])
                    v: pl.Tile[[32, 128], pl.FP32, pl.Mem.Vec] = pl.exp(r)
                    f: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.tile.reshape(v, [64, 64])
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(ok)
                    s: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.add(e, f)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(s, [i * 64, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        self._assert_declined(Before)

    def test_author_declared_allocation_is_left_alone(self):
        """A binding the author wrote stays the author's."""

        # The declared pl.MemRef slot authoring lives only in LHS annotations —
        # a parser-owned channel (see tests/ut/ir/parser/test_multi_buffer_codegen.py);
        # the tracing surface has no producer-side spelling yet, so this
        # fixture collects through the parser markers with the round-trip
        # carry spelling.
        @pl.program_parser
        class Before:
            @pl.function_parser
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                for i, (acc,) in pl.pipeline(0, 4, 1, stage=2, init_values=(out,)):
                    t: pl.Tile[[64, 64], pl.FP32, pl.MemRef("mine", slots=2)[i % 2], pl.Mem.Vec] = pl.load(
                        a, [i * 64, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i * 64, 0], acc)
                    y = pl.yield_(nxt)
                return y

        after = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        bound = _slotted_memrefs(after)
        assert set(bound) == {"t"}
        assert bound["t"].base_.name_hint == "mine", "the pass must not re-base the author's allocation"


class TestChaining:
    """The two passes are complementary, not alternatives."""

    def test_declined_loop_is_still_replicated_by_lower_pipeline_loops(self):
        """A step-64 loop takes no slot, so it must still get its F body copies."""

        @pl.program
        class Before:
            @pl.function
            def main(
                self,
                a: pl.Tensor[[256, 64], pl.FP32],
                out: pl.Out[pl.Tensor[[256, 64], pl.FP32]],
            ) -> pl.Tensor[[256, 64], pl.FP32]:
                (acc,) = pl.intrin.carry(out)
                for i in pl.pipeline(0, 256, 64, stage=2):
                    t: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.load(
                        a, [i, 0], [64, 64], target_memory=pl.Mem.Vec
                    )
                    e: pl.Tile[[64, 64], pl.FP32, pl.Mem.Vec] = pl.exp(t)
                    nxt: pl.Tensor[[256, 64], pl.FP32] = pl.store(e, [i, 0], acc)
                    pl.intrin.yield_(nxt)
                (y,) = pl.intrin.leak()
                return y

        before_loads = _load_count(Before)
        after_slots = _run_to_slots(Before, passes.MemoryPlanner.PTOAS)
        with passes.PassContext([], memory_planner=passes.MemoryPlanner.PTOAS):
            replicated = passes.lower_pipeline_loops()(after_slots)
        assert _load_count(replicated) == 2 * before_loads


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
