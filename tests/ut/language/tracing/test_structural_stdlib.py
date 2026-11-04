# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Characterization of the structural stdlib surface on the trace lane.

The behavior pinned here is the contract the authoring-tier simplification
must not change: the ``deps=`` channel (the one dependency discipline the
dispatch and scope spellings share) and the scope body shapes (the deferred
open, the direct-dispatch body, the carrier wrap). The recipe-tier parity
gates in ``test_recipe.py`` pin the construct *conversions*; this module pins
the value shapes those conversions lower onto, which nothing on the trace lane
covered before — in particular the Form-1 per-element ``deps=`` desugaring,
which had coverage on the parse lane only.

Two harnesses, one per lane the shapes live on: the submit spellings through
the jit envelope (the dispatch effect's production lane), the scope spellings
through the raw collector trace (the frontend's IR before the build's passes
outline the carriers — the same harness ``test_recipe.py`` uses).
"""

import types
from typing import TypeAlias

import pypto.language as pl
import pytest
import torch
from pypto import backend
from pypto.backend import BackendType
from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.tracing.collector.driver import instrument
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

F32 = torch.float32
T8x8: TypeAlias = pl.Tensor[[8, 8], pl.FP32]
OUT8x8: TypeAlias = pl.Out[T8x8]


@pytest.fixture(autouse=True)
def _setup_backend():
    """Configure the backend the compile-for-test lane needs (the jit conftest's)."""
    backend.reset_for_testing()
    backend.set_backend_type(BackendType.Ascend910)
    yield
    backend.reset_for_testing()


@pl.function(type=pl.FunctionType.InCore)
def _dispatch_leaf(
    x: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]
) -> pl.Tensor[[8, 8], pl.FP32]:
    """The submitted callee: a marked InCore kernel."""
    t = pl.add(pl.load(x, [0, 0], [8, 8]), pl.load(x, [0, 0], [8, 8]))
    return pl.store(t, [0, 0], out)


def _small_tensor() -> torch.Tensor:
    return torch.randn(8, 8, dtype=F32)


def _small_out() -> torch.Tensor:
    return torch.empty(8, 8, dtype=F32)


def _entry_program(entry):
    """The traced entry's program (the jit envelope's build lane)."""
    bridged, _ = entry._bridge_args((_small_tensor(), _small_out()))
    return entry._build_program(bridged)


def _trace(
    kernel: types.FunctionType,
    annotations: dict[str, object],
    *,
    directions: dict[str, ir.ParamDirection] | None = None,
) -> ir.Function:
    """Trace ``kernel`` through the collector with a fresh one-shot PtoSema.

    The raw frontend IR, before the build's passes outline the carriers — the
    same harness ``test_recipe.py`` traces its parity kernels with. A
    ``pl.Scalar[pl.TASK_ID]`` parameter binds a TaskId the body can depend
    on, the way a captured submit TaskId crosses in production kernels.
    """
    sema = PtoSema(name=kernel.__name__)
    facades = sema.bind_parameters(annotations, directions=directions or {}, scope=kernel.__globals__)
    frame = sema.frame
    assert frame is not None
    token = set_active_frame(frame)
    try:
        instrument(kernel, sema)(*facades)
    finally:
        reset_active_frame(token)
    return sema.result()


def _all_functions(program: ir.Program) -> list[ir.Function]:
    return list(program.functions.values())


class _SubmitCollect(ir.IRVisitor):
    """The ``ir.Submit`` statements of one function, in body order."""

    def __init__(self) -> None:
        super().__init__()
        self.submits: list[ir.Submit] = []

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        if isinstance(op.value, ir.Submit):
            self.submits.append(op.value)
        super().visit_assign_stmt(op)


class _ScopeCollect(ir.IRVisitor):
    """The scope statements of one function, in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.scopes: list[ir.ScopeStmt] = []

    def visit_in_core_scope_stmt(self, op: ir.InCoreScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_in_core_scope_stmt(op)

    def visit_spmd_scope_stmt(self, op: ir.SpmdScopeStmt) -> None:
        self.scopes.append(op)
        super().visit_spmd_scope_stmt(op)


def _submits_of(program: ir.Program) -> list[ir.Submit]:
    submits: list[ir.Submit] = []
    for fn in _all_functions(program):
        assert fn.body is not None
        collector = _SubmitCollect()
        collector.visit_stmt(fn.body)
        submits.extend(collector.submits)
    return submits


def _scope_kinds(fn: ir.Function) -> list[type]:
    """The scope statement kinds of one traced function, in traversal order."""
    assert fn.body is not None
    collector = _ScopeCollect()
    collector.visit_stmt(fn.body)
    return [type(op) for op in collector.scopes]


def _first_scope(fn: ir.Function) -> ir.ScopeStmt:
    """The first scope statement of one traced function."""
    assert fn.body is not None
    collector = _ScopeCollect()
    collector.visit_stmt(fn.body)
    assert collector.scopes
    return collector.scopes[0]


def _manual_dep_edges(scope: object) -> list:
    """The ``manual_dep_edges`` attr of one scope statement ([] without one)."""
    attrs = dict(getattr(scope, "attrs", []) or [])
    return list(attrs.get("manual_dep_edges", []))


def _buffer_chain(fn: ir.Function, end: int) -> list[tuple[str, int]]:
    """The synthesized deps-buffer statement shape ending at Var ``end``.

    The Form-1 desugaring's shape: one ``array.create`` followed by one
    ``array.update_element`` per entry, each rebinding the buffer. Returned as
    ``(op name, argument count)`` pairs — the structure, not the temp names,
    because the two spellings (submit and scope) mint different ones.
    """
    links: dict[int, tuple[str, int, int | None]] = {}
    create = ir.get_op("array.create").name
    update = ir.get_op("array.update_element").name

    class _Link(ir.IRVisitor):
        def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
            value = op.value
            if isinstance(value, ir.Call) and value.op.name in (create, update):
                prev = id(value.args[0]) if value.op.name == update else None
                links[id(op.var)] = (value.op.name, len(value.args), prev)
            super().visit_assign_stmt(op)

    assert fn.body is not None
    _Link().visit_stmt(fn.body)
    shape: list[tuple[str, int]] = []
    current: int | None = end
    while current in links:
        name, argc, prev = links[current]
        shape.append((name, argc))
        if prev is None:
            break
        current = prev
    shape.reverse()
    return shape


def _entry_function(program: ir.Program, name: str) -> ir.Function:
    by_name = {gv.name: fn for gv, fn in program.functions.items()}
    assert name in by_name, f"{name} not in {sorted(by_name)}"
    return by_name[name]


# -----------------------------------------------------------------------------------------------------------
# The deps channel — the submit spellings (the jit envelope's build lane)
# -----------------------------------------------------------------------------------------------------------


class TestSubmitDeps:
    def test_a_scalar_task_id_rides_as_one_direct_edge(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            _, t1 = pl.submit(_dispatch_leaf, a, out, deps=[t0])
            return out

        submits = _submits_of(_entry_program(entry))
        assert len(submits) == 2
        assert len(submits[1].deps) == 1
        dep = submits[1].deps[0]
        assert isinstance(dep, ir.Var)
        assert isinstance(dep.type, ir.ScalarType) and dep.type.dtype == pl.TASK_ID

    def test_a_whole_task_id_array_rides_as_one_direct_edge(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            tids = pl.array.create(1, pl.TASK_ID)
            tids[0] = t0
            _, t1 = pl.submit(_dispatch_leaf, a, out, deps=[tids])
            return out

        submits = _submits_of(_entry_program(entry))
        assert len(submits) == 2
        assert len(submits[1].deps) == 1
        dep = submits[1].deps[0]
        assert isinstance(dep, ir.Var)
        assert isinstance(dep.type, ir.ArrayType) and dep.type.dtype == pl.TASK_ID

    def test_per_element_entries_synthesize_one_deps_array(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            tids = pl.array.create(2, pl.TASK_ID)
            tids[0] = t0
            tids[1] = t0
            _, t1 = pl.submit(_dispatch_leaf, a, out, deps=[tids[0], tids[1]])
            return out

        program = _entry_program(entry)
        submits = _submits_of(program)
        assert len(submits) == 2
        # The whole list is replaced by one synthesized buffer Var.
        assert len(submits[1].deps) == 1
        dep = submits[1].deps[0]
        assert isinstance(dep, ir.Var)
        assert isinstance(dep.type, ir.ArrayType) and dep.type.dtype == pl.TASK_ID
        # The buffer's construction: array.create(2) then two update_element
        # rebinds, one per entry.
        assert _buffer_chain(_entry_function(program, "entry"), id(dep)) == [
            ("array.create", 1),
            ("array.update_element", 3),
            ("array.update_element", 3),
        ]

    def test_none_entries_drop_from_the_edge_list(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            _, t1 = pl.submit(_dispatch_leaf, a, out, deps=[t0, None])
            return out

        submits = _submits_of(_entry_program(entry))
        assert len(submits[1].deps) == 1  # the None sentinel contributes no edge

    def test_mixing_a_whole_array_with_elements_is_loud(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            tids = pl.array.create(1, pl.TASK_ID)
            tids[0] = t0
            with pytest.raises(TypeError, match="cannot mix a whole TASK_ID array"):
                pl.submit(_dispatch_leaf, a, out, deps=[tids, tids[0]])
            return out

        entry.compile_for_test(_small_tensor(), _small_out())

        @pl.jit_trace
        def entry_swapped(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            tids = pl.array.create(1, pl.TASK_ID)
            tids[0] = t0
            with pytest.raises(TypeError, match="cannot mix a whole TASK_ID array"):
                pl.submit(_dispatch_leaf, a, out, deps=[tids[0], tids])
            return out

        entry_swapped.compile_for_test(_small_tensor(), _small_out())

    def test_a_non_taskid_value_is_loud(self):
        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            with pytest.raises(TypeError, match="must be a TaskId variable"):
                pl.submit(_dispatch_leaf, a, out, deps=[a])
            return out

        entry.compile_for_test(_small_tensor(), _small_out())


# -----------------------------------------------------------------------------------------------------------
# The deps channel — the scope spellings (the raw collector trace)
# -----------------------------------------------------------------------------------------------------------


class TestScopeDeps:
    def test_spmd_scope_deps_ride_manual_dep_edges(self):
        def kernel(
            x: T8x8,
            out: OUT8x8,
            tid: pl.Scalar[pl.TASK_ID],
        ):
            with pl.spmd(2, name_hint="grid", deps=[tid]):
                i = pl.tile.get_block_idx()
                t = pl.load(x, [i * 4, 0], [4, 8])
                pl.store(t, [i * 4, 0], out)

        fn = _trace(
            kernel,
            {"x": T8x8, "out": OUT8x8, "tid": pl.Scalar[pl.TASK_ID]},
            directions={"out": ir.ParamDirection.Out},
        )
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt, ir.InCoreScopeStmt]
        edges = _manual_dep_edges(_first_scope(fn))
        assert len(edges) == 1
        assert isinstance(edges[0], ir.Var)
        assert isinstance(edges[0].type, ir.ScalarType) and edges[0].type.dtype == pl.TASK_ID

    def test_spmd_scope_per_element_deps_synthesize_the_buffer(self):
        def kernel(
            x: T8x8,
            out: OUT8x8,
            tid: pl.Scalar[pl.TASK_ID],
        ):
            tids = pl.array.create(2, pl.TASK_ID)
            tids[0] = tid
            tids[1] = tid
            with pl.spmd(2, name_hint="grid", deps=[tids[0], tids[1]]):
                i = pl.tile.get_block_idx()
                t = pl.load(x, [i * 4, 0], [4, 8])
                pl.store(t, [i * 4, 0], out)

        fn = _trace(
            kernel,
            {"x": T8x8, "out": OUT8x8, "tid": pl.Scalar[pl.TASK_ID]},
            directions={"out": ir.ParamDirection.Out},
        )
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt, ir.InCoreScopeStmt]
        edges = _manual_dep_edges(_first_scope(fn))
        # The whole list is replaced by the one synthesized buffer Var, exactly
        # as the submit spelling's deps= is.
        assert len(edges) == 1
        buffer = edges[0]
        assert isinstance(buffer, ir.Var)
        assert isinstance(buffer.type, ir.ArrayType) and buffer.type.dtype == pl.TASK_ID
        assert _buffer_chain(fn, id(buffer)) == [
            ("array.create", 1),
            ("array.update_element", 3),
            ("array.update_element", 3),
        ]

    def test_at_scope_per_element_deps_synthesize_the_buffer(self):
        def kernel(
            x: T8x8,
            out: OUT8x8,
            tid: pl.Scalar[pl.TASK_ID],
        ):
            tids = pl.array.create(1, pl.TASK_ID)
            tids[0] = tid
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="fuse", deps=[tids[0]]):
                t = pl.load(x, [0, 0], [8, 8])
                pl.store(pl.add(t, t), [0, 0], out)

        fn = _trace(
            kernel,
            {"x": T8x8, "out": OUT8x8, "tid": pl.Scalar[pl.TASK_ID]},
            directions={"out": ir.ParamDirection.Out},
        )
        assert _scope_kinds(fn) == [ir.InCoreScopeStmt]
        edges = _manual_dep_edges(_first_scope(fn))
        assert len(edges) == 1
        buffer = edges[0]
        assert isinstance(buffer, ir.Var)
        assert isinstance(buffer.type, ir.ArrayType) and buffer.type.dtype == pl.TASK_ID

    def test_spmd_scope_rejects_a_mixed_list(self):
        def kernel(
            x: T8x8,
            out: OUT8x8,
            tid: pl.Scalar[pl.TASK_ID],
        ):
            tids = pl.array.create(1, pl.TASK_ID)
            tids[0] = tid
            with pytest.raises(TypeError, match="cannot mix a whole TASK_ID array"):
                with pl.spmd(2, deps=[tids, tids[0]]):
                    pl.store(pl.load(x, [0, 0], [8, 8]), [0, 0], out)

        _trace(
            kernel,
            {"x": T8x8, "out": OUT8x8, "tid": pl.Scalar[pl.TASK_ID]},
            directions={"out": ir.ParamDirection.Out},
        )

    def test_the_submit_and_scope_spellings_agree_on_the_buffer_shape(self):
        """One channel: the two spellings' Form-1 desugarings are the same shape.

        The submit and scope spellings lower ``deps=`` through two trace-side
        implementations today; this pin is the seam a single shared lowering
        must keep — same statement kinds, same argument shapes, one buffer
        Var replacing the whole list.
        """

        def via_scope(
            x: T8x8,
            out: OUT8x8,
            tid: pl.Scalar[pl.TASK_ID],
        ):
            tids = pl.array.create(2, pl.TASK_ID)
            tids[0] = tid
            tids[1] = tid
            with pl.spmd(2, name_hint="grid", deps=[tids[0], tids[1]]):
                i = pl.tile.get_block_idx()
                pl.store(pl.load(x, [i * 4, 0], [4, 8]), [i * 4, 0], out)

        fn = _trace(
            via_scope,
            {"x": T8x8, "out": OUT8x8, "tid": pl.Scalar[pl.TASK_ID]},
            directions={"out": ir.ParamDirection.Out},
        )
        edges = _manual_dep_edges(_first_scope(fn))
        assert len(edges) == 1
        scope_shape = _buffer_chain(fn, id(edges[0]))

        @pl.jit_trace
        def via_submit(a: T8x8, out: OUT8x8):
            _, t0 = pl.submit(_dispatch_leaf, a, out)
            tids = pl.array.create(2, pl.TASK_ID)
            tids[0] = t0
            tids[1] = t0
            _, t1 = pl.submit(_dispatch_leaf, a, out, deps=[tids[0], tids[1]])
            return out

        program = _entry_program(via_submit)
        submits = _submits_of(program)
        assert len(submits[1].deps) == 1
        submit_shape = _buffer_chain(_entry_function(program, "via_submit"), id(submits[1].deps[0]))
        assert (
            scope_shape
            == submit_shape
            == [
                ("array.create", 1),
                ("array.update_element", 3),
                ("array.update_element", 3),
            ]
        )

    def test_a_non_taskid_value_is_loud_on_a_scope(self):
        def kernel(x: T8x8, out: OUT8x8):
            with pytest.raises(TypeError, match="TaskId"):
                with pl.spmd(2, deps=[x]):  # pyright: ignore[reportArgumentType] -- a non-TaskId dep is the subject
                    pl.store(pl.load(x, [0, 0], [8, 8]), [0, 0], out)

        _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})


# -----------------------------------------------------------------------------------------------------------
# The scope body shapes — the deferred open and the dispatch body
# -----------------------------------------------------------------------------------------------------------


class TestScopeBodyShapes:
    def test_an_empty_at_scope_still_emits_its_scope(self):
        """The declarable-scope contract: a body that emits nothing still gets its scope."""

        def kernel(x: T8x8, out: OUT8x8):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="empty"):
                pass
            pl.store(pl.load(x, [0, 0], [8, 8]), [0, 0], out)

        fn = _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
        assert _scope_kinds(fn) == [ir.InCoreScopeStmt]
        assert _first_scope(fn).name_hint == "empty"

    def test_an_empty_spmd_body_emits_the_dispatch_scope_only(self):
        """A body that never reads the block index leaves the carrier unopened."""

        def kernel(x: T8x8, out: OUT8x8):
            with pl.spmd(2, name_hint="empty"):
                pass
            pl.store(pl.load(x, [0, 0], [8, 8]), [0, 0], out)

        fn = _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt]
        assert _first_scope(fn).name_hint == "empty_spmd"

    def test_a_single_call_body_takes_the_direct_dispatch_shape(self):
        """A one-call body dispatches directly: the call sits in the Spmd scope, no carrier.

        Pinned on the envelope's build lane (the callee is a marked kernel,
        so the body is a real call site): the dispatch scope holds exactly the
        one call and no InCore carrier is outlined around it.
        """

        @pl.jit_trace
        def entry(a: T8x8, out: OUT8x8):
            with pl.spmd(2):
                _dispatch_leaf(a, out)
            return out

        fn = _entry_function(_entry_program(entry), "entry")
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt]
        spmd = _first_scope(fn)
        # The single call is the dispatch body itself — an EvalStmt in the
        # Spmd scope, no InCore carrier around it.
        assert isinstance(spmd.body, ir.EvalStmt)

    def test_a_multi_statement_body_wraps_in_the_carrier(self):
        """An inline multi-statement body is auto-outlined into the InCore carrier.

        The carrier's open trigger is the body's per-block-index read — the
        tile-scope read (``pl.tile.get_block_idx()``), the spelling the
        production grids use.
        """

        def kernel(x: T8x8, out: OUT8x8):
            with pl.spmd(2, name_hint="grid"):
                i = pl.tile.get_block_idx()
                t = pl.load(x, [i * 4, 0], [4, 8])
                pl.store(t, [i * 4, 0], out)

        fn = _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt, ir.InCoreScopeStmt]

    @pytest.mark.parametrize("read", [pl.tile.get_block_idx, pl.get_block_idx, pl.tensor.get_block_idx])
    def test_every_block_index_spelling_triggers_the_carrier(self, read):
        """The open trigger is the semantic read, not the frontend spelling.

        ``pl.get_block_idx`` / ``pl.tensor.get_block_idx`` produce the
        ``tensor.get_block_idx`` op, which lowers to the tile op only in
        ``ConvertTensorToTileOps`` — after the carrier decision runs — so the
        frontend-side probe has to recognize it by its own op name, the same
        lenient-by-name rule the parser's static ``spmd_body_reads_block_idx``
        applies. All three spellings therefore trace the same wrapped shape:
        the dispatch scope with the InCore carrier nested inside.
        """

        def kernel(x: T8x8, out: OUT8x8):
            with pl.spmd(2, name_hint="grid"):
                t = pl.load(x, [read() * 4, 0], [4, 8])  # the read nested in the offsets
                pl.store(t, [read() * 4, 0], out)

        fn = _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt, ir.InCoreScopeStmt]

    def test_an_unrelated_tensor_read_does_not_trigger_the_carrier(self):
        """The probe is the block-index family, not any tensor-scope op.

        A body whose reads are plain loads keeps the direct shape — the
        statements land in the Spmd scope and no InCore carrier is outlined
        around them.
        """

        def kernel(x: T8x8, out: OUT8x8):
            with pl.spmd(2, name_hint="flat"):
                t = pl.load(x, [0, 0], [4, 8])
                t2 = pl.load(x, [4, 0], [4, 8])
                pl.store(pl.add(t, t2), [0, 0], out)

        fn = _trace(kernel, {"x": T8x8, "out": OUT8x8}, directions={"out": ir.ParamDirection.Out})
        assert _scope_kinds(fn) == [ir.SpmdScopeStmt]
