# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Direct acceptance for the PTO trace semantics (``PtoSema`` / ``PtoFrame``).

These tests drive the collector against the PTO semantics without the JIT
envelope or the function adapter: a Sema, a frame installed as the ambient
trace channel, and the instrumented kernel. That boundary matters — the JIT
suite proves the policy on top, and the collector suites prove host execution —
so the statements and bindings produced here are the only place the two claims
meet.

Each assertion pins IR structure, not merely "it ran": a host-executed branch or
an unrolled loop produces a smaller program that still looks plausible.
"""

import ast
import pathlib
import types
from typing import Any

import pypto.language as pl
import pytest
from pypto.language.ambient import active_frame, reset_active_frame, set_active_frame
from pypto.language.constructs.loops import RangeIterator
from pypto.language.protocol import _ExistingVarBinding
from pypto.language.tracing.collector.driver import instrument
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import DataType, ir

F32 = pl.FP32
SHAPE = [16, 16]
TENSOR = pl.Tensor[SHAPE, F32]
OUT_TENSOR = pl.Out[pl.Tensor[SHAPE, F32]]


class _Collect(ir.IRVisitor):
    """Collect the structural statements of one body in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.assigns: list[ir.AssignStmt] = []
        self.ifs: list[ir.IfStmt] = []
        self.fors: list[ir.ForStmt] = []
        self.returns: list[ir.ReturnStmt] = []

    def visit_assign_stmt(self, op: ir.AssignStmt) -> None:
        self.assigns.append(op)
        super().visit_assign_stmt(op)

    def visit_if_stmt(self, op: ir.IfStmt) -> None:
        self.ifs.append(op)
        super().visit_if_stmt(op)

    def visit_for_stmt(self, op: ir.ForStmt) -> None:
        self.fors.append(op)
        super().visit_for_stmt(op)

    def visit_return_stmt(self, op: ir.ReturnStmt) -> None:
        self.returns.append(op)
        super().visit_return_stmt(op)


def _trace(
    kernel: types.FunctionType,
    annotations: dict[str, object],
    *,
    directions: dict[str, ir.ParamDirection] | None = None,
    return_type: ir.Type | None = None,
) -> tuple[ir.Function, PtoSema]:
    """Trace ``kernel`` through the collector with a fresh one-shot PtoSema."""
    sema = PtoSema(name=kernel.__name__, return_type=return_type)
    facades = sema.bind_parameters(annotations, directions=directions or {}, scope=kernel.__globals__)
    frame = sema.frame
    assert frame is not None
    token = set_active_frame(frame)
    try:
        instrument(kernel, sema)(*facades)
    finally:
        reset_active_frame(token)
    return sema.result(), sema


def _collect(fn: ir.Function) -> _Collect:
    assert fn.body is not None
    return _collect_stmt(fn.body)


def _collect_stmt(stmt: ir.Stmt) -> _Collect:
    collector = _Collect()
    collector.visit_stmt(stmt)
    return collector


def test_parameter_directions_reach_the_function_signature():
    """Out/InOut declarations survive binding as IR parameter directions."""

    def kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        acc: pl.InOut[pl.Tensor[[16, 16], pl.FP32]],
    ):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, tile), [0, 0], out)
        return out

    fn, _ = _trace(
        kernel,
        {"x": TENSOR, "out": OUT_TENSOR, "acc": pl.InOut[pl.Tensor[SHAPE, F32]]},
        directions={
            "out": ir.ParamDirection.Out,
            "acc": ir.ParamDirection.InOut,
        },
    )

    assert [p.name_hint for p in fn.params] == ["x", "out", "acc"]
    assert list(fn.param_directions) == [
        ir.ParamDirection.In,
        ir.ParamDirection.Out,
        ir.ParamDirection.InOut,
    ]


def test_a_rebind_assigns_through_the_bound_var():
    """The frame keeps one Var per name: a rebind targets the same variable.

    A definition is an ``AssignStmt`` introducing a fresh Var, so the
    definition/rebind distinction is *identity*, not node kind: the rebind must
    reuse the first binding's Var (assign-through). A fresh Var here would mean
    the name was forked, which silences later reads of the earlier value.
    """

    def kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(kernel, {"x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    t_binds = [stmt for stmt in stmts.assigns if stmt.var.name_hint == "t"]
    assert len(t_binds) == 2
    assert t_binds[0].var is t_binds[1].var, (
        "a same-type rebind must assign through the existing Var, not fork a new one"
    )
    assert "t" in _walk_var_names(t_binds[1].value)


def test_returning_a_declared_out_parameter_is_recorded():
    """The entry's return names the declared output buffer, not a new value."""

    def kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, tile), [0, 0], out)
        return out

    fn, _ = _trace(kernel, {"x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.returns) == 1
    returned = stmts.returns[0].value
    assert len(returned) == 1
    assert isinstance(returned[0], ir.Var)
    assert returned[0].name_hint == "out"


def test_runtime_predicate_stages_an_if_stmt():
    """A predicate over a runtime scalar becomes an IR branch, not a host one."""

    def kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        if n > 0:
            t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.ifs) == 1
    branch = stmts.ifs[0]
    condition = branch.condition
    # A comparison is a dedicated IR node (``ir.Gt``), not a Call; what matters
    # is that it is staged and reads the runtime parameter.
    assert not isinstance(condition, (ir.ConstBool, ir.ConstInt)), (
        "the predicate must stay a runtime conditional, not a host-folded constant"
    )
    assert "n" in _walk_var_names(condition), (
        f"the staged predicate must read the runtime parameter, got {condition}"
    )
    assert _collect_stmt(branch.then_body).assigns, "the then-branch must hold the staged rebind"


def test_runtime_loop_stages_a_for_stmt_with_a_runtime_extent():
    """A runtime bound stays a loop extent instead of being host-unrolled.

    The carry itself (``iter_args``/``return_vars``) is created later, by SSA
    conversion and incore outlining, so the raw trace pins the looser but
    decisive fact: one loop whose extent is the runtime parameter.
    """

    def kernel(
        n: pl.Scalar[pl.INDEX],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        for _i in pl.range(n):
            t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(kernel, {"n": pl.Scalar[pl.INDEX], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.fors) == 1, "the loop must stay one staged loop, not N host iterations"
    loop = stmts.fors[0]
    assert loop.kind == ir.ForKind.Sequential
    assert "n" in _walk_var_names(loop.stop), (
        f"the loop extent must read the runtime parameter, got {loop.stop}"
    )
    assert _collect_stmt(loop.body).assigns, "the body must hold the staged rebind"


def test_a_typed_int64_bound_types_the_loop_var_int64():
    """The tracing lane rebuilds an INT64 loop var from typed INT64 bounds.

    The parser lane infers the loop-var dtype from explicitly typed ConstInt
    bounds (print→parse roundtrip fidelity); the tracing descriptor must apply
    the same inference so identical bounds produce the identical loop-var
    dtype in both lanes.
    """

    def _int64_bound_kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        for i in pl.range(pl.const(8, pl.INT64)):
            t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_int64_bound_kernel, {"x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.fors) == 1
    loop = stmts.fors[0]
    assert isinstance(loop.stop, ir.ConstInt)
    assert isinstance(loop.loop_var.type, ir.ScalarType)
    assert loop.loop_var.type.dtype == DataType.INT64, (
        f"an INT64-typed bound must rebuild an INT64 loop var, got {loop.loop_var.type}"
    )


def test_a_bare_int_bound_keeps_the_loop_var_index():
    """A bare Python int bound still maps to an INDEX loop var."""

    def _bare_int_kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        for i in pl.range(8):
            t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_bare_int_kernel, {"x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.fors) == 1
    loop = stmts.fors[0]
    assert isinstance(loop.loop_var.type, ir.ScalarType)
    assert loop.loop_var.type.dtype == DataType.INDEX, (
        f"a bare Python int bound must keep the INDEX loop var, got {loop.loop_var.type}"
    )


def test_escaping_exception_clears_the_frame_and_aborts_the_sema():
    """A failing trace leaves no ambient frame and no reusable Sema."""

    def kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        raise ValueError("boom")

    sema = PtoSema(name="boom")
    facades = sema.bind_parameters({"x": TENSOR, "out": OUT_TENSOR}, scope=kernel.__globals__)
    frame = sema.frame
    assert frame is not None
    token = set_active_frame(frame)
    try:
        with pytest.raises(ValueError, match="boom"):
            instrument(kernel, sema)(*facades)
    finally:
        reset_active_frame(token)

    assert active_frame() is None
    with pytest.raises(RuntimeError, match="no built function"):
        sema.result()
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.begin_frame()


def test_a_staged_logical_op_carries_the_expression_row():
    """``and``/``or``/``not`` carry the pinned row, not ``Span.unknown()``.

    The parser lane passes the boolean expression's span into ``ir.And`` /
    ``ir.Or`` / ``ir.Not``; the tracing lane resolves the same location from
    the instrumented row's pin (the methods' explicit ``span`` argument wins
    when a caller provides one).
    """

    def _logical_ops_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        if (n > 0) and not (n > 10):
            t = pl.add(t, t)
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_logical_ops_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.ifs) == 1
    branch = stmts.ifs[0]
    boolean_row = _boolop_row("_logical_ops_kernel")

    # ``a and not b`` lowers to And(a, Not(b)) — both must carry the row.
    and_span = branch.condition.span
    assert and_span is not None
    assert (str(and_span.filename), and_span.begin_line) == (__file__, boolean_row), (
        f"the And op must carry the expression's row, got {(and_span.filename, and_span.begin_line)}"
    )
    assert isinstance(branch.condition, ir.And)
    not_operand = branch.condition.right
    assert isinstance(not_operand, ir.Not)
    not_span = not_operand.span
    assert not_span is not None
    assert (str(not_span.filename), not_span.begin_line) == (__file__, boolean_row), (
        f"the Not op must carry the expression's row, got {(not_span.filename, not_span.begin_line)}"
    )


class _VarNames(ir.IRVisitor):
    """Collect the ``name_hint`` of every Var an IR subtree mentions."""

    def __init__(self) -> None:
        super().__init__()
        self.names: list[str] = []

    def visit_var(self, op: ir.Var) -> None:
        self.names.append(op.name_hint)
        super().visit_var(op)


def _walk_var_names(node: ir.Expr | ir.Stmt) -> list[str]:
    """Every Var name mentioned by ``node``, including inside nested calls."""
    collector = _VarNames()
    if isinstance(node, ir.Expr):
        collector.visit_expr(node)
    elif isinstance(node, ir.Stmt):
        collector.visit_stmt(node)
    return collector.names


# ---------------------------------------------------------------------------
# The explicit carry/leak channel — the Frame operations the removed
# ``pl.intrin`` surface drove (``publish_iter_args`` / ``collect_yield`` /
# ``create_leak_var`` / ``complete_leak``). Reconstructed here in miniature so
# the result-binding path of the staged constructs stays acceptance-tested
# against the semantics that own it.
# ---------------------------------------------------------------------------


class _CarryGroup:
    """One ``carry(*values)`` group; binding every proxy publishes the iter_args."""

    def __init__(self, initial: list[Any]) -> None:
        self.initial = initial
        self.slots: list[Any] = [None] * len(initial)
        self.filled = [False] * len(initial)


class _CarryProxy:
    """One channel position of a carry group (an IterArg binding proxy)."""

    def __init__(self, group: _CarryGroup, index: int) -> None:
        self._group = group
        self._index = index

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        init_expr = self._group.initial[self._index].unwrap()
        iter_arg = ir.IterArg(name, init_expr.type, init_expr, ir.Span.unknown())
        self._group.slots[self._index] = iter_arg
        self._group.filled[self._index] = True
        if all(self._group.filled):
            ctx.publish_iter_args(tuple(self._group.slots))
        return _ExistingVarBinding(iter_arg)


def _carry(*values: Any) -> tuple[Any, ...]:
    group = _CarryGroup(list(values))
    return tuple(_CarryProxy(group, i) for i in range(len(values)))


class _YieldMarker:
    """The expression-statement yield marker (``__trace_stmt__``)."""

    def __init__(self, operands: list[Any]) -> None:
        self.operands = operands

    def __trace_stmt__(self, ctx: Any) -> None:
        ctx.collect_yield(self.operands)


def _yield(*values: Any) -> Any:
    return _YieldMarker(list(values))


class _LeakGroup:
    """One ``leak()`` group; binding every proxy completes the result bindings."""

    def __init__(self, n: int, construct: Any) -> None:
        self.n = n
        self.construct = construct
        self.slots: list[Any] = [None] * n
        self.filled = [False] * n


class _LeakProxy:
    """One result position of a leak group (a result-binding proxy).

    The ``_bundle`` attribute is the Frame's recognition marker
    (``_ResultBindingProxy``): a pending result construct admits only these
    proxies as bindings.
    """

    def __init__(self, group: _LeakGroup, index: int) -> None:
        self._bundle = group
        self._index = index

    def __trace_bind__(self, ctx: Any, name: str) -> Any:
        var = ctx.create_leak_var(self._bundle, name, self._index)
        self._bundle.slots[self._index] = var
        self._bundle.filled[self._index] = True
        if all(self._bundle.filled):
            ctx.complete_leak(self._bundle)
        return _ExistingVarBinding(var)


def _leak() -> tuple[Any, ...]:
    frame = active_frame()
    assert frame is not None
    construct = frame.pending_result_construct()
    assert construct is not None
    arity = construct.result_arity
    assert arity is not None
    group = _LeakGroup(arity, construct)
    return tuple(_LeakProxy(group, i) for i in range(group.n))


class _ExitRecordingRange(RangeIterator[pl.Scalar]):
    """A ``pl.range`` whose descriptor close records, to pin the pairing.

    The trace iterable is the for-construct's descriptor: ``begin`` fires at
    the loop-target binding and ``__trace_exit__`` must fire exactly once at
    the close — on the deferred result-binding path no less than the plain
    one.
    """

    exits: int = 0

    def __trace_exit__(self, ctx: Any) -> None:
        _ExitRecordingRange.exits += 1
        super().__trace_exit__(ctx)


def _kernel_line(marker: str, kernel_name: str) -> int:
    """The source line of the ``marker``-named call inside ``kernel_name``.

    The kernels are nested in their test functions, so the walk visits every
    function definition at any depth and the distinct kernel name selects
    the right one.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text())
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fn = next(n for n in fns if n.name == kernel_name)
    rows = [
        call.lineno
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and (getattr(call.func, "id", None) or getattr(call.func, "attr", None)) == marker
    ]
    assert len(rows) == 1, f"expected exactly one {marker!r} call inside {kernel_name}, got rows {rows}"
    return rows[0]


def _boolop_row(kernel_name: str) -> int:
    """The source line of the single ``ast.BoolOp`` inside ``kernel_name``.

    ``and``/``or`` are operators, not calls, so ``_kernel_line``'s call walk
    cannot see them; this walks the same function set for the boolean
    expression's own row.
    """
    tree = ast.parse(pathlib.Path(__file__).read_text())
    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    fn = next(n for n in fns if n.name == kernel_name)
    rows = [op.lineno for op in ast.walk(fn) if isinstance(op, ast.BoolOp)]
    assert len(rows) == 1, f"expected exactly one BoolOp inside {kernel_name}, got rows {rows}"
    return rows[0]


def test_a_type_changing_rebind_honors_the_bind_boundary_span():
    """The fresh ``let`` of a type-changing rebind carries the boundary's span.

    Every neighbouring rebind path passes ``span=stmt_span``; the pin
    fallback in the builder masks the omission only while the pinned row
    equals the boundary span. Driven directly with an explicit span and no
    pin, the rebind must honor the caller's span — not fall back to frame
    capture, which attributes the statement to the Sema's own source file.
    """

    sema = PtoSema(name="rebind_span")
    facades = sema.bind_parameters({"x": TENSOR}, scope={})
    frame = sema.frame
    assert frame is not None
    token = set_active_frame(frame)
    try:
        frame.bind(None, "t", facades[0])
        frame.bind(None, "t", pl.const(1.0, pl.FP32), span=ir.Span("user.py", 42, 0))
    finally:
        reset_active_frame(token)
        frame.finish()
    stmts = _collect(sema.result())

    t_binds = [stmt for stmt in stmts.assigns if stmt.var.name_hint == "t"]
    assert len(t_binds) == 2
    assert t_binds[0].var is not t_binds[1].var, "a type-changing rebind starts a fresh Var"
    span = t_binds[1].span
    assert span is not None
    assert (str(span.filename), span.begin_line) == ("user.py", 42), (
        f"the fresh let must carry the boundary span, got {(span.filename, span.begin_line)}"
    )


def test_an_explicit_carried_for_pairs_the_descriptor_exit_at_the_result_binding():
    """The deferred close runs the iterable's ``__trace_exit__`` exactly once.

    A result-producing ``close()`` returns before the descriptor exit (the
    loop must stay open for the return-var bindings); the final result
    binding owns the pairing. The result yield and the loop's closing row
    carry their pinned locations — the same resolution the plain close path
    gets — instead of degrading to unknown.
    """

    def _explicit_carry_kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        (acc,) = _carry(pl.load(x, [0, 0], [16, 16]))
        for i in _ExitRecordingRange(4):
            acc = pl.add(acc, acc)
            _yield(acc)
        (o,) = _leak()
        pl.store(o, [0, 0], out)
        return out

    _ExitRecordingRange.exits = 0
    fn, _ = _trace(_explicit_carry_kernel, {"x": TENSOR, "out": OUT_TENSOR})
    assert _ExitRecordingRange.exits == 1, (
        "the result-binding completion must run the descriptor's __trace_exit__ "
        "exactly once — the plain end_for_loop bypass would leave it unpaired"
    )

    stmts = _collect(fn)
    assert len(stmts.fors) == 1
    loop = stmts.fors[0]
    assert [v.name_hint for v in loop.return_vars] == ["o"], (
        "the leak binding must become the loop's return var"
    )
    yield_row = _kernel_line("_yield", "_explicit_carry_kernel")
    leak_row = _kernel_line("_leak", "_explicit_carry_kernel")
    all_yields = _collect_yields(fn)
    assert len(all_yields) == 1
    span = all_yields[0].span
    assert span is not None
    assert (str(span.filename), span.begin_line) == (__file__, yield_row)
    assert loop.span is not None
    assert (str(loop.span.filename), loop.span.begin_line, loop.span.end_line) == (
        __file__,
        _kernel_line("_ExitRecordingRange", "_explicit_carry_kernel"),
        leak_row,
    )


def test_a_result_producing_if_pins_the_result_rows():
    """A staged if's result yield and deferred close carry their rows."""

    def _result_if_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        if n > 0:
            t = pl.add(t, t)
            _yield(t)
        (o,) = _leak()
        pl.store(o, [0, 0], out)
        return out

    fn, _ = _trace(_result_if_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)

    assert len(stmts.ifs) == 1
    branch = stmts.ifs[0]
    assert [v.name_hint for v in branch.return_vars] == ["o"]
    all_yields = _collect_yields(fn)
    assert len(all_yields) == 1
    span = all_yields[0].span
    assert span is not None
    assert (str(span.filename), span.begin_line) == (__file__, _kernel_line("_yield", "_result_if_kernel"))
    assert branch.span is not None
    assert branch.span.end_line == _kernel_line("_leak", "_result_if_kernel"), (
        "the deferred close must resolve the pinned row (the result binding), "
        "not degrade the if's closing position to unknown"
    )


def test_a_zero_operand_yield_pins_the_backedge_row():
    """The while backedge terminator carries the yield's own row.

    A zero-operand ``yield_`` is a user statement — the row pin is its
    location, and the synthesized ``YieldStmt`` must resolve it rather than
    degrade to unknown (the parser lane pins the same row).
    """

    def _zero_yield_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        while n > 0:
            t = pl.add(t, t)
            _yield()
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_zero_yield_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})

    all_yields = _collect_yields(fn)
    assert len(all_yields) == 1
    span = all_yields[0].span
    assert span is not None
    assert (str(span.filename), span.begin_line) == (
        __file__,
        _kernel_line("_yield", "_zero_yield_kernel"),
    )


class _Yields(ir.IRVisitor):
    """Collect every ``YieldStmt`` in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.yields: list[ir.YieldStmt] = []

    def visit_yield_stmt(self, op: ir.YieldStmt) -> None:
        self.yields.append(op)
        super().visit_yield_stmt(op)


def _collect_yields(fn: ir.Function) -> list[ir.YieldStmt]:
    assert fn.body is not None
    collector = _Yields()
    collector.visit_stmt(fn.body)
    return collector.yields


class _OpCalls(ir.IRVisitor):
    """Collect every ``Call`` expression in traversal order."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[ir.Call] = []

    def visit_call(self, op: ir.Call) -> None:
        self.calls.append(op)
        super().visit_call(op)


def test_a_one_sided_host_bind_rejects_at_the_first_post_join_read():
    """A host value bound on one branch of a staged if cannot be read after
    the join.

    Host values mint no IR, so the IR verifier's use-after-def check — the
    documented delegation for a one-sided *staged* bind — never fires for
    them: the kept literal would silently inline into every post-join read,
    baking a branch-dependent shape or host control flow that always takes
    one path.
    """

    def _one_sided_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        if n > 0:
            scale = 2
        t = pl.load(x, [0, 0], [scale, scale])
        pl.store(t, [0, 0], out)
        return out

    with pytest.raises(TypeError, match="bound on only one branch"):
        _trace(_one_sided_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})


def test_branch_scratch_and_agreeing_host_binds_stay_silent():
    """The join check still accepts the legal host patterns.

    Per-branch host scratch read only inside its branch is legal (the poison
    defers to the first post-join read, which never happens), and a host
    literal that both branches bind identically is path-independent and
    stays silent.
    """

    def _legal_host_binds_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        if n > 0:
            scratch = 2
            t = pl.add(t, pl.const(scratch, pl.FP32))
        if n > 0:
            factor = 3
        else:
            factor = 3
        t = pl.add(t, pl.const(factor, pl.FP32))
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_legal_host_binds_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)
    assert len(stmts.ifs) == 2
    assert len(stmts.assigns) == 3, (
        "the load and both branch-independent adds must survive: none of the host binds is control-dependent"
    )


def test_the_minted_task_invalid_carries_the_bind_row():
    """``prev = None`` reifies to ``system.task_invalid`` with the bind row.

    The parser lane lowers the same None sentinel with the source span; the
    tracing lane must resolve the pinned row instead of degrading the minted
    expression to ``Span.unknown()``.
    """

    def _none_sentinel_kernel(
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        prev = None  # noqa: F841 - the sentinel bind itself is under test: it mints system.task_invalid
        pl.store(t, [0, 0], out)
        return out

    fn, _ = _trace(_none_sentinel_kernel, {"x": TENSOR, "out": OUT_TENSOR})
    calls = _OpCalls()
    assert fn.body is not None
    calls.visit_stmt(fn.body)
    invalid = [call for call in calls.calls if call.op.name == ir.get_op("system.task_invalid").name]
    assert len(invalid) == 1
    tree = ast.parse(pathlib.Path(__file__).read_text())
    fn_node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_none_sentinel_kernel"
    )
    row = next(
        node.lineno
        for node in ast.walk(fn_node)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and node.value.value is None
    )
    span = invalid[0].span
    assert span is not None
    assert (str(span.filename), span.begin_line) == (__file__, row), (
        f"the minted task_invalid must carry the bind row, got {(span.filename, span.begin_line)}"
    )


def test_a_leak_result_var_carries_the_result_binding_row():
    """The Var minted for an explicit staged result resolves the pinned row.

    The surrounding region rows (yields, the deferred close) already resolve
    the pin; the result Var must not be the one minted node left on
    ``Span.unknown()``.
    """

    def _leak_span_kernel(
        n: pl.Scalar[pl.INT32],
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
    ):
        t = pl.load(x, [0, 0], [16, 16])
        if n > 0:
            t = pl.add(t, t)
            _yield(t)
        (o,) = _leak()
        pl.store(o, [0, 0], out)
        return out

    fn, _ = _trace(_leak_span_kernel, {"n": pl.Scalar[pl.INT32], "x": TENSOR, "out": OUT_TENSOR})
    stmts = _collect(fn)
    assert len(stmts.ifs) == 1
    var = stmts.ifs[0].return_vars[0]
    span = var.span
    assert span is not None
    assert (str(span.filename), span.begin_line) == (__file__, _kernel_line("_leak", "_leak_span_kernel")), (
        f"the minted leak Var must carry the result-binding row, got {(span.filename, span.begin_line)}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
