# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collector contract tests — ownership, lifetime, exceptional exit,
result binding, and one-representative-iteration behavior of the
interpreter-owned staged-execution design.

These are functional conformance tests over the real tracing stack
(PtoSema/PtoFrame/PtoConstructs through the trace interpreter); the
collector-side unit contract (Construct stack, typed results, trusted queries)
lives in ``tests/ut/language/tracer/collector/test_interpreter.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import pathlib

import pypto.language as pl
import pypto.language.tracer.collector.interpreter as collector_interpreter
import pytest
from pypto.language import ambient
from pypto.language.ambient import active_frame
from pypto.language.tracer import semantics
from pypto.language.tracer.collector.interpreter import InterpreterError
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.language.typing import annotation_to_type
from pypto.pypto_core import DataType, ir

# ---------------------------------------------------------------------------
# one-shot Sema, fresh Frame per trace
# ---------------------------------------------------------------------------


def _kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y = pl.add(x, x)
    return y


def test_ptosema_reuse_rejected_after_success():
    """A mutable Sema is one-shot: after a successful trace, begin_frame is
    rejected rather than reset."""
    sema = PtoSema(name="probe")
    sema.bind_parameters({"x": pl.Tensor([64], pl.FP32, _annotation_only=True)})
    sema.begin_frame().finish()
    assert sema.result().name == "probe"
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.begin_frame()


def test_ptosema_reuse_rejected_after_failure():
    """Reuse after a failed trace is likewise rejected (the Sema is never
    reset — the failed trace's builder is discarded with it)."""

    def boom(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y = pl.add(x, x)
        raise ValueError(f"mid-trace {y!r}")

    sema = PtoSema(
        name="probe",
        return_type=annotation_to_type(pl.Tensor[[64], pl.FP32]),
    )
    facades = sema.bind_parameters({"x": pl.Tensor([64], pl.FP32, _annotation_only=True)})
    with pytest.raises(ValueError, match="mid-trace"):
        from pypto.language.tracer.collector.driver import instrument  # noqa: PLC0415

        instrument(boom, sema)(*facades)
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.begin_frame()


def test_ptosema_one_shot_after_inline_body():
    """The inline entrance is a function trace too: ``finish_inline_body``
    locks the one-shot state (previously ``begin_frame()`` after an inline
    body returned the already-finished Frame)."""
    sema = PtoSema(name="inline_probe")
    sema.bind_parameters({"x": pl.Tensor([8], pl.FP32, _annotation_only=True)})
    sema.finish_inline_body("pass")
    assert sema.result().name == "inline_probe"
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.begin_frame()


def test_ptosema_one_shot_after_inline_body_failure(monkeypatch):
    """The lock is taken before the first builder mutation, so an inline
    body that fails mid-emission also rejects reuse."""
    sema = PtoSema(name="inline_probe")
    sema.bind_parameters({"x": pl.Tensor([8], pl.FP32, _annotation_only=True)})

    def boom(body, lang):
        raise RuntimeError("inline failed")

    monkeypatch.setattr(sema._builder, "inline_stmt", boom)
    with pytest.raises(RuntimeError, match="inline failed"):
        sema.finish_inline_body("pass")
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.begin_frame()


def test_ptosema_inline_body_is_a_started_trace():
    """``finish_inline_body`` is an entrance of its own: it rejects an
    already-started trace — a previous inline body, or a frame already
    handed to an interpreter-driven trace through ``begin_frame`` — with the
    one-shot contract error rather than reaching the builder."""
    sema = PtoSema(name="inline_probe")
    sema.bind_parameters({"x": pl.Tensor([8], pl.FP32, _annotation_only=True)})
    sema.finish_inline_body("pass")
    with pytest.raises(RuntimeError, match="one-shot"):
        sema.finish_inline_body("pass")

    sema2 = PtoSema(name="inline_probe2")
    sema2.bind_parameters({"x": pl.Tensor([8], pl.FP32, _annotation_only=True)})
    sema2.begin_frame()  # the interpreter-driven entrance claimed the trace
    with pytest.raises(RuntimeError, match="one-shot"):
        sema2.finish_inline_body("pass")


def test_each_trace_gets_fresh_sema_and_frame():
    """The trace driver acquires a fresh one-shot Sema per independent
    trace (the decorator adapter), so repeated traces never share Frame
    state."""
    f1 = function_trace(_kernel)
    f2 = function_trace(_kernel)
    assert f1 is not f2
    # the two traces produced structurally identical, independent functions
    ir.assert_structural_equal(f1, f2, enable_auto_mapping=True)


# ---------------------------------------------------------------------------
# exceptional exit: skip finalization, propagate unchanged
# ---------------------------------------------------------------------------


def _raising_in_scopes(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(2):
            y = pl.add(x, x)
            raise ValueError(f"boom {y!r}")
    return x


def test_escaping_exception_preserves_type_and_value():
    with pytest.raises(ValueError, match="boom") as excinfo:
        function_trace(_raising_in_scopes)
    assert "pl.add" not in str(excinfo.value)  # the original message unchanged


def test_escaping_exception_skips_finalization():
    """No Interpreter.close, no Frame finish, no staged close: the one-shot
    Sema never produces a result."""
    sema = PtoSema(
        name="probe",
        return_type=annotation_to_type(pl.Tensor[[64], pl.FP32]),
    )
    facades = sema.bind_parameters({"x": pl.Tensor([64], pl.FP32, _annotation_only=True)})
    with pytest.raises(ValueError):
        # drive the raising kernel through the real trace driver with this sema
        from pypto.language.tracer.collector.driver import instrument  # noqa: PLC0415

        instrument(_raising_in_scopes, sema)(*facades)
    with pytest.raises(RuntimeError, match="never ran"):
        sema.result()


def test_active_frame_reset_after_success_and_exception():
    """``active_frame()`` is installed for one function execution and reset
    unconditionally."""
    assert active_frame() is None
    function_trace(_kernel)
    assert active_frame() is None
    with pytest.raises(ValueError):
        function_trace(_raising_in_scopes)
    assert active_frame() is None


def test_half_descriptors_rejected_at_construction():
    """The Frame validates the complete trace-protocol surface at construct
    construction: a trace context manager with an opening hook but no
    ``__trace_exit__``
    is a contextual TypeError at the boundary, not a late bare
    ``AttributeError`` from the construct's close."""

    class HalfForDescriptor:
        def __trace_iter__(self, ctx):
            return []

    class HalfWithDescriptor:
        def __trace_enter__(self, ctx, as_name=None, single_call_body=False):
            return self

    def for_kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        for i in HalfForDescriptor():
            pass
        return x

    def with_kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with HalfWithDescriptor():
            pass
        return x

    with pytest.raises(TypeError, match="__trace_exit__"):
        function_trace(for_kernel)
    with pytest.raises(TypeError, match="__trace_exit__"):
        function_trace(with_kernel)


def test_malformed_trace_bind_result_rejected():
    """A ``__trace_bind__`` hook may return an ``ir.Var``, a facade,
    or ``None`` (decline) — anything else is a contextual TypeError at the
    Frame. It previously fell through to the ordinary bind and silently
    downgraded the name to a host value."""

    class BadBinding:
        def __trace_bind__(self, ctx, name):
            return 42

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        _ = BadBinding()  # the malformed hook result is the subject
        return x

    with pytest.raises(TypeError, match="__trace_bind__ returned an unsupported result"):
        function_trace(kernel)


def test_bindings_without_bind_protocol_rejected_at_construction():
    """A loop-target binding proxy receives the loop's deferred region
    construction and a with
    ``as``-target binding proxy receives the target bind — both through
    ``__trace_bind__``.
    A trace-protocol object handing back a bare object previously installed a
    staged
    for-construct whose region never opened while the trace finished
    successfully (silently omitted IR); an ``as``-target silently became a
    host value."""

    class BareBindingForDescriptor:
        def __trace_iter__(self, ctx):
            return object()  # no __trace_bind__

        def __trace_exit__(self, ctx):
            return None

    class BareAsWithDescriptor:
        def __trace_enter__(self, ctx, as_name=None, single_call_body=False):
            return object()  # no __trace_bind__

        def __trace_exit__(self, ctx):
            return None

    def for_kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        for i in BareBindingForDescriptor():
            pass
        return x

    def with_kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with BareAsWithDescriptor() as _:  # the bare as-target is the subject
            pass
        return x

    with pytest.raises(TypeError, match="without __trace_bind__"):
        function_trace(for_kernel)
    with pytest.raises(TypeError, match="without __trace_bind__"):
        function_trace(with_kernel)


def test_zero_operand_yield_targets_the_nearest_eligible_scope():
    """Ordered target selection: a search may cross
    non-eligible constructs but never skips a nearer eligible target. The
    zero-operand yield terminator's walk visits the interpreter's active
    constructs innermost-first: a loop with explicit `iter_args` (accepting
    yield terminators) or a
    result-producing if between the
    yield and an outer while/bare if with inferred loop-carried values
    stops the walk and reports
    the arity mismatch on that exact construct; a while or bare if with
    inferred loop-carried values nearer than an outer loop with explicit
    `iter_args` is the legal terminator target
    (the former independent while/if lookups each skipped other kinds,
    letting an outer while win over a nearer loop with explicit
    `iter_args`)."""

    def natural_while_wrapping_explicit(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        n = pl.const(3, pl.INT32)
        while x < n:
            x = pl.add(x, pl.const(1, pl.INT32))
            acc0 = pl.const(0, pl.INT32)
            (acc,) = pl.intrin.carry(acc0)
            for i in pl.range(1):  # explicit iter_args — nearer, accepts yield terminators
                acc = pl.add(acc, i)
                pl.intrin.yield_()  # must mismatch the for's arity of 1
                pl.intrin.yield_(acc)
            (out,) = pl.intrin.leak()
            _ = out
        return x

    def staged_if_wrapping_explicit(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if x > 0:
            acc0 = pl.const(0, pl.INT32)
            (acc,) = pl.intrin.carry(acc0)
            for i in pl.range(1):  # explicit — nearer than the outer bare if
                acc = pl.add(acc, i)
                pl.intrin.yield_()
                pl.intrin.yield_(acc)
            (out,) = pl.intrin.leak()
            _ = out
        return x

    def explicit_wrapping_natural_while(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        for i in pl.range(2):  # explicit outer
            n = pl.const(3, pl.INT32)
            while x < n:  # inner while with inferred loop-carried values — the nearer eligible target
                x = pl.add(x, pl.const(1, pl.INT32))
                pl.intrin.yield_()  # the inner while's backedge — legal
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    def bare_if_inside_natural_while(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        n = pl.const(3, pl.INT32)
        while x < n:
            if x > 1:  # bare staged if — nearer than the outer while
                x = pl.add(x, pl.const(1, pl.INT32))
                pl.intrin.yield_()  # the if's void branch terminator
            pl.intrin.yield_()  # the while's backedge (the if already closed)
        return x

    def natural_for_is_crossed(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        n = pl.const(9, pl.INT32)
        while x < n:
            for i in pl.range(2):  # a for with inferred loop-carried values — no yield channel, crossed
                x = pl.add(x, pl.const(1, pl.INT32))
                pl.intrin.yield_()  # crosses the for → the while's backedge
        return x

    def host_while_is_crossed(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        t = 0
        while t < 2:  # host — a host construct, not a terminator target
            t += 1
            pl.intrin.yield_()  # nothing staged to terminate — rejected
        return x

    for kernel in (natural_while_wrapping_explicit, staged_if_wrapping_explicit):
        with pytest.raises(TypeError, match="arity"):
            function_trace(kernel)

    fn = function_trace(explicit_wrapping_natural_while)
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1
    assert any(isinstance(s, ir.YieldStmt) and not s.value for s in _walk(whiles[0]))

    fn = function_trace(bare_if_inside_natural_while)  # the lowered ctrl-flow shape
    ifs = [s for s in _walk(fn.body) if isinstance(s, ir.IfStmt)]
    assert len(ifs) == 1
    assert sum(1 for s in _walk(ifs[0]) if isinstance(s, ir.YieldStmt)) == 1
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1
    yields_in_while = [s for s in _walk(whiles[0]) if isinstance(s, ir.YieldStmt)]
    assert len(yields_in_while) == 2  # the if's terminator and the backedge

    fn = function_trace(natural_for_is_crossed)  # CROSS constructs are crossed
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1
    assert any(isinstance(s, ir.YieldStmt) and not s.value for s in _walk(whiles[0]))

    with pytest.raises(TypeError, match="no region accepts a zero-operand yield terminator"):
        function_trace(host_while_is_crossed)


def test_zero_operand_yield_without_selection_does_not_fall_back(monkeypatch):
    """No zero-operand-yield selection means no region accepts a
    zero-operand terminator
    — never a fallback to the result-yield routing. A staged construct that
    advertises the ordinary yield interface but does not opt into the
    zero-operand protocol (``zero_operand_yield`` → CROSS) previously had its
    zero-operand marker
    delivered through ``on_yield([])``, silently emitting a zero-arity region."""
    from pypto.language.tracer import constructs as regions  # noqa: PLC0415

    monkeypatch.setattr(
        regions.PtoIfConstruct, "zero_operand_yield", lambda self: regions._ZeroOperandYieldAction.CROSS
    )

    def kernel(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if x > 0:  # staged if — accepts yield operands, but not zero-operand now
            pl.intrin.yield_()
        return x

    with pytest.raises(TypeError, match="no region accepts a zero-operand yield terminator"):
        function_trace(kernel)


def test_zero_operand_yield_is_a_structural_terminator():
    """A zero-operand yield is structural, never a result: the staged
    while with inferred loop-carried values (no iter_args) backedge
    terminator (the ctrl-flow
    transform's lowered form) or a bare staged if's zero-operand branch
    terminator. Both emit at the current position; every other shape is
    rejected — the function body has no region to terminate, and a
    result-producing loop accepting yield operands reports its arity mismatch."""

    def top_level(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        pl.intrin.yield_()
        return x

    def while_backedge(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        n = pl.const(3, pl.INT32)
        while x < n:
            x = pl.add(x, pl.const(1, pl.INT32))
            pl.intrin.yield_()  # the staged while's backedge
        return x

    def if_terminator(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if x < 0:  # a bare staged if — the zero-operand yield terminator ends the branch
            _ = pl.add(x, pl.const(1, pl.INT32))
            pl.intrin.yield_()
        return x

    def explicit_arity(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        for i in pl.range(4):  # explicit iter_args — accepts yield operands, one result
            acc = pl.add(acc, i)
            pl.intrin.yield_()  # zero operands against a one-result region
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="no region accepts a zero-operand yield terminator"):
        function_trace(top_level)
    fn = function_trace(while_backedge)
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1
    assert any(isinstance(s, ir.YieldStmt) for s in _walk(whiles[0]))
    fn = function_trace(if_terminator)
    ifs = [s for s in _walk(fn.body) if isinstance(s, ir.IfStmt)]
    assert len(ifs) == 1
    assert any(isinstance(s, ir.YieldStmt) for s in _walk(ifs[0]))
    with pytest.raises(TypeError, match="arity"):
        function_trace(explicit_arity)


def test_entered_context_manager_receives_exception_triple():
    """Python unwinds every successfully entered ``with`` scope through its
    normal context-manager semantics while the exception escapes — a host
    context manager observes the exact exception triple."""
    seen: list[tuple] = []

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            seen.append((exc_type, exc_val is not None))
            return False

    def kernel(cm, x):
        with cm:
            y = x + 1
            raise KeyError("inside")
        return y

    from pypto.language.tracer.collector import instrument  # noqa: PLC0415

    with pytest.raises(KeyError, match="inside"):
        instrument(kernel)(Probe(), 5)
    assert seen == [(KeyError, True)]


# ---------------------------------------------------------------------------
# staged while representative iteration, pending result construct
# ---------------------------------------------------------------------------


def _while_staged(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    n = pl.const(3, pl.INT32)
    while x < n:
        x = pl.add(x, pl.const(1, pl.INT32))
    return x


def test_staged_while_traces_one_iteration():
    """The staged while body traces one representative iteration (one
    WhileStmt, one body), regardless of the host predicate's truth count."""
    fn = function_trace(_while_staged)
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1


def _walk(node) -> list:
    out = [node]
    for name in ("body", "else_body", "then_body"):
        child = getattr(node, name, None)
        if child is not None:
            out.extend(_walk(child))
    stmts = getattr(node, "stmts", None)
    if stmts is not None:
        for s in stmts:
            out.extend(_walk(s))
    return out


def _explicit_for(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    acc0 = pl.const(0, pl.INT32)
    (acc,) = pl.intrin.carry(acc0)
    for i in pl.range(4):
        acc = pl.add(acc, i)
        pl.intrin.yield_(acc)
    (out,) = pl.intrin.leak()
    return out


def test_explicit_for_pending_result():
    """The explicit-`iter_args` for closes as the interpreter's exact top
    pending result construct; the result bindings finish it (one ForStmt,
    one YieldStmt, loop-carried channel)."""
    fn = function_trace(_explicit_for)
    for_stmts = [s for s in _walk(fn.body) if isinstance(s, ir.ForStmt)]
    assert len(for_stmts) == 1
    yields = [s for s in _walk(fn.body) if isinstance(s, ir.YieldStmt)]
    assert len(yields) == 1


def test_witnessed_leak_addresses_its_exact_region():
    """The witnessed ``pl.intrin.leak`` form binds to the exact region it was
    created against (the type witnesses give Pyright the result types): the
    result-binding group records the staged construct, and the binding
    proxies' binds find that same construct.
    """

    def kernel(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak(pl.Scalar[pl.INT32])
        return out

    fn = function_trace(kernel)
    for_stmts = [s for s in _walk(fn.body) if isinstance(s, ir.ForStmt)]
    assert len(for_stmts) == 1
    yields = [s for s in _walk(fn.body) if isinstance(s, ir.YieldStmt)]
    assert len(yields) == 1


def test_witnessed_leak_rejected_without_staged_region():
    """``pl.intrin.leak`` addresses the staged region it is created against:
    the
    witnessed form with no result-producing construct staged is rejected at
    creation, not at a later bind."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        pl.intrin.leak(pl.Scalar[pl.INT32])  # nothing staged
        return n

    with pytest.raises(RuntimeError, match="no pending result construct"):
        function_trace(kernel)


def test_leak_bundle_identity_is_a_frame_precondition():
    """The Frame's ``pl.intrin.leak`` operations require the result-binding
    group's exact region — the
    direct preconditions: a group that does not
    address the innermost active construct is rejected at ``create_leak_var`` and
    ``complete_leak``. The public intrinsic creates its groups against
    the staged region, so these preconditions are the boundary the binding
    proxies bind through."""
    import types  # noqa: PLC0415

    from pypto.language.tracer.collector.interface import (  # noqa: PLC0415
        AwaitingResultBindings,
        BranchDisposition,
    )
    from pypto.language.tracer.collector.interpreter import Interpreter  # noqa: PLC0415

    class _PendingResultRegion:
        kind = "for"
        completed = False
        result_arity = 1

        def is_staged(self):
            return True

        def close(self):
            return AwaitingResultBindings(BranchDisposition.NOT_APPLICABLE)

        def create_leak_var(self, name, index):
            del index
            return ir.Var(name, annotation_to_type(pl.Scalar[pl.INT32]), ir.Span.unknown())

        def complete(self, slots):
            self.completed_slots = slots

    sema = PtoSema(name="probe")
    sema.bind_parameters({"x": pl.Tensor([8], pl.FP32, _annotation_only=True)})
    frame = sema.frame
    assert frame is not None
    rt = Interpreter(sema)  # type: ignore[arg-type]
    rt.enter_frame()
    region = _PendingResultRegion()
    rt._construct_stack.append(region)  # type: ignore[arg-type]
    rt.exit_construct()  # the region's result-producing close enters the result-binding phase
    assert rt.pending_result() is region

    good = types.SimpleNamespace(construct=region, n=1, filled=[True])
    bad = types.SimpleNamespace(construct=object(), n=1, filled=[True])
    var = frame.create_leak_var(good, "y", 0)  # the exact region → delegates
    assert var.name_hint == "y"
    good.slots = [var]
    frame.complete_leak(good)  # commits the exact region
    assert region.completed_slots == [var]
    with pytest.raises(TypeError, match="innermost active construct"):
        frame.create_leak_var(bad, "y", 0)
    with pytest.raises(TypeError, match="exact pending result construct"):
        frame.complete_leak(bad)


def test_outer_else_rejects_inner_pending_result_construct():
    """An outer staged if's else cannot run while an inner explicit if
    (yield, no result binding) left a pending result construct as the
    interpreter's top: the
    pending result construct is itself an IfConstruct, so without the guard the branch switch
    would ``begin_else`` on the *staged* region and only fail later on
    the per-branch locals bookkeeping. The rejection names the pending
    result construct and fires
    before any mutation."""

    def kernel(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if x > 0:
            if x > 1:
                y = pl.add(x, pl.const(1, pl.INT32))
                pl.intrin.yield_(y)
            # no result binding: the inner if stays as the pending result construct
        else:
            x = pl.add(x, pl.const(2, pl.INT32))
        return x

    with pytest.raises(InterpreterError, match="eval_else: result bindings for the staged if"):
        function_trace(kernel)


def test_result_binding_phase_rejects_every_frame_mutator():
    """Every named Frame mutator rejects before mutating while the exact
    top pending result construct is in its result-binding phase — including
    the paths reached only through ambient stdlib code (an
    expression-statement marker's ``__trace_stmt__`` hook is the one
    expression-statement channel the phase admits, so it is the probe)."""

    probe: dict[str, object] = {}

    class MutatorProbe:
        """An expression-statement marker that calls every mutating
        Frame/TraceContext entry point inside the result-binding phase,
        catching each rejection;
        the trace then completes, proving no partial mutation."""

        def __call__(self) -> MutatorProbe:
            return self

        def __trace_stmt__(self, ctx: object) -> None:
            frame = ctx._frame  # type: ignore[attr-defined]
            probe["temp_seq_before"] = frame._temp_seq
            attempts = {
                "let_temp": lambda: frame.let_temp(
                    ir.ConstInt(1, DataType.INT32, ir.Span.unknown()), "probe"
                ),
                "ctx_let_temp": lambda: ctx.let_temp(
                    ir.ConstInt(1, DataType.INT32, ir.Span.unknown()), "probe"
                ),
                "ctx_eval": lambda: ctx.eval(ir.ConstInt(1, DataType.INT32, ir.Span.unknown())),
                "publish_iter_args": lambda: frame.publish_iter_args(()),
                "collect_yield": lambda: frame.collect_yield([]),
            }
            rejected: list[str] = []
            for name, call in attempts.items():
                try:
                    call()
                except TypeError:
                    rejected.append(name)
            probe["rejected"] = rejected
            probe["temp_seq_after"] = frame._temp_seq

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        MutatorProbe()  # the result-binding phase: the explicit-`iter_args` for awaits its result bindings
        (out,) = pl.intrin.leak()
        return out

    fn = function_trace(kernel)
    assert fn is not None  # the trace completed — no partial mutation
    assert probe["rejected"] == [
        "let_temp",
        "ctx_let_temp",
        "ctx_eval",
        "publish_iter_args",
        "collect_yield",
    ]
    assert probe["temp_seq_after"] == probe["temp_seq_before"]


def test_unconsumed_carry_is_loud():
    """Success leaves no unconsumed carry — the first intervening semantic
    event rejects it (the adjacency contract); with no event at all, the
    function finish is the backstop."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        _ = acc  # an ordinary bind between the carry and any loop opener
        return n

    with pytest.raises(TypeError, match="immediately following loop opener"):
        function_trace(kernel)


def test_carry_with_no_following_event_rejected_at_finish():
    """The finish backstop: a carry published as the function's last
    statement is rejected at finish, not silently dropped."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)

    with pytest.raises(TypeError, match="never claimed"):
        function_trace(kernel)


# ---------------------------------------------------------------------------
# carry adjacency: the pending iter_args group must meet the immediately-
# following loop opener (the next semantic event)
# ---------------------------------------------------------------------------


def test_carry_adjacency_rejects_intervening_bind():
    """An ordinary bind between the carry binds and the loop opener
    rejects (the bundle must meet the immediately-following opener)."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        gap = pl.const(7, pl.INT32)  # noqa: F841 — the intervening bind is the subject
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="an ordinary bind of 'gap' intervenes"):
        function_trace(kernel)


def test_carry_adjacency_rejects_intervening_expression_statement():
    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        pl.add(acc, acc)  # a call expression statement — not a loop opener
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="an expression statement intervenes"):
        function_trace(kernel)


def test_carry_adjacency_rejects_intervening_with():
    class HostCM:
        def __enter__(self) -> HostCM:
            return self

        def __exit__(self, *exc: object) -> bool:
            return False

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        with HostCM():
            pass
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="a with intervenes"):
        function_trace(kernel)


def test_carry_adjacency_rejects_intervening_if():
    """Both modes reject: a host if is as much an intervening event as
    a staged one."""

    def kernel_host(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        if _host_true():  # a host (host-truth) if — not a loop opener
            pass
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    def kernel_staged(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        if n > 1:
            pass
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    for kernel in (kernel_host, kernel_staged):
        with pytest.raises(TypeError, match="an if intervenes"):
            function_trace(kernel)


def test_carry_adjacency_rejects_second_carry():
    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        cnt0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        (cnt,) = pl.intrin.carry(cnt0)
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="intervenes"):
        function_trace(kernel)


def test_carry_adjacency_allows_comments():
    """Comments between the carry and its loop are annotation state —
    they attach to the loop opener naturally."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        # the loop-var comment must survive the adjacency window
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    fn = function_trace(kernel)
    assert fn is not None


def test_carry_adjacency_immediate_staged_while_claims():
    """An immediately-following staged while claims the pending loop-carried
    arguments at construct creation (the explicit-while state lives on the
    construct, not the
    Frame)."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        while acc < n:
            acc = pl.add(acc, pl.const(1, pl.INT32))
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    fn = function_trace(kernel)
    assert fn is not None
    whiles = [s for s in _walk(fn.body) if isinstance(s, ir.WhileStmt)]
    assert len(whiles) == 1


_HOST_LIMIT = 3  # a host global: read natively, so a host while predicate without an instrumented bind


def _host_true() -> bool:
    return True  # a host truth source for host-if predicates


def _host_false() -> bool:
    return False  # a host falsity source: a never-taken host branch


def test_claimed_carry_rejects_host_first_predicate():
    """A while that declares pending loop-carried arguments must activate on
    its first predicate — a host first predicate never references the
    carried variables and is rejected loudly instead of drifting to a later
    opener."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        while _HOST_LIMIT > 0:  # host predicate — the claim never activates
            break
        return n

    with pytest.raises(TypeError, match="must activate on its first predicate"):
        function_trace(kernel)


def test_deferred_for_activation_binds_target_once():
    """The loop-target binding proxy retains its exact PtoForConstruct and
    activates the bound target exactly once; a later alias bind of the loop
    var is an ordinary ``AssignStmt`` rebind."""

    def kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        acc = pl.const(0.0, pl.FP32)
        for i in pl.range(4):
            acc = pl.add(acc, pl.const(1.0, pl.FP32))
            j = i  # an alias bind of the loop var — an ordinary bind
            acc = pl.add(acc, pl.cast(j, pl.FP32))
        return acc

    fn = function_trace(kernel)
    names = {v.name_hint for v in _vars(fn)}
    assert "i" in names  # the loop var's staged local
    assert "j" in names  # the alias's staged local (a second, ordinary let)


def _vars(fn) -> list:
    out: list = []

    class V(ir.IRVisitor):
        def visit_assign_stmt(self, op):
            out.append(op.var)
            super().visit_assign_stmt(op)

        def visit_for_stmt(self, op):
            out.append(op.loop_var)
            super().visit_for_stmt(op)

    V().visit_function(fn)
    return out


def test_deferred_for_activation_failure_propagates(monkeypatch):
    """A deferred for activation that fails — before and after the trace
    iterable's builder mutation — propagates to the outside stdlib with no
    ``end_*`` repair of the partial builder state: the one-shot
    Sema never produces a result."""

    def kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        acc = pl.const(0.0, pl.FP32)
        for i in pl.range(4):
            acc = pl.add(acc, pl.const(1.0, pl.FP32))
        return acc

    from pypto.language.tracer import constructs as regions  # noqa: PLC0415
    from pypto.language.tracer.collector.driver import instrument  # noqa: PLC0415

    orig = regions.PtoForConstruct.activate_target

    def drive(fail_after_begin: bool) -> None:
        def boom(self, ctx, name):
            if fail_after_begin:
                orig(self, ctx, name)  # begin_for_loop lands on the builder…
            raise RuntimeError("activation failed")

        monkeypatch.setattr(regions.PtoForConstruct, "activate_target", boom)
        sema = PtoSema(name="probe", return_type=annotation_to_type(pl.Tensor[[64], pl.FP32]))
        facades = sema.bind_parameters({"x": pl.Tensor([64], pl.FP32, _annotation_only=True)})
        with pytest.raises(RuntimeError, match="activation failed"):
            instrument(kernel, sema)(*facades)
        with pytest.raises(RuntimeError, match="never ran"):
            sema.result()

    drive(fail_after_begin=False)
    drive(fail_after_begin=True)


# ---------------------------------------------------------------------------
# staged loop ``else`` is rejected (the ``has_else`` fact)
# ---------------------------------------------------------------------------


def _while_else_staged(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
    while n > 0:
        y = pl.add(x, x)
        n = pl.add(n, pl.const(-1, pl.INT32))
    else:  # noqa: PLW0120 — the loop-else is the subject under test
        y = pl.add(x, x)
    return y


def _for_else_staged(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.range(4):
        y = pl.add(x, x)
    else:  # noqa: PLW0120 — the loop-else is the subject under test
        y = pl.add(x, x)
    return y


def test_staged_while_else_rejected():
    """A staged while cannot collect an else suite — its staged break
        suppresses the native keyword, so the host loop always exits through
        the representative-iteration predicate and the else would be collected
        unconditionally, contradicting Python's skipped-by-break semantics
    ."""
    with pytest.raises(RuntimeError, match="cannot have an 'else' suite"):
        function_trace(_while_else_staged)


def test_staged_for_else_rejected():
    """As the while case: a staged for's close into an else suite is a
    loud rejection before any else statement is collected."""
    with pytest.raises(RuntimeError, match="cannot have an 'else' suite"):
        function_trace(_for_else_staged)


def test_host_loop_else_kept_in_trace():
    """A host loop (host predicate / host iterable) keeps its native
    else: host semantics run and the trace completes (no IR for the loop)."""

    def kernel(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        total = 0
        while total < 3:  # host predicate — host ints
            total += 1
        else:  # noqa: PLW0120 — host loop-else is the subject under test
            total += 10
        return pl.add(x, x) if total else x

    fn = function_trace(kernel)
    assert fn is not None  # traced to completion with the host else intact


# ---------------------------------------------------------------------------
# Definition of Done — static guards: removed mirrors stay removed
# ---------------------------------------------------------------------------


def test_no_doc_citations_in_production_code():
    """Production code states its contracts directly: citations of this
    repository's design documents in docstrings/comments couple code to a
    document that changes on a different cadence, and an editor reader
    cannot follow a repo-relative path anyway. New code must not cite
    ``docs/`` (records of the external design-notes line — RFC/ADR numbers
    living outside this repository — are annotated, not path-cited)."""
    import subprocess  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[3] / "python"
    result = subprocess.run(
        ["rg", "-l", r"docs/(design|en|zh-cn)/", str(root), "-g", "*.py", "-g", "*.cpp", "-g", "*.h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1, (
        f"production code cites design documents:\n{result.stdout}\nstate the contract in code terms instead"
    )


def _resolved_import_targets(node: ast.Import | ast.ImportFrom, package: str) -> tuple[str, ...]:
    """Return fully qualified modules imported by an AST import node."""
    if isinstance(node, ast.Import):
        return tuple(alias.name for alias in node.names)

    module = node.module or ""
    if node.level:
        module = importlib.util.resolve_name("." * node.level + module, package)
    if not module:
        return ()
    return (module, *(f"{module}.{alias.name}" for alias in node.names))


def test_relative_frontend_imports_are_resolved_for_layering_checks():
    """Relative imports resolving to either frontend are visible to the guard."""
    tracer_tree = ast.parse("from ..tracer import constructs")
    parser_tree = ast.parse("from . import parser")

    tracer_targets = _resolved_import_targets(tracer_tree.body[0], "pypto.language.constructs")
    parser_targets = _resolved_import_targets(parser_tree.body[0], "pypto.language")

    assert "pypto.language.tracer" in tracer_targets
    assert "pypto.language.parser" in parser_targets


def test_shared_language_layer_has_no_runtime_frontend_imports():
    """The language layers sit below the frontends' entrance policy.

    A runtime import of ``pypto.language.tracer`` or ``pypto.language.parser``
    from shared code would invert the layering: shared modules would load
    frontend machinery on import, and a frontend could stop being
    replaceable. Imports under ``if TYPE_CHECKING:`` are typing-only and
    stay allowed. The sibling frontends do not import each other either —
    shared seams (``language/ambient.py``, ``language/protocol.py``,
    ``language/diagnostics.py``) carry what both need.

    The entrance tier adds two zones above the shared layer:
    ``language/entrance`` (the foundation — it may drive the tracing
    collector and Sema, never the parser) and ``tracer/decorators`` (the
    tracing frontend's entrance policy — it composes the foundation). The
    collector + language machinery below the foundation (``tracer`` outside
    ``decorators``) imports no entrance policy, and the parser is taught
    neither tracing nor entrance semantics.
    ``pypto/language/__init__.py`` is exempt: the package facade dispatches
    ``pl.function``/``pl.program`` to the tracing frontend by design."""
    language_root = pathlib.Path(__file__).resolve().parents[3] / "python" / "pypto" / "language"
    facade = language_root / "__init__.py"
    violations = []
    for path in sorted(language_root.rglob("*.py")):
        rel = path.relative_to(language_root)
        if path == facade:
            continue
        zone = next((p for p in rel.parts if p in ("tracer", "parser", "entrance")), "shared")
        if zone == "tracer" and len(rel.parts) > 1 and rel.parts[1] == "decorators":
            # The tracing frontend's entrance policy lives in the tracer tree
            # but sits above the foundation.
            zone = "tracer-decorators"
        package = "pypto.language"
        if rel.parts[:-1]:
            package += "." + ".".join(rel.parts[:-1])
        tree = ast.parse(path.read_text(), filename=str(path))
        # Nodes under an `if TYPE_CHECKING:` guard are typing-only.
        typing_only: set[int] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.If)
                and isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
            ):
                typing_only.update(id(sub) for sub in ast.walk(node))
        for node in ast.walk(tree):
            if id(node) in typing_only or not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            targets = _resolved_import_targets(node, package)
            for target in targets:
                forbidden = {
                    "shared": ("tracer", "parser", "entrance"),
                    "tracer": ("parser", "entrance"),
                    "tracer-decorators": ("parser",),
                    "parser": ("tracer", "entrance"),
                    "entrance": ("parser",),
                }[zone]
                hit = next(
                    (
                        f
                        for f in forbidden
                        if target == f"pypto.language.{f}" or target.startswith(f"pypto.language.{f}.")
                    ),
                    None,
                )
                if hit is not None:
                    violations.append(f"{rel}:{node.lineno}: {zone} imports pypto.language.{hit}")
    assert not violations, "runtime frontend imports invert the language layering:\n" + "\n".join(violations)


# Files that still cite records of the *external* design-notes line by
# section (``ADR 27 §3.2``, ``doc 31 §0.1``, ``RFC #1300 §2.4`` — records
# that live outside this repository and are annotated, not path-cited).
# The set must not grow — new files state contracts in code terms (the
# doc altitude map: docs/README.md).
_GRANDFATHERED_SECTION_CITATIONS = frozenset(
    {
        "bindings/modules/ir.cpp",
        "bindings/modules/passes.cpp",
        "pypto/ir/builder.py",
        "pypto/ir/pass_manager.py",
        "pypto/language/tracer/decorators/function.py",
        "pypto/language/distributed/op/tensor_ops.py",
        "pypto/language/distributed/typing/__init__.py",
        "pypto/language/constructs/loops.py",
        "pypto/language/constructs/scopes.py",
        "pypto/language/constructs/submit.py",
        "pypto/language/intrin.py",
        "pypto/language/op/marker.py",
        "pypto/language/op/system_ops.py",
        "pypto/language/op/tensor_ops.py",
        "pypto/language/typing/tuple.py",
        "pypto/language/typing/type_utils.py",
        "pypto/pypto_core/passes.pyi",
    }
)


def test_no_new_section_citations_in_production_code():
    """Section citations (``NN §N``) couple code to documents on a
    different cadence — the same principle as
    ``test_no_doc_citations_in_production_code``. The files above are
    grandfathered; a new file joining the list is the failure."""
    import subprocess  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[3] / "python"
    result = subprocess.run(
        ["rg", "-l", r"\d+ §", str(root)],
        capture_output=True,
        text=True,
        check=False,
    )
    cited: set[str] = set()
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            if line:
                cited.add(str(pathlib.Path(line).relative_to(root)))
    new = cited - _GRANDFATHERED_SECTION_CITATIONS
    assert not new, (
        f"new production files cite documents by section:\n{sorted(new)}\n"
        "state the contract in code terms instead (rationale: the doc "
        "altitude map, docs/README.md)"
    )


def test_removed_ordering_mirrors_stay_removed():
    """The retired Sema/Frame mirrors and lifecycle flags must not be
    reintroduced (static checks that removed mirrors and lifecycle flags
    are not read)."""
    for removed in ("Sentinel", "_Handle"):
        assert not hasattr(collector_interpreter, removed), removed
    for removed in ("_open_stack", "_loop_carriers", "_cf_depth", "_staged", "_host_bound"):
        assert not hasattr(semantics.PtoFrame, removed), removed
        assert not hasattr(semantics.PtoSema, removed), removed
    # the ambient entry is the Frame, never the Sema
    for removed in ("active_trace", "current_split_aiv_mode", "set_split_aiv_mode"):
        assert not hasattr(ambient, removed), removed
    assert hasattr(ambient, "active_frame")


def test_active_frame_is_the_only_ambient_entry():
    """No scope-spanning ambient state: the funnel module carries exactly
    one contextvar (the active Frame)."""
    contextvars = [name for name in vars(ambient) if name.startswith("_") and name.isupper()]
    assert contextvars == ["_ACTIVE_FRAME"], contextvars


# ---------------------------------------------------------------------------
# regression: nested-if merges and native-transfer cleanup
# ---------------------------------------------------------------------------


def test_nested_staged_if_merge_stays_branch_local():
    """A nested staged if inside an outer staged branch: the inner
    merge stays branch-local — the outer else path never observes values
    bound only on the outer then path."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if n > 0:
            if n > 1:
                y = pl.add(n, pl.const(1, pl.INT32))
            else:
                y = pl.const(2, pl.INT32)
        else:
            y = pl.const(0, pl.INT32)
        return y

    fn = function_trace(kernel)
    assert fn is not None


def test_suppressing_cm_owns_ir_consequences():
    """Invariant 21 pinned in code terms: a suppressing context manager
    owns the IR consequences of the suppressed dynamic path. The staged
    region it suppressed stays open (its staged close was skipped during
    the unwind), so the resumed statements land *inside* that region —
    there is no trace-time detection or recovery by design, and default
    IR property verification accepts the shape."""

    class Suppressor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return True

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with Suppressor():
            with pl.at(level=pl.Level.CORE_GROUP):
                y = pl.add(x, x)
                raise ValueError(f"boom {y!r}")
        z = pl.add(x, x)
        return z

    fn = function_trace(kernel)
    # the resumed statements (z's add, the return) landed inside the
    # still-open region — the consequence the suppressing manager owns
    regions = [s for s in _walk(fn.body) if isinstance(s, ir.InCoreScopeStmt)]
    assert len(regions) == 1
    inner = _walk(regions[0])
    assert any(isinstance(s, ir.ReturnStmt) for s in inner)
    assert "z" in {v.name_hint for v in _vars(fn)}
    # and no trace-time or default-verification rejection exists for it
    from pypto.pypto_core.passes import (  # noqa: PLC0415
        PropertyVerifierRegistry,
        get_default_verify_properties,
    )

    PropertyVerifierRegistry.verify_or_throw(
        get_default_verify_properties(), ir.Program([fn], "probe", ir.Span.unknown())
    )


def test_staged_with_not_finalized_during_unwind():
    """An exception unwinding through a staged ``with`` skips its
    staged close entirely: the trace context manager's ``__trace_exit__``
    must not run — the one-shot Sema is discarded with the exception."""
    exits: list[object] = []

    class RecordingDescriptor:
        def __trace_iter__(self, ctx):  # pragma: no cover - not a trace iterable
            raise AssertionError("not a for trace iterable")

        def __trace_enter__(self, ctx, as_name=None, single_call_body=False):
            return self

        def __trace_exit__(self, ctx):
            exits.append(ctx)

    context_manager = RecordingDescriptor()

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with context_manager:
            y = pl.add(x, x)
            raise ValueError(f"boom {y!r}")
        return x  # pragma: no cover - unreachable

    with pytest.raises(ValueError, match="boom"):
        function_trace(kernel)
    assert exits == []


def test_host_with_still_receives_exception_triple():
    """The P1b skip is staged-only: a host ``with`` still receives
    its exception triple and suppression exactly once during the unwind
    (regression guard for the _exit_with_construct branch)."""

    class Suppressing:
        def __init__(self):
            self.triples = []

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.triples.append(exc)
            return True  # suppress

    sup = Suppressing()

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with sup:
            raise KeyError("inside")
        return x  # pragma: no cover - suppressed path resumes here

    from pypto.language.tracer.collector import instrument  # noqa: PLC0415

    fn_result = instrument(kernel)(pl.Tensor([8], pl.FP32, _annotation_only=True))
    assert fn_result is not None  # the suppression resumed the trace
    assert len(sup.triples) == 1
    assert sup.triples[0][0] is KeyError


def test_native_break_cleans_stale_constructs_in_staged_if():
    """A native break crossing a staged if inside a loop-``else``: the
    if is wrapped in the interpreter's unwind scope guard, so the keyword's
    unwind closes it at the if's exact lexical boundary — the per-branch
    locals mappings commit and the enclosing structure stays balanced (the
    stale loop previously swallowed the enclosing close)."""

    def for_else_kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if n > 0:
            for j in [1, 2, 3]:
                if j == 2:
                    break
            else:  # noqa: PLW0120 — the skipped-else path is the subject
                pass
            y = pl.add(n, pl.const(1, pl.INT32))
        else:
            y = pl.const(0, pl.INT32)
        return y

    def while_else_kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if n > 0:
            t = 0
            while t < 3:
                t += 1
                break
            else:  # noqa: PLW0120 — the skipped-else path is the subject
                t = 99
            y = pl.add(n, pl.const(t, pl.INT32))
        else:
            y = pl.const(0, pl.INT32)
        return y

    for kernel in (for_else_kernel, while_else_kernel):
        fn = function_trace(kernel)
        assert fn is not None


def test_native_continue_cleans_stale_constructs_in_staged_if():
    """As the break case: a native continue unwinds the instrumented
    constructs enclosing it inside the loop, and their unwind scope guards
    close each one at its exact lexical boundary."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if n > 0:
            total = 0
            for j in [1, 2, 3]:
                if j == 2:
                    continue
                total += 1
            y = pl.add(n, pl.const(total, pl.INT32))
        else:
            y = pl.const(0, pl.INT32)
        return y

    fn = function_trace(kernel)
    assert fn is not None


def test_native_break_through_with_keeps_cm_protocol():
    """A native break unwinding a host ``with`` inside a loop-``else``:
    the context manager receives exactly one ``__exit__`` per entry, at
    the with's own lexical boundary in the unwind (the generated with
    adapter — not any pre-keyword cleanup — drives the close)."""

    class Probe:
        def __init__(self):
            self.exits = 0

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.exits += 1
            return False

    probe = Probe()

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        if n > 0:
            for j in [1, 2, 3]:
                with probe:
                    if j == 2:
                        break
            else:  # noqa: PLW0120 — the skipped-else path is the subject
                pass
            y = pl.add(n, pl.const(1, pl.INT32))
        else:
            y = pl.const(0, pl.INT32)
        return y

    function_trace(kernel)
    assert probe.exits == 2  # one per host entry (j=1 normal, j=2 break-unwind)


# ---------------------------------------------------------------------------
# Unwind scope guards: exceptional exit and suppression crossing a guard
# ---------------------------------------------------------------------------


def test_escaping_exception_through_unwind_guard():
    """An exception escaping through an unwind scope guard skips normal
    finalization exactly as elsewhere: the guard no-ops on the exception
    triple, the one-shot Sema never produces a result, and the original
    exception propagates unchanged."""

    def kernel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        total = pl.const(0.0, pl.FP32)
        for j in [1, 2]:  # host loop; its unwind guard wraps the while rows
            if j == 1:
                if x[0] > 0.0:  # staged if inside the guarded loop body
                    raise ValueError("boom inside guard")
                break  # makes both ifs and the loop transfer-affected
            total = pl.add(total, pl.const(1.0, pl.FP32))
        return total

    sema = PtoSema(
        name="probe",
        return_type=annotation_to_type(pl.Tensor[[64], pl.FP32]),
    )
    facades = sema.bind_parameters({"x": pl.Tensor([64], pl.FP32, _annotation_only=True)})
    with pytest.raises(ValueError, match="boom inside guard"):
        from pypto.language.tracer.collector.driver import instrument  # noqa: PLC0415

        instrument(kernel, sema)(*facades)
    with pytest.raises(RuntimeError, match="never ran"):
        sema.result()


def test_suppressing_cm_after_guarded_staged_if_owns_consequences():
    """Invariant 21 pinned for the unwind scope guard: a suppressing context
    manager owns the IR consequences of the suppressed path. The guard
    no-ops during the unwind (the staged if stays open), the resumed
    statements land inside the still-open region, and the guard's later
    sweep closes the stale construct — completing the trace with the
    consequence owned, never a trace-time recovery."""

    class Suppressor:
        def __enter__(self) -> Suppressor:
            return self

        def __exit__(self, *exc: object) -> bool:
            return True  # suppress everything

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        for j in [1, 2]:  # the loop guard's sweep closes what the suppression left
            with Suppressor():
                if x[0] > 0.0:  # staged if; the break below guards it
                    raise ValueError("suppressed")
                break
        return pl.add(x, x)

    fn = function_trace(kernel)
    assert fn is not None
    # one staged if per loop entry (the all-paths collection of the
    # host two-iteration loop): the second iteration's statements
    # collected inside the first entry's still-open region — the owned
    # consequence — and the guard's sweep closed both before the return
    ifs = [s for s in _walk(fn.body) if isinstance(s, ir.IfStmt)]
    assert len(ifs) == 2
    first, second = ifs
    assert second in _walk(first)  # the resumed iteration landed inside
    top_level = [s for s in fn.body if isinstance(s, ir.ReturnStmt)]
    assert len(top_level) == 1  # the return collected after the sweep


def test_staged_transfer_in_inner_loop_else_is_precise():
    """The staged variant of the else-suite transfer: a break in an
    inner loop's else suite crossing a staged if is rejected by the
    transfer rule itself (the alternate staged path was not
    collected) — not by stale-construct corruption (the walker previously
    counted the else suite inside the loop, so no guards were emitted
    and the stale constructs corrupted routing instead)."""

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        for outer in [1, 2, 3]:
            if x[0] > 0.0:  # staged: the break crosses its then-branch
                for inner in [1, 2]:
                    z = pl.add(x, x)  # noqa: F841 — the instrumented body is the subject
                else:  # noqa: PLW0120 — the else-suite transfer is the subject
                    break  # targets the outer loop
            w = pl.add(x, x)
        return w

    with pytest.raises(InterpreterError, match="staged Construct"):
        function_trace(kernel)


# ---------------------------------------------------------------------------
# Third review round — boundary-crossing pending result constructs,
# residual carried-value paths
# ---------------------------------------------------------------------------


def test_with_boundary_rejects_inner_pending_result():
    """A with whose body stages an inner pending result construct cannot
    close across it: the result bindings must finish inside the with
    (previously the exit took the out-of-order path and everything after
    the ``pl.intrin.leak`` collected inside the still-open region)."""

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            acc0 = pl.const(0, pl.INT32)
            (acc,) = pl.intrin.carry(acc0)
            for i in pl.range(4):
                acc = pl.add(acc, i)
                pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()  # outside the with — the crossing
        return x

    with pytest.raises(InterpreterError, match="finish the result bindings inside the with"):
        function_trace(kernel)


def test_leak_inside_with_closes_the_region_before_the_boundary():
    """The legal placement: the result bindings finish the pending result
    construct inside the with, so the region closes at its boundary and
    the return collects outside it."""

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            acc0 = pl.const(0, pl.INT32)
            (acc,) = pl.intrin.carry(acc0)
            for i in pl.range(4):
                acc = pl.add(acc, i)
                pl.intrin.yield_(acc)
            (out,) = pl.intrin.leak()
        return x

    fn = function_trace(kernel)
    assert fn is not None
    regions = [s for s in _walk(fn.body) if isinstance(s, ir.InCoreScopeStmt)]
    assert len(regions) == 1
    # the return is NOT inside the region — the boundary closed before it
    assert not any(isinstance(s, ir.ReturnStmt) for s in _walk(regions[0]))


def test_guarded_if_boundary_rejects_inner_pending_result():
    """An unwind scope guard's sweep rejects an inner construct's pending
    result construct sitting above the construct's own boundary: only the
    construct's own result-producing close may defer to the result
    bindings that follow it."""

    def kernel(x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
        for outer in [1, 2]:  # host loop: its guard sweeps after the loop
            if x[0] > 0.0:  # staged if, guarded (the break in its subtree)
                if _host_false():  # host: never taken, still triggers guarding
                    break
                acc0 = pl.const(0, pl.INT32)
                (acc,) = pl.intrin.carry(acc0)
                for i in pl.range(4):
                    acc = pl.add(acc, i)
                    pl.intrin.yield_(acc)  # stages the pending result construct; no leak inside
        return x

    with pytest.raises(InterpreterError, match="finish the result bindings inside the construct"):
        function_trace(kernel)


def test_carry_adjacency_rejects_intervening_host_for():
    """A host for cannot declare pending loop-carried arguments (a host
    iterable has no iter-arg channel) — even a zero-iteration one
    intervenes."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        for j in []:  # host, zero-iteration
            pass
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="a host for .* intervenes"):
        function_trace(kernel)


def test_carry_adjacency_rejects_intervening_temporary_emission():
    """let_temp (and the TraceContext emission paths) are mutating Frame
    operations: a temporary emission between the carry binds and the
    opener rejects before mutating."""

    def kernel(n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
        acc0 = pl.const(0, pl.INT32)
        (acc,) = pl.intrin.carry(acc0)
        active_frame().let_temp(ir.ConstInt(1, DataType.INT32, ir.Span.unknown()), "gap")
        for i in pl.range(4):
            acc = pl.add(acc, i)
            pl.intrin.yield_(acc)
        (out,) = pl.intrin.leak()
        return out

    with pytest.raises(TypeError, match="a temporary emission .* intervenes"):
        function_trace(kernel)
