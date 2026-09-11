# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``Marker`` + standalone sub-kernel calls.

The standalone-call anchors: the executed trace with the callee ``Marker``
in the caller's scope emits the detached ``ir.Call`` — matching the
parser frontend's ``_parse_external_function_call`` shape raw for
straight-line callers and post-canonical for call-in-loop. The callee's
``ir.Function`` equals ``function_parser(callee)`` (incl. ``Out`` param
directions). Guardrails (span-pointed): arg-count mismatch, kwargs
rejected, host arg loud.

The two-way build pattern: a caller defined inside the test references the
callee by a *closure-captured* name (calls execute
natively, the name resolves from the closure cell). The test binds the
ir.Function for the AST parse, then rebinds the ``Marker`` for the trace —
the closure cell is shared, so both sides see their own binding (the
placeholder-era parse-time artifact is gone: the executed path needs no
parse-time resolution).
"""

from collections.abc import Callable
from typing import Any

import pypto.language as pl
import pytest
from pypto.language.op.marker import Marker
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_mark, function_trace, trace, trace_program
from pypto.pypto_core import DataType, ir


def _mark(f: "Callable[..., Any] | None" = None, **kw: Any) -> Any:
    """The @pl.function(...) decorator form for
    function_mark (the flip's _function wrapper maps type= -> function_type=).

    Typed ``-> Any``: the decorated value is an ``ir.Function`` at module
    scope and a callable ``Marker`` inside a program class, and the call
    sites exercise kwargs (``device=``/``attrs=``) the raw Python signature
    cannot express — the same surface ``DSLFunction`` models for
    ``pl.function``."""
    kw = {"function_type": kw.pop("type")} if "type" in kw else kw
    if f is not None:
        return function_mark(f, **kw)

    def deco(g) -> Any:
        return function_mark(g, **kw)

    return deco


def softmax_tile_fn(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.mul(t, 2.0)
    pl.store(s, [0, 0], x)
    return x


def helper_fn(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.add(x, 1.0)


# ===========================================================================
# The standalone-call anchor (raw-equal, straight-line)
# ===========================================================================


def test_standalone_call_anchor_raw_equal():
    """`out = softmax_tile(y)` emits `Call(GlobalVar("softmax_tile_fn"), ...)`
    with the deduced return type — raw-equal to the parser frontend's
    ``_parse_external_function_call`` shape. The Marker's gvar uses the
    callee's *definition* name (``softmax_tile_fn``), matching the AST."""
    softmax_tile = function_parser(softmax_tile_fn)  # the AST side's binding

    def caller(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        t = pl.load(x, [0, 0], [64, 64])
        y = pl.store(t, [0, 0], x)
        return softmax_tile(y)

    ast_caller = function_parser(caller)
    softmax_tile = Marker(softmax_tile_fn)  # the new side's binding
    new_caller = trace(caller)

    ir.assert_structural_equal(new_caller, ast_caller, enable_auto_mapping=True)

    # the Call's gvar is the definition name and the return type is deduced
    calls: list[ir.Call] = []

    class _V(ir.IRVisitor):
        def visit_call(self, op):
            if op.op.name == "softmax_tile_fn":
                calls.append(op)
            super().visit_call(op)

    _V().visit_function(new_caller)
    assert len(calls) == 1
    assert calls[0].op.name == "softmax_tile_fn"
    assert isinstance(calls[0].type, ir.TensorType)


def test_callee_ir_equals_function_parser():
    """The callee's ir.Function (the executed trace) equals
    ``function_parser(callee)`` — the second half of the anchor."""
    new = function_trace(softmax_tile_fn)
    ir.assert_structural_equal(new, function_parser(softmax_tile_fn), enable_auto_mapping=True)


def test_marker_self_sufficient_deduction():
    """The Marker's deduction reads the *declared* signature — never
    the callee's traced body. The detached Call's return type is deduced
    from the Marker's params + `->` return types alone."""
    marker = Marker(softmax_tile_fn)
    t = ir.TensorType([ir.ConstInt(64, DataType.INDEX, ir.Span.unknown())], DataType.FP32, None)
    arg = pl.Tensor(expr=ir.Var("t", t, ir.Span.unknown()))
    result = marker(arg)
    assert isinstance(result, pl.Tensor)
    call = result.unwrap()
    assert isinstance(call, ir.Call)
    assert call.op.name == "softmax_tile_fn"
    assert isinstance(call.type, ir.TensorType)
    # the declared return type flows into the Call (the annotation-only
    # facades are read at construction — the marker never touched a body)
    assert call.type.dtype == DataType.FP32


# ===========================================================================
# Fire-and-forget call → EvalStmt
# ===========================================================================


def test_call_expression_statement_eval_stmt():
    """A fire-and-forget call emits `EvalStmt(Call(...))` — matching
    the parser frontend (the call is still typed; deduction runs regardless of
    statement context)."""
    helper = function_parser(helper_fn)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        helper(x)  # fire-and-forget
        return x

    ast_caller = function_parser(caller)
    helper = Marker(helper_fn)
    new_caller = trace(caller)

    ir.assert_structural_equal(new_caller, ast_caller, enable_auto_mapping=True)
    assert "helper_fn(x)" in str(new_caller)  # expression statement, no let


def test_write_expression_statement_eval_stmt():
    """A side-effect write as an expression statement still lands as ``EvalStmt``.

    ``pl.write`` returns a typed destination facade for value positions, but
    a call as an expression statement remains an evaluated side effect."""

    def kernel(
        src: pl.Tensor[[16, 16], pl.FP16], dst: pl.Tensor[[16, 16], pl.FP16]
    ) -> pl.Tensor[[16, 16], pl.FP16]:
        t: pl.Tile[[16, 16], pl.FP16] = pl.load(src, [0, 0], [16, 16])
        val: pl.Scalar[pl.FP16] = pl.read(t, [0, 0])
        pl.write(t, [0, 1], val)  # raw-Call expression statement
        result: pl.Tensor[[16, 16], pl.FP16] = pl.store(t, [0, 0], dst)
        return result

    new = trace(kernel)
    text = str(new)
    assert "tile.write" in text
    assert "pl.tile.write(t, [0, 1], val)" in text


# ===========================================================================
# Call-in-loop (the Call lands in the loop scope; post-canonical)
# ===========================================================================


def test_call_in_loop_raw():
    """A call inside a staged loop body: the Call is a let in the loop
    scope, raw-equal to the parser frontend. The post-canonical call regime is
    deferred to the program anchors: a standalone external call is
    NOT roundtrip-stable under the tests/ut roundtrip instrument (the
    printed bare-name callee cannot reparse — the AST's own limitation),
    so any pass over the standalone caller trips the
    conftest's RoundtripInstrument; the program shape (self.method,
    resolved via global_vars) is roundtrip-stable and lands with the
    two-pass envelope."""
    helper = function_parser(helper_fn)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        acc = x
        for i in pl.range(2):
            acc = helper(acc)
        return acc

    ast_caller = function_parser(caller)
    helper = Marker(helper_fn)
    new_caller = trace(caller)

    ir.assert_structural_equal(new_caller, ast_caller, enable_auto_mapping=True)


# ===========================================================================
# Out-param deduce + the callee anchor (arg-direction-based return type)
# ===========================================================================


def kernel_out_fn(
    x: pl.Tensor[[64], pl.FP32], out: pl.Out[pl.Tensor[[64], pl.FP32]]
) -> pl.Tensor[[64], pl.FP32]:
    t = pl.load(x, [0], [64])
    out2 = pl.store(t, [0], out)
    return out2


def test_out_param_callee_anchor():
    """An `Out`-param callee: the executed trace creates the param with
    direction Out (recovered from the source — the runtime annotations
    cannot carry it), so the callee's ir.Function matches function_parser
    including param_directions."""
    new = function_trace(kernel_out_fn)
    ast_ = function_parser(kernel_out_fn)
    assert [d for d in new.param_directions] == [ir.ParamDirection.In, ir.ParamDirection.Out]
    assert [d for d in new.param_directions] == [d for d in ast_.param_directions]
    ir.assert_structural_equal(new, ast_, enable_auto_mapping=True)


def test_out_param_call_deduce():
    """A call to an Out-param callee: the return type is deduced from the
    declared `->` via the args (the Marker's params include the Out param —
    plain calls pair args 1:1, no submit skipping)."""
    kernel_out = function_parser(kernel_out_fn)

    def caller(x: pl.Tensor[[64], pl.FP32], dst: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return kernel_out(x, dst)

    ast_caller = function_parser(caller)
    kernel_out = Marker(kernel_out_fn)
    new_caller = trace(caller)
    ir.assert_structural_equal(new_caller, ast_caller, enable_auto_mapping=True)


def test_param_rebind_reuses_cell():
    """`out = pl.store(..., out)` rebinds a param — the
    rebinding reuses the param's ``ir.Var`` (an ``AssignStmt``, not a second
    ``LetStmt``), both in the callee's own body and in a caller's arg."""
    # callee-side: the Out param is rebound through its ``ir.Var``
    new = function_trace(kernel_out_fn)
    assert "out2" in str(new)  # sanity: the body traced

    # caller-side: a param passed to the callee and rebound via the call
    kernel_out = function_parser(kernel_out_fn)

    def caller(x: pl.Tensor[[64], pl.FP32], out: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        r = kernel_out(x, out)
        out = r  # rebind the param's staged local
        return out

    ast_caller = function_parser(caller)
    kernel_out = Marker(kernel_out_fn)
    new_caller = trace(caller)
    ir.assert_structural_equal(new_caller, ast_caller, enable_auto_mapping=True)


# ===========================================================================
# Guardrails (span-pointed, parser-aligned)
# ===========================================================================


def _arg_facade() -> pl.Tensor:
    t = ir.TensorType([ir.ConstInt(64, DataType.INDEX, ir.Span.unknown())], DataType.FP32, None)
    return pl.Tensor(expr=ir.Var("t", t, ir.Span.unknown()))


def test_marker_arg_count_guard():
    marker = Marker(helper_fn)
    with pytest.raises(TypeError, match="expects 1 argument"):
        marker(_arg_facade(), _arg_facade())


def test_marker_kwargs_guard():
    marker = Marker(helper_fn)
    with pytest.raises(TypeError, match="keyword arguments are not accepted"):
        marker(_arg_facade(), extra=_arg_facade())  # type: ignore[call-arg]


def test_marker_host_arg_lifted_at_call_boundary():
    """A host ``int`` reaching a sub-kernel call is lifted to a typed
    constant at the operand boundary (the AST parses a literal
    arg the same way); a genuinely unsupported host value stays a loud
    error (the ``host value`` guard)."""
    marker = Marker(helper_fn)
    call = marker(42)
    assert isinstance(call.unwrap(), ir.Call)
    arg = call.unwrap().args[0]
    assert isinstance(arg, ir.ConstInt)
    assert arg.value == 42

    with pytest.raises(TypeError, match="host value"):
        marker("not a staged value")  # type: ignore[arg-type]


def test_marker_unknown_callee_loud():
    """An unknown callee name at the call site is the host's own
    UnboundLocalError discipline (the name never resolved to a Marker) —
    the parser frontend's equivalent is its parse-time undefined-name error."""

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return not_a_callee(x)  # type: ignore[reportUndefinedVariable]  # noqa: F821 -- loud unknown-callee probe

    with pytest.raises(NameError, match="not_a_callee"):
        trace(caller)


# ===========================================================================
# The mark-vs-eager split
# ===========================================================================


def test_function_mark_standalone_eager_traces():
    """A standalone kernel eager-traces — `function_mark(k)` returns the
    ir.Function (zero migration for the
    standalone users)."""
    fn = function_mark(softmax_tile_fn)
    assert isinstance(fn, ir.Function)
    assert fn.name == "softmax_tile_fn"


def test_function_mark_method_marks():
    """Inside a program class a method *marks* — `function_mark`
    returns a callable Marker (the class decorator defers the trace)."""

    class _Prog:
        def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.add(x, 1.0)

    marked = function_mark(_Prog.helper)
    assert isinstance(marked, Marker)
    assert marked.name == "helper"
    # the leading self is excluded from the declared signature
    assert marked.param_names == ["x"]


# ===========================================================================
# The sub-kernel kwarg surface (device= / attrs=)
# ===========================================================================


def _find_call_attrs(fn, name: str) -> dict:
    """Walk a function body for the first call to ``name`` and return its
    attrs dict (the printer does not render call attrs — assert the IR
    directly)."""
    from pypto.pypto_core import ir as _ir  # noqa: PLC0415 - local: avoid cycle-ish import placement

    found = []

    def walk(stmt):
        if isinstance(stmt, _ir.AssignStmt) and isinstance(stmt.value, _ir.Call):
            if stmt.value.op.name == name:
                found.append(stmt.value)
        if isinstance(stmt, _ir.SeqStmts):
            for s in stmt.stmts:
                walk(s)
            return
        body = getattr(stmt, "body", None)
        if isinstance(body, _ir.SeqStmts):
            for s in body.stmts:
                walk(s)

    walk(fn.body)
    assert found, f"no {name} call in {fn.name} body"
    return found[0].attrs or {}


def test_marker_device_kwarg_on_orch_callee():
    """``device=`` lands on an Orchestrator callee's call as the dispatch
    attr (the AST's accepted-kwarg set); a non-Orchestrator callee
    rejects it loudly (the callee's *effective* role derives from its
    function type — Orchestration → Orchestrator, the C++ Function's own
    derivation)."""

    class Prog:
        @_mark(type=ir.FunctionType.Orchestration, role=ir.Role.Orchestrator)
        def chip_orch(self, data: pl.Tensor[[256], pl.FP32]) -> pl.Tensor[[256], pl.FP32]:
            return data

        @_mark(type=ir.FunctionType.Orchestration, role=ir.Role.Orchestrator)
        def main(self, data: pl.Tensor[[256], pl.FP32]) -> pl.Tensor[[256], pl.FP32]:
            out = self.chip_orch(data, device=0)
            return out

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    assert "device" in _find_call_attrs(main, "chip_orch")


def test_marker_device_kwarg_derived_role():
    """``@pl.function(type=pl.FunctionType.Orchestration)`` (no explicit
    role) still accepts ``device=`` — the effective role derives from the
    function type, matching the AST's ``_reject_unknown_call_kwargs``."""

    class Prog:
        @_mark(type=ir.FunctionType.Orchestration)
        def chip_orch(self, data: pl.Tensor[[256], pl.FP32]) -> pl.Tensor[[256], pl.FP32]:
            return data

        @_mark(type=ir.FunctionType.Orchestration)
        def main(self, data: pl.Tensor[[256], pl.FP32]) -> pl.Tensor[[256], pl.FP32]:
            out = self.chip_orch(data, device=0)
            return out

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    assert "device" in _find_call_attrs(main, "chip_orch")


def test_marker_device_kwarg_rejected_on_subworker():
    """A non-Orchestrator callee rejects ``device=`` loudly (the AST's
    allowed-kwarg filter — ``device=`` is legal only for Orchestrators)."""

    class Prog:
        @_mark
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @_mark(type=ir.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out = self.stage(x, device=0)
            return out

    with pytest.raises(TypeError, match="device= is only legal"):
        trace_program(Prog)


def test_marker_attrs_kwarg():
    """``attrs=`` (call-site metadata) lands on the call; unknown kwargs are
    loud (the AST's allowed set is exactly {attrs} + submit/spmd + device)."""

    class Prog:
        @_mark
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @_mark(type=ir.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out = self.stage(x, attrs={"foo": 1})
            return out

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    assert _find_call_attrs(main, "stage").get("foo") == 1


def test_marker_unknown_kwarg_rejected():
    class Prog:
        @_mark
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @_mark(type=ir.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out = self.stage(x, bogus=1)
            return out

    with pytest.raises(TypeError, match="bogus"):
        trace_program(Prog)
