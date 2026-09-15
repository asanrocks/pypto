# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Inline-as-sub-kernel.

An inline callee is a sub-kernel: a `Call(GlobalVar, …)` at the call site
+ a `FunctionType.Inline` function in the Program, spliced by the backend
`inline_functions` pass (the first pipeline pass). The anchor runs
the NEW caller through `canonicalize(fn, inline=True, callees=[...])` (the
harness's inline regime) and compares against the AST caller — which is
*parse-time-spliced* (`_parse_inline_call`).

The flagged risk: the pass's splice (alpha-rename +
param-substitute) vs the AST parser's parse-time splice. Probed with
non-trivial bodies (locals, rebinds, CF, loop) — the post-canonical SSA
rename washes out the pass's `_inline0` suffix, so both sides converge;
no divergence surfaced (the pass is load-bearing, its shape wins — the
precedent would have been to document it).

The one divergence the splice probe did surface is the **return arity** of a
multi-value trailing return: the pass wires the callee's values into the
caller's `ReturnStmt`, so the parse-time expansion has to splice the
written-out `return a, b` rather than a tuple value (**F16**, fixed — anchored
below). The same return written through a tuple temporary keeps the parser's
`MakeTuple` binding on both lanes (**F37**, fixed — anchored below). An early
return under staged control flow has no splice semantics on either lane, so
the parser rejects it the way the pass and the `@pl.expand` entrance do
(**F17**, fixed — pinned below); a dead second top-level return is the F15
shape on this seam and is fixed the same way (**F38**, anchored below).
"""

import pypto.language as pl
import pytest
from pypto import InternalError
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.op.marker import Marker
from pypto.language.parser import function_parser, inline
from pypto.language.tracer.decorators.function import function_trace, trace
from pypto.pypto_core import ir, passes

from ._harness import canonicalize

_TENSOR_ADDS = ir.get_op("tensor.adds").name


def _body_stmts(fn: ir.Function) -> list[ir.Stmt]:
    """A function body's statement list (a single stmt is not wrapped)."""
    body = fn.body
    return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]


def _inline_anchor(helper_fn, n_params: int = 1):
    """The two-way inline anchor. The caller is defined HERE so its
    ``helper`` reference captures THIS frame's ``helper`` cell — the two
    helpers rebind that cell (Marker for the trace, InlineFunction for the
    AST parse) between the sides. ``n_params`` picks the caller's arity."""
    callee_new = function_trace(helper_fn, function_type=ir.FunctionType.Inline)
    helper = Marker(helper_fn, function_type=ir.FunctionType.Inline)

    if n_params == 1:

        def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return helper(x)

    else:

        def caller(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
            return helper(x, n)

    caller_new = trace(caller)
    spliced = canonicalize(caller_new, inline=True, callees=[callee_new])

    helper = inline(helper_fn)  # the AST side: parse-time splice
    ast_caller = function_parser(caller)
    c_ast = canonicalize(ast_caller)

    ir.assert_structural_equal(spliced, c_ast, enable_auto_mapping=True)


# ===========================================================================
# The inline-call anchor (post-inline_functions caller equality, post-canonical)
# ===========================================================================


def h_add_one(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    y = pl.add(x, 1.0)
    return y


def test_inline_call_anchor():
    """`return h_add_one(x)` — the new caller after `inline_functions` (the
    Inline callee spliced in, then the canonical pair) equals the AST
    caller (parse-time-spliced body, canonicalized)."""
    _inline_anchor(h_add_one)


def h_rebind(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    t = pl.add(x, 1.0)
    t = pl.add(t, x)  # a local + a rebind in the inline body
    return t


def test_inline_anchor_rebind_body():
    """The inline-splice probe: a local + rebind body — the pass's splice and the AST's
    parse-time splice converge post-canonical."""
    _inline_anchor(h_rebind)


def h_cf(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
    if n != 0:
        x = pl.add(x, 1.0)
    return x


def test_inline_anchor_cf_body():
    """The inline-splice probe: a staged-if body in the inline callee."""
    _inline_anchor(h_cf, n_params=2)


def h_loop(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    acc = x
    for i in pl.range(2):
        acc = pl.add(acc, acc)
    return acc


def test_inline_anchor_loop_body():
    """The inline-splice probe: a loop body in the inline callee."""
    _inline_anchor(h_loop)


def h_multi_value(
    x: pl.Tensor[[64], pl.FP32],
) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
    a = pl.add(x, 1.0)
    b = pl.mul(x, 2.0)
    return a, b


def _inline_multi_value_anchor(helper_fn):
    """The two-way anchor for a multi-value trailing return: ``_inline_anchor``
    with the two-value ``->`` the caller has to declare (the tracing entry is
    the type-sound one, and the declared arity is what makes the caller's
    ``return helper(x)`` well-formed on both lanes)."""
    callee_new = function_trace(helper_fn, function_type=ir.FunctionType.Inline)
    helper = Marker(helper_fn, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return helper(x)

    caller_new = trace(caller)
    spliced = canonicalize(caller_new, inline=True, callees=[callee_new])

    helper = inline(helper_fn)  # the AST side: parse-time splice
    ast_caller = function_parser(caller)
    c_ast = canonicalize(ast_caller)

    ir.assert_structural_equal(spliced, c_ast, enable_auto_mapping=True)


def test_inline_multi_value_trailing_return_by_both_lanes():
    """**F16.** `return h_multi_value(x)` where the inline callee's trailing
    return is multi-value: the pass splice wires the callee's two trailing
    values into the caller as a *two-value* ``ReturnStmt``, and the parser's
    parse-time expansion of the same call is the written-out body — `return
    a, b` — which is the multi-value ``ReturnStmt`` too. The expansion used to
    collapse the tuple into one ``MakeTuple`` value
    (``_inline_return_expr``'s ``ast.Tuple`` arm), which changed the caller's
    return arity relative to the un-expanded body; §9 (*'neither frontend
    creates a fictional staged-tuple value'*) settles it. Raw: both callers
    return two values, neither a ``MakeTuple``; `_inline_multi_value_anchor`
    then asserts the post-canonical equality on top."""
    helper = Marker(h_multi_value, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return helper(x)

    spliced = _inline_spliced(h_multi_value, caller)
    (new_ret,) = [s for s in _body_stmts(spliced) if isinstance(s, ir.ReturnStmt)]
    assert len(new_ret.value) == 2

    helper = inline(h_multi_value)
    ast_caller = function_parser(caller)
    (ast_ret,) = [s for s in _body_stmts(ast_caller) if isinstance(s, ir.ReturnStmt)]
    assert len(ast_ret.value) == 2
    assert not any(isinstance(v, ir.MakeTuple) for v in ast_ret.value)
    ir.assert_structural_equal(spliced, ast_caller, enable_auto_mapping=True)

    _inline_multi_value_anchor(h_multi_value)


def test_inline_nested_multi_value_trailing_return_by_both_lanes():
    """**F16, through a second inline callee.** The same multi-value trailing
    return reaches the root caller through an inline callee whose whole body is
    `return h_inner(x)`: both lanes splice to a fixpoint, so the root caller
    keeps the innermost trailing return's arity instead of gaining a tuple
    value on the way. The parser side is the inline-mode `parse_return` taking
    the nested expansion's value list (`_inline_return_values`), so a
    `parse_expression`-only expansion of the nested call would be rejected
    here."""

    def h_inner(
        x: pl.Tensor[[64], pl.FP32],
    ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        a = pl.add(x, 1.0)
        b = pl.mul(x, 2.0)
        return a, b

    def h_outer(
        x: pl.Tensor[[64], pl.FP32],
    ) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return h_inner(x)

    inner_fn, outer_fn = h_inner, h_outer
    # The rebind resolves the two callee cells per lane: the Marker for the
    # trace below, the InlineFunction for the AST parse (same trick as
    # ``_inline_anchor``, one level deeper).
    h_inner = Marker(inner_fn, function_type=ir.FunctionType.Inline)
    h_outer = Marker(outer_fn, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return h_outer(x)

    caller_new = trace(caller)
    inner_new = function_trace(inner_fn, function_type=ir.FunctionType.Inline)
    outer_new = function_trace(outer_fn, function_type=ir.FunctionType.Inline)
    prog = ir.Program([caller_new, inner_new, outer_new], "p", ir.Span.unknown())
    spliced = passes.inline_functions()(prog)["caller"]
    assert spliced is not None

    h_inner = inline(inner_fn)
    h_outer = inline(outer_fn)
    ast_caller = function_parser(caller)

    ir.assert_structural_equal(spliced, ast_caller, enable_auto_mapping=True)


# ===========================================================================
# The Inline callee's Function type + the pass's contract
# ===========================================================================


def test_inline_callee_function_type():
    """The callee's ir.Function is FunctionType.Inline — the marked
    callee flows Marker → PtoSema.__init__(function_type=Inline) →
    ib.function(type=Inline)."""
    callee = function_trace(h_add_one, function_type=ir.FunctionType.Inline)
    assert callee.func_type == ir.FunctionType.Inline


def test_inline_functions_removes_callee_and_splices():
    """The pass's contract: after splicing, the Inline callee is gone from
    the Program and the caller's Call is replaced by the spliced body."""
    callee = function_trace(h_add_one, function_type=ir.FunctionType.Inline)
    helper = Marker(h_add_one, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return helper(x)

    caller_fn = trace(caller)
    prog = ir.Program([caller_fn, callee], "p", ir.Span.unknown())
    out = passes.inline_functions()(prog)
    assert {f.name for f in out.functions.values()} == {"caller"}
    printed = str(out["caller"])
    assert "h_add_one" not in printed
    assert "tensor.adds" in printed


# ===========================================================================
# Inline→inline cycle (the pass's loud error)
# ===========================================================================


def test_inline_cycle_loud():
    """An Inline→Inline call cycle is the pass's loud ValueError (the pass
    detects cycles in the FunctionType::Inline call graph)."""

    def h2(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return h1(x)  # noqa: F821 - resolved below via the closure rebind

    h2_marker = Marker(h2, function_type=ir.FunctionType.Inline)

    def h1(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return h2(x)

    h2 = h2_marker  # so h1's body resolves the sibling at trace time
    h1_marker = Marker(h1, function_type=ir.FunctionType.Inline)
    h1 = h1_marker

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return h1(x)

    caller_fn = trace(caller)
    callee1 = function_trace(h1_marker.kernel, function_type=ir.FunctionType.Inline)
    callee2 = function_trace(h2_marker.kernel, function_type=ir.FunctionType.Inline)
    prog = ir.Program([caller_fn, callee1, callee2], "p", ir.Span.unknown())
    with pytest.raises(ValueError, match="Cycle detected"):
        passes.inline_functions()(prog)


# ===========================================================================
# The return-shape divergences (R9)
# (F16's multi-value trailing return, F37's tuple temporary and F38's
# dead tail are anchored above; F17's early return is a rejection pair)
# ===========================================================================


def _inline_spliced(helper_fn, caller_fn):
    """The pass-spliced caller (raw), for the divergence pins.

    Unlike ``_inline_anchor`` this keeps the pre-canonical shapes: the
    divergence is a ``ReturnStmt`` arity difference, which the SSA rename
    does not change but the extra machinery would obscure.
    """
    callee_new = function_trace(helper_fn, function_type=ir.FunctionType.Inline)
    caller_new = trace(caller_fn)
    prog = ir.Program([caller_new, callee_new], "p", ir.Span.unknown())
    spliced = passes.inline_functions()(prog)["caller"]
    assert spliced is not None
    return spliced


def h_early_return(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
    if n != 0:
        return pl.add(x, 1.0)  # early return inside staged control flow
    return pl.mul(x, 2.0)


def test_inline_early_return_rejected_by_both_lanes():
    """**F17 fixed — the parser rejects the early return both tracing
    entrances reject.**

    An inline callee with a ``return`` under staged control flow has no splice
    semantics: the backend pass rejects the shape after cloning (``Inline
    function '...' contains a non-trailing ReturnStmt; only a single trailing
    return is supported``, ``CloneInlineBody`` in
    ``inline_functions_pass.cpp``), and the ``@pl.expand`` entrance rejects the
    same body at trace time (``an expansion body must end in one trailing
    return``, ``Sema.return_``).

    The parser's expansion used to capture the nested return's expression into
    ``_inline_return_values`` and then overwrite it with the callee's trailing
    return, leaving the branch **empty** and silently losing its ``add`` (the
    caller always carried the trailing ``mul``). ``_expand_inline_call`` now
    rejects a ``return`` that is not a direct statement of the callee's body
    (``_reject_inline_nested_return``), so the spelling is a rejection pair
    with no accepted form: the pass at splice time, the parser at expansion
    time."""
    helper = Marker(h_early_return, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
        return helper(x, n)

    callee_new = function_trace(h_early_return, function_type=ir.FunctionType.Inline)
    prog = ir.Program([trace(caller), callee_new], "p", ir.Span.unknown())
    with pytest.raises(InternalError, match="non-trailing ReturnStmt"):
        passes.inline_functions()(prog)

    helper = inline(h_early_return)
    with pytest.raises(ParserSyntaxError, match="non-trailing ReturnStmt"):
        function_parser(caller)


def h_dead_tail(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.add(x, 1.0)
    return pl.mul(x, 2.0)  # unreachable after the function-level return


def h_dead_tail_nested(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
    return pl.add(x, 1.0)
    if n != 0:  # unreachable after the function-level return
        return pl.mul(x, 2.0)


def test_inline_dead_tail_after_trailing_return_by_both_lanes():
    """**F38 fixed — the dead second top-level ``return`` is dropped by both
    lanes.**

    A second top-level ``return`` after the inline callee's trailing return is
    the F15 shape on the inline-expansion seam: the trailing return is the host
    exit (``_return_row``'s in-host-mode guard), so the traced callee is one
    ``ReturnStmt`` carrying ``add`` and the pass splices that. The parser's
    ``_expand_inline_call`` used to keep walking the callee's body, and
    ``parse_return`` overwrites ``_inline_return_values`` on every return, so
    the caller carried the dead ``return pl.mul(...)`` instead. The inline body
    walk now applies the same stop-after-return rule as ``parse_function``
    (F15's ``stop_after_return``): the first top-level return ends the walk, so
    the caller carries the reached ``add`` on both lanes. Raw: neither lane's
    caller has the dead return's op, and the spliced caller equals the parsed
    one."""
    helper = Marker(h_dead_tail, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return helper(x)

    callee_new = function_trace(h_dead_tail, function_type=ir.FunctionType.Inline)
    assert [type(s) for s in _body_stmts(callee_new)] == [ir.ReturnStmt]
    prog = ir.Program([trace(caller), callee_new], "p", ir.Span.unknown())
    spliced = passes.inline_functions()(prog)["caller"]
    (new_ret,) = [s for s in _body_stmts(spliced) if isinstance(s, ir.ReturnStmt)]
    assert new_ret.value[0].op.name == _TENSOR_ADDS

    helper = inline(h_dead_tail)
    ast_caller = function_parser(caller)
    (ast_ret,) = [s for s in _body_stmts(ast_caller) if isinstance(s, ir.ReturnStmt)]
    assert ast_ret.value[0].op.name == _TENSOR_ADDS
    ir.assert_structural_equal(spliced, ast_caller, enable_auto_mapping=True)


def test_inline_dead_tail_nested_return_by_both_lanes():
    """**F38, the dead *nested* return** — the same rule scopes the
    non-trailing-return rejection.

    A dead tail can hold a nested ``return`` too. F17's rejection scan used to
    cover the whole inline body, so it rejected such a tail the traced lane
    never executes (the ``if`` sits after the body's direct return), while the
    traced callee is one ``ReturnStmt`` carrying ``add`` and the pass splices
    that. The scan now covers the collected prefix (``_inline_body_prefix``),
    so the rejection stays on the shape the pass can actually see and both
    lanes inline the reached ``add``. A live nested return keeps being rejected
    (``test_inline_early_return_rejected_by_both_lanes``)."""
    helper = Marker(h_dead_tail_nested, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64], pl.FP32]:
        return helper(x, n)

    callee_new = function_trace(h_dead_tail_nested, function_type=ir.FunctionType.Inline)
    assert [type(s) for s in _body_stmts(callee_new)] == [ir.ReturnStmt]
    prog = ir.Program([trace(caller), callee_new], "p", ir.Span.unknown())
    spliced = passes.inline_functions()(prog)["caller"]
    (new_ret,) = [s for s in _body_stmts(spliced) if isinstance(s, ir.ReturnStmt)]
    assert new_ret.value[0].op.name == _TENSOR_ADDS

    helper = inline(h_dead_tail_nested)
    ast_caller = function_parser(caller)
    (ast_ret,) = [s for s in _body_stmts(ast_caller) if isinstance(s, ir.ReturnStmt)]
    assert ast_ret.value[0].op.name == _TENSOR_ADDS
    ir.assert_structural_equal(spliced, ast_caller, enable_auto_mapping=True)


def h_temp_tuple_return(
    x: pl.Tensor[[64], pl.FP32],
) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
    tmp = (pl.add(x, 1.0), pl.mul(x, 2.0))
    return tmp


def test_inline_temp_tuple_trailing_return_by_both_lanes():
    """**F37 fixed — the tuple temporary stays one staged tuple value.**

    The same multi-value trailing return as F16, written through a tuple
    temporary (`tmp = (a, b); return tmp`) instead of the tuple literal. The
    parser's expansion binds ``tmp`` to one staged ``MakeTuple`` and returns
    the ``tmp`` Var, so the spliced caller keeps the temporary and a
    single-value return. The tracing lane used to read ``(a, b)`` as a host
    container of facades and flatten the binding at the return, dropping the
    temporary and returning the callee's two values. R4 (the parser is
    normative wherever it accepts valid Python) settles the lane: a tuple
    whose elements are all staged facades *is* the staged tuple value
    ``pl.Tuple(a, b)`` when bound, so `return tmp` is one value on both lanes.
    Raw: both callers bind ``tmp`` to a ``MakeTuple`` and return one Var;
    `_inline_multi_value_anchor` then asserts the post-canonical equality on
    top."""
    helper = Marker(h_temp_tuple_return, function_type=ir.FunctionType.Inline)

    def caller(x: pl.Tensor[[64], pl.FP32]) -> tuple[pl.Tensor[[64], pl.FP32], pl.Tensor[[64], pl.FP32]]:
        return helper(x)

    spliced = _inline_spliced(h_temp_tuple_return, caller)
    spliced_stmts = _body_stmts(spliced)
    (new_ret,) = [s for s in spliced_stmts if isinstance(s, ir.ReturnStmt)]
    assert len(new_ret.value) == 1  # the `tmp` Var, not a flattened pair
    assert isinstance(new_ret.value[0], ir.Var)
    assert any(isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.MakeTuple) for s in spliced_stmts)

    helper = inline(h_temp_tuple_return)
    ast_caller = function_parser(caller)
    ast_stmts = _body_stmts(ast_caller)
    (ast_ret,) = [s for s in ast_stmts if isinstance(s, ir.ReturnStmt)]
    assert len(ast_ret.value) == 1  # one `tmp` value
    assert isinstance(ast_ret.value[0], ir.Var)
    assert any(isinstance(s, ir.AssignStmt) and isinstance(s.value, ir.MakeTuple) for s in ast_stmts)
    ir.assert_structural_equal(spliced, ast_caller, enable_auto_mapping=True)

    _inline_multi_value_anchor(h_temp_tuple_return)
