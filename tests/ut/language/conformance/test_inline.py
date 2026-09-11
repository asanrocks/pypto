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
"""

import pypto.language as pl
import pytest
from pypto.language.op.marker import Marker
from pypto.language.parser import function_parser, inline
from pypto.language.tracer.decorators.function import function_trace, trace
from pypto.pypto_core import ir, passes

from ._harness import canonicalize


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
