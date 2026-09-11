# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.program`` (static, two-pass).

The two-pass envelope: Pass 1 collects the marked methods (class-body
order) into the gvar/Marker registry; Pass 2 traces each with the
``self``-proxy in scope (host-param passthrough), in source order (no
topo-order; the Markers are self-sufficient). Post-trace: the
declared-return-type check and the recursion guard.

Anchors: ``program_conformance`` vs ``pl.program_parser`` (raw + canonical —
roundtrip-stable in programs, self.method resolves via global_vars). The
forward-ref program is a *targeted* anchor: the tracing frontend's
entry-before-helper call emits the helper's *declared* return type while
the AST's carries UnknownType — the documented
improvement, asserted directly rather than via structural equality.
"""

import pypto.language as pl
import pytest
from pypto.language.parser import program_parser
from pypto.language.tracer.decorators.function import trace_program
from pypto.pypto_core import DataType, ir

from ._harness import program_conformance


class SoftmaxProgram:
    @pl.function
    def scale_tile(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        t = pl.load(x, [0, 0], [64, 64])
        s = pl.mul(t, 2.0)
        pl.store(s, [0, 0], x)
        return x

    @pl.function
    def entry(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.scale_tile(x)


class ForwardRefProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return self.scale_tile(x)

    @pl.function
    def scale_tile(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return pl.add(x, 1.0)


# ===========================================================================
# The two-method program anchor
# ===========================================================================


def test_program_anchor_raw():
    """``trace_program(cls)`` == ``program_parser(cls)`` raw: the two-pass
    envelope traces every marked method; the cross-function call (via the
    self-proxy's sibling Marker) matches the AST's ``_parse_kernel_call``."""
    program_conformance(SoftmaxProgram, trace_program)


def test_program_anchor_canonical():
    """The post-canonical regime over the whole Program — roundtrip-stable
    because self.method resolves via the Program's global_vars (unlike the
    standalone external call)."""
    program_conformance(SoftmaxProgram, trace_program, canonical=True)


def test_program_function_set_matches_ast():
    """All marked methods are traced — the function set matches the AST
    frontend's collection (class-body order, both methods present)."""
    new = trace_program(SoftmaxProgram)
    ast_ = program_parser(SoftmaxProgram)
    assert {f.name for f in new.functions.values()} == {f.name for f in ast_.functions.values()}
    assert new.name == "SoftmaxProgram"


# ===========================================================================
# The forward-ref program (the declared-type improvement)
# ===========================================================================


def _gvar_calls(fn: ir.Function) -> list[ir.Call]:
    found: list[ir.Call] = []

    class _V(ir.IRVisitor):
        def visit_call(self, op):
            if isinstance(op.op, ir.GlobalVar):
                found.append(op)
            super().visit_call(op)

    _V().visit_function(fn)
    return found


def test_forward_ref_emits_declared_return_type():
    """Entry (defined before helper) is traced in
    class-body order; its call emits with the helper's *declared* return
    type — the Marker is self-sufficient, so the not-yet-traced body
    is never consulted. The AST's forward-ref call carries UnknownType
    — this anchor records the improvement as a targeted
    assertion rather than a structural-equality gate."""
    new = trace_program(ForwardRefProgram)
    ast_ = program_parser(ForwardRefProgram)

    entry_new = new["entry"]
    assert entry_new is not None  # Program.__getitem__ is typed Optional
    (call,) = _gvar_calls(entry_new)
    assert call.op.name == "scale_tile"
    assert isinstance(call.type, ir.TensorType)  # the declared `->`, not UnknownType
    assert call.type.dtype == DataType.FP32

    # the AST's gap, pinned as evidence (not as a target)
    entry_ast = ast_["entry"]
    assert entry_ast is not None
    (ast_call,) = _gvar_calls(entry_ast)
    assert isinstance(ast_call.type, ir.UnknownType)

    # the non-forward function matches the AST exactly
    scale_new = new["scale_tile"]
    scale_ast = ast_["scale_tile"]
    assert scale_new is not None and scale_ast is not None
    ir.assert_structural_equal(scale_new, scale_ast, enable_auto_mapping=True)


# ===========================================================================
# The declared-return-type check
# ===========================================================================


class WrongReturnProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP16]:
        return x  # x is FP32 — the declared FP16 is a lie


def test_declared_return_type_check():
    """The declared `->` is checked against the trace — a mismatch is
    a loud 'the declared return type was a lie'."""
    with pytest.raises(TypeError, match="declared"):
        trace_program(WrongReturnProgram)


class CorrectReturnProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return x


def test_declared_return_type_check_passes():
    """The check does not misfire on a correct declaration."""
    new = trace_program(CorrectReturnProgram)
    assert {f.name for f in new.functions.values()} == {"entry"}


class NoReturnProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # type: ignore[reportReturnType] -- deliberately no return; the Sema check catches it
        t = pl.load(x, [0], [64])
        pl.store(t, [0], x)  # no return — the declared `->` is a lie


def test_declared_return_type_check_no_return():
    """A declared `->` whose trace produced no return value is the
    same lie — loud (the gap pyright surfaces statically, the Sema check
    surfaces at trace time)."""
    with pytest.raises(TypeError, match="declared"):
        trace_program(NoReturnProgram)


# ===========================================================================
# The recursion guard
# ===========================================================================


class RecursiveProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return self.entry(x)


class MutualProgram:
    @pl.function
    def a(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return self.b(x)

    @pl.function
    def b(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return self.a(x)


def test_recursion_guard_self_call():
    """A self-call is a cycle — loud (recursion is forbidden). A
    *check* after the trace (class-body order already handled forward
    refs), never an ordering."""
    with pytest.raises(TypeError, match="cycle"):
        trace_program(RecursiveProgram)


def test_recursion_guard_mutual():
    with pytest.raises(TypeError, match="a -> b -> a"):
        trace_program(MutualProgram)


# ===========================================================================
# The self-proxy: sibling calls, chaining, multiple siblings
# ===========================================================================


class MultiCallProgram:
    @pl.function
    def helper(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return pl.add(x, 1.0)

    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        a = self.helper(x)
        b = self.helper(a)
        return self.helper(b)  # the result of a sibling call feeds the next


def test_multiple_sibling_calls():
    """`self.helper(x)` twice plus a chained call — each emits a detached
    Call from the sibling Marker; the result facade of one call feeds the
    next (attribute-on-result)."""
    new = trace_program(MultiCallProgram)
    ast_ = program_parser(MultiCallProgram)
    ir.assert_structural_equal(new, ast_, enable_auto_mapping=True)


class UnknownSiblingProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return self.nope(x)  # type: ignore[reportAttributeAccessIssue]  # noqa: F821 -- unknown-sibling probe


def test_unknown_sibling_loud():
    """An unknown sibling on the proxy is loud (the parser frontend's
    undefined-program-function discipline, re-skinned)."""
    with pytest.raises(AttributeError, match="nope"):
        trace_program(UnknownSiblingProgram)
