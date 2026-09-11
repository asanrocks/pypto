# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Remaining structures (`while`, `and`/`or`/`not`) + hardening.

The value algebra: ``logical_and/or/not`` build the first-class
``ir.And``/``Or``/``Not`` nodes (no re-lowering); the RHS is a
single-expr thunk driven by ``Frame``'s short-circuit resolvers. ``while``
makes a lazy staging decision on the hoisted per-site thunk (evaluate
True once, then False to exit). Hardening: the typed staged initialization
merge guardrail (a bare literal at a staged join errors loudly,
interpreter-side over the instrumented locals) and the auto-unwind
exception path (no half-built C++ builder).
"""

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import DataType, ir

from ._harness import conformance

build = function_trace


def k_while(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_nested_while(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], m: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        while m > 0:
            acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_while_break(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        acc = pl.add(acc, acc)
        break
    pl.store(acc, [0, 0], x)
    return x


def k_and(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0 and k < 3:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_or_not(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if (k == 0 or k > 5) and not (k == 3):
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_if_in_for_stress(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        if k == 0:
            acc = pl.add(acc, acc)
        else:
            acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_while_anchor():
    """The lazy staging decision (True-then-False) on the hoisted per-site
    thunk; the body traces one representative iteration and the lowering
    pass restructures."""
    conformance(k_while, build)
    conformance(k_while, build, canonical=True)


def test_nested_while_anchor():
    conformance(k_nested_while, build)


def test_while_break_anchor():
    conformance(k_while_break, build)


def test_and_anchor():
    """The first-class ir.And, raw-equal to the parser frontend."""
    conformance(k_and, build)


def test_or_not_anchor():
    conformance(k_or_not, build)


def test_if_in_for_stress_anchor():
    """The joint stress test: conditional loop-carried with both
    branches — convert_to_ssa's loop-carried business."""
    conformance(k_if_in_for_stress, build)
    conformance(k_if_in_for_stress, build, canonical=True)


# —— the value algebra (direct API) ——


def test_logical_nodes_first_class():
    """logical_and/or/not yield the first-class ir.And/Or/Not nodes."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    a = pl.Scalar(expr=ir.Var("a", ir.ScalarType(DataType.BOOL), ir.Span.unknown()))
    b = pl.Scalar(expr=ir.Var("b", ir.ScalarType(DataType.BOOL), ir.Span.unknown()))
    assert isinstance(frame.logical_and(a, b).unwrap(), ir.And)
    assert isinstance(frame.logical_or(a, b).unwrap(), ir.Or)
    assert isinstance(frame.logical_not(a).unwrap(), ir.Not)


def k_and_host_rhs(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0 and True:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_or_host_rhs(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0 or 1:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_and_non_bool(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], m: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k and m == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_not_non_bool(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if not k:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def test_logical_host_operand_anchor():
    """A host operand at a logical operator is
    lifted to a typed Const at the operand boundary (bool→ConstBool,
    int→ConstInt(INDEX)), matching the parser frontend's ``parse_constant``.
    The left operand picks the behavior: ``k == 0`` is staged →
    the operator emits IR; the host RHS is normalized, so ``k == 0 and True``
    and ``k == 0 or 1`` build the first-class And/Or node, raw-equal to the
    parser frontend (a host RHS is never staged)."""
    conformance(k_and_host_rhs, build)
    conformance(k_or_host_rhs, build)


def test_logical_host_operand_normalized():
    """Direct API: ``Frame`` lifts a host RHS
    of a logical operator — bool→ConstBool, int→ConstInt(INDEX),
    float→ConstFloat — so a staged BOOL left + host right yields the
    first-class node with the Const operand (never an implicit retype)."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    a = pl.Scalar(expr=ir.Var("a", ir.ScalarType(DataType.BOOL), ir.Span.unknown()))
    node = frame.logical_and(a, True).unwrap()
    assert isinstance(node, ir.And)
    assert isinstance(node.right, ir.ConstBool) and node.right.value is True
    node = frame.logical_or(a, 1).unwrap()
    assert isinstance(node, ir.Or)
    assert isinstance(node.right, ir.ConstInt) and node.right.value == 1
    node = frame.logical_or(a, 1.5).unwrap()
    assert isinstance(node, ir.Or)
    assert isinstance(node.right, ir.ConstFloat) and node.right.value == 1.5


def test_logical_non_bool_scalar_loud_error():
    """A non-bool Scalar coerced to bool is a verifier error — host
    truthiness must not flow into staged values; write the explicit test
    (x != 0)."""
    for kernel in (k_and_non_bool, k_not_non_bool):
        with pytest.raises(TypeError, match="verifier error"):
            function_trace(kernel)


# —— hardening ——


def k_typed_staged_init(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = 0  # host bind — no staged local
    for i in pl.range(4):
        # staged rebind needs a typed staged local → loud error
        acc = pl.add(  # pyright: ignore[reportCallIssue, reportArgumentType]
            acc, acc
        )
    pl.store(acc, [0, 0], x)  # pyright: ignore[reportArgumentType]
    return x


def test_typed_staged_init_merge_guardrail():
    """A bare literal at a staged join errors loudly — the
    host bind emitted no staged local, so the loop-carried derivation would
    silently inline the literal."""
    with pytest.raises(TypeError, match="typed staged initialization"):
        function_trace(k_typed_staged_init)


def k_auto_unwind(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        raise RuntimeError("boom")
    pl.store(acc, [0, 0], x)
    return x


def test_auto_unwind_exception_path():
    """An exception mid-body LIFO-closes every open builder context
    (no half-built C++ builder / segfault) and surfaces the original error."""
    with pytest.raises(RuntimeError, match="boom"):
        function_trace(k_auto_unwind)


def test_auto_unwind_mid_loop():
    """The exception path through an open loop scope."""

    def k_boom_loop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        acc = pl.load(x, [0, 0], [64, 64])
        for i in pl.range(4):
            acc = pl.add(acc, acc)
            raise RuntimeError("loop boom")
        pl.store(acc, [0, 0], x)
        return x

    with pytest.raises(RuntimeError, match="loop boom"):
        function_trace(k_boom_loop)


# —— staged return inside staged control flow: per-branch ReturnStmt — the
#    value joins across paths via the SSA pass;
#    normalize_return_order + DCE absorb the liveness ——


def k_ret_in_if(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        return x  # staged return inside a staged if — per-branch ReturnStmt
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_ret_in_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.range(4):
        return x  # staged return inside a staged for — per-branch ReturnStmt
    return x


def k_ret_in_while(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    while k < 3:
        return x  # staged return inside a staged while — per-branch ReturnStmt
    return x


def test_staged_return_in_if_ok():
    """A staged return inside staged control flow
    is recorded in-branch as a per-branch ReturnStmt (matching the parser);
    the SSA pass derives the join, normalize_return_order + DCE absorb
    liveness. The interpreter's ``in_host_mode`` guard suppresses the host
    return while a staged construct is open, so the host continues past the
    return and the dead tail is collected faithfully."""
    conformance(k_ret_in_if, build)


def test_staged_return_in_for_ok():
    """As ``test_staged_return_in_if_ok`` but inside a staged for."""
    conformance(k_ret_in_for, build)


def test_staged_return_in_while_ok():
    """As ``test_staged_return_in_if_ok`` but inside a staged while."""
    conformance(k_ret_in_while, build)


def k_top_return(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x  # top-level staged return — OK (no construct open)


def test_top_level_staged_return_ok():
    """A top-level staged return (no staged control flow open) is the
    supported case — the host ``return`` fires (``in_host_mode`` is True)
    and ``Frame`` emits the ``ReturnStmt`` at function level."""
    conformance(k_top_return, build)


def k_ret_in_at_region(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
        return x  # a pl.at region is not control flow — top-level return
    return x


def test_staged_return_in_at_region_ok():
    """A ``pl.at`` region is not control flow — a return inside it
    is the top-level return (the loud error is staged-control-flow
    only)."""
    conformance(k_ret_in_at_region, build)


def k_ret_in_at_inside_if(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        with pl.at(level=pl.Level.CORE_GROUP):
            return x  # a region inside staged control flow — per-branch ReturnStmt
    return x


def k_ret_in_at_inside_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.range(4):
        with pl.at(level=pl.Level.CORE_GROUP):
            return x  # a region inside staged control flow — per-branch ReturnStmt
    return x


def test_staged_return_in_region_inside_cf_ok():
    """A staged return inside a ``pl.at``
    region nested in staged control flow is recorded in-branch (the region
    is not control flow, but the enclosing ``if``/``for`` is — the
    per-branch ReturnStmt lands inside the ScopeStmt inside the
    IfStmt/ForStmt, matching the parser)."""
    for kernel in (k_ret_in_at_inside_if, k_ret_in_at_inside_for):
        conformance(kernel, build)
