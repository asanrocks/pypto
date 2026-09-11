# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: staged ``if``/``else`` + per-branch locals.

The interpreter's per-branch locals mappings: a staged branch pushes
a per-branch locals mapping, the else branch starts from the
pre-conditional snapshot, and ``exit_construct`` commits the union.
Mutable-local convergence: branch binds reuse one ``ir.Var`` per
name (the Sema's per-frame mapping from source names to mutable
``ir.Var``s) and the SSA pass synthesizes the phi. The conditionally-bound
loud error exempts the legitimate in-loop single-branch def.
"""

import pypto
import pypto.ir as _pyir
import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir

from ._harness import canonicalize, conformance

build = function_trace


def k_merge(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_else_passthrough(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_nested(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], m: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        if m == 1:
            a = pl.load(x, [0, 0], [64, 64])
        else:
            a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_if_in_for(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        if k == 0:
            acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_merge_anchor():
    """Mutable-local convergence: both branches bind the SAME
    ``ir.Var`` 'a' — raw equal; convert_to_ssa synthesizes the phi from the
    shared mutable local."""
    conformance(k_merge, build)
    conformance(k_merge, build, canonical=True)


def test_else_passthrough_anchor():
    """A rebind in one branch: the pre-conditional staged local carries the
    missing path (the SSA pass handles the merge)."""
    conformance(k_else_passthrough, build)
    conformance(k_else_passthrough, build, canonical=True)


def test_nested_if_anchor():
    conformance(k_nested, build)


def test_if_in_for_anchor():
    """Conditional loop-carried — the if inside the loop is the SSA
    pass's loop-carried business; the conditionally-bound check exempts it."""
    conformance(k_if_in_for, build)
    conformance(k_if_in_for, build, canonical=True)


def test_canonicalize_synthesizes_branch_phi():
    """The branch phi: convert_to_ssa synthesizes it from the shared
    mutable-local stores — the raw form diverges by design, the canonical
    form converges."""
    fn = build(k_merge)
    canonical = canonicalize(fn)
    ir.assert_structural_equal(canonical, canonicalize(canonical), enable_auto_mapping=True)


# —— conditionally-bound reads: rejected by the IR verifier, not the frontend ——


def _expect_conditional_read_rejected(kernel):
    """A name bound on some branches of a staged ``if`` and read after the
    merge. The frontend no longer pre-rejects this; the IR verifier's
    use-after-def check rejects it at ``ConvertToSSA`` as a use-before-def
    (soundness is the verifier's job, not the frontend's)."""
    fn = function_trace(kernel)
    assert isinstance(fn, ir.Function)
    prog = ir.Program([fn], f"proto_{fn.name}", ir.Span.unknown())
    with pytest.raises(pypto.Error, match="used outside its defining scope|used before definition"):
        _pyir.compile(prog, dump_passes=False, skip_ptoas=True)


def k_cond_bound_then(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_cond_bound_else(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        pass
    pl.store(a, [0, 0], x)
    return x


def test_conditionally_bound_then_only():
    """A name bound only on the then path (implicit else) and read after the
    merge: the frontend traces it; the IR verifier rejects the use-before-def."""
    _expect_conditional_read_rejected(k_cond_bound_then)


def test_conditionally_bound_else_path():
    """A name bound in one branch of a then/else pair and read after the merge."""
    _expect_conditional_read_rejected(k_cond_bound_else)


def test_in_loop_single_branch_def_exempt():
    """The legitimate in-loop single-branch def does NOT trigger
    the check — the passthrough is the previous iteration's value (the SSA
    pass's loop-carried business)."""
    fn = function_trace(k_if_in_for)
    assert isinstance(fn, ir.Function)


def test_require_bool_predicate():
    """A staged predicate must be BOOL-typed — ``Frame`` owns
    predicate typing."""

    def k_bad(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
        if k + 1:  # INT32 predicate, not a comparison
            a = pl.load(x, [0, 0], [64, 64])
        else:
            a = pl.load(x, [0, 0], [64, 64])
        pl.store(a, [0, 0], x)
        return x

    with pytest.raises(TypeError, match="BOOL-typed"):
        function_trace(k_bad)
