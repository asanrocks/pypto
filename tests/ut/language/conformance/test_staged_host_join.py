# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Typed staged initialization: staged-ness stability at joins.

A host bind emits no IR, so a name whose staged local must be re-read after
a control-flow join (if merge, loop back-edge) silently loses the staged
value when a host literal replaces it. The tracer enforces both directions
of a staged-ness flip under active staged control flow:

- host prior + staged rebind (the original §4 guard), and
- staged prior + host rebind (demotion), plus a cross-branch disagreement
  check at the if merge — the then mapping is swapped out at ``enter_else``,
  so the bind-time guards cannot see the other branch.

Top-level sequential flips stay legal: with no join in sight, host values
follow host rules and each read sees its sequential Python value.
"""

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir


def _expect_typed_staged_init_error(kernel) -> None:
    with pytest.raises(TypeError, match="typed staged initialization"):
        function_trace(kernel)


# —— cross-branch disagreement: the if merge rejects staged-vs-host ——


def k_staged_then_host(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    visible = pl.min(n, 8)
    if visible > 4:
        valid_len = pl.min(8, visible - 4)  # staged
    else:
        valid_len = 0  # host
    return pl.fillpad(pl.set_validshape(a, 1, valid_len), pad_value=pl.PadValue.min)


def k_host_then_staged(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    visible = pl.min(n, 8)
    if visible > 4:
        valid_len = 0  # host
    else:
        valid_len = pl.min(8, visible - 4)  # staged
    return pl.fillpad(pl.set_validshape(a, 1, valid_len), pad_value=pl.PadValue.min)


def k_mixed_join_then_rebind(
    a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]
) -> pl.Tensor[[4, 8], pl.FP32]:
    if n > 4:
        x = pl.const(2, pl.INDEX)
    else:
        x = 1
    x = pl.add(x, 1)  # a staged rebind cannot rescue a poisoned join
    return pl.fillpad(pl.set_validshape(a, 1, x), pad_value=pl.PadValue.min)


def test_if_join_staged_then_host_branch():
    """The reported phi-bug shape: staged in then, host literal in else.

    Before the merge check, the else-side host binding won the unchecked
    union merge and the join compiled with a constant `0`, dropping the
    then-branch computation entirely.
    """
    _expect_typed_staged_init_error(k_staged_then_host)


def test_if_join_host_then_staged_branch():
    """The mirrored ordering: previously accepted at trace time and only
    rejected late by the SSA verifier's use-before-def check."""
    _expect_typed_staged_init_error(k_host_then_staged)


def test_mixed_join_then_staged_rebind():
    """A staged rebind after the join closes cannot launder the poisoned
    merge — the join itself is rejected first."""
    _expect_typed_staged_init_error(k_mixed_join_then_rebind)


# —— staged-ness demotion under active staged control flow ——


def k_demote_in_branch(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    x = pl.const(2, pl.INDEX)
    if n > 4:
        x = 1  # demotion: post-join reads would inline 1 regardless of n
    return pl.fillpad(pl.set_validshape(a, 1, x), pad_value=pl.PadValue.min)


def k_demote_in_for(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    acc = pl.const(0, pl.INDEX)
    for i in pl.range(n):
        acc = pl.add(acc, 1)
        acc = 5  # demotion kills the loop-carried accumulator
    return pl.fillpad(pl.set_validshape(a, 1, acc), pad_value=pl.PadValue.min)


def k_demote_in_while(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    pos = pl.const(0, pl.INDEX)
    while pos < n:
        pos = pl.add(pos, 2)
        pos = 7  # demotion: post-loop reads would inline 7
    return pl.fillpad(pl.set_validshape(a, 1, pos), pad_value=pl.PadValue.min)


def k_demote_in_nested_if(
    a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32], m: pl.Scalar[pl.INT32]
) -> pl.Tensor[[4, 8], pl.FP32]:
    x = pl.const(2, pl.INDEX)
    if n > 4:
        if m > 4:
            x = 1  # demotion deep inside poisons the outer join
        else:
            x = pl.add(x, 1)
    return pl.fillpad(pl.set_validshape(a, 1, x), pad_value=pl.PadValue.min)


def test_demote_staged_name_in_if_branch():
    _expect_typed_staged_init_error(k_demote_in_branch)


def test_demote_staged_name_in_for_body():
    """§4's acc_loop_bad failure mode via rebind instead of initialization:
    the accumulator's AssignStmts become dead and every iteration reads the
    inlined literal."""
    _expect_typed_staged_init_error(k_demote_in_for)


def test_demote_staged_name_in_while_body():
    _expect_typed_staged_init_error(k_demote_in_while)


def test_demote_staged_name_in_nested_if():
    _expect_typed_staged_init_error(k_demote_in_nested_if)


# —— legal programs: sequential flips and per-branch host scratch ——


def k_toplevel_host_then_staged(a: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
    x = 1  # host bind, dead from the IR's point of view
    x = pl.const(2, pl.INDEX)
    return pl.fillpad(pl.set_validshape(a, 1, x), pad_value=pl.PadValue.min)


def k_toplevel_staged_then_host(a: pl.Tensor[[4, 8], pl.FP32]) -> pl.Tensor[[4, 8], pl.FP32]:
    x = pl.const(2, pl.INDEX)
    x = 1  # sequential Python semantics: later reads see 1
    return pl.fillpad(pl.set_validshape(a, 1, x), pad_value=pl.PadValue.min)


def k_branch_host_scratch(
    a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]
) -> pl.Tensor[[4, 8], pl.FP32]:
    # Host scratch per branch (shape computation) is legal as long as the
    # name is not read after the join: host values cannot depend on a
    # staged predicate, so only same-branch reads are sound.
    if n > 4:
        cols = 4
        _ = pl.load(a, [0, 0], [1, cols])  # emits the phi-unsafe read; value unused
    else:
        cols = 8
        _ = pl.load(a, [0, 0], [1, cols])
    return pl.fillpad(pl.set_validshape(a, 1, 1), pad_value=pl.PadValue.min)


def k_phi_fixed(a: pl.Tensor[[4, 8], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[4, 8], pl.FP32]:
    visible = pl.min(n, 8)
    if visible > 4:
        valid_len = pl.min(8, visible - 4)
    else:
        valid_len = pl.const(0, pl.INDEX)  # typed staged on every path
    return pl.fillpad(pl.set_validshape(a, 1, valid_len), pad_value=pl.PadValue.min)


def test_toplevel_flips_stay_legal():
    """No join in sight: host values follow host rules and each read sees
    its sequential Python value. The guards are join-scoped, not global."""
    assert isinstance(function_trace(k_toplevel_host_then_staged), ir.Function)
    assert isinstance(function_trace(k_toplevel_staged_then_host), ir.Function)


def test_branch_host_scratch_stays_legal():
    assert isinstance(function_trace(k_branch_host_scratch), ir.Function)


def test_fixed_phi_join_converges():
    """The documented fix: a typed constant on the host side makes the name
    staged on every path; the join threads the converged variable instead
    of inlining a literal."""
    fn = function_trace(k_phi_fixed)
    assert isinstance(fn, ir.Function)
    assert "set_validshape(a, 1, valid_len)" in str(fn), str(fn)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
