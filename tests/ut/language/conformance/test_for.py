# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: staged ``for`` loops.

The tracing frontend emits **no eager iter-args**: the loop-carried value
is the self-referential ``AssignStmt`` update to a mutable local, and the
SSA pass derives it. Anchors assert raw equality for the loop shape; the canonical
stability probe pins the post-canonical regime. Host ranges
unroll natively (no ForStmt; the parser frontend rejects host ranges, so
this is a behavior test, not an anchor).
"""

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import ir

from ._harness import canonicalize, conformance


class PtoSemaProbe(PtoSema):
    """Minimal PtoSema alias for the direct-API tests."""


build = function_trace


def k_loop_accum(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_loop_staged_bound(
    x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(n):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_loop_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
        break
    pl.store(acc, [0, 0], x)
    return x


def k_loop_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
        continue
    pl.store(acc, [0, 0], x)
    return x


def k_gemm_tile_loop(
    x: pl.Tensor[[64, 128], pl.FP16], w: pl.Tensor[[128, 64], pl.FP16]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        acc = pl.matmul(pl.load(x, [0, 0], [64, 64]), pl.load(w, [0, 0], [64, 64]))
        for i in pl.range(2):
            t = pl.load(x, [0, i * 64], [64, 64])
            pl.store(t, [0, i * 64], x)
        pl.store(acc, [0, 0], x)
    return x


def test_loop_accum_anchor():
    """The loop-carried accumulator: self-referential ``AssignStmt`` update,
    no iter-args — raw-equal to the parser frontend's simple for."""
    conformance(k_loop_accum, build)


def test_loop_staged_bound_anchor():
    conformance(k_loop_staged_bound, build)


def test_loop_break_anchor():
    """A staged break records a BreakStmt (the lowering pass
    restructures); the raw shape matches the parser frontend's break_stmt."""
    conformance(k_loop_break, build)


def test_loop_continue_anchor():
    conformance(k_loop_continue, build)


def test_gemm_tile_loop_anchor():
    """A real loop kernel: the gemm with a loop over tiles, inside a pl.at
    scope — the loop var flows into tile offsets (i * 64)."""
    conformance(k_gemm_tile_loop, build)


def test_loop_canonical_stability():
    """The post-canonical regime: both frontends' loop IR survive the
    pass pipeline and stay equal after canonicalize (SSA derives the
    loop-carried value from the ``AssignStmt`` update)."""
    conformance(k_loop_accum, build, canonical=True)
    conformance(k_gemm_tile_loop, build, canonical=True)


def test_canonicalize_derives_loop_carried():
    """The SSA pass derives the loop-carried value — the
    canonical form has a loop-carried iter_arg the raw form lacks."""
    fn = build(k_loop_accum)
    canonical = canonicalize(fn)
    with pytest.raises(ValueError):
        ir.assert_structural_equal(fn, canonical, enable_auto_mapping=True)
    ir.assert_structural_equal(canonical, canonicalize(canonical), enable_auto_mapping=True)


# —— the loop-target binding-proxy surface ——


def test_loop_var_cell_registered_without_let():
    """The loop-target binding-proxy bind registers the loop var's
    ``ir.Var`` directly (no ``LetStmt``) — the
    loop var IS that ``ir.Var``, matching the parser frontend's define_var."""
    fn = function_trace(k_loop_accum)
    # the loop var 'i' appears in the ForStmt's loop_var; no let for 'i'
    printed = str(fn)
    assert "for i in pl.range(4):" in printed
    assert isinstance(fn.body, ir.SeqStmts)  # sanity
    assert list(fn.body)  # sanity


def k_host_range(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in range(3):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_host_range_unrolls():
    """C6: a host range unrolls natively — the body runs N×, no
    ForStmt is emitted, and the loop var is a host value (no staged local)."""
    fn = build(k_host_range)
    printed = str(fn)
    assert "pl.range" not in printed
    # the unrolled body ran 3× (three adds)
    assert printed.count("pl.tile.add") == 3


def test_open_for_rejects_non_trace_iterable():
    """A pl.const(5) reaching create_for_construct is a loud 'not iterable'."""
    sema = PtoSemaProbe(name="probe")
    frame = sema.begin_frame()
    try:
        frame.create_for_construct(None, pl.const(5, pl.INDEX))
        raise AssertionError("expected TypeError")
    except TypeError as e:
        assert "not iterable" in str(e)
    frame.finish()
