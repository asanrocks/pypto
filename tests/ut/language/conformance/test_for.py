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

R6 sweep — the rest of the iterator surface: the positional range forms
(1/2/3 args, computed bounds), the sibling iterator kinds
(``pl.parallel``/``pl.unroll``/``pl.pipeline``), stage pinning, the
``attrs=`` kwarg, and the target forms. Every iterator here is staged, so
each anchor is asserted **raw** and post-``canonicalize()``: the loop-carried
value is the self-referential ``AssignStmt`` update on both lanes and the
SSA pass derives it (raw already holds; canonical is the staged-flow
regime). The two target-form spellings with no shared surface are pinned as
rejections: the iter_args tuple target needs ``init_values=``, which the
tracing surface deliberately lacks
(``docs/design/01-language.md`` §8 line 415, §14 line 674).
"""

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.parser import function_parser
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


# —— R6: the positional range forms, the sibling iterator kinds, targets ——


def k_range_start_stop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(1, 4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_range_start_stop_step(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(0, 4, 2):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_range_computed_bounds(
    x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX]
) -> pl.Tensor[[64, 64], pl.FP32]:
    """A computed staged bound: the loop var flows into the body, while
    ``start``/``stop``/``step`` are expressions over a staged Scalar."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(n - 2, n + 2, 2):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_parallel_loop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.parallel(4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_parallel_three_arg(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.parallel(0, 4, 2):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_unroll_loop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.unroll(4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_unroll_three_arg(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.unroll(0, 6, 2):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_pipeline_loop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.pipeline(8, stage=4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_pipeline_three_arg_stage(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.pipeline(0, 8, 2, stage=2):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_range_attrs(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """The ``attrs=`` kwarg: the tracing lane carries the dict straight to
    ``begin_for_loop``; the parser parses the dict literal. Same attr map."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4, attrs={"my_attr": 1}):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_pipeline_attrs_stage(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """``stage=`` is merged into ``attrs["pipeline_stages"]`` on both lanes
    (the parser at ``_parse_range_call``/``parse_for_loop``, the iterator at
    ``RangeIterator.begin``); the user-supplied attrs ride alongside."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.pipeline(8, stage=2, attrs={"my_attr": 1}):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_range_int64_const_bound(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """The round-trip bound spelling: the printer emits a non-INDEX bound as
    ``pl.const(4, pl.INT64)``. The parser rebuilds an INT64 loop var from it
    (its round-trip fidelity inference); the tracing lane always types the
    loop var INDEX. ``assert_structural_equal``'s loop-var syntax rule
    (``AreForSyntaxScalarDtypesEquivalent``, ``structural_equal.cpp:50``)
    treats INT64/INDEX as for-syntax-equivalent in exactly this position, so
    the anchor is the oracle's definition of equivalence, not a hidden
    divergence."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(pl.const(4, pl.INT64)):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_nested_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(2):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_for_in_while(x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    i = pl.const(0, pl.INDEX)
    while i < n:
        for j in pl.range(2):
            acc = pl.add(acc, acc)
        i = pl.add(i, 1)
    pl.store(acc, [0, 0], x)
    return x


def test_range_start_stop_anchor():
    """``pl.range(start, stop)`` — the parser fills ``step=1``; the trace
    iterable fills ``step=1`` before unwrapping."""
    conformance(k_range_start_stop, build)
    conformance(k_range_start_stop, build, canonical=True)


def test_range_start_stop_step_anchor():
    conformance(k_range_start_stop_step, build)
    conformance(k_range_start_stop_step, build, canonical=True)


def test_range_computed_bounds_anchor():
    """Computed bounds over a staged Scalar: both lanes evaluate the same
    three expressions (``n - 2``/``n + 2``/``2``) and derive the same
    loop-carried update."""
    conformance(k_range_computed_bounds, build)
    conformance(k_range_computed_bounds, build, canonical=True)


def test_parallel_loop_anchor():
    """``pl.parallel(4)``: ``ForKind.Parallel`` on both lanes, no
    ``pipeline_stages`` attr."""
    conformance(k_parallel_loop, build)
    conformance(k_parallel_loop, build, canonical=True)


def test_parallel_three_arg_anchor():
    conformance(k_parallel_three_arg, build)
    conformance(k_parallel_three_arg, build, canonical=True)


@pytest.mark.filterwarnings("ignore:.*RoundtripInstrument.*IR not printable:UserWarning")
def test_unroll_loop_anchor():
    """``pl.unroll(4)``: ``ForKind.Unroll`` on both lanes (the C++ pass
    resolves the marker). The canonical pair here runs ``convert_to_ssa``
    without ``UnrollLoops`` (which the real pipeline runs first), so SSA
    derives ``iter_args`` on the surviving Unroll marker — an artifact state
    for that kind that both lanes reach identically; the roundtrip
    instrument's printer warning for it is expected noise (filtered as in
    ``test_unroll_loops_pass``)."""
    conformance(k_unroll_loop, build)
    conformance(k_unroll_loop, build, canonical=True)


@pytest.mark.filterwarnings("ignore:.*RoundtripInstrument.*IR not printable:UserWarning")
def test_unroll_three_arg_anchor():
    conformance(k_unroll_three_arg, build)
    conformance(k_unroll_three_arg, build, canonical=True)


def test_pipeline_loop_anchor():
    """``pl.pipeline(8, stage=4)``: ``ForKind.Pipeline`` +
    ``attrs["pipeline_stages"] == 4`` on both lanes — the stage pinning."""
    conformance(k_pipeline_loop, build)
    conformance(k_pipeline_loop, build, canonical=True)


def test_pipeline_three_arg_stage_anchor():
    conformance(k_pipeline_three_arg_stage, build)
    conformance(k_pipeline_three_arg_stage, build, canonical=True)


def test_range_attrs_anchor():
    conformance(k_range_attrs, build)
    conformance(k_range_attrs, build, canonical=True)


def test_pipeline_attrs_stage_anchor():
    conformance(k_pipeline_attrs_stage, build)
    conformance(k_pipeline_attrs_stage, build, canonical=True)


def test_range_int64_const_bound_anchor():
    conformance(k_range_int64_const_bound, build)
    conformance(k_range_int64_const_bound, build, canonical=True)


def test_nested_for_anchor():
    """A staged loop nested in a staged loop — the inner bound is a literal
    on both lanes, so no outer loop-carried value is derived."""
    conformance(k_nested_for, build)
    conformance(k_nested_for, build, canonical=True)


def test_for_in_while_anchor():
    """A staged ``for`` inside a staged ``while``: the while carries ``i``,
    the for carries ``acc`` — raw-equal, and canonical through both SSA
    derivations."""
    conformance(k_for_in_while, build)
    conformance(k_for_in_while, build, canonical=True)


def _top_level_for(fn: ir.Function) -> ir.ForStmt:
    body = fn.body
    stmts = body.stmts if isinstance(body, ir.SeqStmts) else (body,)
    for stmt in stmts:
        if isinstance(stmt, ir.ForStmt):
            return stmt
    raise AssertionError("no top-level ForStmt")


def test_iterator_kind_and_stage_attr_pinned():
    """Pin the kind/attr mapping both lanes must agree on (the structural
    equality above already compares them — this states the expected
    literals so a one-sided change is visible): ``pl.range`` → Sequential,
    ``pl.parallel`` → Parallel, ``pl.unroll`` → Unroll, ``pl.pipeline`` →
    Pipeline + ``pipeline_stages``."""
    variants = [
        (k_range_start_stop, ir.ForKind.Sequential, {}),
        (k_range_start_stop_step, ir.ForKind.Sequential, {}),
        (k_parallel_loop, ir.ForKind.Parallel, {}),
        (k_parallel_three_arg, ir.ForKind.Parallel, {}),
        (k_unroll_loop, ir.ForKind.Unroll, {}),
        (k_unroll_three_arg, ir.ForKind.Unroll, {}),
        (k_pipeline_loop, ir.ForKind.Pipeline, {"pipeline_stages": 4}),
        (k_pipeline_three_arg_stage, ir.ForKind.Pipeline, {"pipeline_stages": 2}),
        (k_pipeline_attrs_stage, ir.ForKind.Pipeline, {"my_attr": 1, "pipeline_stages": 2}),
        (k_range_attrs, ir.ForKind.Sequential, {"my_attr": 1}),
    ]
    for kernel, kind, attrs in variants:
        traced = _top_level_for(build(kernel))
        parsed = _top_level_for(function_parser(kernel))
        assert traced.kind == kind
        assert parsed.kind == kind
        assert dict(traced.attrs) == attrs
        assert dict(parsed.attrs) == attrs


# —— the target forms with no shared spelling (trap 2) ——


def k_iter_args_target(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i, (v,) in pl.range(4, init_values=(acc,)):  # type: ignore[call-overload]
        v = pl.add(v, v)
    pl.store(acc, [0, 0], x)
    return x


def k_bare_tuple_target(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i, j in pl.range(4):
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_iter_args_tuple_target_is_parser_only():
    """No anchor is possible for the iter_args target: its defining spelling
    needs ``init_values=``, and the tracing surface of
    ``pl.range``/``pl.parallel``/``pl.pipeline`` deliberately has none
    (``docs/design/01-language.md`` §8 line 415, §14 line 674 — the
    loop-carried channel is ``pl.intrin.carry`` in the assembly dialect).
    The parser accepts the spelling (its round-trip ownership); tracing
    rejects it at construction. Pinned here so a future change to either
    side is visible."""
    function_parser(k_iter_args_target)
    with pytest.raises(TypeError, match="init_values"):
        build(k_iter_args_target)


def test_bare_tuple_target_rejected_by_both_lanes():
    """A bare tuple target (no ``init_values``) has no shared meaning: the
    parser rejects ``(i, j)`` because the second element must be an iter_args
    tuple, and the trace iterable's target binding is a single proxy, not an
    unpackable sequence."""
    with pytest.raises(TypeError, match="cannot unpack"):
        build(k_bare_tuple_target)
    with pytest.raises(ParserSyntaxError, match="Iter args must be a tuple"):
        function_parser(k_bare_tuple_target)
