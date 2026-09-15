# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: the SPMD dispatch surface + the explicit AIV-split region (R11).

Both frontends route the SPMD scope through one handler per lane. The parser
classifies the with-form body **statically** in ``_emit_spmd_body``
(``ast_parser.py:4755``: an explicit ``pl.at(CORE_GROUP)`` carrier, else a body
that ``_spmd_body_reads_block_idx`` (``:4578``, an ``ast.walk`` over the whole
suite) reads the per-block index → one ``InCoreScopeStmt`` carrier around the
whole body, else a direct-dispatch body with no wrapper), and builds the
for-form in ``_parse_spmd_for_loop`` (``:5036``: always
``Spmd(body=InCore(loop_var = get_block_idx; body))``). The tracing lane opens
the scope through ``spmd.__trace_enter__`` / ``spmd.begin``
(``constructs/scopes.py:961`` / ``:1130``) and classifies the body the same
static way: the rewriter's ``single_call_body`` (the direct-dispatch shape) and
``reads_block_idx`` (its mirror of ``_spmd_body_reads_block_idx``, an
``ast.walk`` over the whole suite) facts decide the wrap *before* the body's
first statement; ``_maybe_open_spmd_carriers`` (``tracer/semantics.py:1005``)
remains only as the backstop for a read the AST walk cannot see (an aliased
``get_block_idx`` call). ``pl.split_aiv`` is
``constructs/scopes.py:1257`` (bare top-level → synthesized ``InCore``
wrapper, inside an open InCore scope or an InCore-typed function → emitted in
place) against ``_parse_split_aiv_for_loop`` (``ast_parser.py:5145``, arms
keyed on ``_is_inside_scope(InCore)`` / ``func_type == InCore``).

**Regime.** The SPMD scope is a scope, not a ``ForStmt``, so the anchors carry
no iter_arg/phi territory: raw equality is the load-bearing half. Every anchor
here is asserted **raw**, and **raw + post-``canonicalize()``** wherever the
body (or an enclosing construct) carries staged control flow — the SSA pass has
loop-carried values to derive there, and both lanes must reach the same
canonical shape. Both regimes land green for every anchor below.

**The carrier boundary is the same static, body-wide classification on both
lanes (F23, the migration review's F1).** The parser's
``_spmd_body_reads_block_idx`` walks the whole suite, and ``_emit_spmd_body``
wraps on that fact; the rewriter's ``reads_block_idx`` fact (threaded to the
with-construct through ``rt.on_begin_with``) is its tracing-side mirror, so
the carrier opens at the body's first statement whenever the body reads the
index anywhere — not at the first top-level read. The four spellings the
review pinned red are anchors below:

- a statement before the read — it stays inside the carrier;
- a read nested in staged control flow — the carrier opens at the loop;
- a read inside a ``pl.split_aiv`` region, with a statement between regions —
  the region is emitted in place (no per-region wrapper) and the
  between-statement stays inside the one carrier (the fixture shape from
  ``tests/ut/codegen/test_orchestration_returned_param_map.py``, whose
  migrated text had to state the carrier explicitly);
- a body that is a *single* statement reading the index — the read fact wins
  over the rewriter's ``single_call_body`` direct-dispatch shape, so the outer
  ``SpmdScopeStmt`` keeps the *split* hint (``w_spmd``).

``docs/design/04-migration-guide.md`` §6.2 (line 333) states the contract —
*'The with-form `pl.spmd` body — including multi-statement bodies reading
`get_block_idx()` — is wrapped correctly by the tracer'*.

The no-shared-spelling variants are pinned at the foot of the file:
``deps=`` without ``as tid`` (an **intended divergence**, documented in
``docs/en/dev/language/02-manual_dependencies.md:36``), the bare body the
parser rejects (needs-decision), nested ``pl.split_aiv`` (F24), and the
rejection parities both lanes share verbatim. The explicit-carrier spelling — a
body that is a lone nested ``pl.at(level=pl.Level.CORE_GROUP, ...)`` — has one
residual lane difference (filed by the fix campaign's A2, **F36**): the outer
``SpmdScopeStmt.name_hint``, which the parser passes through unsplit and the
tracing lane splits. Both lanes accept the spelling, the statements and the
carrier are identical, and no pass reads the outer hint, so R5
(``docs/fix-construct-parity.md`` §4) classifies the difference as a
**naming-only, accepted divergence** — the witness is
``test_carrier_explicit_at_hint_accepted_drift``, not a fix target. The same
fixture exposes a printer-coverage hole shared by both lanes: the for-form
fall-through drops the **carrier's** hint on a read-first body (**F40**,
``test_explicit_carrier_print_drops_the_carrier_hint``).

The ``k_saiv_in_at_staged_for`` anchor also has a focused printer round-trip
regression: its canonical loop yield is nested inside a ``pl.split_aiv`` region.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir, passes

from ._harness import canonicalize, conformance, program_conformance

_TILE_GET_SUBBLOCK_IDX = ir.get_op("tile.get_subblock_idx").name
_TILE_GET_BLOCK_IDX = ir.get_op("tile.get_block_idx").name

build = function_trace


def _spmd(fn: ir.Function) -> ir.SpmdScopeStmt:
    """The function's first ``SpmdScopeStmt`` (the anchors all hold exactly one)."""
    found = [s for s in fn.body if isinstance(s, ir.SpmdScopeStmt)]
    assert found, "no SpmdScopeStmt in the traced/parsed function"
    return found[0]


def _incore_scopes(fn: ir.Function) -> list[ir.InCoreScopeStmt]:
    """Every ``InCoreScopeStmt`` in the function (nested ones included)."""
    return [s for s in _stmts(fn) if isinstance(s, ir.InCoreScopeStmt)]


def _incore_chain(spmd: ir.SpmdScopeStmt) -> list[ir.InCoreScopeStmt]:
    """The nested ``InCoreScopeStmt`` chain under a ``SpmdScopeStmt``'s body."""
    chain: list[ir.InCoreScopeStmt] = []
    node: ir.Stmt = spmd.body
    while isinstance(node, ir.InCoreScopeStmt):
        chain.append(node)
        node = node.body
    return chain


def _assert_subblock_idx_prologue(region: ir.SplitAivScopeStmt) -> None:
    """The region's first statement is ``loop_var = pl.tile.get_subblock_idx()``."""
    body = region.body
    first = body.stmts[0] if isinstance(body, ir.SeqStmts) else body
    assert isinstance(first, ir.AssignStmt), type(first).__name__
    assert isinstance(first.value, ir.Call) and first.value.op.name == _TILE_GET_SUBBLOCK_IDX


def _stmts(fn: ir.Function) -> list[ir.Stmt]:
    """Flatten a function body into a statement list (SeqStmts included)."""
    out: list[ir.Stmt] = []

    def walk(node: object) -> None:
        if isinstance(node, ir.SeqStmts):
            for s in node.stmts:
                walk(s)
        elif isinstance(node, ir.ScopeStmt):
            out.append(node)
            walk(node.body)

    for s in fn.body:
        walk(s)
    return out


# ===========================================================================
# The for-form: ``for i in pl.spmd(N):``
# ===========================================================================


def k_spmd_for(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4):
        x = pl.add(x, x)
    return x


def k_spmd_for_named(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.spmd(4, name_hint="q"):
        y = pl.add(x, pl.cast(i, pl.FP32))
        x = pl.add(y, 1.0)
    return x


def k_spmd_for_sync(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, sync_start=True):
        x = pl.add(x, x)
    return x


def k_spmd_for_split(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, name_hint="q", optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        x = pl.add(x, x)
    return x


def k_spmd_for_slot(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, optimizations=[pl.cross_core_slot(slot_num=3)]):
        x = pl.add(x, x)
    return x


def k_spmd_for_split_slot(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)]):
        x = pl.add(x, x)
    return x


def k_spmd_for_aer(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, allow_early_resolve=True):
        x = pl.add(x, x)
    return x


def k_spmd_for_predicate(
    x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]
) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(4, predicate=(rc[0] > 0)):
        x = pl.add(x, x)
    return x


def k_spmd_for_staged_core(x: pl.Tensor[[64], pl.FP32], n: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(n):
        x = pl.add(x, x)
    return x


def k_spmd_for_body_if(x: pl.Tensor[[64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
    for _i in pl.spmd(2, name_hint="w"):
        if k > 0:
            x = pl.add(x, x)
    return x


def k_spmd_for_in_staged_for(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for _k in pl.range(2):
        for i in pl.spmd(2, name_hint="w"):
            x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def test_spmd_for_anchor():
    conformance(k_spmd_for, build)


def test_spmd_for_named_anchor():
    conformance(k_spmd_for_named, build)


def test_spmd_for_sync_anchor():
    conformance(k_spmd_for_sync, build)


def test_spmd_for_split_anchor():
    conformance(k_spmd_for_split, build)


def test_spmd_for_slot_anchor():
    """``pl.cross_core_slot`` lands as the *inner* InCore's ``slot_num`` attr on
    both lanes (the docstring of the shared fold says ``pl.split`` only, the
    parser's ``_parse_optimizations_list`` accepts both)."""
    conformance(k_spmd_for_slot, build)


def test_spmd_for_split_slot_anchor():
    """Both optimization entries on one line — the entry fold's order/attr
    behaviour is shared (``_fold_optimization_entries`` vs
    ``_parse_optimizations_list``)."""
    conformance(k_spmd_for_split_slot, build)


def test_spmd_for_aer_anchor():
    """``allow_early_resolve`` on the for-form. This is where the two lanes'
    *shared* ``_submit_only_scope_attrs`` helper earns its keep: the for-form
    used to silently drop the kwarg (the DSV4 parity verifier's
    ``expert_shared`` divergence) while the with-form carried it."""
    conformance(k_spmd_for_aer, build)


def test_spmd_for_predicate_anchor():
    conformance(k_spmd_for_predicate, build)


def test_spmd_for_staged_core_anchor():
    """A staged ``core_num`` (``pl.Scalar`` param): ``_pin_spmd_core_num``
    unwraps the facade, the parser resolves the same Var from the scope."""
    conformance(k_spmd_for_staged_core, build)


def test_spmd_for_body_if_anchor():
    """A staged ``if`` inside the for-form body — the canonical pair has the
    branch phi to derive on both lanes."""
    conformance(k_spmd_for_body_if, build)
    conformance(k_spmd_for_body_if, build, canonical=True)


def test_spmd_for_in_staged_for_anchor():
    """A for-form SPMD scope nested in a staged ``pl.range`` loop (raw and
    canonical: the enclosing loop carries a value)."""
    conformance(k_spmd_for_in_staged_for, build)
    conformance(k_spmd_for_in_staged_for, build, canonical=True)


def test_spmd_for_loop_var_binds_block_idx():
    """Belt to the conformance braces: the for-form's loop variable is bound by
    ``loop_var = pl.tile.get_block_idx()`` as the *first* statement of the
    nested InCore carrier (the parser's ``_parse_spmd_for_loop`` shape)."""
    spmd = _spmd(build(k_spmd_for_named))
    assert isinstance(spmd.body, ir.InCoreScopeStmt), type(spmd.body).__name__
    incore = spmd.body
    prologue = incore.body
    assert isinstance(prologue, (ir.AssignStmt, ir.SeqStmts)), type(prologue).__name__
    first = prologue.stmts[0] if isinstance(prologue, ir.SeqStmts) else prologue
    assert isinstance(first, ir.AssignStmt)
    assert isinstance(first.value, ir.Call) and first.value.op.name == _TILE_GET_BLOCK_IDX


# ===========================================================================
# The with-form: TaskId capture, deps edges, and the cluster boundary
# ===========================================================================


def k_spmd_capture_deps(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.spmd(2, name_hint="a") as t1:
        i = pl.tile.get_block_idx()
        x = pl.add(x, pl.cast(i, pl.FP32))
    with pl.spmd(2, name_hint="b", deps=[t1]) as t2:
        j = pl.tile.get_block_idx()
        x = pl.add(x, pl.cast(j, pl.FP32))
    with pl.at(level=pl.Level.CORE_GROUP, deps=[t2]):
        y = pl.load(x, [0], [64])
        pl.store(y, [0], x)
    return x


def k_spmd_capture_array(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    tids = pl.array.create(2, pl.TASK_ID)
    with pl.spmd(2, name_hint="a") as t1:
        i = pl.tile.get_block_idx()
        x = pl.add(x, pl.cast(i, pl.FP32))
    tids[0] = t1
    return x


def k_spmd_in_cluster(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.cluster():
        with pl.spmd(2, name_hint="w"):
            i = pl.tile.get_block_idx()
            x = pl.add(x, pl.cast(i, pl.FP32))
    return x


class KForFormDispatch:
    """The for-form's single-call dispatch body (program-level: the callee is
    reached through ``self``)."""

    @pl.function
    def callee(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return pl.add(x, 1.0)

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for _i in pl.spmd(2, name_hint="w"):
            x = self.callee(x)
        return x


class KCaptureSplitSlot:
    """``as tid`` + both optimization entries on the capture form: the split
    and slot land on the auto-synthesized InCore wrapper, ``task_id_var`` on
    the Spmd scope (the canonical attr order)."""

    @pl.function
    def callee(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        return pl.add(x, 1.0)

    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(
            2,
            name_hint="w",
            optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.cross_core_slot(slot_num=4)],
        ) as _tid:
            x = self.callee(x)
        return x


def test_spmd_capture_deps_anchor():
    """Two captured dispatches chained by ``deps=`` and consumed by a
    ``pl.at(deps=)`` — the TaskId binding proxy and the ``manual_dep_edges``
    threading agree on both lanes (raw and canonical)."""
    conformance(k_spmd_capture_deps, build)
    conformance(k_spmd_capture_deps, build, canonical=True)


def test_spmd_capture_array_anchor():
    """The captured TaskId stored into a ``pl.array.create(N, pl.TASK_ID)`` —
    the array-write path of the capture binding."""
    conformance(k_spmd_capture_array, build)


def test_spmd_in_cluster_anchor():
    """A cluster-nested plain with-form whose body reads the index: the
    cluster-nested scope is the parser's inline-body wrap here, and the
    tracing lane opens the same carrier (raw and canonical)."""
    conformance(k_spmd_in_cluster, build)
    conformance(k_spmd_in_cluster, build, canonical=True)


def test_spmd_for_dispatch_program_anchor():
    """Program-level for-form with a ``self.callee`` dispatch body — the
    for-form's always-wrap shape, cross-function."""
    program_conformance(KForFormDispatch, trace_program)
    program_conformance(KForFormDispatch, trace_program, canonical=True)


def test_spmd_capture_split_slot_program_anchor():
    """Capture + split + slot on the single-call body (the four-cell matrix's
    wrapped row, now with the capture's attr order)."""
    program_conformance(KCaptureSplitSlot, trace_program)
    program_conformance(KCaptureSplitSlot, trace_program, canonical=True)


# ===========================================================================
# ``pl.split_aiv``: the first-class AIV-split region
# ===========================================================================


def k_saiv_up_down(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_none(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_left_right(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.LEFT_RIGHT):
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_in_at(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            y = pl.load(x, [0], [64])
            x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_then_stmt(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    y = pl.load(x, [0], [64])
    x = pl.store(pl.add(y, 1.0), [0], x)
    return x


def k_saiv_in_staged_for(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for k in pl.range(2):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            y = pl.load(x, [0], [64])
            x = pl.store(pl.add(y, pl.cast(k + aiv, pl.FP32)), [0], x)
    return x


def k_saiv_in_at_staged_for(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        for k in pl.range(2):
            for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
                y = pl.load(x, [0], [64])
                x = pl.store(pl.add(y, pl.cast(k + aiv, pl.FP32)), [0], x)
    return x


class KSaivInInCoreFunction:
    """The parser's arm 2: a region written at the top of an InCore-typed
    function is emitted **bare** (no synthesized wrapper) so a printed outlined
    ``*_incore_0`` re-parses to the same IR."""

    @pl.function(type=pl.FunctionType.InCore)
    def k(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            y = pl.load(x, [0], [64])
            x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
        return x


def test_split_aiv_up_down_anchor():
    conformance(k_saiv_up_down, build)


def test_split_aiv_none_anchor():
    conformance(k_saiv_none, build)


def test_split_aiv_left_right_anchor():
    """Sibling regions may carry different modes (multi-mode); here the single
    region pins ``LEFT_RIGHT`` on the region node, not on a function flag."""
    conformance(k_saiv_left_right, build)


def test_split_aiv_in_at_anchor():
    """Already inside an open CORE_GROUP InCore scope → emitted in place on
    both lanes (no second wrapper; the ``NoNestedInCore``-safe flatten)."""
    conformance(k_saiv_in_at, build)


def test_split_aiv_then_stmt_anchor():
    """A statement after the bare top-level region stays outside it on both
    lanes — the wrapper covers the region, not the rest of the function."""
    conformance(k_saiv_then_stmt, build)


def test_split_aiv_in_staged_for_anchor():
    """A region inside a staged ``pl.range`` loop (raw and canonical)."""
    conformance(k_saiv_in_staged_for, build)
    conformance(k_saiv_in_staged_for, build, canonical=True)


def test_split_aiv_in_at_staged_for_anchor():
    """A region inside a staged loop inside a CORE_GROUP scope: the region is
    flattened in place and the loop-carried rebinding is equal in both the raw
    and canonical regimes."""
    conformance(k_saiv_in_at_staged_for, build)
    conformance(k_saiv_in_at_staged_for, build, canonical=True)


def test_split_aiv_in_at_canonical_roundtrip():
    """F26 regression: the canonical yield nested in a ``SplitAivScopeStmt``
    keeps its assignment LHS so the printed program parses back identically."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        traced = canonicalize(build(k_saiv_in_at_staged_for))
        program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    printed = program.as_python()
    assert "= pl.yield_(" in printed
    reparsed = pl.parse(printed)
    ir.assert_structural_equal(program, reparsed, enable_auto_mapping=True)


def test_split_aiv_in_incore_function_program_anchor():
    """Arm 2, program-level: the InCore function's region is bare on both
    lanes (``ctx.is_inside_incore()`` OR ``function_type == InCore``)."""
    program_conformance(KSaivInInCoreFunction, trace_program)


def test_split_aiv_region_shape():
    """Belt to the braces: the region is a ``SplitAivScopeStmt`` carrying the
    mode, whose loop variable is bound to ``pl.tile.get_subblock_idx()``; and
    there is exactly ONE enclosing InCore scope on each spine — the synthesized
    wrapper for the bare top-level form, the ``pl.at(CORE_GROUP)`` carrier
    itself for the in-place form (no ``InCore(InCore(...))`` doubling)."""
    fn = build(k_saiv_up_down)
    scopes = [s for s in _stmts(fn) if isinstance(s, ir.SplitAivScopeStmt)]
    assert scopes and scopes[0].split == ir.SplitMode.UP_DOWN
    assert len(_incore_scopes(fn)) == 1
    assert isinstance(_incore_scopes(fn)[0].body, ir.SplitAivScopeStmt)
    _assert_subblock_idx_prologue(scopes[0])

    inner = build(k_saiv_in_at)
    assert [s for s in _stmts(inner) if isinstance(s, ir.SplitAivScopeStmt)], (
        "expected the region inside the pl.at scope"
    )
    # The pl.at(CORE_GROUP) InCore scope IS the carrier: exactly one, and no
    # InCore nested directly in another (the shape NoNestedInCore forbids).
    assert len(_incore_scopes(inner)) == 1
    assert not any(isinstance(s.body, ir.InCoreScopeStmt) for s in _incore_scopes(inner))


# ===========================================================================
# The CORE_GROUP carrier boundary — F23 (the migration review's F1)
# ===========================================================================


def k_carrier_read_first(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        i = pl.tile.get_block_idx()
        y = pl.load(x, [0, 0], [64, 64])
        z = pl.add(y, pl.cast(i, pl.FP32))
        pl.store(z, [0, 0], x)
    return x


def k_carrier_pre_stmt(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        y = pl.load(x, [0, 0], [64, 64])
        i = pl.tile.get_block_idx()
        z = pl.add(y, pl.cast(i, pl.FP32))
        pl.store(z, [0, 0], x)
    return x


def k_carrier_single_read(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        _i = pl.tile.get_block_idx()
    return x


def k_carrier_read_in_staged_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        for j in pl.range(2):
            i = pl.tile.get_block_idx()
            y = pl.load(x, [0, 0], [64, 64])
            z = pl.add(y, pl.cast(i + j, pl.FP32))
            pl.store(z, [0, 0], x)
    return x


def k_carrier_split_aiv_between(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            i = pl.tile.get_block_idx()
            y = pl.load(x, [0, 0], [64, 64])
            z = pl.add(y, pl.cast(i + aiv, pl.FP32))
            pl.store(z, [0, 0], x)
        pl.system.syncall()
        w = pl.load(x, [0, 0], [64, 64])
        pl.store(pl.add(w, 1.0), [0, 0], x)
    return x


def test_carrier_read_first_anchor():
    """The boundary's *equivalent* half: when the read is the body's first
    statement the parser's static wrap and the tracing lane's trigger fire at
    the same point (raw and canonical)."""
    conformance(k_carrier_read_first, build)
    conformance(k_carrier_read_first, build, canonical=True)


def test_carrier_pre_read_statement_anchor():
    """A statement *before* the first ``get_block_idx`` read.

    ``_spmd_body_reads_block_idx`` walks the whole suite, so the parser wraps
    the entire body in one ``InCoreScopeStmt``; the tracing lane's
    ``reads_block_idx`` fact is the same static, body-wide classification, so
    its carrier opens at the body's first statement rather than at the read.
    """
    conformance(k_carrier_pre_stmt, build)


def test_carrier_single_read_name_hint_anchor():
    """A body that is exactly one ``i = pl.tile.get_block_idx()`` statement.

    The read fact wins over the direct-dispatch shape (``single_call_body``),
    so the body is *inline* on both lanes: the split hints (``w_spmd`` outer /
    ``w`` carrier) land on the scopes, where the pin recorded the tracing
    lane's unsplit outer hint.
    """
    conformance(k_carrier_single_read, build)
    assert _spmd(build(k_carrier_single_read)).name_hint == "w_spmd"


def test_carrier_read_inside_staged_flow_anchor():
    """A read nested in staged control flow (raw and canonical).

    The static classification sees the nested read, so the carrier opens at
    the loop's entry on both lanes — the tracing lane's dynamic trigger used
    to decline here and leave the whole body bare in the Spmd scope.
    """
    conformance(k_carrier_read_in_staged_for, build)
    conformance(k_carrier_read_in_staged_for, build, canonical=True)


def test_carrier_split_aiv_between_anchor():
    """The migration review's F1 fixture shape.

    A body whose only read of the per-block index is *inside* a
    ``pl.split_aiv`` region, with a statement between the region and the rest.
    Both lanes wrap the whole body once, so the region is emitted in place (no
    per-region synthesized wrapper) and the between-statements are inside the
    same carrier — exactly the shape
    ``tests/ut/codegen/test_orchestration_returned_param_map.py`` had to state
    explicitly (its migrated body nests
    ``pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_fused")``).
    """
    conformance(k_carrier_split_aiv_between, build)
    fn = build(k_carrier_split_aiv_between)
    assert isinstance(_spmd(fn).body, ir.InCoreScopeStmt), "one carrier wraps the whole body"
    assert [s for s in _stmts(fn) if isinstance(s, ir.SplitAivScopeStmt)], (
        "the region is emitted in place inside the carrier"
    )


@pytest.mark.parametrize(
    "kernel",
    [
        k_carrier_pre_stmt,
        k_carrier_single_read,
        k_carrier_read_in_staged_for,
        k_carrier_split_aiv_between,
    ],
    ids=["pre_stmt", "single_read", "read_in_staged_for", "split_aiv_between"],
)
def test_carrier_boundary_roundtrip(kernel):
    """F23 regression: every carrier-boundary spelling prints text the parser
    re-reads to the same IR. The pin recorded three of the four breaking the
    print → reparse fixpoint (the carrier boundary landed mid-body)."""
    traced = build(kernel)
    program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    reparsed = pl.parse(program.as_python())
    ir.assert_structural_equal(program, reparsed, enable_auto_mapping=True)


def k_carrier_explicit_at(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c"):
            y = pl.load(x, [0, 0], [64, 64])
            z = pl.add(y, 1.0)
            pl.store(z, [0, 0], x)
    return x


def k_carrier_explicit_at_read(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c"):
            i = pl.tile.get_block_idx()
            y = pl.load(x, [0, 0], [64, 64])
            z = pl.add(y, pl.cast(i, pl.FP32))
            pl.store(z, [0, 0], x)
    return x


def k_carrier_explicit_at_read_tid(
    x: pl.Tensor[[64, 64], pl.FP32],
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(2, name_hint="w") as _tid:
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="c"):
            i = pl.tile.get_block_idx()
            y = pl.load(x, [0, 0], [64, 64])
            z = pl.add(y, pl.cast(i, pl.FP32))
            pl.store(z, [0, 0], x)
    return x


@pytest.mark.parametrize(
    "kernel", [k_carrier_explicit_at, k_carrier_explicit_at_read], ids=["dispatch_body", "index_reading"]
)
def test_carrier_explicit_at_hint_accepted_drift(kernel):
    """**F36 — accepted divergence under R5 (naming-only), not a fix target.

    A body that supplies its own CORE_GROUP carrier (a lone nested
    ``pl.at(level=pl.Level.CORE_GROUP, ...)`` — the parser's
    ``_explicit_incore_carrier`` shape) keeps the ``pl.spmd(name_hint=...)``
    unsplit on the parser lane, because the nested scope *is* the carrier; the
    tracing lane's split-hint rule (``not single_call_body``) reserves the
    ``_spmd`` suffix for the outer scope regardless.

    The two IRs differ in exactly one string: the outer ``SpmdScopeStmt``'s
    ``name_hint`` (the assertion below pins that the failing path is
    ``body[0].name_hint``, i.e. nothing else drifted). Both lanes accept the
    spelling, the statements and the explicit carrier are identical, the hint
    names the outlined function and nothing else (``OutlineClusterScopes``
    consults its ``_spmd_`` *suffix* only when the hint is empty), and each
    lane's printed text re-parses without error — so this is R5's trivial,
    naming-only drift (``docs/fix-construct-parity.md`` §4: 'trivial drift ...
    is fine'), not wrong codegen. Fixing it would need a third static body fact
    threaded through the rewriter (the parser's explicit-carrier check, which
    needs the ``pl.at`` level resolved) for a renamed kernel, which R5 forbids.

    The assertions stay as the accepted-divergence witness; do not tighten one
    lane to make this test disappear. The fixture's *shared* printer hole is a
    separate finding (F40)."""
    traced = build(kernel)
    parsed = function_parser(kernel)
    assert _spmd(traced).name_hint == "w_spmd"
    assert _spmd(parsed).name_hint == "w"
    with pytest.raises(ValueError, match=r"body\[0\]\.name_hint"):
        ir.assert_structural_equal(traced, parsed, enable_auto_mapping=True)


def test_explicit_carrier_print_keeps_the_carrier_hint():
    """**F40 regression — the explicit carrier's hint survives print → reparse.**

    A read-first body that supplies its own CORE_GROUP carrier used to print
    through the printer's for-form sniff
    (``IRPythonPrinter::VisitStmt_(SpmdScopeStmtPtr)``'s ``tile.get_block_idx``
    check) as ``for i in pl.spmd(n, name_hint=<the Spmd hint>)`` — a shape with
    no slot for the carrier's own hint. Re-parsing synthesised the for-form
    carrier (``name_hint`` = the split-inner name, ``"w"``), so the
    user-written ``pl.at(level=pl.Level.CORE_GROUP, name_hint="c")`` lost its
    name on both lanes identically (a printer/parser-coverage hole, the F39
    class, not a parity divergence); on the parser lane the outer hint
    additionally re-split (``"w"`` → ``"w_spmd"``) because the for-form always
    wraps.

    The printer now spells the carrier out as a nested
    ``pl.at(level=pl.Level.CORE_GROUP, name_hint=...)`` whenever the for-form
    desugaring would not reproduce its hint, so each lane's printed text
    re-parses to the same IR."""
    for build_fn, lane in ((build, "tracing"), (function_parser, "parser")):
        original = build_fn(k_carrier_explicit_at_read)
        (carrier,) = _incore_scopes(original)
        assert carrier.name_hint == "c", lane
        program = ir.Program([original], f"p_{original.name}_{lane}", ir.Span.unknown())
        reparsed = pl.parse(program.as_python())[original.name]
        (reparsed_carrier,) = _incore_scopes(reparsed)
        assert reparsed_carrier.name_hint == "c", lane
        prologue = reparsed_carrier.body
        first = prologue.stmts[0] if isinstance(prologue, ir.SeqStmts) else prologue
        assert isinstance(first, ir.AssignStmt)
        assert isinstance(first.value, ir.Call) and first.value.op.name == _TILE_GET_BLOCK_IDX, lane
        assert first.var.name_hint == "i", lane  # the body itself survived the round trip
        # The whole statement survives, outer hint included: the for-form's
        # implicit re-split ("w" -> "w_spmd") is gone with it.
        ir.assert_structural_equal(original, reparsed, enable_auto_mapping=True)


def test_captured_spmd_explicit_carrier_wraps_once_by_both_lanes():
    """**F44 regression — the captured explicit carrier wraps once, on both lanes.**

    ``with pl.spmd(n, name_hint=...) as tid:`` whose body supplies its own
    ``pl.at(level=pl.Level.CORE_GROUP, name_hint="c")`` carrier used to nest
    twice on the tracing lane. The ``as tid`` target binding emits no IR, but it
    still ran the frame's pre-emit flush, and the wrapped with-form's pending
    carrier (``_carrier_flush_on_emit``) opened there — before the body reached
    its own region — so ``cancel_pending_carrier`` could no longer absorb it
    (``_flush_deferred_scopes``' ``entering_incore`` branch only saw a pending
    entry). The tracing IR was ``Spmd{InCore("w"){InCore("c"){...}}}`` where
    the parser's ``_emit_spmd_body`` treats the nested ``pl.at`` as *the* carrier
    and emits ``Spmd{InCore("c"){...}}`` — the shape
    ``cancel_pending_carrier``'s own docstring says the tracing lane intends.

    Both lanes now emit one carrier (``["c"]``) and each lane's printed text
    re-parses to the same IR (F40's printer fix makes both lanes fixpoints).
    The outer ``SpmdScopeStmt.name_hint`` still differs (``"w_spmd"`` vs
    ``"w"``) — F36's accepted naming-only drift, unchanged by this fix."""
    traced = build(k_carrier_explicit_at_read_tid)
    parsed = function_parser(k_carrier_explicit_at_read_tid)
    assert [c.name_hint for c in _incore_chain(_spmd(parsed))] == ["c"]
    assert [c.name_hint for c in _incore_chain(_spmd(traced))] == ["c"]
    for original, lane in ((parsed, "parser"), (traced, "tracing")):
        program = ir.Program([original], f"p_{original.name}_{lane}", ir.Span.unknown())
        reparsed = pl.parse(program.as_python())[original.name]
        assert [c.name_hint for c in _incore_chain(_spmd(reparsed))] == ["c"], lane
        # The statement survives print → reparse on both lanes; only the outer
        # Spmd hint differs *between* lanes (F36).
        ir.assert_structural_equal(original, reparsed, enable_auto_mapping=True)


# ===========================================================================
# No-shared-spelling variants
# ===========================================================================


def k_spmd_deps_without_capture(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.spmd(2, name_hint="a", deps=[None]):
        i = pl.tile.get_block_idx()
        x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def k_spmd_deps_for_form(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.spmd(2, deps=[None]):
        x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def k_spmd_bare_body(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        y = pl.add(x, 1.0)
        x = pl.add(y, 2.0)
    return x


def k_spmd_bare_body_single_call(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.spmd(2, name_hint="w"):
        _y = pl.tensor.adds(x, 1.0)
    return x


def k_saiv_nested(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
        for b in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            y = pl.load(x, [0], [64])
            x = pl.store(pl.add(y, pl.cast(b, pl.FP32)), [0], x)
    return x


def k_saiv_n_not_two(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(4, mode=pl.SplitMode.NONE):
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_no_mode(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2):  # type: ignore[call-arg]
        y = pl.load(x, [0], [64])
        x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_saiv_with_function_split(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        for aiv in pl.split_aiv(2, mode=pl.SplitMode.NONE):
            y = pl.load(x, [0], [64])
            x = pl.store(pl.add(y, pl.cast(aiv, pl.FP32)), [0], x)
    return x


def k_cluster_spmd_aer(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.cluster():
        with pl.spmd(2, allow_early_resolve=True):
            i = pl.tile.get_block_idx()
            x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def k_cluster_spmd_capture(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.cluster():
        with pl.spmd(2) as _tid:
            i = pl.tile.get_block_idx()
            x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def k_spmd_zero_cores(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.spmd(0):
        x = pl.add(x, pl.cast(i, pl.FP32))
    return x


def test_spmd_deps_without_capture_is_anchored():
    """``deps=`` without ``as tid`` traces like the parser — anchored.

    The parser (the round-trip owner) accepts both spellings and synthesizes
    the TaskId Var the outliner needs, so the tracer accepts them too
    (an uncaptured scope still lowers to a Submit; ``manual_dep_edges`` rides
    the Spmd scope). The earlier tracing-only rejection — once recorded as an
    "intended divergence" in the dev docs — broke the corpus's
    ``gate.py`` / ``expert_routed.py`` shapes and contradicted the
    conformance principle (accept what the parser accepted); the docs now
    state the permissive rule."""
    for kernel in (k_spmd_deps_without_capture, k_spmd_deps_for_form):
        assert function_parser(kernel) is not None  # the parser lane accepts
        conformance(kernel, build)


def test_spmd_bare_body_is_tracing_accepted():
    """**needs-decision.**

    ``docs/en/dev/language/01-statements.md`` (the ``pl.spmd`` section) and
    ``docs/en/user/language/04-scopes.md:140`` state the contract: a body that
    neither reads the per-block index nor dispatches a ``self.<kernel>(...)``
    call is rejected — *'rejected — every block would be doing identical
    work'*. The parser enforces it; the tracing lane accepts the body and emits
    the bare ``Spmd`` scope, and the printed text cannot be re-parsed. The
    tracing suite that owns this construct
    (``test_spmd_with_form.py``) records the omission as deliberate — the
    property survives to the IR and its diagnostic home should be a
    pass/verifier check — but no design doc or manifest sanctions it. Recorded
    here without a fix."""
    for kernel in (k_spmd_bare_body, k_spmd_bare_body_single_call):
        with pytest.raises(ParserSyntaxError, match="identical work"):
            function_parser(kernel)
        traced = build(kernel)
        assert isinstance(_spmd(traced).body, ir.Stmt)
        with pytest.raises(ParserSyntaxError, match="identical work"):
            pl.parse(str(traced))


def test_nested_split_aiv_is_tracing_accepted():
    """**F24 — a bug, pinned, not an anchor.**

    The parser rejects a ``pl.split_aiv`` region inside another one
    (``_parse_split_aiv_for_loop``: *'a split_aiv body already represents the
    two AIV lanes, so it cannot contain another split_aiv loop'*). The tracing
    ``split_aiv.begin`` has no nesting guard, so the region nests and the
    printed text cannot be re-parsed — the print→parse fixpoint breaks the same
    way F12's stray ``break`` does. Delete the tracing-side assertions when
    F24 lands."""
    traced = build(k_saiv_nested)
    nested = [s for s in _stmts(traced) if isinstance(s, ir.SplitAivScopeStmt)]
    assert nested, "the tracing lane should have emitted a region"
    assert any(
        isinstance(s, ir.ScopeStmt) and isinstance(s.body, ir.SplitAivScopeStmt) for s in _stmts(traced)
    ), "the tracing lane should have emitted nested regions"
    with pytest.raises(ParserSyntaxError, match="nested 'for ... in pl.split_aiv"):
        function_parser(k_saiv_nested)
    with pytest.raises(ParserSyntaxError, match="nested 'for ... in pl.split_aiv"):
        pl.parse(str(traced))


def test_split_aiv_n_and_mode_rejected_by_both_lanes():
    """No accepted spelling exists: ``n`` is hardware-fixed at 2 and ``mode``
    is required, so the two lanes reject with their own diagnostics (tracing
    raises ``ValueError`` at construction / ``TypeError`` on the missing
    argument, the parser ``ParserSyntaxError``)."""
    with pytest.raises(ValueError, match="must be the integer 2"):
        build(k_saiv_n_not_two)
    with pytest.raises(ParserSyntaxError, match="requires n == 2"):
        function_parser(k_saiv_n_not_two)
    with pytest.raises(TypeError, match="missing 1 required positional argument"):
        build(k_saiv_no_mode)
    with pytest.raises(ParserSyntaxError, match=r"requires mode="):
        function_parser(k_saiv_no_mode)


def test_split_aiv_with_function_split_rejected_by_both_lanes():
    """One gate both lanes share verbatim: a region under a function-level
    ``pl.split(...)`` would be silently dropped by the per-region lowering."""
    with pytest.raises(ParserSyntaxError, match="mutually exclusive AIV-split"):
        build(k_saiv_with_function_split)
    with pytest.raises(ParserSyntaxError, match="mutually exclusive AIV-split"):
        function_parser(k_saiv_with_function_split)


def test_cluster_nested_spmd_submit_only_kwargs_rejected_by_both_lanes():
    """A cluster-nested ``pl.spmd`` never lowers to a ``Submit``, so the
    Submit-only kwargs and the TaskId capture are rejected by both lanes."""
    with pytest.raises(TypeError, match="cannot be nested inside `pl.cluster"):
        build(k_cluster_spmd_aer)
    with pytest.raises(ParserSyntaxError, match="cannot be nested inside `pl.cluster"):
        function_parser(k_cluster_spmd_aer)
    with pytest.raises(TypeError, match="cannot capture a TaskId"):
        build(k_cluster_spmd_capture)
    with pytest.raises(ParserSyntaxError, match="cannot capture a TaskId"):
        function_parser(k_cluster_spmd_capture)


def test_spmd_core_num_validation_rejected_by_both_lanes():
    """``_parse_and_validate_core_num`` (parser) / ``_pin_spmd_core_num``
    (tracing): the same positivity rule, one message each."""
    with pytest.raises(ValueError, match="must be a positive integer"):
        build(k_spmd_zero_cores)
    with pytest.raises(ParserSyntaxError, match="must be a positive integer"):
        function_parser(k_spmd_zero_cores)
