# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.spmd`` Submit-only kwargs and ``pl.at``'s ``windowize``: form-parity anchors.

The bug class these pin: a construct kwarg stored in ``__init__`` consumed on
one emission path but silently dropped on a sibling path (or accepted where
the parser rejects). ``pl.spmd`` has two lowering paths — the with-form's
``__trace_enter__`` and the for-form's ``begin`` — and the for-form used to
build its Spmd scope with no attrs at all, dropping ``allow_early_resolve``
and ``predicate`` (the DSV4 parity verifier's ``expert_shared`` divergence;
``allow_early_resolve`` appears on the for-form in 8+ corpus files). The
same sweep found three more in the family:

* ``pl.at(..., windowize=True)`` — ``self.windowize`` was never read.
* for-form / plain-with-form ``deps=`` — silently accepted where the parser
  rejects (``deps=`` is legal only on the ``as tid`` capture form).
* cluster-nested ``pl.spmd`` with Submit-only kwargs (or ``as tid``) —
  accepted at trace time, then failing late in ``OutlineClusterScopes`` as
  an internal error where the parser rejects at parse time.

The fixes route both ``pl.spmd`` forms through one attrs helper
(``_submit_only_scope_attrs``) and mirror the parser's form-surface
rejections at trace time. These anchors hold the line: emission cases via
``conformance`` (tracer vs parser, raw structural equality), rejection
cases via ``TypeError`` with the parser's message wording.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir

from ._harness import conformance

build = function_trace


# ===========================================================================
# Emission parity — the for-form carries the same scope attrs as the parser
# ===========================================================================


def k_spmd_for_aer(a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
    for i in pl.spmd(2, allow_early_resolve=True):
        off = i * 8
        t = pl.load(a, [off, 0], [8, 16])
        pl.store(pl.tile.add(t, t), [off, 0], a)
    return a


def k_spmd_for_predicate(
    a: pl.Tensor[[16, 16], pl.FP32], rc: pl.Tensor[[16, 16], pl.INT32]
) -> pl.Tensor[[16, 16], pl.FP32]:
    for i in pl.spmd(2, predicate=(rc[0, 0] > 0)):
        off = i * 8
        t = pl.load(a, [off, 0], [8, 16])
        pl.store(pl.tile.add(t, t), [off, 0], a)
    return a


def k_spmd_with_aer_predicate(
    a: pl.Tensor[[16, 16], pl.FP32], rc: pl.Tensor[[16, 16], pl.INT32]
) -> pl.Tensor[[16, 16], pl.FP32]:
    with pl.spmd(2, allow_early_resolve=True, predicate=(rc[0, 0] > 0)):
        i = pl.tile.get_block_idx()
        off = i * 8
        t = pl.load(a, [off, 0], [8, 16])
        pl.store(pl.tile.add(t, t), [off, 0], a)
    return a


def k_at_windowize(a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
        i = pl.tile.get_block_idx()
        off = i * 8
        t = pl.load(a, [off, 0], [8, 16])
        pl.store(pl.tile.add(t, t), [off, 0], a)
    return a


def test_for_form_allow_early_resolve():
    """``for i in pl.spmd(n, allow_early_resolve=True)`` threads the attr onto
    the Spmd scope exactly as the parser does (the pre-fix tracer dropped it —
    the with/for sibling paths drifted)."""

    conformance(k_spmd_for_aer, build)


def test_for_form_predicate():
    """The for-form's dispatch predicate rides on the Spmd scope like the
    parser's ``_parse_spmd_for_loop`` (pre-fix: silently dropped)."""

    conformance(k_spmd_for_predicate, build)


def test_with_form_submit_only_kwargs():
    """The with-form pin: ``allow_early_resolve`` + ``predicate`` on one scope —
    guards the shared attrs helper against with-form regressions too."""

    conformance(k_spmd_with_aer_predicate, build)


def test_at_windowize():
    """``pl.at(..., windowize=True)`` lands the ``windowize`` scope attr
    (appended after ``slot_num``, the parser's attr order)."""

    conformance(k_at_windowize, build)


# ===========================================================================
# Rejection parity — accept what the parser accepted, reject what it rejected
# ===========================================================================


class _ForFormDeps:
    @pl.function
    def stage(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.manual_scope():
            a, tid = pl.spmd_submit(self.stage, a, core_num=1)
            for i in pl.spmd(2, deps=[tid]):
                off = i * 8
                t = pl.load(a, [off, 0], [8, 16])
                pl.store(pl.tile.add(t, t), [off, 0], a)
        return a


class _PlainWithFormDeps:
    @pl.function
    def stage(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.manual_scope():
            a, tid = pl.spmd_submit(self.stage, a, core_num=1)
            with pl.spmd(2, deps=[tid]):  # no `as tid` — deps= is not legal here
                i = pl.tile.get_block_idx()
                off = i * 8
                t = pl.load(a, [off, 0], [8, 16])
                pl.store(pl.tile.add(t, t), [off, 0], a)
        return a


class _ClusterNestedAer:
    @pl.function
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.cluster():
            with pl.spmd(2, allow_early_resolve=True):
                i = pl.tile.get_block_idx()
                off = i * 8
                t = pl.load(a, [off, 0], [8, 16])
                pl.store(pl.tile.add(t, t), [off, 0], a)
        return a


class _ClusterNestedAsTid:
    @pl.function
    def main(self, a: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.cluster():
            with pl.spmd(2) as tid:
                i = pl.tile.get_block_idx()
                off = i * 8
                t = pl.load(a, [off, 0], [8, 16])
                pl.store(pl.tile.add(t, t), [off, 0], a)
                _ = tid
        return a


def test_for_form_deps_rejected():
    """The for-form captures no TaskId, so ``deps=`` has nowhere to hang its
    edges — rejected by keyword presence (``deps=[]`` included), mirroring the
    parser's ``allow_deps=False`` (pre-fix: silently accepted and dropped)."""

    with pytest.raises(TypeError, match="pl.spmd\\(\\) does not accept 'deps=' here"):
        trace_program(_ForFormDeps)


def test_plain_with_form_deps_rejected():
    """``deps=`` is legal only on the ``as tid`` capture form (pre-fix: the
    tracer threaded ``manual_dep_edges`` where the parser rejects)."""

    with pytest.raises(TypeError, match="pl.spmd\\(\\) does not accept 'deps=' here"):
        trace_program(_PlainWithFormDeps)


def test_cluster_nested_submit_only_kwargs_rejected():
    """A cluster-nested ``pl.spmd`` is unwrapped into the Group function and
    never produces a Submit — the Submit-only kwargs would be lost. Rejected
    at trace time (pre-fix: accepted, then a late internal error in
    ``OutlineClusterScopes``)."""

    with pytest.raises(TypeError, match="cannot be nested inside `pl.cluster\\(\\)`"):
        trace_program(_ClusterNestedAer)


def test_cluster_nested_as_tid_rejected():
    """``as tid`` inside ``pl.cluster()``: same never-produces-a-Submit
    reasoning, mirroring the parser's ``_parse_spmd_scope_with_tid`` guard."""

    with pytest.raises(TypeError, match="cannot capture a TaskId when nested"):
        trace_program(_ClusterNestedAsTid)


def test_submit_only_attrs_survive_on_the_scope():
    """Direct IR pin (belt to the conformance braces): the for-form's Spmd
    scope carries the attrs the corpus's ``expert_shared`` shape needs."""

    fn = build(k_spmd_for_aer)
    found = _spmd_attrs(fn)
    assert found and all("allow_early_resolve" in a for a in found), found


def _spmd_attrs(fn: ir.Function) -> list:
    found: list = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.SpmdScopeStmt):
            found.append([a if isinstance(a, str) else a[0] for a in (stmt.attrs or [])])
        for attr in ("body", "stmts"):
            children = getattr(stmt, attr, None)
            if children is None:
                continue
            for child in children if isinstance(children, (list, tuple)) else [children]:
                if isinstance(child, ir.Stmt):
                    walk(child)

    for s in fn.body or []:
        walk(s)
    return found
