# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.at(..., predicate=)``: the dispatch-predicate scope form (gap K8, task F4).

The bug class this pins: a parser-era construct kwarg with placement gates
that the tracing frontend dropped wholesale — ``pl.at(predicate=)`` did not
exist on the tracing ``at`` at all (``TypeError`` at trace time), so the
predicated outlined block had no tracing spelling. The fix threads the kwarg
through the tracing ``AtContext`` with the parser's placement gates
(``ast_parser.py::_parse_at_predicate``) re-encoded at the trace-protocol
boundary:

* ``level=pl.Level.CORE_GROUP`` only — any other level builds a Hierarchy
  scope, which is never outlined into a task dispatch, so the predicate
  would be silently dropped;
* not nested inside ``pl.cluster()`` / ``pl.spmd()`` / another CORE_GROUP
  ``pl.at`` — the inner dispatch folds into the enclosing task and codegen
  emits ``set_predicate`` from the outer call only (nesting inside a
  *non*-CORE_GROUP ``pl.at`` stays legal: a Hierarchy scope is not a task
  wrapper);
* the enclosing function must be one ``OutlineIncoreScopes`` rewrites
  (``Opaque`` / ``Inline`` / orchestration-like) — a scope written in a
  ``Group`` / ``Spmd`` / InCore-family kernel body never becomes a dispatch;
* the predicate itself must be a staged value.

Emission cases run through ``conformance`` (tracer vs parser, structural
equality — the kwarg lowers to the ``predicate`` scope attr in the parser's
canonical position); rejection cases assert ``TypeError`` with the parser's
message wording. The end-to-end lowering (``CoreTaskPredicate`` +
``Arg::set_predicate`` in orchestration C++, identical to the
``pl.spmd_submit`` sibling) is pinned in
``tests/ut/codegen/test_predicate_codegen.py``.
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir

from ._harness import conformance

build = function_trace


# ===========================================================================
# Emission parity — the predicate rides the InCore scope like the parser's
# ===========================================================================


def k_at_predicate_plain(
    x: pl.Tensor[[64, 64], pl.FP32], rc: pl.Tensor[[64, 64], pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0, 0] > 0)):
        t = pl.load(x, [0, 0], [32, 32])
        pl.store(t, [0, 0], x)
    return x


def k_at_predicate_capture(
    x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]
) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="producer") as g_tid:
        t = pl.add(x, x)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[g_tid], predicate=(rc[0] > 0)) as tid:
        y = pl.add(t, 1.0)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid]):
        z = pl.add(y, 2.0)
    return z


def k_at_predicate_aer(
    x: pl.Tensor[[64, 64], pl.FP32], rc: pl.Tensor[[64, 64], pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True, predicate=(rc[0, 0] > 0)):
        t = pl.load(x, [0, 0], [32, 32])
        pl.store(t, [0, 0], x)
    return x


def k_at_predicate_in_hierarchy(
    x: pl.Tensor[[64, 64], pl.FP32], rc: pl.Tensor[[64, 64], pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    # A Hierarchy scope is not a task wrapper: the inner InCore scope is still
    # outlined into its own Submit, so the predicate keeps its carrier.
    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
        with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0, 0] > 0)):
            t = pl.load(x, [0, 0], [32, 32])
            pl.store(t, [0, 0], x)
    return x


def test_at_predicate_plain_anchor():
    """The no-capture form: the predicate lands as the scope's ``predicate``
    attr, forcing the Submit shape exactly as the parser's ``_parse_at_meta``."""

    conformance(k_at_predicate_plain, build)


def test_at_predicate_capture_anchor():
    """``deps=`` + ``predicate=`` + ``as tid`` on one scope — the full
    doc-example shape (a producer TaskId in deps, the predicate over a tensor,
    the dispatch TaskId captured for a later dep)."""

    conformance(k_at_predicate_capture, build)


def test_at_predicate_aer_anchor():
    """``allow_early_resolve`` + ``predicate`` together — pins the canonical
    attr order (``allow_early_resolve`` before ``predicate``, both before
    ``slot_num`` / ``windowize``)."""

    conformance(k_at_predicate_aer, build)


def test_at_predicate_in_hierarchy_anchor():
    """Nesting inside a *non*-CORE_GROUP ``pl.at`` stays legal (the parser's
    gate rejects InCore / Cluster / Spmd nesting only)."""

    conformance(k_at_predicate_in_hierarchy, build)


def test_at_predicate_scope_attr_position():
    """Direct IR pin (belt to the conformance braces): the InCore scope
    carries ``("predicate", <Expr>)`` after ``allow_early_resolve``."""

    fn = build(k_at_predicate_aer)
    found = _incore_attr_names(fn)
    assert found, "no InCoreScopeStmt in the traced function"
    names = found[0]
    assert "predicate" in names, names
    assert names.index("allow_early_resolve") < names.index("predicate"), names


def _incore_attr_names(fn: ir.Function) -> list[list[str]]:
    found: list[list[str]] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.InCoreScopeStmt):
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


# ===========================================================================
# Rejection parity — the parser's placement gates, re-encoded at trace time
# ===========================================================================


class _HierarchyLevel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator, predicate=(rc[0] > 0)):
            y = pl.add(x, x)
        return y


class _ClusterNested:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.cluster():
            with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0] > 0)):
                y = pl.add(x, x)
        return y


class _SpmdNested:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.spmd(2):
            with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0] > 0)):
                y = pl.add(x, x)
        return y


class _AtNested:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP):
            with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0] > 0)):
                y = pl.add(x, x)
        return y


class _IncoreFunction:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, predicate=(rc[0] > 0)):
            y = pl.add(x, x)
        return y


class _HostPredicate:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32], rc: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.FP32]:
        with pl.at(level=pl.Level.CORE_GROUP, predicate=True):
            y = pl.add(x, x)
        return y


def test_hierarchy_level_rejected():
    """A non-CORE_GROUP ``pl.at`` builds a Hierarchy scope, which is never
    outlined into a task dispatch — the predicate would be lost."""

    with pytest.raises(TypeError, match="pl.at\\(predicate=\\.\\.\\.\\) is only supported with"):
        trace_program(_HierarchyLevel)


def test_cluster_nested_rejected():
    """A cluster-nested InCore scope is unwrapped into the Group function and
    never produces a Submit — rejected at trace time, not as a late internal
    error in ``OutlineClusterScopes``."""

    with pytest.raises(TypeError, match="cannot be nested inside `pl.cluster\\(\\)`"):
        trace_program(_ClusterNested)


def test_spmd_nested_rejected():
    """The inner dispatch folds into the enclosing Spmd wrapper — codegen
    emits ``set_predicate`` from the outer call only."""

    with pytest.raises(TypeError, match="cannot be nested inside `pl.spmd\\(\\)`"):
        trace_program(_SpmdNested)


def test_at_nested_rejected():
    """Nesting inside another CORE_GROUP ``pl.at`` folds the inner dispatch
    into the enclosing kernel body."""

    with pytest.raises(
        TypeError, match="cannot be nested inside `another pl.at\\(level=pl.Level.CORE_GROUP\\)`"
    ):
        trace_program(_AtNested)


def test_incore_function_rejected():
    """``OutlineIncoreScopes`` skips ``Group`` / ``Spmd`` / InCore-family
    bodies, so a scope written there never becomes a dispatch (the parser's
    function-type gate)."""

    with pytest.raises(TypeError, match="is not supported inside a InCore function"):
        trace_program(_IncoreFunction)


def test_host_predicate_rejected():
    """The predicate must be a staged value — a host bool/int has no IR
    carrier (the spmd scope form's rule)."""

    with pytest.raises(TypeError, match="pl.at\\(predicate=\\.\\.\\.\\) must be a staged value"):
        trace_program(_HostPredicate)
