# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: the generic scopes (the R10 sweep) — ``pl.scope`` /
``pl.manual_scope`` / ``pl.graph`` / ``pl.cluster``.

Both frontends are thin here. The parser routes every ``with pl.<generic>``
through ``_parse_scope_body`` (``ast_parser.py:5365``) — the single
``builder.scope(...)`` injection point, entered by ``_parse_scope``
(``:4064``, ``mode=`` kwarg), ``_parse_manual_scope`` (``:4090``, the
``mode=MANUAL`` alias, both via ``_emit_runtime_scope`` ``:4101``),
``_parse_legacy_scope`` (``:4141``, ``pl.cluster``) and ``_parse_graph_scope``
(``:5421``). The tracing lane routes the same ``with`` through the rewriter's
``_with_row`` (``rewriter.py:1485``) into ``Sema.create_with_construct``
(``semantics.py:1548``) → ``PtoWithConstruct`` (``tracer/constructs.py:576``),
whose ``__trace_enter__`` hooks live on the descriptors in
``constructs/scopes.py`` (``scope`` ``:222``, ``graph`` ``:309``,
``manual_scope`` ``:448``, ``cluster`` ``:536``); both lanes end in the same
``IRBuilder.begin_scope`` / ``end_scope`` pair (``ir/builder.py:222``,
``:254``) and emit the same ``RuntimeScopeStmt`` / ``GraphScopeStmt`` /
``ClusterScopeStmt``.

**Regime.** Every straight-line scope anchor is asserted **raw**: the scope is
a transparent region on both sides (no ``iter_arg``/phi is involved), so the
two frontends must emit the identical statement. Anchors whose scope body
contains staged control flow (a ``for`` with a carried value, a branch with
divergent binds) and anchors placed *inside* staged control flow are asserted
**raw *and* post-``canonicalize()``** — raw holds (the scope is still
transparent), and canonical is the staged-flow regime proper (the SSA pass
promotes the shared mutable locals across the region). One anchor is raw-only:
a scope *inside* a loop body whose rebinding is loop-carried (``k_manual_in_loop``)
— both lanes' canonical IR is equal, but the post-SSA shape hits a printer gap
(F22, pinned below), so the canonical half is pinned rather than asserted
through the harness.

The variants that cannot be anchored, and why (all pinned below):

- ``pl.dump_tag(x)`` before a ``pl.graph`` region is an anchor (**F18**):
  the parser reaches the forward-sticky merge through
  ``_parse_scope_body`` (``_merge_forward_sticky_dump``, ``ast_parser.py:5313``)
  and the tracing ``graph`` hook now carries the same merge. The attr round-trips
  as a ``dumps=[...]`` kwarg on the scope (**F39**, fixed: the printer emits it
  and both lanes' parser rebuilds it);
- ``pl.scope(mode=<anything but the enum>)``: the tracing ``scope``
  descriptor compares ``mode`` against the enum in its hook
  (``constructs/scopes.py:269`` / ``:273``), so a non-enum value silently
  selects AUTO and skips the ``auto_scope=False`` opt-in the parser enforces
  — **F19**;
- the runtime's manual-scope nesting rules (MANUAL-in-MANUAL, and
  AUTO-in-MANUAL under ``auto_scope=False``): no accepted spelling exists on
  either lane — both reject the pair (**F20**, fixed; rejection pins below);
- a non-identifier ``pl.graph`` name / ``pl.cluster(name_hint=...)``: the
  parser validates it (``_parse_scope_name_hint``, ``:4039``), the tracing
  descriptors do not — **F21**;
- a loop-carried value updated inside a manual runtime scope: the printer
  drops the yield assignment LHS (it keeps it only for an AUTO
  ``RuntimeScopeStmt``) and the roundtrip instrument rejects the text —
  **F22**, printer-side, both frontends' canonical IR equal;
- the lenient spellings ``pl.graph(name="x")`` and
  ``pl.scope(pl.ScopeMode.MANUAL)`` are tracing-only (the parser requires the
  positional / keyword form respectively) — same IR, **needs-decision**
  (acceptance-only, pinned);
- ``pl.cluster(name_hint=1)`` fails late in nanobind
  (``begin_scope(): incompatible function arguments``) instead of at the
  frontend — both lanes reject, **needs-decision** (diagnostic quality,
  pinned).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserSyntaxError
from pypto.language.parser import function_parser, parse_program, program_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir, passes

from ._harness import canonicalize, conformance, program_conformance

build = function_trace


def _scopes(fn: ir.Function, kind: ir.ScopeKind) -> list[ir.ScopeStmt]:
    """Every ``ScopeStmt`` of ``kind`` in the function body (DFS, source order)."""
    found: list[ir.ScopeStmt] = []

    def walk(stmt: ir.Stmt) -> None:
        if isinstance(stmt, ir.ScopeStmt) and stmt.scope_kind == kind:
            found.append(stmt)
        if isinstance(stmt, ir.SeqStmts):
            for child in stmt.stmts:
                walk(child)
        else:
            body = getattr(stmt, "body", None)
            if isinstance(body, ir.Stmt):
                walk(body)

    walk(fn.body)
    return found


def _only_scope(fn: ir.Function, kind: ir.ScopeKind) -> ir.ScopeStmt:
    found = _scopes(fn, kind)
    assert len(found) == 1, f"expected exactly one {kind.name} scope, got {len(found)}"
    return found[0]


# ===========================================================================
# pl.scope — the AUTO/MANUAL runtime scope
# ===========================================================================


def k_scope_manual(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(mode=pl.ScopeMode.MANUAL):
        a = pl.add(a, a)
    return a


def k_scope_manual_fence(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    b = pl.add(a, a)
    with pl.scope(mode=pl.ScopeMode.MANUAL):
        b = pl.mul(b, b)
    return pl.add(b, b)


def k_scope_manual_around_loop(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(mode=pl.ScopeMode.MANUAL):
        for _ in pl.range(4):
            a = pl.add(a, a)
    return a


def test_scope_manual_anchor():
    """``mode=MANUAL``: ``RuntimeScopeStmt(manual=True)`` on both lanes."""
    conformance(k_scope_manual, build)


def test_scope_manual_fence_anchor():
    """Statements around the scope stay in the enclosing body — the scope is
    a transparent region, not a function boundary."""
    conformance(k_scope_manual_fence, build)


def test_scope_manual_around_loop_anchor():
    """A staged loop inside the scope: the SSA pass promotes the carried local
    identically on both lanes."""
    conformance(k_scope_manual_around_loop, build)
    conformance(k_scope_manual_around_loop, build, canonical=True)


class ScopeAutoOff:
    """The AUTO spelling, only legal under ``auto_scope=False``."""

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.scope():
            x = pl.add(x, x)
        return x


class ScopeAutoOffExplicit:
    """``mode=pl.ScopeMode.AUTO`` is the same scope as the bare form."""

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.scope(mode=pl.ScopeMode.AUTO):
            x = pl.add(x, x)
        return x


def test_scope_auto_off_anchor():
    """``with pl.scope():`` under ``auto_scope=False``: both lanes record the
    hand-placed AUTO scope plus the ``auto_scope=False`` function attr."""
    program_conformance(ScopeAutoOff, trace_program)


def test_scope_auto_off_explicit_anchor():
    program_conformance(ScopeAutoOffExplicit, trace_program)


# ===========================================================================
# pl.manual_scope — the MANUAL alias
# ===========================================================================


def k_manual_scope(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        a = pl.add(a, a)
    return a


def k_manual_leak(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        b = pl.add(a, a)
    return pl.add(b, b)


def k_manual_shadow(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    b = pl.add(a, a)
    with pl.manual_scope():
        b = pl.mul(b, b)
    return pl.add(b, b)


def k_manual_around_loop(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        for _ in pl.range(4):
            a = pl.add(a, a)
    return a


def k_manual_in_loop(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    for _ in pl.range(4):
        with pl.manual_scope():
            a = pl.add(a, a)
    return a


def k_manual_cond_bind(a: pl.Tensor[[16], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[16], pl.FP32]:
    if k == 0:
        with pl.manual_scope():
            b = pl.add(a, a)
    else:
        b = pl.add(a, a)
    return b


def k_manual_nested_if(a: pl.Tensor[[16], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        if k == 0:
            a = pl.add(a, a)
        else:
            a = pl.mul(a, a)
    return a


def k_return_in_manual(a: pl.Tensor[[16], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        if k == 0:
            return pl.add(a, a)
        a = pl.mul(a, a)
    return a


def test_manual_scope_anchor():
    """Bare ``pl.manual_scope()`` == ``pl.scope(mode=MANUAL)`` on both lanes
    (``RuntimeScopeStmt(manual=True)``)."""
    conformance(k_manual_scope, build)


def test_manual_scope_leaks_body_locals_anchor():
    """A name defined only in the scope body is visible after it — the parser's
    ``exit_scope(leak_vars=True)`` and the tracing frame's region-local commit
    agree."""
    conformance(k_manual_leak, build)


def test_manual_scope_shadow_anchor():
    """A rebind of an outer local inside the scope updates that local on both
    lanes (not a fresh scope-local)."""
    conformance(k_manual_shadow, build)


def test_manual_scope_around_loop_anchor():
    conformance(k_manual_around_loop, build)
    conformance(k_manual_around_loop, build, canonical=True)


def test_manual_scope_in_loop_anchor():
    """The scope sits inside the staged loop: raw-equal (the region is
    transparent to the loop-carried rebinding).

    The canonical half cannot be asserted here: after SSA the loop-carried
    ``pl.yield_`` lands *inside* the manual ``RuntimeScopeStmt`` body, and
    ``IRPythonPrinter::VisitStmtBody`` special-cases only the
    ``manual_=false`` scope, so the printed text drops the
    ``var = pl.yield_(...)`` assignment LHS and the roundtrip instrument
    rejects it. Both lanes' canonical IR is equal — pinned as F22 in
    ``test_scope_in_loop_canonical_is_printer_gap``."""
    conformance(k_manual_in_loop, build)


def test_scope_in_loop_canonical_is_printer_gap():
    """F22 — a loop-carried value updated inside a *manual* runtime scope
    prints a bare ``pl.yield_(...)`` (the printer's ``VisitStmtBody`` keeps the
    assignment LHS only for an AUTO ``RuntimeScopeStmt``,
    ``src/ir/transforms/python_printer.cpp:2659``), and the parser rejects that
    text. The frontends agree: with the roundtrip instrument bypassed, both
    lanes' post-SSA IR is structurally equal."""
    with passes.PassContext([], passes.VerificationLevel.NONE):
        traced = canonicalize(function_trace(k_manual_in_loop))
        parsed = canonicalize(function_parser(k_manual_in_loop))
        ir.assert_structural_equal(traced, parsed, enable_auto_mapping=True)
        program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    printed = program.as_python()
    assert "pl.yield_(" in printed
    assert "= pl.yield_(" not in printed
    with pytest.raises(ParserSyntaxError, match="assignment-form pl.yield_"):
        parse_program(printed)


def test_manual_scope_cond_bind_anchor():
    """A scope inside one branch of a staged ``if``: the divergent bind
    converges through SSA on both lanes."""
    conformance(k_manual_cond_bind, build)
    conformance(k_manual_cond_bind, build, canonical=True)


def test_manual_scope_nested_if_anchor():
    """A staged ``if`` inside the scope region."""
    conformance(k_manual_nested_if, build)
    conformance(k_manual_nested_if, build, canonical=True)


def test_return_in_manual_scope_anchor():
    """A return inside the scope body: both lanes record the ``ReturnStmt``
    inline at its source position (no region-specific exit lowering)."""
    conformance(k_return_in_manual, build)
    conformance(k_return_in_manual, build, canonical=True)


class ManualInOrch:
    """The documented placement: a runtime scope in an Orchestration entry."""

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.manual_scope():
            x = pl.add(x, x)
        return x


class ManualInInCore:
    """Both lanes accept a runtime scope in a device-side body (the class
    docstring's placement rule is not enforced by either frontend)."""

    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.manual_scope():
            x = pl.add(x, x)
        return x


def test_manual_scope_in_orchestration_anchor():
    program_conformance(ManualInOrch, trace_program)


def test_manual_scope_in_incore_anchor():
    program_conformance(ManualInInCore, trace_program)


# ===========================================================================
# pl.graph — the recordable orchestration region
# ===========================================================================


def k_graph(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph("decoder"):
        a = pl.add(a, a)
    return a


def k_graph_in_loop(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    for _ in pl.range(4):
        with pl.graph("decoder"):
            a = pl.add(a, a)
    return a


def test_graph_anchor():
    """``with pl.graph("decoder"):`` — same ``GraphScopeStmt`` (name hint is
    the region name) on both lanes."""
    conformance(k_graph, build)


def test_graph_in_loop_anchor():
    """The documented usage: a graph region inside the per-layer staged
    loop."""
    conformance(k_graph_in_loop, build)
    conformance(k_graph_in_loop, build, canonical=True)


class GraphThenCluster:
    """A cluster region inside a graph region — the two-outliner nesting."""

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.graph("decoder"):
            with pl.cluster(name_hint="grp"):
                x = pl.add(x, x)
        return x


def test_graph_then_cluster_anchor():
    program_conformance(GraphThenCluster, trace_program)


# ===========================================================================
# pl.cluster — the co-scheduled AIC/AIV region
# ===========================================================================


def k_cluster(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster():
        a = pl.add(a, a)
    return a


def k_cluster_named(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(name_hint="grp"):
        a = pl.add(a, a)
    return a


def k_cluster_nested(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster():
        with pl.cluster(name_hint="inner"):
            a = pl.add(a, a)
    return a


def k_cluster_around_loop(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(name_hint="grp"):
        for _ in pl.range(4):
            a = pl.add(a, a)
    return a


def k_manual_in_cluster(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster():
        with pl.manual_scope():
            a = pl.add(a, a)
    return a


class ClusterInOrch:
    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.cluster(name_hint="grp"):
            x = pl.add(x, x)
        return x


def test_cluster_anchor():
    """Bare ``pl.cluster()`` → ``ClusterScopeStmt(name_hint="")``."""
    conformance(k_cluster, build)


def test_cluster_named_anchor():
    conformance(k_cluster_named, build)


def test_cluster_nested_anchor():
    """A cluster inside a cluster (the parser object has no depth rule for
    this kind on either lane)."""
    conformance(k_cluster_nested, build)


def test_cluster_around_loop_anchor():
    conformance(k_cluster_around_loop, build)
    conformance(k_cluster_around_loop, build, canonical=True)


def test_manual_scope_in_cluster_anchor():
    """A runtime scope nested in a cluster region."""
    conformance(k_manual_in_cluster, build)


def test_cluster_in_orchestration_anchor():
    program_conformance(ClusterInOrch, trace_program)


# ===========================================================================
# No-shared-spelling variants (pinned, no frontend edited)
# ===========================================================================


class GraphDumpTag:
    """F18 anchor: a ``pl.dump_tag`` before a ``pl.graph`` region, whose
    forward-sticky tagged set must merge onto the GraphScopeStmt on both lanes."""

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        pl.dump_tag(x)
        with pl.graph("decoder"):
            x = pl.add(x, x)
        return x


class GraphDumpTagCluster:
    """The same marker on a cluster scope is equivalent on both lanes (the
    tracing cluster hook merges the sticky set — ``constructs/scopes.py:575``)."""

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        pl.dump_tag(x)
        with pl.cluster(name_hint="grp"):
            x = pl.add(x, x)
        return x


def _only_main(fn: ir.Program) -> ir.Function:
    return next(f for f in fn.functions.values() if f.name == "main")


def test_graph_dump_tag_anchor():
    """F18 (fixed) — ``pl.dump_tag`` before a ``pl.graph`` region: the
    forward-sticky tagged tensors merge onto the ``GraphScopeStmt`` on both
    lanes. The parser reaches the merge through ``_parse_scope_body`` →
    ``_merge_forward_sticky_dump``; the tracing ``graph.__trace_enter__``
    carries the same merge, exactly like the ``cluster`` hook already did."""
    program_conformance(GraphDumpTag, trace_program)
    traced = trace_program(GraphDumpTag)
    parsed = program_parser(GraphDumpTag)
    traced_graph = _only_scope(_only_main(traced), ir.ScopeKind.Graph)
    parsed_graph = _only_scope(_only_main(parsed), ir.ScopeKind.Graph)
    assert traced_graph.attrs.get("dump_vars") is not None
    assert parsed_graph.attrs.get("dump_vars") is not None


def test_scope_dump_vars_roundtrip_anchor():
    """F39 (fixed) — a scope-level ``dump_vars`` attr is print-stable. The
    printer emits it as a ``dumps=[...]`` kwarg on ``pl.graph`` / ``pl.cluster``
    — the scope-side selective-dump surface ``pl.at`` already has — and both
    lanes' parser rebuilds the attr, so ``parse(as_python())`` is a fixed point
    on both lanes (they emit the same IR; the printer was the half that
    dropped it)."""
    for cls, kind in ((GraphDumpTag, ir.ScopeKind.Graph), (GraphDumpTagCluster, ir.ScopeKind.Cluster)):
        program = program_parser(cls)
        scope = _only_scope(_only_main(program), kind)
        assert scope.attrs.get("dump_vars") is not None
        printed = program.as_python()
        assert "dumps=[x]" in printed, printed
        reparsed = parse_program(printed)
        reparsed_scope = _only_scope(_only_main(reparsed), kind)
        assert reparsed_scope.attrs.get("dump_vars") is not None
        ir.assert_structural_equal(program, reparsed, enable_auto_mapping=True)
        # The tracing lane emits the same IR, so the printed form is its
        # fixpoint too.
        traced = trace_program(cls)
        ir.assert_structural_equal(traced, parse_program(traced.as_python()), enable_auto_mapping=True)


def k_graph_dumps(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph("decoder", dumps=[a]):
        a = pl.add(a, a)
    return a


def k_cluster_dumps(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(name_hint="grp", dumps=[a]):
        a = pl.add(a, a)
    return a


def k_cluster_dumps_unnamed(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(dumps=[a]):
        a = pl.add(a, a)
    return a


@pytest.mark.parametrize("kernel", [k_graph_dumps, k_cluster_dumps, k_cluster_dumps_unnamed])
def test_explicit_scope_dumps_anchor(kernel: Callable[..., Any]) -> None:
    """F39 (fixed) — the explicit ``dumps=`` spelling on a graph / cluster
    scope: an anchor on both lanes (the parser accepts the kwarg and records
    ``dump_vars``; the tracing descriptor does the same), and the printed form
    re-parses to the same function."""
    conformance(kernel, build)
    traced = function_trace(kernel)
    program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    printed = program.as_python()
    assert "dumps=[a]" in printed, printed
    ir.assert_structural_equal(program, parse_program(printed), enable_auto_mapping=True)


def test_cluster_scope_keeps_sticky_dump_vars():
    """The cluster counterpart is equivalent — the divergence is graph-only."""
    traced = trace_program(GraphDumpTagCluster)
    parsed = program_parser(GraphDumpTagCluster)
    traced_cluster = _only_scope(_only_main(traced), ir.ScopeKind.Cluster)
    parsed_cluster = _only_scope(_only_main(parsed), ir.ScopeKind.Cluster)
    assert traced_cluster.attrs.get("dump_vars") is not None
    assert parsed_cluster.attrs.get("dump_vars") is not None
    ir.assert_structural_equal(traced, parsed, enable_auto_mapping=True)


def k_scope_mode_int(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(mode=1):  # type: ignore[arg-type]
        a = pl.add(a, a)
    return a


def k_scope_mode_str(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(mode="MANUAL"):  # type: ignore[arg-type]
        a = pl.add(a, a)
    return a


@pytest.mark.parametrize("kernel", [k_scope_mode_int, k_scope_mode_str])
def test_scope_mode_is_not_validated_by_tracing(kernel: Callable[..., Any]):
    """F19 — a non-enum ``mode=`` silently selects AUTO on the tracing lane
    (``constructs/scopes.py:273``: ``manual=(self.mode == ScopeMode.MANUAL)``
    is False for ``1`` / ``"MANUAL"``, and the AUTO gate at ``:269`` passes
    because ``self.mode == ScopeMode.AUTO`` is False too), skipping the
    ``auto_scope=False`` gate the parser enforces. Both the emitted scope kind
    and the print -> reparse fixpoint diverge, so no anchor exists."""
    traced = function_trace(kernel)
    scope = _only_scope(traced, ir.ScopeKind.Runtime)
    assert isinstance(scope, ir.RuntimeScopeStmt)
    assert scope.manual is False  # silently AUTO, in a default auto_scope=True body
    with pytest.raises(ParserSyntaxError, match="Expected pl.ScopeMode"):
        function_parser(kernel)
    program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    with pytest.raises(ParserSyntaxError, match="auto_scope=False"):
        parse_program(program.as_python())


def k_manual_in_manual(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope():
        with pl.manual_scope():
            a = pl.add(a, a)
    return a


def k_scope_manual_in_manual(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(mode=pl.ScopeMode.MANUAL):
        with pl.scope(mode=pl.ScopeMode.MANUAL):
            a = pl.add(a, a)
    return a


@pytest.mark.parametrize("kernel", [k_manual_in_manual, k_scope_manual_in_manual])
def test_nested_manual_scope_rejected_by_both_lanes(kernel: Callable[..., Any]):
    """F20 (fixed) — a manual scope may not nest inside another manual scope
    (the runtime forbids it). Both lanes enforce the same rule: the parser
    through ``_manual_scope_depth`` (``ast_parser.py:4155``), the tracing
    descriptors through ``ctx.is_inside_manual_scope()`` (the builder's
    open-scope mirror), so ``pl.manual_scope()`` and
    ``pl.scope(mode=pl.ScopeMode.MANUAL)`` reject the nesting identically."""
    match = "manual scope may not be nested"
    with pytest.raises(ParserSyntaxError, match=match):
        function_trace(kernel)
    with pytest.raises(ParserSyntaxError, match=match):
        function_parser(kernel)


class AutoInManual:
    """F20 (second arm, fixed): under ``auto_scope=False`` the AUTO gate passes,
    but an AUTO scope inside a manual scope is still rejected by both lanes
    (the runtime forbids AUTO-in-MANUAL)."""

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.manual_scope():
            with pl.scope():
                x = pl.add(x, x)
        return x


def test_auto_scope_in_manual_rejected_by_both_lanes():
    """F20 (second arm, fixed) — the parser's ``_manual_scope_depth`` gate at
    ``ast_parser.py:4177`` and the tracing ``scope`` hook's
    ``is_inside_manual_scope()`` gate both reject the AUTO-in-MANUAL spelling
    with the same message."""
    match = "AUTO scope may not be nested inside a manual scope"
    with pytest.raises(ParserSyntaxError, match=match):
        trace_program(AutoInManual)
    with pytest.raises(ParserSyntaxError, match=match):
        program_parser(AutoInManual)


class ManualInAuto:
    """The complement the F20 gate must not over-reach: under
    ``auto_scope=False`` a MANUAL scope inside an AUTO one is legal (only
    MANUAL-in-MANUAL and AUTO-in-MANUAL are forbidden by the runtime), so the
    gate keys on the enclosing scope's *manual* flag."""

    @pl.function(type=pl.FunctionType.Orchestration, auto_scope=False)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.scope():
            with pl.manual_scope():
                x = pl.add(x, x)
        return x


def test_manual_scope_inside_auto_scope_anchor():
    """AUTO-then-MANUAL stays a shared accepted spelling after F20: the new
    gate reads the open-scope mirror's manual flags, not the presence of any
    runtime scope."""
    program_conformance(ManualInAuto, trace_program)


def k_graph_bad_name(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph("has space"):
        a = pl.add(a, a)
    return a


def k_cluster_bad_name(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(name_hint="has space"):
        a = pl.add(a, a)
    return a


@pytest.mark.parametrize(
    ("kernel", "kind", "match"),
    [
        (k_graph_bad_name, ir.ScopeKind.Graph, "valid non-keyword identifier"),
        (k_cluster_bad_name, ir.ScopeKind.Cluster, "valid non-keyword identifier"),
    ],
)
def test_scope_name_hint_validation_is_parser_only(
    kernel: Callable[..., Any], kind: ir.ScopeKind, match: str
):
    """F21 — ``_parse_scope_name_hint`` (``ast_parser.py:4039``) validates the
    graph region name / cluster ``name_hint`` as a non-keyword identifier; the
    tracing descriptors only check emptiness (graph) or nothing (cluster), so
    a non-identifier reaches the IR and the printed form fails to re-parse."""
    traced = function_trace(kernel)
    assert _only_scope(traced, kind).name_hint == "has space"
    with pytest.raises(ParserSyntaxError, match=match):
        function_parser(kernel)
    program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    with pytest.raises(ParserSyntaxError, match=match):
        parse_program(program.as_python())


def k_graph_keyword_name(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph(name="decoder"):  # type: ignore[call-arg]
        a = pl.add(a, a)
    return a


def k_scope_positional_mode(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.scope(pl.ScopeMode.MANUAL):  # type: ignore[misc]
        a = pl.add(a, a)
    return a


@pytest.mark.parametrize(
    ("kernel", "kind"),
    [(k_graph_keyword_name, ir.ScopeKind.Graph), (k_scope_positional_mode, ir.ScopeKind.Runtime)],
)
def test_lenient_generic_scope_spellings_are_tracing_only(kernel: Callable[..., Any], kind: ir.ScopeKind):
    """needs-decision — ``pl.graph(name="decoder")`` (keyword) and
    ``pl.scope(pl.ScopeMode.MANUAL)`` (positional) are accepted by the tracing
    signatures and rejected by the parser's explicit arg forms. The IR is the
    same as the canonical spelling and re-parses cleanly, so this is an
    acceptance-only gap with no doc deciding it."""
    traced = function_trace(kernel)
    _only_scope(traced, kind)
    with pytest.raises(ParserSyntaxError):
        function_parser(kernel)
    program = ir.Program([traced], f"p_{traced.name}", ir.Span.unknown())
    parse_program(program.as_python())  # the canonical spelling round-trips


def k_cluster_int_name_hint(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster(name_hint=1):  # type: ignore[arg-type]
        a = pl.add(a, a)
    return a


def test_cluster_non_string_name_hint_fails_late():
    """needs-decision — a non-string ``name_hint`` is rejected by both lanes,
    but tracing's rejection surfaces from nanobind
    (``begin_scope(): incompatible function arguments``) rather than as a
    frontend diagnostic, so an anchor is impossible (no accepted spelling)."""
    with pytest.raises(TypeError, match="incompatible function arguments"):
        function_trace(k_cluster_int_name_hint)
    with pytest.raises(ParserSyntaxError, match="must be a string literal"):
        function_parser(k_cluster_int_name_hint)


# ===========================================================================
# Rejection parity (both lanes loud on the same source — no anchor possible)
# ===========================================================================


def k_graph_in_graph(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph("outer"):
        with pl.graph("inner"):
            a = pl.add(a, a)
    return a


def k_graph_in_at(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        with pl.graph("g"):
            a = pl.add(a, a)
    return a


def k_graph_in_cluster(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster():
        with pl.graph("g"):
            a = pl.add(a, a)
    return a


@pytest.mark.parametrize("kernel", [k_graph_in_graph, k_graph_in_at, k_graph_in_cluster])
def test_graph_placement_rejected_by_both_lanes(kernel: Callable[..., Any]):
    """The graph placement rules are mirrored verbatim (the tracing ``graph``
    hook's forbidden-kind / nesting checks mirror ``_parse_graph_scope``)."""
    match = "cannot be nested inside"
    with pytest.raises(ParserSyntaxError, match=match):
        function_trace(kernel)
    with pytest.raises(ParserSyntaxError, match=match):
        function_parser(kernel)


class GraphInInCore:
    @pl.function(type=pl.FunctionType.InCore)
    def main(self, x: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
        with pl.graph("g"):
            x = pl.add(x, x)
        return x


def test_graph_in_device_body_rejected_by_both_lanes():
    """A graph region in a device-side body: both lanes name the function
    type."""
    match = "not valid inside a InCore"
    with pytest.raises(ParserSyntaxError, match=match):
        trace_program(GraphInInCore)
    with pytest.raises(ParserSyntaxError, match=match):
        program_parser(GraphInInCore)


def k_graph_empty_name(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph(""):
        a = pl.add(a, a)
    return a


def test_graph_empty_name_rejected_by_both_lanes():
    """An empty region name is loud on both lanes (different exception types:
    the tracing ctor raises before the trace, the parser at its name check)."""
    with pytest.raises(ValueError, match="must not be empty"):
        function_trace(k_graph_empty_name)
    with pytest.raises(ParserSyntaxError, match="must not be empty"):
        function_parser(k_graph_empty_name)


def k_scope_as_target(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.manual_scope() as _t:  # type: ignore[var-annotated]
        a = pl.add(a, a)
    return a


def k_cluster_as_target(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.cluster() as _t:  # type: ignore[var-annotated]
        a = pl.add(a, a)
    return a


def k_graph_as_target(a: pl.Tensor[[16], pl.FP32]) -> pl.Tensor[[16], pl.FP32]:
    with pl.graph("g") as _t:  # type: ignore[var-annotated]
        a = pl.add(a, a)
    return a


@pytest.mark.parametrize("kernel", [k_scope_as_target, k_cluster_as_target, k_graph_as_target])
def test_scope_as_target_rejected_by_both_lanes(kernel: Callable[..., Any]):
    """An ``as`` target binds a producer TaskId, which only ``pl.at`` /
    ``pl.spmd`` have: both lanes reject it on every generic scope."""
    with pytest.raises(TypeError, match="is not supported"):
        function_trace(kernel)
    with pytest.raises(ParserSyntaxError, match="is not supported"):
        function_parser(kernel)
