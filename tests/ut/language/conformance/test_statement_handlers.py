# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: the statement-position handlers + function attrs + optimization
entries (R12).

Six surfaces share one property — the directive is *consumed at collection* and
leaves no statement of its own:

- ``pl.static_print`` / ``pl.static_assert`` — parser-only diagnostics
  (``_handle_static_print`` ``ast_parser.py:3001`` /
  ``_handle_static_assert`` ``:3028``) against the tracing no-op
  (``compat/diagnostics.py``).
- ``pl.dump_tag`` — a forward-sticky Var set (parser
  ``_handle_dump_tag`` ``ast_parser.py:3085`` + ``_dump_tagged_vars`` merge in
  ``_parse_kernel_call`` ``:7001``; tracing ``Sema.record_dump_tag``
  ``semantics.py:363`` + ``DumpTagMarker`` consumed by ``bare_stmt`` and merged
  by ``Marker.__call__`` ``op/marker.py:340``).
- ``pl.set_cache_policy`` — attaches to the enclosing scope's
  ``cache_policy_vars`` attr (parser ``_collect_scope_cache_policy_decls`` /
  ``_handle_set_cache_policy`` ``ast_parser.py:3170``; tracing
  ``Sema.route_cache_policy`` ``semantics.py:1095`` +
  ``_DeferredDeclarableScope``).
- ``pl.func_attr`` — merges a dict into ``Function.attrs`` (parser
  ``_consume_func_attr_prologue`` ``ast_parser.py:1063`` +
  ``_parse_func_attr_value`` ``:1174``; tracing ``func_attr`` /
  ``_FuncAttrMarker`` ``constructs/func_attrs.py`` →
  ``TraceContext.collect_func_attr`` ``protocol.py:383``).
- ``optimizations=[...]`` — the ``pl.at`` / ``pl.spmd`` entry list (parser
  ``_parse_optimizations_list`` ``ast_parser.py:3821`` /
  ``_parse_spmd_optimizations_list`` ``:3908``; tracing
  ``_fold_optimization_entries`` ``constructs/scopes.py:623``).

**Regime.** A statement handler that emits no IR makes both lanes' bodies
identical: the anchors are asserted **raw** (no pass state). The staged-control
anchors (a diagnostic inside a ``pl.range`` loop, ``set_cache_policy`` in the
for-form ``pl.spmd`` body) additionally assert post-``canonicalize()`` — the SSA
pass has loop-carried values to derive there. The ``optimizations=`` anchors on
a scope are **raw + canonical** for the same reason R11's ``pl.at`` anchors are:
the scope's attrs are not pass state, but the surrounding loop-carried value is.
Linear-function attrs (``pl.func_attr``) are raw-only: ``convert_to_ssa`` does
not touch ``Function.attrs``.

**The one intended divergence is the parser-only diagnostic.** ``static_print`` /
``static_assert`` are the *AST parser's* diagnostics: under tracing the call is
a **silent runtime no-op** — nothing prints, and a failing ``static_assert``
does not raise (``docs/design/00-tracing.md:111``; the user doc
``docs/en/user/language/05-directives.md:18-25``; recorded as the by-design gap
**K4** in ``docs/migration-known-gaps.md``; ``docs/design/04-migration-guide.md``
§4.4). The IR is identical on both lanes (nothing is emitted), so the anchor
lands green and the *effect* divergence is pinned rather than anchored. Three
acceptance-only gaps have no doc sanction and are **needs-decision** (the
``static_print`` arity check; ``set_cache_policy`` must precede the scope's
first statement on the tracing lane; ``func_attr`` takes a pre-built dict on the
tracing lane).
"""

from __future__ import annotations

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserError, ParserSyntaxError
from pypto.language.parser import function_parser, program_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir

from ._harness import conformance, program_conformance

build = function_trace

#: ``pl.split(slot_num=...)`` is deprecated but both lanes still accept it; the
#: warning is the entry constructor's, not the construct's.
pytestmark = pytest.mark.filterwarnings("ignore:pl.split")


# ===========================================================================
# static_print / static_assert — parser-only diagnostics, no IR on either lane
# ===========================================================================


def k_static_print_label(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_print("label", x)
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_static_print_fstring(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    pl.static_print(f"t = {t}")
    pl.store(t, [0, 0], x)
    return x


def k_static_assert_true(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_assert(True)
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_static_assert_msg(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_assert(True, "always true")
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_static_assert_literal_cmp(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_assert(64 > 32)  # noqa: PLR0133 - the literal comparison is the point
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_static_diag_in_loop(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.range(2):
        pl.static_print("iter", x)
        pl.static_assert(True)
        t = pl.load(x, [i, 0], [1, 64])
        pl.store(t, [i, 0], x)
    return x


def test_static_print_label_anchor():
    """A string label + a tensor: both lanes consume the marker and emit no IR —
    the remaining body is raw-equal."""
    conformance(k_static_print_label, build)


def test_static_print_fstring_anchor():
    """The parser's f-string formatter lowers ``{expr}`` to the same (absent) IR
    under tracing, where the call is a no-op."""
    conformance(k_static_print_fstring, build)


def test_static_assert_true_anchor():
    conformance(k_static_assert_true, build)


def test_static_assert_msg_anchor():
    conformance(k_static_assert_msg, build)


def test_static_assert_literal_cmp_anchor():
    """A host-evaluable comparison literal: the parser's ``try_eval_expr`` path
    passes; the tracing no-op ignores it."""
    conformance(k_static_assert_literal_cmp, build)


def test_static_diag_in_loop_anchor():
    """Both diagnostics inside a staged loop (raw + canonical — the loop carries
    the body's value)."""
    conformance(k_static_diag_in_loop, build)
    conformance(k_static_diag_in_loop, build, canonical=True)


# —— the intended divergence: the check/print side effect is parser-only ——


def k_false_guard(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_assert(False, "nope")
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_static_print_noargs(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.static_print()
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def test_static_assert_false_is_silent_under_tracing():
    """**Intended divergence** — ``pl.static_print`` / ``pl.static_assert`` are
    the AST parser's diagnostics: under tracing they are silent runtime no-ops
    (``docs/design/00-tracing.md:111``; ``docs/en/user/language/05-directives.md:18-25``;
    the by-design gap **K4**, ``docs/migration-known-gaps.md``; ``docs/design/04-migration-guide.md``
    §4.4). A false condition therefore collects fine under tracing and raises in
    the parser; the emitted IR is the same (nothing) on both lanes. Pinned, not
    an anchor."""
    traced = function_trace(k_false_guard)
    assert "pl.static_assert(" not in str(traced)
    with pytest.raises(ParserError, match="static_assert failed: nope"):
        function_parser(k_false_guard)


def test_static_print_argument_validation_is_parser_only():
    """**needs-decision** (acceptance-only). ``pl.static_print()`` with no
    argument: the parser rejects (``requires at least 1 argument``), the tracing
    no-op accepts (the ``*args`` signature). No doc decides the arity check; the
    no-op contract explains the behavior but not the validation."""
    assert function_trace(k_static_print_noargs) is not None
    with pytest.raises(ParserSyntaxError, match=r"requires at least 1 argument"):
        function_parser(k_static_print_noargs)


# ===========================================================================
# pl.dump_tag — forward-sticky Var set, merged into a consuming kernel call
# ===========================================================================


class DumpTagForwardSticky:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        pl.dump_tag(x)
        pl.dump_tag(out)
        self.kernel(x, out)
        return out


class DumpTagRebindInvalidation:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[64, 64], pl.FP32], out: pl.Tensor[[64, 64], pl.FP32]
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        pl.dump_tag(x)
        y = pl.add(x, 1.0)
        self.kernel(y, out)  # y is a new Var — the prior tag does not cover it
        return out


def _kernel_call(fn: ir.Function) -> ir.Call:
    calls = [s.expr for s in fn.body if isinstance(s, ir.EvalStmt) and isinstance(s.expr, ir.Call)]
    assert len(calls) == 1, f"expected one kernel call, got {len(calls)}"
    return calls[0]


def _only_main(prog: ir.Program) -> ir.Function:
    return next(f for f in prog.functions.values() if f.name == "main")


def test_dump_tag_forward_sticky_anchor():
    """Two tags before one kernel call: both tagged Vars are merged into the
    Call's ``dump_vars`` attr, by Var identity, on both lanes (raw + program
    canonical). The explicit attr assertions keep the anchor non-vacuous."""
    program_conformance(DumpTagForwardSticky, trace_program)
    program_conformance(DumpTagForwardSticky, trace_program, canonical=True)

    traced = _kernel_call(_only_main(trace_program(DumpTagForwardSticky)))
    parsed = _kernel_call(_only_main(program_parser(DumpTagForwardSticky)))
    assert len(traced.attrs["dump_vars"]) == 2
    assert len(parsed.attrs["dump_vars"]) == 2


def test_dump_tag_rebind_invalidation_anchor():
    """A tag on ``x`` does not cover the derived ``y``: neither lane merges a
    ``dump_vars`` attr onto the consumer (the doc's Var-identity rule)."""
    program_conformance(DumpTagRebindInvalidation, trace_program)

    traced = _kernel_call(_only_main(trace_program(DumpTagRebindInvalidation)))
    parsed = _kernel_call(_only_main(program_parser(DumpTagRebindInvalidation)))
    assert "dump_vars" not in traced.attrs
    assert "dump_vars" not in parsed.attrs


def _dump_tag_in_incore(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.dump_tag(x)
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def test_dump_tag_outside_orchestration_rejected_by_both_lanes():
    """The marker is Orchestration/Inline-only; both lanes reject a kernel-body
    use (the parser at ``_handle_dump_tag`` ``ast_parser.py:3085``, tracing in
    ``Sema.record_dump_tag`` ``semantics.py:368``)."""
    with pytest.raises(ParserSyntaxError, match="only valid inside an Orchestration or Inline"):
        function_parser(_dump_tag_in_incore)
    with pytest.raises(TypeError, match="only valid inside an Orchestration or Inline"):
        function_trace(_dump_tag_in_incore)


# ===========================================================================
# pl.set_cache_policy — scope attr, consumed at collection
# ===========================================================================


def k_cache_policy_at(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_cache_policy_spmd_with(
    x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.spmd(4):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        i = pl.tile.get_block_idx()
        t = pl.load(x, [i, 0], [1, 64])
        pl.store(t, [i, 0], out)
    return out


def k_cache_policy_spmd_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.spmd(4):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        t = pl.load(x, [i, 0], [1, 64])
        pl.store(t, [i, 0], x)
    return x


def k_cache_policy_dup(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_cache_policy_at_anchor():
    """The marker joins the ``pl.at`` scope's ``cache_policy_vars`` attr before
    the body runs — raw-equal."""
    conformance(k_cache_policy_at, build)


def test_cache_policy_spmd_with_anchor():
    """The same marker on the with-form ``pl.spmd`` InCore carrier."""
    conformance(k_cache_policy_spmd_with, build)


def test_cache_policy_spmd_for_anchor():
    """The for-form ``pl.spmd`` body (raw + canonical — the loop carries the
    body's value)."""
    conformance(k_cache_policy_spmd_for, build)
    conformance(k_cache_policy_spmd_for, build, canonical=True)


def test_cache_policy_duplicate_anchor():
    """A repeated declaration for the same binding *and the same policy* is
    redundant, not an error — both lanes keep the first
    (``docs/en/dev/language/05-cache-policy.md`` § *Where the declaration may be
    written*; the parser at ``_collect_scope_cache_policy_decls``, tracing at
    ``_buffer_cache_policy``). A differing policy is a contradiction on both
    lanes (pinned below)."""
    conformance(k_cache_policy_dup, build)


# —— no-shared-spelling variants (pinned) ——


def k_cache_policy_conflict(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        pl.set_cache_policy(x, pl.CachePolicy.DEFAULT)
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_cache_policy_outside(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_cache_policy_late(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        pl.store(t, [0, 0], x)
    return x


def k_cache_policy_graph(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.graph("g"):
        pl.set_cache_policy(x, pl.CachePolicy.BYPASS)
        y = pl.add(x, 1.0)
    return y


def k_cache_policy_first_arg_not_name(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.set_cache_policy(x[0], pl.CachePolicy.BYPASS)  # type: ignore[arg-type]
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_cache_policy_must_precede_scope_first_statement():
    """**needs-decision** (acceptance-only, same IR). The parser pre-scans the
    scope body's top-level statements and hoists the marker
    (``docs/en/dev/language/05-cache-policy.md`` § *Where the declaration may be
    written*), so a declaration after the body's first statement parses; the
    tracing lane is the stricter, loud form (``semantics.py:1128`` requires the
    declaration to precede the deferred scope's first IR-emitting statement)
    and rejects it. No doc sanction is stated for the ordering rule."""
    assert function_parser(k_cache_policy_late) is not None
    with pytest.raises(ParserSyntaxError, match="must precede the scope's first statement"):
        function_trace(k_cache_policy_late)


def test_cache_policy_conflicting_policies_rejected_by_both_lanes():
    """A differing policy for the same binding is a contradiction, not a
    redundancy — both lanes reject it with the same message
    (``_buffer_cache_policy``, ``constructs/scopes.py:1758``)."""
    with pytest.raises(ParserSyntaxError, match="conflicting policies"):
        function_parser(k_cache_policy_conflict)
    with pytest.raises(ParserSyntaxError, match="conflicting policies"):
        function_trace(k_cache_policy_conflict)


def test_cache_policy_outside_scope_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match="must be a standalone statement directly inside"):
        function_parser(k_cache_policy_outside)
    with pytest.raises(ParserSyntaxError, match="must be a standalone statement directly inside"):
        function_trace(k_cache_policy_outside)


def test_cache_policy_non_declarable_scope_rejected_by_both_lanes():
    """A ``pl.graph`` body has no GM read to govern: both lanes reject with the
    same "nothing to attach to" message (the parser's ``_collect_scope_cache_policy_decls``,
    tracing's ``route_cache_policy``)."""
    with pytest.raises(ParserSyntaxError, match="nothing to attach to on this Graph scope"):
        function_parser(k_cache_policy_graph)
    with pytest.raises(ParserSyntaxError, match="nothing to attach to on this Graph scope"):
        function_trace(k_cache_policy_graph)


def test_cache_policy_first_argument_must_be_a_name():
    """Both lanes reject a non-Name first argument (parser:
    ``first argument must be a bare variable name``; tracing: the marker names
    no binding). Pinned, not an anchor — the error types differ."""
    with pytest.raises(ParserSyntaxError, match="must be a bare variable name"):
        function_parser(k_cache_policy_first_arg_not_name)
    with pytest.raises(Exception, match="names a bound tensor variable"):
        function_trace(k_cache_policy_first_arg_not_name)


# ===========================================================================
# pl.func_attr — a prologue directive merged into Function.attrs
# ===========================================================================


def k_func_attr_split(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"split": pl.SplitMode.UP_DOWN})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_split_none(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"split": pl.SplitMode.NONE})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_scalars(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"flag": True, "name": "abc", "ratio": 1.5, "n": 4})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_directions(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"dirs": [pl.adir.input, pl.adir.output]})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_param_ref(
    w: pl.Tensor[[64, 64], pl.FP32], x: pl.Tensor[[64, 64], pl.FP32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"stationary": w})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_ref_list(
    w: pl.Tensor[[64, 64], pl.FP32], x: pl.Tensor[[64, 64], pl.FP32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"refs": [w]})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_multi(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"a": 1})
    pl.func_attr({"b": 2})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def test_func_attr_split_anchor():
    """The enum spelling: both lanes store the lowered integer under ``split``."""
    conformance(k_func_attr_split, build)


def test_func_attr_split_none_anchor():
    """``SplitMode.NONE`` is dropped on both lanes (a non-canonical spelling of
    "no split" the printer filters) — the function carries no ``split`` attr."""
    conformance(k_func_attr_split_none, build)
    traced = function_trace(k_func_attr_split_none)
    assert "split" not in traced.attrs


def test_func_attr_scalars_anchor():
    """Host scalars (bool/int/float/str) pass through as themselves."""
    conformance(k_func_attr_scalars, build)


def test_func_attr_directions_anchor():
    """A ``pl.adir`` list reconstructs to ``ArgDirection`` values on the parser
    and coerces to the same list on the tracing lane."""
    conformance(k_func_attr_directions, build)


def test_func_attr_param_ref_anchor():
    """A bare name is a parameter reference: both lanes store the param ``Var``
    (the parser's ``_parse_attr_value``, tracing's ``_coerce_attr_value``)."""
    conformance(k_func_attr_param_ref, build)


def test_func_attr_ref_list_anchor():
    """A list of parameter references — the one non-reserved list shape both
    lanes accept."""
    conformance(k_func_attr_ref_list, build)


def test_func_attr_multiple_directives_anchor():
    """Two prologue directives merge (the builder's unique-key check permits
    distinct keys) — raw-equal."""
    conformance(k_func_attr_multi, build)


# —— no-shared-spelling variants (pinned) ——


def k_func_attr_late(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    pl.func_attr({"a": 1})
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_dup(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"a": 1})
    pl.func_attr({"a": 2})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_decorator_only(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"auto_scope": False})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def k_func_attr_int_list(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr({"dims": [1, 2, 3]})
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def test_func_attr_must_be_first_statement():
    """The prologue rule is shared verbatim: a directive after another statement
    is rejected by both lanes with the same message."""
    with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):
        function_parser(k_func_attr_late)
    with pytest.raises(ParserSyntaxError, match="must appear before every other statement"):
        function_trace(k_func_attr_late)


def test_func_attr_duplicate_key_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match="Duplicate function attribute 'a'"):
        function_parser(k_func_attr_dup)
    with pytest.raises(ParserSyntaxError, match="Duplicate function attribute 'a'"):
        function_trace(k_func_attr_dup)


def test_func_attr_decorator_only_key_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match="cannot be set with pl.func_attr"):
        function_parser(k_func_attr_decorator_only)
    with pytest.raises(ParserSyntaxError, match="cannot be set with pl.func_attr"):
        function_trace(k_func_attr_decorator_only)


def test_func_attr_int_list_rejected_by_both_lanes():
    """An arbitrary-key integer list reaches the builder's attr coercion on both
    lanes and is rejected there (not a frontend divergence)."""
    with pytest.raises(Exception, match="Unsupported list element type for key: dims"):
        function_parser(k_func_attr_int_list)
    with pytest.raises(Exception, match="Unsupported list element type for key: dims"):
        function_trace(k_func_attr_int_list)


# —— the pre-built-dict acceptance gap ——


_PREBUILT_ATTRS = {"a": 1}


def k_func_attr_prebuilt(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    pl.func_attr(_PREBUILT_ATTRS)
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], x)
    return x


def test_func_attr_prebuilt_dict_is_tracing_only():
    """**needs-decision** (acceptance-only, same IR). The parser requires a dict
    *literal* (``_merge_func_attr_directive`` ``ast_parser.py:1123``, so it can
    resolve bare-name references syntactically); the tracing marker receives the
    runtime dict and accepts any mapping. ``function_trace``'s attrs equal the
    literal spelling's parse."""
    traced = function_trace(k_func_attr_prebuilt)
    assert traced.attrs == {"a": 1}
    with pytest.raises(ParserSyntaxError, match="argument must be a dict literal"):
        function_parser(k_func_attr_prebuilt)


# ===========================================================================
# optimizations=[...] — the fully-qualified spellings and the deprecated alias
# ===========================================================================


def k_opt_fq_split(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.optimizations.split(pl.SplitMode.UP_DOWN)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_opt_fq_slot(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.optimizations.cross_core_slot(slot_num=3)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_opt_deprecated_slot(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.NONE, slot_num=3)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_optimizations_fully_qualified_split_anchor():
    """The ``pl.optimizations.split(...)`` spelling (accepted by the parser's
    ``_is_pl_optimization_call``) folds to the same scope ``split`` metadata
    (raw + canonical)."""
    conformance(k_opt_fq_split, build)
    conformance(k_opt_fq_split, build, canonical=True)


def test_optimizations_fully_qualified_slot_anchor():
    """The ``pl.optimizations.cross_core_slot(...)`` spelling folds to the same
    ``slot_num`` scope attr."""
    conformance(k_opt_fq_slot, build)


def test_optimizations_deprecated_split_slot_anchor():
    """``pl.split(mode, slot_num=N)`` feeds the same ``slot_num`` attr on both
    lanes (the deprecated alias; the parser's ``_try_parse_pl_split`` and tracing's
    ``Split.slot_num``)."""
    conformance(k_opt_deprecated_slot, build)


# —— rejection parities ——


def k_opt_duplicate_split(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN), pl.split(pl.SplitMode.LEFT_RIGHT)],
    ):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_opt_bad_entry(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=["nope"]):  # type: ignore[list-item]
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_opt_split_hierarchy(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(
        level=pl.Level.HOST,
        role=pl.Role.Orchestrator,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN)],
    ):
        y = pl.add(x, 1.0)
    return y


def test_optimizations_duplicate_split_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match=r"Duplicate 'pl.split"):
        function_parser(k_opt_duplicate_split)
    with pytest.raises(TypeError, match="Duplicate pl.split"):
        function_trace(k_opt_duplicate_split)


def test_optimizations_bad_entry_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match="Unsupported entry"):
        function_parser(k_opt_bad_entry)
    with pytest.raises(TypeError, match="Unsupported entry"):
        function_trace(k_opt_bad_entry)


def test_optimizations_split_on_hierarchy_rejected_by_both_lanes():
    with pytest.raises(ParserSyntaxError, match="only supported with level=pl.Level.CORE_GROUP"):
        function_parser(k_opt_split_hierarchy)
    with pytest.raises(TypeError, match="only supported with level=pl.Level.CORE_GROUP"):
        function_trace(k_opt_split_hierarchy)
