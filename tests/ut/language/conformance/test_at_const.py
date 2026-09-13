# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""`pl.at` + the operand boundary deltas + the exhaustive family
anchors.

The collection path is type-complete: anchors across the op families (tile,
tensor, unified/matmul, scalar, const) prove ``is_staged`` +
``bind``/``bare_stmt`` handle every facade kind uniformly — the ops are
already shared, so per-op implementation is not the point.

R11 adds the remaining ``pl.at`` **kwarg surface** (the anchors below):
``allow_early_resolve``, ``windowize``, the ``cross_core_slot`` optimization
entry, ``deps=``/``no_dep_args=``/``dumps=``, the ``as tid`` capture, the
positional ``level`` form, ``pl.at`` nested in a non-CORE_GROUP ``pl.at``, and
``pl.at`` inside staged control flow. All are asserted **raw**, plus
post-``canonicalize()`` where a ``deps=`` chain or an enclosing staged loop
gives the SSA pass loop-carried values to derive. The descriptor side is
``at.__trace_enter__`` / ``_collect_scope_attrs``
(``constructs/scopes.py:1624`` / ``:1582``) and the parser's
``_parse_at_kwargs`` / ``_dispatch_at_keyword`` / ``_parse_at_meta``
(``ast_parser.py:3707`` / ``:3743``); the attr order is shared, so a print →
reparse cycle reproduces the list.

Four ``pl.at`` kwarg spellings have no shared shape — the parser validates more
than the tracing descriptor does, so each is pinned rather than anchored:
``name_hint=`` is not checked as an identifier (**F25**, a bug: the traced text
cannot be re-parsed; the R10 sibling of the ``graph`` / ``cluster`` gap filed as
F21), and the bool-literal / list-literal / list-shape checks
(``windowize=1``, ``allow_early_resolve=1``, a pre-built ``optimizations=``
list, ``deps=None``) are parser-only acceptance gaps with **needs-decision** and
no IR consequence.
"""

import pypto.language as pl
import pytest
from pypto.language.diagnostics import ParserError, ParserSyntaxError
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_trace
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import ir

from ._harness import conformance

build = function_trace


# —— the pl.at scope ——


def k_at_incore(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        s = pl.add(t, t)
        pl.store(s, [0, 0], x)
    return x


def k_at_hierarchy(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_split(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(pl.SplitMode.UP_DOWN)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_at_incore_anchor():
    conformance(k_at_incore, build)


def test_at_hierarchy_anchor():
    conformance(k_at_hierarchy, build)


def test_at_split_anchor():
    conformance(k_at_split, build)


# —— the typed-constant reification (pl.const) ——


def k_const_int(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    c = pl.const(0, pl.FP32)
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, c)
    pl.store(s, [0, 0], x)
    return x


def k_const_float(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    c = pl.const(1.5, pl.FP32)
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.add(t, c)
    pl.store(s, [0, 0], x)
    return x


def k_const_index(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    # INDEX is the offset dtype: an INDEX constant is legal as a load offset,
    # while a tile/tensor scalar instruction would reject it (upstream #2132 —
    # the operand boundary asks for an explicit cast otherwise).
    c = pl.const(0, pl.INDEX)
    t = pl.load(x, [c, 0], [64, 64])
    s = pl.add(t, t)
    pl.store(s, [0, 0], x)
    return x


def test_const_int_anchor():
    conformance(k_const_int, build)


def test_const_float_anchor():
    conformance(k_const_float, build)


def test_const_index_anchor():
    conformance(k_const_index, build)


def test_const_reified_to_scalar():
    """pl.const yields a typed Scalar wrapping the Const node (the executed
    path's reification of the parser's intercept) — a bind therefore
    emits a typed ``LetStmt``, satisfying the typed staged initialization
    requirement."""
    c = pl.const(0, pl.FP32)
    assert isinstance(c, pl.Scalar)
    expr = c.unwrap()
    assert isinstance(expr, ir.ConstInt) and expr.dtype == pl.FP32
    cf = pl.const(1.5, pl.FP16)
    assert isinstance(cf.unwrap(), ir.ConstFloat)
    assert isinstance(pl.const(0, pl.INDEX).unwrap(), ir.ConstInt)


# —— the exhaustive family anchors (the collection path is uniform) ——


def k_tile_chain(x: pl.Tensor[[64, 128], pl.FP16]) -> pl.Tensor[[64, 128], pl.FP16]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.tile.muls(t, 2.0)
    r = pl.tile.adds(s, 1.0)
    pl.store(r, [0, 0], x)
    return x


def k_tensor_ops(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    y = pl.tensor.add(x, x)
    z = pl.tensor.muls(y, 1.0)
    return z


# noqa: E501
def k_matmul(
    x: pl.Tensor[[64, 128], pl.FP16], w: pl.Tensor[[128, 64], pl.FP16]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    b = pl.load(w, [0, 0], [64, 64])
    acc = pl.matmul(a, b)
    pl.store(acc, [0, 0], x)
    return x


def k_gemm(
    x: pl.Tensor[[64, 128], pl.FP16], w: pl.Tensor[[128, 64], pl.FP16]
) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        a = pl.load(x, [0, 0], [64, 64])
        b = pl.load(w, [0, 0], [64, 64])
        acc = pl.matmul(a, b)
        pl.store(acc, [0, 0], x)
    return x


def test_tile_chain_anchor():
    conformance(k_tile_chain, build)


def test_tensor_ops_anchor():
    conformance(k_tensor_ops, build)


def test_matmul_anchor():
    conformance(k_matmul, build)


def test_gemm_anchor():
    """The gemm kernel: straight-line + the pl.at scope, no loop (the first real
    kernel shape)."""
    conformance(k_gemm, build)


# —— guardrails ——


def test_open_with_rejects_non_at():
    """A host context declines at ``create_with_construct`` (None — the
    interpreter wraps the host context manager); only the ``pl.at`` trace
    context manager is staged."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    assert frame.create_with_construct(None, object()) is None
    frame.finish()


def test_at_deps_deferred():
    """``pl.at(deps=…)`` is collect-tier: the trace context manager
    accepts the full signature and enters the scope ignoring the deps — the
    marker, not an exception, is the contract. The deps graph is the Part 4+
    slice."""
    sema = PtoSema(name="probe")
    frame = sema.begin_frame()
    bag = pl.at(pl.Level.CORE_GROUP, deps=[None])
    opened = frame.create_with_construct(None, bag)
    assert opened is not None  # the staged ``with`` construct opened (deps ignored)
    opened.construct.close()
    frame.finish()


# —— R11: the remaining pl.at kwarg surface ——


def k_at_allow_early_resolve(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=True):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_windowize(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, windowize=True):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_cross_core_slot(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=3)]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_capture_deps(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="producer") as tid:
        t = pl.load(x, [0, 0], [64, 64])
        y = pl.add(t, 1.0)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid]):
        pl.store(pl.add(y, 2.0), [0, 0], x)
    return x


def k_at_no_dep_args_dumps(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, no_dep_args=[x], dumps=[x]):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_hierarchy_named(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator, name_hint="hier"):
        y = pl.add(x, 1.0)
    return y


def k_at_positional_level(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_in_hierarchy(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.HOST, role=pl.Role.Orchestrator):
        with pl.at(level=pl.Level.CORE_GROUP):
            t = pl.load(x, [0, 0], [64, 64])
            pl.store(t, [0, 0], x)
    return x


def k_at_in_staged_for(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    for _i in pl.range(2):
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="t"):
            t = pl.load(x, [k, 0], [64, 64])
            pl.store(t, [k, 0], x)
    return x


def test_at_allow_early_resolve_anchor():
    """The speculative early-dispatch hint: ``("allow_early_resolve", True)`` in
    the parser's canonical attr position, no predicate needed to force it."""
    conformance(k_at_allow_early_resolve, build)
    conformance(k_at_allow_early_resolve, build, canonical=True)


def test_at_windowize_anchor():
    conformance(k_at_windowize, build)


def test_at_cross_core_slot_anchor():
    """``pl.cross_core_slot`` alone (no split entry): the ``slot_num`` attr
    lands on the same InCore scope on both lanes (attr order pinned by
    ``test_at_predicate_scope_attr_position``'s sibling in test_jit)."""
    conformance(k_at_cross_core_slot, build)


def test_at_capture_deps_anchor():
    """``as tid`` + ``deps=`` on a later scope: the TaskId capture placeholder
    and the ``manual_dep_edges`` threading agree (raw and canonical)."""
    conformance(k_at_capture_deps, build)
    conformance(k_at_capture_deps, build, canonical=True)


def test_at_no_dep_args_dumps_anchor():
    """``no_dep_args=`` → ``arg_direction_overrides_vars`` and ``dumps=`` →
    ``dump_vars``, both as Var lists on the scope (raw and canonical)."""
    conformance(k_at_no_dep_args_dumps, build)
    conformance(k_at_no_dep_args_dumps, build, canonical=True)


def test_at_hierarchy_named_anchor():
    """A non-CORE_GROUP level with a role and a name hint — the
    ``HierarchyScopeStmt`` shape."""
    conformance(k_at_hierarchy_named, build)


def test_at_positional_level_anchor():
    """The positional ``pl.at(pl.Level.CORE_GROUP)`` spelling (the form the
    printer emits; ``_parse_at_kwargs`` accepts both)."""
    conformance(k_at_positional_level, build)


def test_at_in_hierarchy_anchor():
    """A CORE_GROUP ``pl.at`` inside a Hierarchy ``pl.at`` — nesting is legal
    there (the predicate/``deps=`` placement gate rejects only Cluster / Spmd /
    CORE_GROUP nesting)."""
    conformance(k_at_in_hierarchy, build)


def test_at_in_staged_for_anchor():
    """A ``pl.at`` scope inside a staged loop (raw and canonical: the loop
    carries a value through the scope)."""
    conformance(k_at_in_staged_for, build)
    conformance(k_at_in_staged_for, build, canonical=True)


# —— the pl.at kwarg-spelling gaps (no shared shape: pins, not anchors) ——


def k_at_bad_name_hint(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="has space"):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_non_bool_windowize(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, windowize=1):  # type: ignore[arg-type]
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_non_bool_aer(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, allow_early_resolve=1):  # type: ignore[arg-type]
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_dynamic_optimizations(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    opts = [pl.split(pl.SplitMode.UP_DOWN)]
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=opts):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def k_at_deps_none(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, deps=None):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], x)
    return x


def test_at_name_hint_validation_is_parser_only():
    """**F25 — a bug, pinned, not an anchor.**

    ``pl.at(name_hint=...)`` reaches ``begin_scope`` unchecked on the tracing
    lane, while the parser runs ``_parse_scope_name_hint``
    (``ast_parser.py:4039``: a non-keyword identifier). The traced text names
    the scope ``"has space"`` and cannot be re-parsed, so the print → reparse
    fixpoint breaks. This is F21's root cause (the ``graph`` / ``cluster``
    descriptors) reaching the ``at`` descriptor as well; the descriptor lives in
    the same file. Delete the tracing-side assertions when F25 lands."""
    traced = build(k_at_bad_name_hint)
    assert "has space" in str(traced)
    with pytest.raises(ParserSyntaxError, match="must be a valid non-keyword identifier"):
        function_parser(k_at_bad_name_hint)
    with pytest.raises(ParserSyntaxError, match="must be a valid non-keyword identifier"):
        pl.parse(str(traced))


def test_at_bool_literal_kwargs_are_parser_only():
    """**needs-decision** (acceptance-only, no IR consequence).

    ``windowize=1`` / ``allow_early_resolve=1``: the parser requires a boolean
    literal (``_dispatch_at_keyword``), the tracing descriptor tests truthiness
    and emits the same ``True`` attr. Both lanes produce the same IR, so no
    anchor is possible and no design doc decides the spelling."""
    for kernel in (k_at_non_bool_windowize, k_at_non_bool_aer):
        assert build(kernel) is not None
        with pytest.raises(ParserSyntaxError, match="must be a boolean literal"):
            function_parser(kernel)


def test_at_optimizations_list_literal_is_parser_only():
    """**needs-decision** (acceptance-only, same IR).

    The parser inspects the AST (``optimizations=`` must be a list *literal* —
    the ``pl.split(...)`` marker resolves no other way), while
    ``at.__trace_enter__`` accepts any list/tuple, so a pre-built list works
    under tracing. The emitted scope is identical."""
    traced = build(k_at_dynamic_optimizations)
    assert "split" in str(traced)
    with pytest.raises(ParserError, match="pl.split"):
        function_parser(k_at_dynamic_optimizations)


def test_at_deps_none_is_parser_only():
    """**needs-decision** (acceptance-only, same IR).

    ``deps=None`` is a falsy no-op on the tracing lane
    (``_collect_scope_attrs`` skips it, no ``manual_dep_edges`` attr) while the
    parser requires a list / tuple / comprehension. ``pl.at`` — unlike
    ``pl.spmd`` — accepts ``deps=`` without ``as tid`` on both lanes, so only
    the argument *type* differs."""
    assert build(k_at_deps_none) is not None
    with pytest.raises(ParserError, match="must be a list / tuple / comprehension"):
        function_parser(k_at_deps_none)
