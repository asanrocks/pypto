# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: returns / exits (the R9 sweep).

Both frontends are thin at the return site. The parser's ``parse_return``
(``ast_parser.py:6039``) lowers ``return e`` to an ``ir.ReturnStmt`` with one
value, ``return a, b`` to a multi-value ``ReturnStmt`` (the ``ast.Tuple``
arm), and a bare ``return`` to a void ``ReturnStmt``. The tracing lane routes
the keyword through the rewriter's ``_return_row`` (``rewriter.py:1625``:
record the value with ``rt.return_`` + a host-execution exit guard) into
``Sema.return_`` / ``_return_impl`` (``semantics.py:1348`` / ``:1379``),
which drops host-valued elements and emits the same ``ReturnStmt`` shapes.
The declared-``->`` contract is the tracing entry's business
(``tracer/decorators/function.py:797``, ``return_contract.py``) and has no
parser counterpart.

**Regime.** Straight-line returns carry no pass state, so the trailing-return
anchors are asserted **raw**. A return that sits inside staged control flow
(the branch/loop anchors) is asserted **raw *and* post-``canonicalize()``**:
raw is the interesting half (both lanes record the ``ReturnStmt`` inline, at
its source position, with no return_var/phi), and canonical is the
staged-flow regime proper (SSA joins the per-path returns — a function whose
branches each return converges through the promoted mutable locals).

The variants that cannot be anchored, and why:

- a statement after a **function-level** (host-executed) ``return``: the
  return ends the traced execution (the rewriter's host guard fires — no
  staged Construct is open), so the tail is never collected. **F15**: the
  parser read source statically and used to emit that dead tail; it now
  stops collecting at the function-level return too, so both lanes agree
  (anchored below). The tail is unreachable by construction — Python would
  never execute it — so no semantics is lost;
- a host-valued return element (``return 1.0``, ``return t, 1.0``): one side
  silently drops the element — no doc decides the spelling, **needs-decision**
  (pinned);
- a list-literal return ``return [a, b]``: the parser's round-trip reading is
  one ``MakeTuple`` value, tracing's ergonomics reading is the host-container
  multi-value return — a dialect collision, **needs-decision** (pinned);
- ``return pl.Tuple(a, b)``: the tracing facade has no parser spelling
  (``Unknown operation 'pl.Tuple'``) — no shared spelling (pinned);
- a declared ``->`` that lies: the tracing entry rejects loudly
  (``docs/design/04-migration-guide.md`` §4.4), the parser is the round-trip
  owner and does not check — an **intended divergence** (pinned);
- ``@pl.expand``: the additive tracing-native entrance
  (``docs/design/03-entrance.md`` §7) has no parser spelling (the parser
  reports ``Unsupported function call``) — no shared spelling (pinned);
- an ``inline`` callee whose body has a multi-value trailing return or an
  early return: **F16** (the pass-spliced tracing caller and the parser's
  parse-time expansion disagreed on the ``ReturnStmt`` arity — fixed, anchored
  in ``test_inline.py``; the tuple-temporary spelling stays divergent as
  **F37**) and **F17** (the early return, pinned in ``test_inline.py``).
"""

import pypto.language as pl
import pytest
from pypto.language.diagnostics import InvalidOperationError, UnsupportedFeatureError
from pypto.language.parser import function_parser, program_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir

from ._harness import conformance

build = function_trace


def _body_stmts(fn: ir.Function) -> list[ir.Stmt]:
    """A function body's statement list (a single stmt is not wrapped)."""
    body = fn.body
    return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]


# ===========================================================================
# The trailing single-return contract (straight-line: raw)
# ===========================================================================


def k_trailing_return(
    x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return o


def test_trailing_return():
    """The plain contract: one trailing ``return``, one ``ReturnStmt``."""
    conformance(k_trailing_return, build)


def k_multi_value_return(x: pl.Tensor[[64, 64], pl.FP32]):
    a = pl.add(x, 1.0)
    b = pl.sub(x, 1.0)
    return a, b  # Python tuple semantics → a multi-value ReturnStmt


def test_multi_value_return():
    """``return a, b`` — the §9 spelling both lanes lower to the IR's
    multi-value ``ReturnStmt`` (not a tuple value)."""
    conformance(k_multi_value_return, build)


def k_scalar_return(x: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    y = pl.add(x, 1)
    return y


def test_scalar_return():
    """A staged scalar return travels the same ReturnStmt path."""
    conformance(k_scalar_return, build)


def k_bare_return(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return  # noqa: PLR1711 - the bare return is the construct under test (void ReturnStmt)


def test_bare_return():
    """A bare ``return`` is a void ``ReturnStmt`` on both lanes."""
    conformance(k_bare_return, build)


def k_no_return(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)


def test_no_return():
    """No trailing return at all — both lanes emit the body with no
    ``ReturnStmt`` (the tracing entry's declared-``->`` check is a separate,
    tracing-only contract, pinned below)."""
    conformance(k_no_return, build)


# ===========================================================================
# Return inside a branch or a loop (staged: raw + canonical)
# ===========================================================================


def k_ret_elif(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
        return o
    elif k == 1:
        return o
    else:
        t2 = pl.load(x, [0, 0], [64, 64])
        pl.store(t2, [0, 0], o)
        return o
    return o


def test_return_in_elif_chain():
    """An early return in each arm of an ``elif`` chain — per-path
    ``ReturnStmt``, joined by the SSA pass."""
    conformance(k_ret_elif, build)
    conformance(k_ret_elif, build, canonical=True)


def k_ret_nested_if_in_for(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.range(4):
        if k == 0:
            return o
    return o


def test_return_in_if_in_for():
    """A return in a branch nested in a staged loop."""
    conformance(k_ret_nested_if_in_for, build)
    conformance(k_ret_nested_if_in_for, build, canonical=True)


def k_ret_loop_tail(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    for i in pl.range(4):
        return o
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
    return o


def test_return_in_loop_with_tail():
    """Statements after a return *inside staged control flow* are still
    collected by both lanes (the host return is suppressed while the
    Construct is open)."""
    conformance(k_ret_loop_tail, build)
    conformance(k_ret_loop_tail, build, canonical=True)


def k_ret_branch_tail(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
):
    if k == 0:
        return o
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
    return o


def test_return_in_branch_with_tail():
    """As above inside a staged ``if`` — deferred staged return ordering."""
    conformance(k_ret_branch_tail, build)
    conformance(k_ret_branch_tail, build, canonical=True)


def k_ret_spmd_for(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    for i in pl.spmd(4):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
        return o
    return o


def test_return_in_spmd_body():
    """A return in the for-form ``pl.spmd`` body — both lanes record it in
    the Spmd region."""
    conformance(k_ret_spmd_for, build)
    conformance(k_ret_spmd_for, build, canonical=True)


def k_ret_at_region(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
        return o
    return o


def test_return_in_at_region():
    """A ``pl.at`` region is not control flow, but it *is* an open Construct:
    the return is recorded in-region on both lanes."""
    conformance(k_ret_at_region, build)
    conformance(k_ret_at_region, build, canonical=True)


def k_ret_at_region_tail(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.load(x, [0, 0], [64, 64])
        pl.store(t, [0, 0], o)
        return o
        t2 = pl.load(x, [0, 0], [64, 64])
        pl.store(t2, [0, 0], o)
    return o


def test_return_in_at_region_with_tail():
    """The tail after a return inside an open region is collected by both
    lanes (the region suppresses the host exit, like a staged Construct)."""
    conformance(k_ret_at_region_tail, build)
    conformance(k_ret_at_region_tail, build, canonical=True)


# ===========================================================================
# ``@pl.expand`` — tracing-native, no parser spelling
# ===========================================================================


@pl.expand
def _exp_add(x, y):
    return pl.add(x, y)


def k_expand_use(x: pl.Tensor[[64], pl.FP32], o: pl.Out[pl.Tensor[[64], pl.FP32]]):
    t = pl.load(x, [0], [64])
    r = _exp_add(t, t)
    pl.store(r, [0], o)
    return o


def k_expansion_written_out(x: pl.Tensor[[64], pl.FP32], o: pl.Out[pl.Tensor[[64], pl.FP32]]):
    t = pl.load(x, [0], [64])
    r = pl.add(t, t)
    pl.store(r, [0], o)
    return o


def test_expand_call_has_no_parser_spelling():
    """``@pl.expand`` is the additive tracing-native entrance
    (``docs/design/03-entrance.md`` §7: *'Values and control have function
    semantics; IR placement has expansion semantics'*), and the AST parser
    *'is not taught entrance or tracing semantics'*
    (``docs/design/04-migration-guide.md`` §8). The same source is rejected by
    the parser, so no ``conformance`` anchor is possible — pinned here."""
    assert function_trace(k_expand_use) is not None
    with pytest.raises(UnsupportedFeatureError, match="Unsupported function call"):
        function_parser(k_expand_use)


def test_expand_emits_the_written_out_body():
    """The expansion contract witness (a targeted pin, not a ``conformance``
    anchor — the two sources differ by construction): the traced expansion is
    structurally the body written out at the call site, which the *parser* can
    parse. This is the equality the missing shared spelling would have
    carried."""
    expanded = function_trace(k_expand_use)
    written_out_ast = function_parser(k_expansion_written_out)
    ir.assert_structural_equal(expanded, written_out_ast, enable_auto_mapping=True)


# ===========================================================================
# The divergences — pinned, never kept as red anchors
# ===========================================================================


def k_early_top(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return o
    t2 = pl.load(x, [0, 0], [64, 64])  # unreachable
    pl.store(t2, [0, 0], o)


def test_function_level_return_drops_unreachable_tail_by_both_lanes():
    """**F15.** A statement after a *function-level* return is unreachable:
    the tracing lane's host-execution guard fires (``rt.in_host_mode()`` is
    True at the function level), so the trailing statements are never
    collected. The parser is the round-trip reader, not an executor, so it
    used to keep the dead tail and emit it as IR (5 body stmts vs 3) — a
    red anchor. The parser now stops collecting at the function-level
    return the way execution does, and the anchor is asserted **raw**
    (straight-line kernel: no staged construct, no pass state)."""
    conformance(k_early_top, build)


def k_two_top_returns(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return o
    return o  # dead second return


def test_function_level_double_return_dropped_by_both_lanes():
    """F15's second face: a second function-level ``return`` is unreachable
    too — the first return ends execution, so only one ``ReturnStmt`` is
    collected on either lane now (raw regime, as above)."""
    conformance(k_two_top_returns, build)


def k_host_literal_return(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return 1.0


def k_mixed_return(x: pl.Tensor[[64, 64], pl.FP32], o: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    t = pl.load(x, [0, 0], [64, 64])
    pl.store(t, [0, 0], o)
    return t, 1.0


def test_host_valued_return_elements_are_tracing_dropped():
    """**needs-decision.** ``_return_impl`` keeps only staged elements
    (``semantics.py:1404``): an all-host return emits *no* ``ReturnStmt``, a
    mixed one keeps only the staged element. The parser lowers the host
    literal to a ``ConstFloat`` ReturnStmt value. §9 says the *staged*
    elements are returned as separate IR values, which leans tracing, but the
    same section says the two frontends must agree on the IR dataflow — no
    doc decides the host element, so the spelling is left unpinned (pinned
    here, not anchored)."""
    new = function_trace(k_host_literal_return)
    ast_ = function_parser(k_host_literal_return)
    assert not any(isinstance(s, ir.ReturnStmt) for s in _body_stmts(new))
    assert any(isinstance(s, ir.ReturnStmt) for s in _body_stmts(ast_))
    new_mixed = function_trace(k_mixed_return)
    ast_mixed = function_parser(k_mixed_return)
    (new_ret,) = [s for s in _body_stmts(new_mixed) if isinstance(s, ir.ReturnStmt)]
    assert len(new_ret.value) == 1
    (ast_ret,) = [s for s in _body_stmts(ast_mixed) if isinstance(s, ir.ReturnStmt)]
    assert len(ast_ret.value) == 2


def k_list_return(x: pl.Tensor[[64, 64], pl.FP32]):
    a = pl.add(x, 1.0)
    b = pl.sub(x, 1.0)
    return [a, b]


def test_list_literal_return_is_a_dialect_collision():
    """**needs-decision.** ``return [a, b]``: tracing reads the host list as
    a container of facades and emits the multi-value ``ReturnStmt``; the
    parser reads it as one ``MakeTuple`` (the printer's own spelling for a
    tuple return — ``_return_impl``'s comment records the same split at
    ``semantics.py:1390``). Both readings are load-bearing for their dialect,
    so the spelling is pinned here rather than anchored."""
    new = function_trace(k_list_return)
    ast_ = function_parser(k_list_return)
    (new_ret,) = [s for s in _body_stmts(new) if isinstance(s, ir.ReturnStmt)]
    (ast_ret,) = [s for s in _body_stmts(ast_) if isinstance(s, ir.ReturnStmt)]
    assert len(new_ret.value) == 2  # host-container multi-value
    assert len(ast_ret.value) == 1  # one MakeTuple
    with pytest.raises(ValueError, match="Structural equality"):
        ir.assert_structural_equal(new, ast_, enable_auto_mapping=True)


def k_pl_tuple_return(x: pl.Tensor[[64, 64], pl.FP32]):
    a = pl.add(x, 1.0)
    b = pl.sub(x, 1.0)
    return pl.Tuple(a, b)


def test_pl_tuple_return_has_no_shared_spelling():
    """``pl.Tuple`` is the tracing-side single-tuple facade; the parser
    reports ``Unknown operation 'pl.Tuple'`` — no shared spelling exists."""
    assert function_trace(k_pl_tuple_return) is not None
    with pytest.raises(InvalidOperationError, match="pl.Tuple"):
        function_parser(k_pl_tuple_return)


class _LyingReturnProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP16]:
        return x  # FP32 value under an FP16 `->`


class _NoReturnProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:  # type: ignore[reportReturnType] - declared `->` never honored
        t = pl.load(x, [0], [64])
        pl.store(t, [0], x)


def test_declared_return_contract_is_tracing_only():
    """**intended divergence.** A declared ``->`` that lies about the traced
    body is a loud tracing-entry error
    (``docs/design/04-migration-guide.md`` §4.4: *'a declared ``->`` that
    lies about what the body produces is a loud decoration-time error'*);
    the parser is the round-trip owner and accepts both spellings. The source
    is shared but one lane rejects it, so no ``conformance`` anchor exists —
    pinned here."""
    with pytest.raises(TypeError, match="declared"):
        trace_program(_LyingReturnProgram)
    with pytest.raises(TypeError, match="declared"):
        trace_program(_NoReturnProgram)
    assert program_parser(_LyingReturnProgram) is not None
    assert program_parser(_NoReturnProgram) is not None
