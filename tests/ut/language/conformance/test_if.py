# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: staged ``if``/``else`` + per-branch locals.

The interpreter's per-branch locals mappings: a staged branch pushes
a per-branch locals mapping, the else branch starts from the
pre-conditional snapshot, and ``exit_construct`` commits the union.
Mutable-local convergence: branch binds reuse one ``ir.Var`` per
name (the Sema's per-frame mapping from source names to mutable
``ir.Var``s) and the SSA pass synthesizes the phi. The conditionally-bound
loud error exempts the legitimate in-loop single-branch def.

The R5 sweep adds the remaining ``if`` variants: the ``elif`` chain (both
frontends nest the chain as an ``IfStmt`` in the previous clause's
``else_body`` — the tracer rewrites the stdlib ``orelse[0] is ast.If``
shape the same way), ``if`` nested in a staged ``while`` and in both
``pl.spmd`` forms, the empty branch, and the one *divergence*: a host
predicate (``gate = 4; if gate > 2:``) is host-executed by tracing, one
path only
(``docs/design/01-language.md`` §5), while the parser stages every ``if``.
All staged anchors here are **raw**-equal (plain syntax leaks locals on
both sides; nothing is yield-derived), so ``canonical`` is asserted as a
redundant stronger form wherever the IR carries staged control flow.
"""

import pypto
import pypto.ir as _pyir
import pypto.language as pl
import pytest
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import function_trace, trace_program
from pypto.pypto_core import ir

from ._harness import canonicalize, conformance, program_conformance

build = function_trace


def k_merge(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_else_passthrough(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_nested(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32], m: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        if m == 1:
            a = pl.load(x, [0, 0], [64, 64])
        else:
            a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_if_in_for(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        if k == 0:
            acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_merge_anchor():
    """Mutable-local convergence: both branches bind the SAME
    ``ir.Var`` 'a' — raw equal; convert_to_ssa synthesizes the phi from the
    shared mutable local."""
    conformance(k_merge, build)
    conformance(k_merge, build, canonical=True)


def test_else_passthrough_anchor():
    """A rebind in one branch: the pre-conditional staged local carries the
    missing path (the SSA pass handles the merge)."""
    conformance(k_else_passthrough, build)
    conformance(k_else_passthrough, build, canonical=True)


def test_nested_if_anchor():
    conformance(k_nested, build)


def test_if_in_for_anchor():
    """Conditional loop-carried — the if inside the loop is the SSA
    pass's loop-carried business; the conditionally-bound check exempts it."""
    conformance(k_if_in_for, build)
    conformance(k_if_in_for, build, canonical=True)


def test_canonicalize_synthesizes_branch_phi():
    """The branch phi: convert_to_ssa synthesizes it from the shared
    mutable-local stores — the raw form diverges by design, the canonical
    form converges."""
    fn = build(k_merge)
    canonical = canonicalize(fn)
    ir.assert_structural_equal(canonical, canonicalize(canonical), enable_auto_mapping=True)


# —— R5: elif chains, if nested in while / spmd, the host-predicate divergence ——


def k_elif_chain(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        a = pl.add(a, a)
    elif k == 1:
        a = pl.sub(a, a)
    else:
        a = pl.mul(a, a)
    pl.store(a, [0, 0], x)
    return x


def k_elif_no_else(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        a = pl.add(a, a)
    elif k == 1:
        a = pl.sub(a, a)
    pl.store(a, [0, 0], x)
    return x


def k_elif_in_while(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        if k == 0:
            acc = pl.add(acc, acc)
        elif k == 1:
            acc = pl.sub(acc, acc)
        else:
            acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_if_else_in_while(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        if k == 0:
            acc = pl.add(acc, acc)
        else:
            acc = pl.sub(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_pass_then_branch(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    a = pl.load(x, [0, 0], [64, 64])
    if k == 0:
        pass
    else:
        a = pl.add(a, a)
    pl.store(a, [0, 0], x)
    return x


def test_elif_chain_anchor():
    """The full ``if``/``elif``/``else`` chain. The parser recurses
    ``parse_if_statement`` on ``orelse[0]`` inside the else scope; the
    rewriter wraps the nested if's rows under the ``eval_else_active``
    guard. Both nest the chain in the outer ``else_body`` — raw-equal, and
    (the chain is staged) canonical-equal."""
    conformance(k_elif_chain, build)
    conformance(k_elif_chain, build, canonical=True)


def test_elif_no_else_anchor():
    """An ``elif`` without a trailing ``else``: the fallthrough is the
    pre-conditional binding of ``a`` on both sides (no branch binds it on
    that path)."""
    conformance(k_elif_no_else, build)
    conformance(k_elif_no_else, build, canonical=True)


def test_elif_in_while_anchor():
    """The chain inside a staged ``while``: a conditional loop-carried
    update through all three paths — the SSA pass's loop-carried business."""
    conformance(k_elif_in_while, build)
    conformance(k_elif_in_while, build, canonical=True)


def test_if_else_in_while_anchor():
    """``if``/``else`` nested in a staged ``while`` — the R5 'if nested in
    while' variant with both branches updating the carried local."""
    conformance(k_if_else_in_while, build)
    conformance(k_if_else_in_while, build, canonical=True)


def test_pass_then_branch_anchor():
    """An empty then-branch: the ``pass`` emits no statement on either
    side, so the ``IfStmt``'s ``then_body`` is empty and the enclosing
    pre-conditional binding carries that path."""
    conformance(k_pass_then_branch, build)


def k_if_in_spmd_for(x: pl.Tensor[[16, 16], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[16, 16], pl.FP32]:
    for _i in pl.spmd(2, name_hint="w"):
        if k == 0:
            x = pl.add(x, x)
        else:
            x = pl.sub(x, x)
    return x


class KIfInSpmdWith:
    @pl.function
    def k(self, x: pl.Tensor[[16, 16], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.spmd(2, name_hint="w"):
            _i = pl.tile.get_block_idx()
            if k == 0:
                x = pl.add(x, x)
            else:
                x = pl.sub(x, x)
        return x


def test_if_in_spmd_for_anchor():
    """``if``/``else`` inside the for-form ``pl.spmd`` body — the
    conditional sits in the Spmd region, raw-equal on both lanes."""
    conformance(k_if_in_spmd_for, build)
    conformance(k_if_in_spmd_for, build, canonical=True)


def test_if_in_spmd_with_anchor():
    """``if``/``else`` inside the with-form ``pl.spmd`` body. A body that
    is not the single-call direct-dispatch shape is the nested InCore wrap
    (see ``test_spmd_with_form``); the conditional is part of that body and
    the four-cell body-shape decision is unchanged by it.
    Program-level: raw-equal, and canonical-equal through the program pair.
    """
    program_conformance(KIfInSpmdWith, trace_program)
    program_conformance(KIfInSpmdWith, trace_program, canonical=True)


# —— host-vs-staged predicate dispatch: an intended divergence ——


def k_host_predicate(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
    gate = 4  # plain Python int — a host value, not a pl.Scalar
    if gate > 2:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [32, 32])
    pl.store(a, [0, 0], x)
    return x


def _walk(node) -> list:
    """Every statement in ``node``'s subtree (the shape ``ir.Stmt``
    exposes: ``body``/``then_body``/``else_body``/``stmts``)."""
    out = [node]
    for name in ("body", "else_body", "then_body"):
        child = getattr(node, name, None)
        if child is not None:
            out.extend(_walk(child))
    stmts = getattr(node, "stmts", None)
    if stmts is not None:
        for s in stmts:
            out.extend(_walk(s))
    return out


def test_host_predicate_dispatch_divergence():
    """**Intended divergence — not an anchor and not a bug.**

    Value-directed staging (``docs/design/01-language.md`` §5, 'Control
    flow: one syntax, two worlds': *'Value-directed staging uses the kind
    of the predicate or iterable'*; §5's example comment *'A host predicate
    (plain Python int/bool) → host if: one path only'*; the §11 table
    repeat at *'if 4 > 2: ... # host predicate → one path, passes
    through'*): tracing executes the matching branch at trace time and
    emits no ``IfStmt``, whereas the AST parser has no host world — it
    stages every ``if`` with a folded constant condition. The two frontends
    therefore cannot agree on this source, and the harness's anchor rule
    (same source, same IR) has no work to do here. This test pins both
    sides' shape so the divergence stays intentional: if tracing ever
    started emitting an ``IfStmt`` for a host predicate (or the parser
    started host-executing), this fails loudly."""
    traced = build(k_host_predicate)
    parsed = function_parser(k_host_predicate)

    # tracing: one path only — the taken branch's bind + the store; no if.
    assert not any(isinstance(s, ir.IfStmt) for s in _walk(traced.body))
    # parser: every `if` is staged — the IfStmt carries both branches.
    if_stmts = [s for s in _walk(parsed.body) if isinstance(s, ir.IfStmt)]
    assert len(if_stmts) == 1
    assert if_stmts[0].then_body and if_stmts[0].else_body


# —— conditionally-bound reads: rejected by the IR verifier, not the frontend ——


def _expect_conditional_read_rejected(kernel):
    """A name bound on some branches of a staged ``if`` and read after the
    merge. The frontend no longer pre-rejects this; the IR verifier's
    use-after-def check rejects it at ``ConvertToSSA`` as a use-before-def
    (soundness is the verifier's job, not the frontend's)."""
    fn = function_trace(kernel)
    assert isinstance(fn, ir.Function)
    prog = ir.Program([fn], f"proto_{fn.name}", ir.Span.unknown())
    with pytest.raises(pypto.Error, match="used outside its defining scope|used before definition"):
        _pyir.compile(prog, dump_passes=False, skip_ptoas=True)


def k_cond_bound_then(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    pl.store(a, [0, 0], x)
    return x


def k_cond_bound_else(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        pass
    pl.store(a, [0, 0], x)
    return x


def test_conditionally_bound_then_only():
    """A name bound only on the then path (implicit else) and read after the
    merge: the frontend traces it; the IR verifier rejects the use-before-def."""
    _expect_conditional_read_rejected(k_cond_bound_then)


def test_conditionally_bound_else_path():
    """A name bound in one branch of a then/else pair and read after the merge."""
    _expect_conditional_read_rejected(k_cond_bound_else)


def test_in_loop_single_branch_def_exempt():
    """The legitimate in-loop single-branch def does NOT trigger
    the check — the passthrough is the previous iteration's value (the SSA
    pass's loop-carried business)."""
    fn = function_trace(k_if_in_for)
    assert isinstance(fn, ir.Function)


def test_require_bool_predicate():
    """A staged predicate must be BOOL-typed — ``Frame`` owns
    predicate typing."""

    def k_bad(x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.FP32]:
        if k + 1:  # INT32 predicate, not a comparison
            a = pl.load(x, [0, 0], [64, 64])
        else:
            a = pl.load(x, [0, 0], [64, 64])
        pl.store(a, [0, 0], x)
        return x

    with pytest.raises(TypeError, match="BOOL-typed"):
        function_trace(k_bad)
