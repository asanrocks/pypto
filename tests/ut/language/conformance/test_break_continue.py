# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Structural: staged ``break`` / ``continue`` (the R7 sweep).

Both frontends record the transfer as a plain ``BreakStmt`` / ``ContinueStmt``
at its source position — the frontend is deliberately thin; all control-flow
restructuring is the passes' business (``docs/design/01-language.md``
Appendix A: *'break/continue lowering'* is *'absorbed by the IR and its
passes'*). The parser handler is ``parse_break`` /
``parse_continue`` (``ast_parser.py:6123``/``:6133``) guarded by
``_validate_loop_control`` (``:6143``, keyed on the ``_loop_kind_stack`` the
loop handlers push). The tracing lane routes the keyword through
``interpreter.break_``/``continue_`` (``interpreter.py:943``/``:956``) →
``_transfer`` (``:961``, nearest loop *construct* on the construct stack) →
``PtoForConstruct._validate_transfer`` (``constructs.py:384``) /
``PtoWhileConstruct.on_break`` (``:517``).

**Regime.** Every anchored spelling below carries staged control flow (a
staged ``for``/``while``, staged in turn by ``docs/design/01-language.md``
§8), so each is asserted **raw**-equal *and* post-``canonicalize()``-equal.
Raw equality is the interesting half: a staged ``break`` suppresses the native
keyword (``if rt.break_(): break`` — ``rewriter._native_transfer_row``), the
loop closes through its trailing ``exit_construct()`` / unwind scope guard, and
both lanes emit the transfer inline with the statements that follow it inside
the same region. Canonical is the staged-flow regime proper: the SSA pass
derives the loop-carried values of a loop whose body contains a transfer.

The variants swept here are R7's three: **nesting** (transfer target selection
in nested ``for``/``while``), **deferred staged exit ordering** (a transfer
inside a staged branch — the branch Construct must close at its lexical
boundary, before the loop's), and **exits inside ``while``**. What this suite
does *not* anchor, and why:

- a transfer into a staged loop's ``else`` suite: both lanes reject the
  staged ``else`` loudly (``docs/design/01-language.md`` §8 *'Loop ``else``
  belongs to host loops only'*) — no shared spelling exists, so the pair is
  pinned as a rejection (F13, fixed below);
- a native transfer targeting a *host* loop across an open staged Construct:
  rejected by tracing (``docs/design/02-collector.md`` §5.4 and §7
  invariant 16), and a host loop is not an anchorable target (the parser
  accepts only ``pl.*`` iterators for ``for``; a host ``while`` carries
  host-valued state — a separate, unshared regime);
- ``break``/``continue`` in ``pl.parallel`` / ``pl.unroll``: rejected by both
  lanes (pinned below);
- the top-level ``pl.spmd`` / ``pl.split_aiv`` bodies: rejected by both lanes
  (F12) — a transfer in a region that lowers to a scope has no loop to bind
  to; pinned below.
"""

import pypto.language as pl
import pytest
from pypto.language.diagnostics import InvalidOperationError, ParserSyntaxError
from pypto.language.parser import function_parser
from pypto.language.tracer.collector.interpreter import InterpreterError
from pypto.language.tracer.decorators.function import function_trace

from ._harness import conformance

build = function_trace


# —— a transfer inside a staged branch: the deferred staged exit ordering ——


def k_break_in_if_in_for(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        if k < 2:
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_continue_in_if_in_for(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        if k < 2:
            acc = pl.add(acc, acc)
            continue
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_break_in_if_in_while(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        if k > 1:
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_continue_in_if_in_while(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        if k > 1:
            acc = pl.add(acc, acc)
            continue
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_deep_nest_break(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    """``for`` > ``while`` > ``if`` > ``break``: three Constructs close in
    LIFO order around one transfer, and the loop after the while still
    collects."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(2):
        while k < 2:
            if k > 0:
                acc = pl.add(acc, acc)
                break
            acc = pl.mul(acc, acc)
        acc = pl.add(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


# —— nesting: the transfer targets the nearest loop, never a convenient outer one ——


def k_nested_for_inner_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_nested_for_inner_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
            continue
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_nested_for_outer_break_after_inner(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
        break
    pl.store(acc, [0, 0], x)
    return x


def k_while_in_while_inner_break(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        while k < 2:
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_while_in_for_inner_break(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        while k < 2:
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_for_in_while_inner_break(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 2:
        for i in pl.range(4):
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


# —— exits inside a while ——


def k_while_bare_continue(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 3:
        acc = pl.add(acc, acc)
        continue
    pl.store(acc, [0, 0], x)
    return x


# —— the staged transfer does not stop collection ——


def k_stmts_after_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """A staged ``break`` suppresses the native keyword, so the trace walks on
    and records the statements after it inside the same loop body — exactly
    as the parser's linear statement walk does."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
        break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_stmts_after_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
        continue
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


# —— transfer gating is the *innermost* loop's kind, in both lanes ——


def k_range_in_parallel_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """The sequential inner loop is the transfer target: the enclosing
    ``pl.parallel`` kind does not reject a ``break`` bound to the inner loop
    (parser: ``_loop_kind_stack[-1]``; tracing: nearest loop Construct)."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.parallel(4):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
            break
    pl.store(acc, [0, 0], x)
    return x


def k_range_in_unroll_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.unroll(4):
        for j in pl.range(2):
            acc = pl.add(acc, acc)
            continue
    pl.store(acc, [0, 0], x)
    return x


def k_pipeline_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """``pl.pipeline`` is a sequential kind: neither lane's transfer gate
    rejects a ``break`` there (the parser's ``_loop_kind_stack`` entry is
    ``"pipeline"``; tracing's descriptor kind is ``"pipeline"``)."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.pipeline(8, stage=4):
        acc = pl.add(acc, acc)
        break
    pl.store(acc, [0, 0], x)
    return x


def k_break_in_spmd_for_nested_in_for(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    """A ``break`` in the for-form ``pl.spmd`` body nested inside a staged
    ``for``: both lanes record it inside the Spmd region (tracing's nearest
    loop Construct is the spmd for-descriptor; the parser's loop-kind stack
    still holds the enclosing ``range``). The standalone spelling — where the
    parser's stack is empty — diverges (F12, pinned at the bottom)."""
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(2):
        for b in pl.spmd(4):
            acc = pl.add(acc, acc)
            break
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


# —— anchors ——


def test_break_in_if_in_for_anchor():
    """The deferred staged exit ordering: the rewriter wraps the staged
    ``if`` in the unwind scope guard (the ``break`` is bound to the enclosing
    loop), so the branch Construct closes at its lexical boundary before the
    loop's. The anchor pins the IR consequence: the ``BreakStmt`` sits inside
    the then-body, ahead of the loop's own close, with the statements after
    the branch still in the loop body."""
    conformance(k_break_in_if_in_for, build)
    conformance(k_break_in_if_in_for, build, canonical=True)


def test_continue_in_if_in_for_anchor():
    conformance(k_continue_in_if_in_for, build)
    conformance(k_continue_in_if_in_for, build, canonical=True)


def test_break_in_if_in_while_anchor():
    conformance(k_break_in_if_in_while, build)
    conformance(k_break_in_if_in_while, build, canonical=True)


def test_continue_in_if_in_while_anchor():
    conformance(k_continue_in_if_in_while, build)
    conformance(k_continue_in_if_in_while, build, canonical=True)


def test_deep_nest_break_anchor():
    conformance(k_deep_nest_break, build)
    conformance(k_deep_nest_break, build, canonical=True)


def test_nested_for_inner_break_anchor():
    """The inner ``for`` is the target; the outer loop's body still collects
    after the inner loop."""
    conformance(k_nested_for_inner_break, build)
    conformance(k_nested_for_inner_break, build, canonical=True)


def test_nested_for_inner_continue_anchor():
    conformance(k_nested_for_inner_continue, build)
    conformance(k_nested_for_inner_continue, build, canonical=True)


def test_nested_for_outer_break_after_inner_anchor():
    conformance(k_nested_for_outer_break_after_inner, build)
    conformance(k_nested_for_outer_break_after_inner, build, canonical=True)


def test_while_in_while_inner_break_anchor():
    """Both loops are ``while``s: the transfer still binds to the innermost
    construct (parser: the inner ``"while"`` entry on the kind stack)."""
    conformance(k_while_in_while_inner_break, build)
    conformance(k_while_in_while_inner_break, build, canonical=True)


def test_while_in_for_inner_break_anchor():
    conformance(k_while_in_for_inner_break, build)
    conformance(k_while_in_for_inner_break, build, canonical=True)


def test_for_in_while_inner_break_anchor():
    conformance(k_for_in_while_inner_break, build)
    conformance(k_for_in_while_inner_break, build, canonical=True)


def test_while_bare_continue_anchor():
    """``continue`` inside a staged ``while`` (the ``break`` counterpart is
    the pre-existing ``k_while_break`` anchor in ``test_while_logical``)."""
    conformance(k_while_bare_continue, build)
    conformance(k_while_bare_continue, build, canonical=True)


def test_stmts_after_break_anchor():
    conformance(k_stmts_after_break, build)
    conformance(k_stmts_after_break, build, canonical=True)


def test_stmts_after_continue_anchor():
    conformance(k_stmts_after_continue, build)
    conformance(k_stmts_after_continue, build, canonical=True)


def test_range_in_parallel_break_anchor():
    conformance(k_range_in_parallel_break, build)
    conformance(k_range_in_parallel_break, build, canonical=True)


@pytest.mark.filterwarnings("ignore:.*RoundtripInstrument.*IR not printable:UserWarning")
def test_range_in_unroll_continue_anchor():
    """A sequential inner loop under ``pl.unroll``: the transfer targets the
    inner loop, and the canonical pair hits the same Unroll-marker artifact as
    the R6 unroll anchors (filtered identically)."""
    conformance(k_range_in_unroll_continue, build)
    conformance(k_range_in_unroll_continue, build, canonical=True)


def test_pipeline_break_anchor():
    conformance(k_pipeline_break, build)
    conformance(k_pipeline_break, build, canonical=True)


def test_break_in_spmd_for_nested_in_for_anchor():
    conformance(k_break_in_spmd_for_nested_in_for, build)
    conformance(k_break_in_spmd_for_nested_in_for, build, canonical=True)


# —— no shared spelling: rejections and the two divergences ——


def k_parallel_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.parallel(4):
        acc = pl.add(acc, acc)
        break
    pl.store(acc, [0, 0], x)
    return x


def k_unroll_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.unroll(4):
        acc = pl.add(acc, acc)
        continue
    pl.store(acc, [0, 0], x)
    return x


def test_parallel_unroll_transfer_rejected_by_both_lanes():
    """The only gate both lanes share verbatim: a transfer is bound to a
    sequential loop kind. ``pl.parallel`` and ``pl.unroll`` reject it with
    the same message on both sides — no accepted spelling, so no anchor."""
    for kernel in (k_parallel_break, k_unroll_continue):
        with pytest.raises(InvalidOperationError, match="not supported in (parallel|unrolled) loops"):
            build(kernel)
        with pytest.raises(InvalidOperationError, match="not supported in (parallel|unrolled) loops"):
            function_parser(kernel)


def k_split_aiv_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        acc = pl.load(x, [0, 0], [64, 64])
        break
    pl.store(acc, [0, 0], x)
    return x


def k_split_aiv_in_for_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(2):
        for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
            acc = pl.add(acc, acc)
            break
    pl.store(acc, [0, 0], x)
    return x


def k_spmd_for_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.spmd(4):
        acc = pl.load(x, [0, 0], [64, 64])
        break
    pl.store(acc, [0, 0], x)
    return x


def k_spmd_for_continue(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    for i in pl.spmd(4):
        acc = pl.load(x, [0, 0], [64, 64])
        continue
    pl.store(acc, [0, 0], x)
    return x


def k_spmd_for_in_parallel_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.parallel(2):
        for j in pl.spmd(4):
            acc = pl.add(acc, acc)
            break
    pl.store(acc, [0, 0], x)
    return x


def test_split_aiv_and_spmd_transfer_rejected_by_both_lanes():
    """**F12 fixed — a transfer in a non-loop region is rejected on both lanes.**

    The parser rejects a transfer in a ``pl.split_aiv`` region at *any*
    nesting (``'break' not supported inside a 'for ... in pl.split_aiv(...)'
    body`` — the sentinel ``_split_aiv_mode_context`` pushes onto
    ``_loop_kind_stack`` shadows an enclosing loop) and one in a top-level
    ``for ... in pl.spmd(...)`` body (``'break' outside loop`` — the spmd
    handler never pushes a loop kind). Both constructs lower to *scopes*, not
    ``ForStmt``s, so the parser's rejection is the intended contract.

    The tracing lane used to accept all of them: ``PtoForConstruct`` was
    selected as the nearest loop Construct and ``_validate_transfer`` fell
    back to kind ``"range"`` for any descriptor without the ``kind`` surface
    (``constructs.py``), covering the spmd for-descriptor and ``pl.split_aiv``
    alike. The recorded ``BreakStmt`` then had no enclosing loop, so the
    printed text could not be re-parsed (the print→parse fixpoint broke).

    The spmd for-descriptor is *transparent*, not itself a target: the parser
    pushes no kind for it, so a transfer in a spmd body nested in a real loop
    binds to that loop and is accepted by both lanes (anchored above). Each
    rejection below therefore only pins the two spellings with no transfer
    target: a body region that lowers to a scope."""
    for lane in (build, function_parser):
        with pytest.raises(InvalidOperationError, match="pl.split_aiv"):
            lane(k_split_aiv_break)
        with pytest.raises(InvalidOperationError, match="pl.split_aiv"):
            lane(k_split_aiv_in_for_break)
        with pytest.raises(InvalidOperationError, match="'break' outside loop"):
            lane(k_spmd_for_break)
        with pytest.raises(InvalidOperationError, match="'continue' outside loop"):
            lane(k_spmd_for_continue)
        with pytest.raises(InvalidOperationError, match="not supported in parallel loops"):
            lane(k_spmd_for_in_parallel_break)


def k_for_else_break(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.range(4):
        acc = pl.add(acc, acc)
        break
    else:
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_while_else_break(
    x: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.INT32]
) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    while k < 4:
        acc = pl.add(acc, acc)
        break
    else:
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_spmd_else(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for i in pl.spmd(4):
        acc = pl.add(acc, acc)
    else:  # noqa: PLW0120 — the loop-else under test
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def k_split_aiv_else(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    acc = pl.load(x, [0, 0], [64, 64])
    for aiv_id in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        acc = pl.add(acc, acc)
    else:  # noqa: PLW0120 — the loop-else under test
        acc = pl.mul(acc, acc)
    pl.store(acc, [0, 0], x)
    return x


def test_loop_else_rejected_by_both_lanes():
    """**F13 fixed — a loop ``else`` suite is rejected on both lanes.**

    ``docs/design/01-language.md`` §8 (*'Loop ``else`` belongs to host
    loops only'*, in the §12 guardrail table as *'``else`` collected
    unconditionally, contradicting skipped-by-``break`` semantics'*): a staged
    ``break`` suppresses the native keyword and a staged loop always exits
    through its predicate after one representative iteration, so an ``else``
    suite would be collected unconditionally rather than skipped by
    ``break``. Tracing rejects the suite at the loop's close — the rewriter
    closes the loop Construct from inside the ``else`` suite and
    ``Interpreter.exit_construct(else_follows=True)`` refuses a staged loop.

    The parser used to accept the loop and never read ``stmt.orelse``, so the
    suite was silently dropped — user code lost with no diagnostic. The
    parsed dialect has no host world, so *every* loop it accepts is staged:
    the rejection is unconditional (no ``break`` required) and covers all
    iterator spellings plus the natural ``while``. Pinned below for a staged
    ``for``, a staged ``while``, and the two scope-lowering descriptors
    (``pl.spmd``, ``pl.split_aiv``); the else body's ``mul`` must never reach
    the parsed IR.
    """
    with pytest.raises(InterpreterError, match="cannot have an 'else' suite"):
        build(k_for_else_break)
    with pytest.raises(ParserSyntaxError, match="cannot have an 'else' suite"):
        function_parser(k_for_else_break)

    with pytest.raises(InterpreterError, match="cannot have an 'else' suite"):
        build(k_while_else_break)
    with pytest.raises(ParserSyntaxError, match="cannot have an 'else' suite"):
        function_parser(k_while_else_break)

    with pytest.raises(InterpreterError, match="cannot have an 'else' suite"):
        build(k_spmd_else)
    with pytest.raises(ParserSyntaxError, match="cannot have an 'else' suite"):
        function_parser(k_spmd_else)

    with pytest.raises(InterpreterError, match="cannot have an 'else' suite"):
        build(k_split_aiv_else)
    with pytest.raises(ParserSyntaxError, match="cannot have an 'else' suite"):
        function_parser(k_split_aiv_else)
