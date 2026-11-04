# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The stdlib submit spellings — policy wrappers over the dispatch effect.

The dispatch family is an authoring effect (see the surface classification in
``docs/en/dev/language/07-authoring-recipes.md``): :func:`pl.submit` and
:func:`pl.spmd_submit` lower to an ``ir.Submit`` — a structure statement — and
the one implementation of that lowering lives in
:mod:`pypto.language.authoring.dispatch`. These wrappers own the public
signatures and their documentation; the recipe-facing spelling is
``ctx.submit(...)`` over the same effect, so the two cannot drift.
"""

from __future__ import annotations

from typing import Any

from pypto.language.ambient import active_frame
from pypto.language.authoring.dispatch import (
    SubmitArg,
    SubmitCallee,
    SubmitEntries,
    SubmitOperand,
    dispatch,
)

__all__ = ["spmd_submit", "submit"]


def submit(
    callee: SubmitCallee,
    *kernel_args: SubmitArg,
    deps: SubmitEntries | None = None,
    dumps: SubmitEntries | None = None,
    allow_early_resolve: bool = False,
    predicate: SubmitOperand | None = None,
    device: SubmitOperand | None = None,
    timing_slot: int | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Submit a kernel and capture its producer TaskId.

    The facade runs the call and returns a real Python tuple
    ``(result_facade, task_id_facade)`` whose elements the rewriter's
    tuple-target routing binds through the Frame (``rt.bind``); the dispatch
    effect builds an :class:`ir.Submit` carrying the dep edges (``deps=``),
    dispatch predicate (``predicate=``), early-resolve hint
    (``allow_early_resolve=``), and dump marks (``dumps=``) — codegen
    packs ``deps`` into ``Arg::set_dependencies`` and the runtime honors
    the predicate at the dispatch point.

    Surface form (must be unpacked as a 2-tuple)::

        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
        (a, b), tid = pl.submit(self.multi_out_kernel, x)
        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid], dumps=[x])
        out, tid    = pl.submit(self.stage1, x, scratch, allow_early_resolve=True)

    The kernel-side ``ir.Submit`` natively returns
    ``Tuple[<kernel return>, TASK_ID]``; element 0 is the tensor result(s),
    element 1 is the producer TaskId (``Scalar[TASK_ID]``). The optional
    ``deps=[...]`` kwarg lists TaskId scalars / arrays this submit must
    wait on. The single-LHS form ``res = pl.submit(...)`` is also accepted
    and binds the whole flat tuple to one variable.

    ``pl.submit`` and its ``deps=`` kwarg work in **both auto and manual
    scope**: ``Arg::set_dependencies`` is orthogonal to OverlapMap
    auto-tracking (final fanin = auto ∪ explicit). In auto scope, use
    ``deps=[...]`` as a precision tool to patch edges the runtime cannot
    infer; in ``pl.manual_scope()``, use it to declare every edge.

    The optional ``dumps=[...]`` kwarg is the submit-side selective tensor
    dump surface (symmetric with ``deps=``): it lists tensor arguments of
    this submit to mark for dump (simpler#844), so an enabled dump pipeline
    filters down to just those bindings. Each entry must be a tensor passed
    positionally to the submitted kernel. ``dumps=`` is the explicit dump
    surface on a submit; the declarative ``pl.dump_tag(t)`` statement feeds the
    same ``dump_vars`` set. These marks only take effect under partial dump
    (``RunConfig.enable_dump_args == 1``); they are a no-op when dump is off
    (``0``) and irrelevant under full dump (``2``), which captures every
    binding regardless.

    The optional ``allow_early_resolve=True`` kwarg (default ``False``) opts
    this task in as a speculative early-dispatch producer (simpler#1065): the
    scheduler may pre-stage this task's consumers onto idle cores before it
    completes, releasing them with a doorbell the instant it finishes. It is a
    producer-side hint — a consumer only pre-stages once *all* of its producers
    are flagged (or already complete). It lowers to
    ``Arg::set_allow_early_resolve(true)`` in orchestration codegen and is a
    pure scheduling optimisation (no effect on results). Pays off on critical
    paths built from many short tasks; harmless otherwise.

    The optional ``predicate=(...)`` kwarg attaches a **dispatch predicate** the
    scheduler evaluates at this task's dispatch point — after its dependencies
    are satisfied, so the value is current without an orchestration-time wait.
    When the comparison is false the task is retired inline (never dispatched to
    a core) while its fanin/fanout still settle, so downstream consumers still
    unlock. Use it to skip work whose need is only known at runtime (e.g. an MoE
    expert with an empty row count)::

        out, tid = pl.spmd_submit(self.expert_ffn, tokens, out, core_num=N,
                                  deps=[gather_tid],
                                  predicate=(row_count[e] > 0))

    The comparison is matched **syntactically, never evaluated**: in this
    position ``row_count[e] > 0`` is a declarative spec handed to the scheduler,
    not a ``tensor.read`` plus a compare. Only ``tensor[indices] OP
    int-literal`` is expressible (one comparison; ``==`` ``!=`` ``>`` ``<``
    ``>=`` ``<=``), mirroring the runtime's single-comparison predicate — no
    chained comparisons, arithmetic, or boolean combination. Reduce anything
    richer to a single gate value in a prior kernel and predicate on that.

    **Contract:** the operand tensor's producer must be one of this submit's
    ``deps=`` (the parser enforces this where statically provable), so the
    dispatch-point read observes the current value.

    The return annotation is ``Any`` (not ``NoReturn``) because the executed
    facade returns a real Python tuple ``(result, task_id)`` — the rewriter's
    tuple-target routing binds each element through the Frame
    (``rt.bind``) — and downstream
    code that does ``out, tid = pl.submit(...)`` would not type-check under
    ``NoReturn``.
    """
    return dispatch(
        callee=callee,
        kernel_args=kernel_args,
        construct="pl.submit",
        frame=active_frame(),
        deps=deps,
        dumps=dumps,
        allow_early_resolve=allow_early_resolve,
        predicate=predicate,
        device=device,
        timing_slot=timing_slot,
        attrs=attrs,
    )


def spmd_submit(
    callee: SubmitCallee,
    *kernel_args: SubmitArg,
    core_num: SubmitOperand | None,
    sync_start: bool = False,
    deps: SubmitEntries | None = None,
    dumps: SubmitEntries | None = None,
    allow_early_resolve: bool = False,
    predicate: SubmitOperand | None = None,
    device: SubmitOperand | None = None,
    timing_slot: int | None = None,
    attrs: dict[str, Any] | None = None,
) -> Any:
    """Launch a kernel as an SPMD task and capture its producer TaskId.

    The facade returns a real Python tuple ``(result, task_id)`` whose
    elements the caller's tuple-target routing binds through the Frame
    (``rt.bind``); the dispatch effect builds an :class:`ir.Submit` carrying the SPMD
    launch spec (``core_num`` / ``sync_start``) alongside the dep edges,
    dispatch predicate, and early-resolve hint. The runtime fans the task
    out across ``core_num`` logical blocks (each kernel reads its block
    index via ``pl.tile.get_block_idx()``).

    It is the SPMD sibling of :func:`submit`: a single orchestration task that
    the runtime fans out across ``core_num`` logical blocks (each kernel reads
    its block index via ``pl.tile.get_block_idx()``). Like :func:`submit` it
    returns one producer TaskId, so the whole dispatch can be named as a
    dependency of later tasks.

    Surface form (must be unpacked as a 2-tuple)::

        out, tid = pl.spmd_submit(self.incore_kernel, x, y, core_num=8)
        out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True,
                                  deps=[prev_tid])

    ``core_num`` is a **required keyword argument** (a positive integer
    expression) — the positional slots are the kernel's own arguments.
    ``sync_start`` (default ``False``) requires all blocks to launch atomically.
    ``deps=[...]``, ``allow_early_resolve=True`` and ``predicate=(...)`` work
    exactly as on :func:`submit` (note: a ``sync_start`` task cannot itself be
    block-by-block pre-staged, but it can still be flagged to let its consumers
    pre-stage). The callee may be an InCore / AIC / AIV kernel or a co-scheduled
    Group.

    Like :func:`submit`, it works in both auto and manual scope; its primary use
    is explicit dependency wiring inside ``pl.manual_scope()``.
    """
    if core_num is None:
        raise TypeError(
            "pl.spmd_submit requires the core_num= keyword (a positive integer "
            "expression — the SPMD block count)"
        )
    return dispatch(
        callee=callee,
        kernel_args=kernel_args,
        construct="pl.spmd_submit",
        frame=active_frame(),
        deps=deps,
        dumps=dumps,
        allow_early_resolve=allow_early_resolve,
        predicate=predicate,
        device=device,
        timing_slot=timing_slot,
        attrs=attrs,
        core_num=core_num,
        sync_start=sync_start,
    )
