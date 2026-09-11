# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``with pl.spmd(...)`` body-shape anchors — the four-cell matrix.

The parser decides the with-form's IR shape statically
(``_emit_spmd_body``): a single call-valued body statement without a split
hint is the direct-dispatch shape (``SpmdScopeStmt(<the call stmt>)``, the
user's unsplit hint — what ``pl.intrin.spmd_direct`` spells in the assembly
dialect); every other body wraps in the nested InCore scope (split name
hints) for ``OutlineIncoreScopes`` to outline. TaskId capture (``as``) is
orthogonal to the body shape.

The tracing frontend cannot know the body shape at ``__trace_enter__`` —
the rewriter computes the same static fact from the with-suite (one
``Assign``/``AnnAssign``/``Expr`` whose value is a ``Call``) and threads it
through ``rt.on_begin_with`` (the ``as_name`` channel). Before that thread
existed, the tracer wrapped on ``as``-capture instead of body shape: the
plain multi-statement form reached ``OutlineClusterScopes`` unwrapped, its
Spmd wrapper carried an inline body, and orchestration codegen's
``FindWrapperInnerCall`` failed (the deepseek v4-pro corpus's
``exp_gate_mm_spmd`` internal error).

Bodies use plain name-target assignments: the full-slice ``y[:] = ...``
subscript-write lowering has a known pre-existing divergence unrelated to
spmd.

The parser additionally *rejects* an inline multi-statement body that
never reads ``pl.tile.get_block_idx()`` (every block would run identical
work). That check is deliberately NOT mirrored in the tracer: the
property survives to the IR — the outlined Spmd kernel simply carries no
block-index input — so its diagnostic home is a pass/verifier check,
which would cover tracing, the assembly dialect, and hand-built IR
uniformly. No such diagnostic exists today (the outlining silently omits
the param); implementing one is C++ work outside this migration's
Python-only contract.
"""

from __future__ import annotations

import pypto.language as pl
from pypto.language.tracer.decorators.function import trace_program

from ._harness import program_conformance


class PlainSingleCall:
    @pl.function
    def callee(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function
    def k(self, y: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.spmd(2, name_hint="w"):
            y = self.callee(y)
        return y


class CapturedSingleCall:
    @pl.function
    def callee(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function
    def k(self, y: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.spmd(2, name_hint="w") as _tid:
            y = self.callee(y)
        return y


class PlainMultiStmt:
    @pl.function
    def callee(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function
    def k(self, y: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.spmd(2, name_hint="w"):
            _i = pl.tile.get_block_idx()
            y = self.callee(y)
        return y


class CapturedMultiStmt:
    @pl.function
    def callee(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        return pl.add(x, x)

    @pl.function
    def k(self, y: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        with pl.spmd(2, name_hint="w") as _tid:
            _i = pl.tile.get_block_idx()
            y = self.callee(y)
        return y


def test_with_form_plain_single_call():
    """Plain + single-call body: the direct-dispatch shape (``Spmd(Call)``
    with the unsplit hint) — capture-free, no InCore wrapper."""

    program_conformance(PlainSingleCall, trace_program)


def test_with_form_captured_single_call():
    """``as``-capture + single-call body: capture is orthogonal — still the
    direct shape, plus the ``task_id_var`` attr. (The pre-fix tracer wrapped
    this case; the parser never does.)"""

    program_conformance(CapturedSingleCall, trace_program)


def test_with_form_plain_multi_stmt():
    """Plain + multi-statement body: the InCore wrap with split name hints
    (``w_spmd`` / ``w``) — the corpus's expert_routed shape (the pre-fix
    tracer left this unwrapped, breaking orchestration codegen)."""

    program_conformance(PlainMultiStmt, trace_program)


def test_with_form_captured_multi_stmt():
    """``as``-capture + multi-statement body: the InCore wrap plus the
    ``task_id_var`` attr."""

    program_conformance(CapturedMultiStmt, trace_program)
