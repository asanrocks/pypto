# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``pl.submit`` / ``pl.spmd_submit`` collect facades and the
``pld.tensor.alloc_window_buffer`` marker.

``pl.submit``/``pl.spmd_submit`` are realized: the call happens, the
signature passes, and ``_lower_submit`` builds an ``ir.Submit`` carrying
the dep edges / dispatch predicate / SPMD launch spec — codegen packs
``deps`` into ``Arg::set_dependencies`` and the runtime honors the
predicate at the dispatch point. They return a real Python tuple
``(result_facade, tid_facade)`` whose elements the rewriter's
 tuple-target routing binds via ``try_bind`` (the Submit projections
 collect). ``pld.tensor.alloc_window_buffer`` remains collect-tier
(the LHS-name channel is a Sema/decorator requirement).

Guardrails pin the AST's arg-surface discipline (accept what the
parser accepted, reject what it rejected): the callee must be a marked
sub-kernel with a declared single ``->`` return type; ``core_num=`` /
``sync_start=`` are ``pl.spmd_submit``-only; unknown kwargs are loud.
"""

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.language.tracer.decorators.function import trace_program
from pypto.pypto_core import ir


def _walk(stmt):
    """Yield all sub-stmts (BFS) for IR-shape assertions on a traced body."""
    out = [stmt]
    if hasattr(stmt, "body") and stmt.body is not None:
        out.extend(_walk(stmt.body))
    if hasattr(stmt, "else_body") and stmt.else_body is not None:
        out.extend(_walk(stmt.else_body))
    if hasattr(stmt, "stmts"):
        for s in stmt.stmts:
            out.extend(_walk(s))
    return out


def test_submit_tuple_binds_collect():
    """``out, tid = pl.submit(...)`` traces: the tuple-element binds land as
    Submit projections (the tracing frontend's "cannot be called directly"
    lines die)."""

    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.submit(self.stage, x)
            return out

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    text = str(main)
    assert "pl.submit(stage, x)" in text  # the Submit projections collect


def test_submit_carries_deps_and_early_resolve():
    """``deps=[prev_tid]`` / ``allow_early_resolve=True`` land on the Submit."""

    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function
        def stage2(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.sub(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            a, a_tid = pl.submit(self.stage, x)
            b, _ = pl.submit(self.stage2, x, deps=[a_tid], allow_early_resolve=True)
            return b

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    text = str(main)
    assert "deps=[a_tid]" in text
    assert "allow_early_resolve=True" in text


def test_spmd_submit_collects():
    """``pl.spmd_submit(self.k, x, core_num=N)`` traces; ``core_num`` is
    required and lands on the Submit."""

    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.spmd_submit(self.stage, x, core_num=8)
            return out

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    text = str(main)
    assert "pl.spmd_submit(stage, x" in text


def test_alloc_window_buffer_collects():
    """``buf = pld.tensor.alloc_window_buffer(...)`` collects and the LHS
    name ``buf`` is injected into the call's ``name=`` kwarg (realized
    — the ``_make_alloc_name_bind`` hook rebinds the placeholder
    at ``try_bind``); ``window(buf, ...)`` references the Var exactly."""

    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            buf = pld.tensor.alloc_window_buffer(4096)  # noqa: F841 - the bind is the point
            return x

    prog = trace_program(Prog)
    main = next(f for f in prog.functions.values() if f.name == "main")
    assert "pld.tensor.alloc_window_buffer" in str(main)
    allocs = [
        s.value
        for s in _walk(main.body)
        if isinstance(s, ir.AssignStmt)
        and isinstance(s.value, ir.Call)
        and s.value.op.name == "pld.tensor.alloc_window_buffer"
    ]
    assert allocs and allocs[0].kwargs.get("name") == "buf"


# —— guardrails (the AST's arg-surface discipline) ——


def test_submit_requires_marked_callee():
    with pytest.raises(TypeError, match="marked sub-kernel"):
        pl.submit(42, pl.const(0, pl.INT32))  # pyright: ignore[reportArgumentType]  # intentional: unmarked callee rejected at trace time


def test_submit_requires_declared_return():
    class Prog:
        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.submit(self.no_return, x)
            return out

        @pl.function
        def no_return(self, x: pl.Tensor[[16], pl.INT32]):  # no -> annotation
            return x

    with pytest.raises(TypeError, match="return type"):
        trace_program(Prog)


def test_spmd_submit_requires_core_num():
    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.spmd_submit(self.stage, x)  # pyright: ignore[reportCallIssue]  # intentional: missing core_num rejected at trace time
            return out

    with pytest.raises(TypeError, match="core_num"):
        trace_program(Prog)


def test_submit_rejects_spmd_kwargs():
    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.submit(self.stage, x, core_num=8)  # pyright: ignore[reportCallIssue]  # intentional: spmd-only kwarg rejected at trace time
            return out

    with pytest.raises(TypeError, match="core_num"):
        trace_program(Prog)


def test_submit_rejects_unknown_kwarg():
    class Prog:
        @pl.function
        def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            return pl.add(x, x)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
            out, tid = pl.submit(self.stage, x, bogus=True)  # pyright: ignore[reportCallIssue]  # intentional: unknown kwarg rejected at trace time
            return out

    with pytest.raises(TypeError, match="bogus"):
        trace_program(Prog)


def test_submit_omits_mid_signature_out_across_the_comm_ctx_suffix():
    """The IR contract the orchestration codegen owns: CommCtx-typed params
    form a trailing suffix, and between the caller-supplied prefix and that
    suffix only Out params may be omitted (runtime-allocated). A mid-signature
    Out is exactly the shape the materialize pass produces and the printer
    round-trips — the authoring rule must accept what the pass emits."""

    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            scratch: pl.Out[pl.Tensor[[4], pl.FP32]],
            data_ctx: pld.CommCtx,
        ) -> pl.Tensor[[4], pl.FP32]:
            return scratch

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            out, _tid = pl.submit(
                self.stage,
                data,
                data_ctx,
                attrs={"arg_directions": [pl.adir.input, pl.adir.scalar]},
            )
            return out

    trace_program(Prog)


def test_submit_still_rejects_omitted_in_param():
    class Prog:
        @pl.function(type=pl.FunctionType.InCore)
        def stage(
            self,
            data: pld.DistributedTensor[[4], pl.FP32],
            scale: pl.Tensor[[4], pl.FP32],
            data_ctx: pld.CommCtx,
        ) -> pl.Tensor[[4], pl.FP32]:
            return data

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, data: pld.DistributedTensor[[4], pl.FP32], data_ctx: pld.CommCtx):
            out, _tid = pl.submit(self.stage, data, data_ctx)
            return out

    with pytest.raises(TypeError, match="missing positional argument 1"):
        trace_program(Prog)
