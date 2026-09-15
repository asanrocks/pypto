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
from pypto.language.parser import program_parser
from pypto.language.tracer.decorators.function import trace_program
from pypto.pypto_core import ir

from ._harness import program_conformance

# Validated at import so a bare name comparison can never drift from the
# registry (see .claude/rules/operator-identity-checks.md).
_ALLOC_WINDOW_BUFFER = ir.get_op("pld.tensor.alloc_window_buffer").name


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
        and s.value.op.name == _ALLOC_WINDOW_BUFFER
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


# ===========================================================================
# R13 conformance anchors — `pl.submit` / `pl.spmd_submit` kwarg surface
# ===========================================================================
#
# A Submit is a *scope-level* construct (an `ir.Submit` node), never a
# `ForStmt`: there is no iter_arg/phi territory, so raw structural equality is
# the load-bearing half for every straight-line program below. An anchor whose
# program carries staged control flow (a Submit inside a staged `pl.range`)
# additionally asserts post-`canonicalize_program()`, the staged-flow regime
# proper — both lanes have loop-carried values for the SSA pass to derive.
#
# Every anchor is a `program_conformance` (both lanes need the callee's
# `ir.Function`/`Marker` in scope, which only the program envelope provides).


class SubmitBasic:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x)
        return out


class SubmitDeps:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function
    def stage2(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.sub(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        a, a_tid = pl.submit(self.stage, x)
        b, _ = pl.submit(self.stage2, a, deps=[a_tid])
        return b


class SubmitDepsPerElement:
    """`deps=[arr[i], ...]` — Form 1, desugared by both lanes into one
    synthesized `Array[K, TASK_ID]` at the current insertion point."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], tids: pl.Array[4, pl.TASK_ID]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, deps=[tids[0], tids[1]])
        return out


class SubmitDepsWholeArray:
    """`deps=[arr]` — the whole TASK_ID array Var passes direct (Var-only
    `Submit.deps`, no desugar)."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], tids: pl.Array[4, pl.TASK_ID]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, deps=[tids])
        return out


class SubmitDepsComprehension:
    """Form 2: `deps=[arr[i] for i in range(N)]` — the parser unrolls the
    comprehension at parse time, the tracing lane evaluates it natively; both
    land the same per-element entries in the same order."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], tids: pl.Array[4, pl.TASK_ID]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, deps=[tids[i] for i in range(2)])
        return out


class SubmitDumps:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, dumps=[x])
        return out


class SubmitEarlyResolve:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, allow_early_resolve=True)
        return out


class SubmitPredicate:
    """`predicate=(rc[0] > 0)` — a plain comparison Expr on `Submit.predicate`,
    never a statement (the parser's declarative single-comparison shape)."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], rc: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, rc)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], rc: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        a, a_tid = pl.submit(self.stage, x, rc)
        b, _ = pl.submit(self.stage, a, rc, deps=[a_tid], predicate=(rc[0] > 0))
        return b


class SubmitTimingSlot:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, timing_slot=3)
        return out


class SubmitAttrsRoundTrip:
    """The round-trip `attrs={...}` channel: an attrs key with a dedicated
    kwarg is setdefault-ed (the dedicated spelling wins on both lanes)."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, attrs={"task_timing_slot": 2})
        return out


class SubmitMultiOutput:
    """A multi-output callee with the nested result target
    `(a, b), tid = pl.submit(...)` — the Submit stays flat
    `Tuple[out0, out1, TASK_ID]` while the Python tuple boundary is nested."""

    @pl.function
    def multi(
        self, x: pl.Tensor[[16], pl.INT32]
    ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32]]:
        return pl.add(x, x), pl.sub(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        (a, b), tid = pl.submit(self.multi, x)
        return a


class SubmitNoDep:
    """`pl.no_dep(x)` at a submit-arg position — per-arg direction override
    (`arg_direction_overrides` attr) threaded by both lanes."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, pl.no_dep(x))
        return out


class SubmitInStagedLoop:
    """A Submit inside a staged `pl.range` — the Submit result is loop-carried,
    so the canonical half exercises the SSA pass's derivation for both lanes."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        acc = x
        for i in pl.range(2):
            acc, tid = pl.submit(self.stage, acc)
        return acc


class SubmitOutOmitted:
    """The submit-only arity window: a trailing `Out` param may be omitted at
    the call site (the backend runtime-allocates it)."""

    @pl.function
    def stage(
        self, x: pl.Tensor[[16, 16], pl.FP32], out: pl.Out[pl.Tensor[[16, 16], pl.FP32]]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        t = pl.load(x, [0, 0], [16, 16])
        return pl.store(t, [0, 0], out)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16, 16], pl.FP32]) -> pl.Tensor[[16, 16], pl.FP32]:
        out, tid = pl.submit(self.stage, x)
        return out


class SpmdSubmitIndexCore:
    """`pl.spmd_submit(..., core_num=<INDEX expression>)`.

    The explicit `pl.const(8, pl.INDEX)` spelling. After the F27 fix the
    documented bare-integer spelling (`SpmdSubmitBareCore` below) is the same
    anchor: the operand-boundary host-literal lift is INDEX on both lanes."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=pl.const(8, pl.INDEX))
        return out


class SpmdSubmitStagedCore:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], n: pl.Scalar[pl.INDEX]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=n)
        return out


class SpmdSubmitSync:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=pl.const(8, pl.INDEX), sync_start=True)
        return out


class SpmdSubmitMetadata:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], rc: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, rc)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], rc: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(
            self.stage,
            x,
            rc,
            core_num=pl.const(8, pl.INDEX),
            dumps=[x],
            allow_early_resolve=True,
            timing_slot=3,
            predicate=(rc[0] > 0),
        )
        return out


class SubmitDevice:
    """`pl.submit(..., device=0)` — F35's spelling of F27's mismatch.

    `device=` rides the same operand boundary (`_opt_expr` -> `_submit_expr`)
    as `core_num=`, so the same host-int lift decides the dtype; the parser's
    `_parse_dispatch_device_kwarg` reads the value through `parse_expression`
    (a bare literal is INDEX)."""

    @pl.function(type=pl.FunctionType.Orchestration)
    def child(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return x

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.child, x, device=0)
        return out


class SpmdSubmitDeps:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        a, a_tid = pl.submit(self.stage, x)
        b, _ = pl.spmd_submit(self.stage, a, core_num=pl.const(4, pl.INDEX), deps=[a_tid])
        return b


def test_submit_basic() -> None:
    program_conformance(SubmitBasic, trace_program)


def test_submit_basic_canonical() -> None:
    program_conformance(SubmitBasic, trace_program, canonical=True)


def test_submit_deps() -> None:
    program_conformance(SubmitDeps, trace_program)


def test_submit_deps_canonical() -> None:
    program_conformance(SubmitDeps, trace_program, canonical=True)


def test_submit_deps_per_element() -> None:
    program_conformance(SubmitDepsPerElement, trace_program)


def test_submit_deps_whole_array() -> None:
    program_conformance(SubmitDepsWholeArray, trace_program)


def test_submit_deps_comprehension() -> None:
    program_conformance(SubmitDepsComprehension, trace_program)


def test_submit_dumps() -> None:
    program_conformance(SubmitDumps, trace_program)


def test_submit_dumps_canonical() -> None:
    program_conformance(SubmitDumps, trace_program, canonical=True)


def test_submit_early_resolve() -> None:
    program_conformance(SubmitEarlyResolve, trace_program)


def test_submit_predicate() -> None:
    program_conformance(SubmitPredicate, trace_program)


def test_submit_predicate_canonical() -> None:
    program_conformance(SubmitPredicate, trace_program, canonical=True)


def test_submit_timing_slot() -> None:
    program_conformance(SubmitTimingSlot, trace_program)


def test_submit_attrs_roundtrip() -> None:
    program_conformance(SubmitAttrsRoundTrip, trace_program)


def test_submit_multi_output() -> None:
    program_conformance(SubmitMultiOutput, trace_program)


def test_submit_multi_output_canonical() -> None:
    program_conformance(SubmitMultiOutput, trace_program, canonical=True)


def test_submit_no_dep() -> None:
    program_conformance(SubmitNoDep, trace_program)


def test_submit_no_dep_canonical() -> None:
    program_conformance(SubmitNoDep, trace_program, canonical=True)


def test_submit_in_staged_loop() -> None:
    program_conformance(SubmitInStagedLoop, trace_program)


def test_submit_in_staged_loop_canonical() -> None:
    program_conformance(SubmitInStagedLoop, trace_program, canonical=True)


def test_submit_out_omitted() -> None:
    program_conformance(SubmitOutOmitted, trace_program)


def test_submit_out_omitted_canonical() -> None:
    program_conformance(SubmitOutOmitted, trace_program, canonical=True)


def test_spmd_submit_index_core() -> None:
    program_conformance(SpmdSubmitIndexCore, trace_program)


def test_spmd_submit_index_core_canonical() -> None:
    program_conformance(SpmdSubmitIndexCore, trace_program, canonical=True)


def test_spmd_submit_bare_core() -> None:
    program_conformance(SpmdSubmitBareCore, trace_program)


def test_spmd_submit_bare_core_canonical() -> None:
    program_conformance(SpmdSubmitBareCore, trace_program, canonical=True)


def test_spmd_submit_staged_core() -> None:
    program_conformance(SpmdSubmitStagedCore, trace_program)


def test_spmd_submit_staged_core_canonical() -> None:
    program_conformance(SpmdSubmitStagedCore, trace_program, canonical=True)


def test_spmd_submit_sync() -> None:
    program_conformance(SpmdSubmitSync, trace_program)


def test_spmd_submit_deps() -> None:
    program_conformance(SpmdSubmitDeps, trace_program)


def test_spmd_submit_metadata() -> None:
    program_conformance(SpmdSubmitMetadata, trace_program)


def test_submit_device() -> None:
    program_conformance(SubmitDevice, trace_program)


def test_submit_device_literal_by_both_lanes() -> None:
    """**F35 (fixed)** — `device=0` stores an INDEX ConstInt on both lanes.

    F35 is F27's mismatch through the same `_submit_expr` host-`int` arm
    (`_opt_expr(device, ...)`): the tracing lane stored INT32 while
    `_parse_dispatch_device_kwarg`'s `parse_expression` produced the ordinary
    parser literal (INDEX). One root cause, one lift: the two F-ids are one
    fix-campaign task and `attrs["device"]` is INDEX on both lanes."""
    new = _first_submit(trace_program(SubmitDevice)["main"])
    ast_ = _first_submit(program_parser(SubmitDevice)["main"])
    assert isinstance((new.attrs or {}).get("device"), ir.ConstInt)
    assert isinstance((ast_.attrs or {}).get("device"), ir.ConstInt)
    assert (new.attrs or {})["device"].type.dtype == pl.INDEX
    assert (ast_.attrs or {})["device"].type.dtype == pl.INDEX
    ir.assert_structural_equal(
        trace_program(SubmitDevice),
        program_parser(SubmitDevice),
        enable_auto_mapping=True,
    )


# ===========================================================================
# R13 no-shared-spelling divergences and red-anchor pins
# ===========================================================================
#
# A divergence is *pinned*, never left as a red anchor: the pin asserts the
# exact shape difference (or the per-lane acceptance) so the finding stays
# reproducible and a fix reports as a failing pin. Every finding is recorded in
# docs/review-constructs.md §19 with its F-id; no frontend file is edited.

from pypto.language.diagnostics import ParserError  # noqa: E402


def _body_stmts(fn: ir.Function) -> list[ir.Stmt]:
    body = fn.body
    return list(body.stmts) if isinstance(body, ir.SeqStmts) else [body]


def _bound_names(fn: ir.Function) -> list[str]:
    names: list[str] = []
    for stmt in _body_stmts(fn):
        if isinstance(stmt, ir.AssignStmt):
            names.append(stmt.var.name_hint)
        names.extend(_bound_names_nested(stmt))
    return names


def _bound_names_nested(stmt: ir.Stmt) -> list[str]:
    names: list[str] = []
    for child in (
        getattr(stmt, "body", None),
        getattr(stmt, "else_body", None),
        *(getattr(stmt, "stmts", None) or []),
    ):
        if isinstance(child, ir.SeqStmts):
            names.extend(_bound_names_nested(child))
        elif isinstance(child, ir.AssignStmt):
            names.append(child.var.name_hint)
            names.extend(_bound_names_nested(child))
    return names


def _first_submit(fn: ir.Function) -> ir.Submit:
    found: list[ir.Submit] = []

    class _V(ir.IRVisitor):
        def visit_submit(self, op):  # noqa: N802 - the visitor API
            found.append(op)
            super().visit_submit(op)

    _V().visit_function(fn)
    assert found, f"no Submit in {fn.name}"
    return found[0]


class SubmitSingleLhs:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        res = pl.submit(self.stage, x)  # noqa: F841 - the single-LHS bind is the shape under test
        return y


def test_single_lhs_submit_by_both_lanes() -> None:
    """**F28 (fixed)** — `res = pl.submit(...)` binds the Submit directly.

    The F28 pin asserted the tracing lane's extra temporary: the single-result
    Submit facade carried a `__trace_bind__` hook that named the node
    `_submit_1` via `ctx.let_temp` and re-wrapped `res` over that temp, so the
    traced body carried one extra Var-to-Var copy. `_lower_submit`'s docstring
    claimed that temp mirrors "the AST parser's `_submit_tmp` shape", but the
    parser's `_submit_tmp` exists only on the two-target *unpacking* path
    (`_parse_submit_assignment`); its single-LHS path
    (`_parse_submit_single_lhs`) binds the Submit directly. R4 makes the
    parser's shape normative, so the hook is gone: the facade binds through
    the ordinary `_default_bind` and both lanes emit one statement
    (`simplify` never folded the copy). The raw witnesses below pin the shape
    on each lane — the harness equality alone would report only a statement
    count."""

    new = trace_program(SubmitSingleLhs)["main"]
    ast_ = program_parser(SubmitSingleLhs)["main"]
    assert len(_body_stmts(new)) == 2
    assert len(_body_stmts(ast_)) == 2
    assert _bound_names(new)[:2] == ["res"]
    assert _bound_names(ast_)[:2] == ["res"]
    program_conformance(SubmitSingleLhs, trace_program)


class SpmdSubmitBareCore:
    """`pl.spmd_submit(..., core_num=8)` — the bare host integer literal.

    The documented bare-integer rule (`docs/en/dev/language/00-python_syntax.md`
    §Expressions: *"A bare integer literal is always INDEX-typed"*) is the
    parser's `parse_constant` lift. The F27 fix makes the tracing operand
    boundary (`_submit_expr`) lift the same host `int` to INDEX, so this bare
    spelling is a plain conformance anchor equal to `SpmdSubmitIndexCore`."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=8)
        return out


def test_bare_core_num_literal_by_both_lanes() -> None:
    """**F27 (fixed)** — a bare `core_num=8` is INDEX-typed on both lanes.

    The F27 pin asserted the buggy shape: `_submit_expr`
    (`constructs/submit.py`) lifted the host `int` to INT32 — printed
    `pl.const(8, pl.INT32)` — while the parser's `parse_constant` lifted the
    same literal to INDEX (printed bare `8`). The lane that moved is tracing
    (R4: the parser's behaviour is normative) and the lift now follows the
    bare-integer rule the frontend's other operand boundaries
    (`typing/_host.py`) already apply, so `Submit.core_num` is an INDEX ConstInt
    on both lanes and the bare spelling joins the explicit
    `pl.const(8, pl.INDEX)` anchor. The same one-line lift also closes the
    unpinned spelling of the same defect (a bare host `int` kernel argument,
    e.g. `pl.submit(self.stage, x, 3)`)."""

    new = _first_submit(trace_program(SpmdSubmitBareCore)["main"])
    ast_ = _first_submit(program_parser(SpmdSubmitBareCore)["main"])
    assert isinstance(new.core_num, ir.ConstInt)
    assert isinstance(ast_.core_num, ir.ConstInt)
    assert new.core_num.type.dtype == pl.INDEX
    assert ast_.core_num.type.dtype == pl.INDEX
    ir.assert_structural_equal(
        trace_program(SpmdSubmitBareCore)["main"],
        program_parser(SpmdSubmitBareCore)["main"],
        enable_auto_mapping=True,
    )


class SubmitBoolArg:
    """`pl.submit(self.stage, x, True)` — the bare host boolean literal.

    The documented boolean rule (`docs/en/dev/language/00-python_syntax.md`:
    *"boolean values require `pl.BOOL`"*) is the parser's `parse_constant`
    lift to `ConstBool`. The F41 fix makes the tracing operand boundary
    (`_submit_expr`) build the same `ConstBool`, so the bare spelling is a
    plain conformance anchor that types its `pl.Scalar[pl.BOOL]` kernel
    parameter."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], flag: pl.Scalar[pl.BOOL]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, True)
        return out


def test_submit_bool_arg() -> None:
    program_conformance(SubmitBoolArg, trace_program)


def test_bool_submit_operand_by_both_lanes() -> None:
    """**F41 (fixed)** — a bare `True` submit operand is `ConstBool` on both
    lanes.

    The F41 pin asserted the buggy shape: `_submit_expr`'s `bool` arm
    (`constructs/submit.py`) lifted the host `bool` to `ConstInt(1, INT32)`
    while the parser's `parse_constant` built `ConstBool`, so the same accepted
    source carried a type lie against the callee's `pl.Scalar[pl.BOOL]` param
    (`docs/en/dev/language/00-python_syntax.md`: *"boolean values require
    `pl.BOOL`"*). Both lanes accept the spelling, so the tracing lane moved
    (R4: the parser is normative) and the lift follows the shared operand
    boundary (`typing/_host.py`'s `coerce_host_scalar`, which checks `bool`
    before `int`)."""

    new = _first_submit(trace_program(SubmitBoolArg)["main"])
    ast_ = _first_submit(program_parser(SubmitBoolArg)["main"])
    assert isinstance(new.args[1], ir.ConstBool)
    assert isinstance(ast_.args[1], ir.ConstBool)
    ir.assert_structural_equal(
        trace_program(SubmitBoolArg)["main"],
        program_parser(SubmitBoolArg)["main"],
        enable_auto_mapping=True,
    )


class SubmitDumpsDuplicate:
    """`pl.submit(..., dumps=[x, x])` — a duplicated dump entry.

    Both lanes accept the spelling; the F29 fix makes the tracing lane drop the
    repeat, so the duplicated form is a plain conformance anchor equal to
    `SubmitDumps`."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, dumps=[x, x])
        return out


def test_dumps_duplicate_entry_by_both_lanes() -> None:
    """**F29 (fixed)** — `dumps=[x, x]` records one `dump_vars` entry on both
    lanes.

    The F29 pin asserted the buggy shape: `_parse_submit_dumps_kwarg` drops the
    repeat by Var identity (and `_parse_kernel_call` rebuilds the vector through
    its positional-arg scan) while the tracing lane's `_coerce_dumps` coerced the
    `dumps=` list verbatim, so the same Var landed twice on the traced
    `Submit.attrs['dump_vars']`.

    The duplicate was not inert, so R5's redundant-drift carve-out does not
    apply: every consumer normalises the vector to an identity set (codegen's
    `CollectDumpVarSet`, the outliner's captured-input index, the printer's
    arg-order emission), so the duplicate has *no printed form* — `dumps=[x, x]`
    printed as `dumps=[x]` and re-parsed deduplicated, which means the traced IR
    failed the gate's own print→parse round-trip invariant (`RoundtripInstrument`,
    `python/pypto/ir/instruments.py`). It is not wrong codegen — `dumps=[x, x]`
    and `dumps=[x]` lower to byte-identical orchestration code (see
    `tests/ut/codegen/test_orchestration_manual_scope.py::
    test_duplicate_dumps_entries_lower_to_one_dump_arg`) — but the round-trip
    break stands, and R5's "trivial drift" reading requires the drift to be
    inert. R4 makes the parser's vector normative, so the lane that moved is
    tracing: `_coerce_dumps` now drops a repeated entry by identity, first
    occurrence kept.

    The order half of the parser's construction (the vector is rebuilt in
    positional-arg order) is a distinct symptom, filed as **F43**; the
    duplicated spelling here is unaffected by it (single entry)."""

    new = _first_submit(trace_program(SubmitDumpsDuplicate)["main"])
    ast_ = _first_submit(program_parser(SubmitDumpsDuplicate)["main"])
    new_dumps = list((new.attrs or {})["dump_vars"])
    ast_dumps = list((ast_.attrs or {})["dump_vars"])
    assert len(new_dumps) == 1
    assert len(ast_dumps) == 1
    assert new_dumps[0].name_hint == ast_dumps[0].name_hint == "x"
    program_conformance(SubmitDumpsDuplicate, trace_program)


def test_dumps_duplicate_entry_by_both_lanes_canonical() -> None:
    """**F29 (fixed)** — the duplicated `dumps=` spelling survives
    `canonicalize_program`."""

    program_conformance(SubmitDumpsDuplicate, trace_program, canonical=True)


class SubmitDumpsReordered:
    """`pl.submit(self.stage, x, y, dumps=[y, x])` over a `(x, y)` submit —
    the source-order spelling the parser rebuilds in positional-arg order.

    The F43 fix makes the tracing lane run the parser's positional-arg scan for
    an explicit-only `dumps=` vector too, so the reordered form is a plain
    conformance anchor equal to `SubmitDumps`."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, y)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, y, dumps=[y, x])
        return out


def test_dumps_reorder_by_both_lanes() -> None:
    """**F43 (fixed)** — the traced `dumps=` vector is rebuilt in
    positional-arg order, like the parser's.

    The F43 pin asserted the buggy shape: `_parse_kernel_call` builds the
    parser's `dump_vars` by scanning the submit's positional args and selecting
    the Vars named in `dumps=` (so the vector is arg-ordered as well as
    identity-deduped, F29's rule), while the tracing lane's
    `_merge_sticky_dump_args` returned an explicit-only vector in source order.
    `dumps=[y, x]` over args `(x, y)` therefore traced `[y, x]` and parsed
    `[x, y]`; the printer emits `dumps=` in arg order, so the traced program
    printed `dumps=[x, y]` and failed its own print→parse round trip
    (`RoundtripInstrument`, `python/pypto/ir/instruments.py`). R4 makes the
    parser's construction normative, so the tracing lane now runs the same
    positional-arg scan for every vector (tag-bearing or not) and the reordered
    spelling is a plain conformance anchor."""

    new = _first_submit(trace_program(SubmitDumpsReordered)["main"])
    ast_ = _first_submit(program_parser(SubmitDumpsReordered)["main"])
    assert [v.name_hint for v in (new.attrs or {})["dump_vars"]] == ["x", "y"]
    assert [v.name_hint for v in (ast_.attrs or {})["dump_vars"]] == ["x", "y"]
    traced = trace_program(SubmitDumpsReordered)
    submit_lines = [ln for ln in str(traced).splitlines() if "pl.submit(" in ln]
    assert submit_lines and "dumps=[x, y]" in submit_lines[0], submit_lines
    ir.assert_structural_equal(traced, pl.parse(str(traced)), enable_auto_mapping=True)
    program_conformance(SubmitDumpsReordered, trace_program)


def test_dumps_reorder_by_both_lanes_canonical() -> None:
    """**F43 (fixed)** — the reordered `dumps=` spelling survives
    `canonicalize_program`."""

    program_conformance(SubmitDumpsReordered, trace_program, canonical=True)


class SubmitDumpTag:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        pl.dump_tag(x)
        out, tid = pl.submit(self.stage, x)
        return out


class SpmdSubmitDumpTag:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        pl.dump_tag(x)
        out, tid = pl.spmd_submit(self.stage, x, core_num=pl.const(8, pl.INDEX))
        return out


class SubmitDumpTagAndDumps:
    """A tagged arg composed with the explicit `dumps=` kwarg: `y` is tagged
    and `x` is listed, over the argument order `(x, y)`."""

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, y)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        pl.dump_tag(y)
        out, tid = pl.submit(self.stage, x, y, dumps=[x])
        return out


def test_dump_tag_merged_into_a_submit_by_both_lanes() -> None:
    """**F30 (fixed)** — the forward-sticky `pl.dump_tag(x)` merge reaches the
    Submit path on both launch forms.

    `_parse_kernel_call` builds the Call/Submit `dump_vars` set from the tagged
    Vars for every call kind (the merge the `Marker.__call__` op boundary also
    performs for a plain call, anchored by
    `test_statement_handlers.py::...dump_tag...`); the tracing `_lower_submit`
    only ever set `dump_vars` from the dedicated `dumps=` kwarg, so the tag on
    a submitted argument was silently dropped: the parser Submit carried one
    `dump_vars` entry and the traced Submit none, and the traced program lost
    the user's declarative dump request. R4 (the parser's vector is normative)
    moves the tracing lane: the lowering now merges the tagged args with the
    parser's positional-arg scan (Var identity deduped), so a tagged arg and an
    explicit `dumps=` entry compose in arg order and the printed `dumps=[...]`
    kwarg re-parses to the same vector."""

    for cls in (SubmitDumpTag, SpmdSubmitDumpTag):
        new = _first_submit(trace_program(cls)["main"])
        ast_ = _first_submit(program_parser(cls)["main"])
        new_dumps = list((new.attrs or {})["dump_vars"])
        ast_dumps = list((ast_.attrs or {})["dump_vars"])
        assert len(new_dumps) == 1
        assert len(ast_dumps) == 1
        assert new_dumps[0].name_hint == ast_dumps[0].name_hint == "x"
        assert "dumps=[x]" in str(trace_program(cls))
        program_conformance(cls, trace_program)

    new = _first_submit(trace_program(SubmitDumpTagAndDumps)["main"])
    ast_ = _first_submit(program_parser(SubmitDumpTagAndDumps)["main"])
    assert [v.name_hint for v in (new.attrs or {})["dump_vars"]] == ["x", "y"]
    assert [v.name_hint for v in (ast_.attrs or {})["dump_vars"]] == ["x", "y"]
    program_conformance(SubmitDumpTagAndDumps, trace_program)


class SubmitPredicateNoDep:
    @pl.function
    def prod(self, rc: pl.Out[pl.Tensor[[16], pl.INT32]]) -> pl.Tensor[[16], pl.INT32]:
        return rc

    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32], rc: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, rc)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        rc, rc_tid = pl.submit(self.prod, x)
        out, tid = pl.submit(self.stage, x, rc, predicate=(rc[0] > 0))
        return out


def test_predicate_deps_contract_is_parser_only() -> None:
    """Documented intentional divergence: the parser performs the
    best-effort producer-in-``deps=`` check, while tracing leaves that
    scheduler contract to the author."""

    trace_program(SubmitPredicateNoDep)  # the tracing lane accepts
    with pytest.raises(ParserError, match="not in deps="):
        program_parser(SubmitPredicateNoDep)


class SubmitMultiOutputSingleTarget:
    @pl.function
    def multi(
        self, x: pl.Tensor[[16], pl.INT32]
    ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32]]:
        return pl.add(x, x), pl.sub(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.multi, x)
        return x


class SubmitMultiOutputTwoStepUnpack:
    """The same single result target, *consumed* by a second-step unpack.

    This is the superset spelling the parser's one-shot surface cannot
    express: the tracing lane's `out` is `_SubmitResult`'s nested outputs
    view, so the caller may name it and unpack it in a second statement
    (F32's R2 verdict, witnessed below)."""

    @pl.function
    def multi(
        self, x: pl.Tensor[[16], pl.INT32]
    ) -> tuple[pl.Tensor[[16], pl.INT32], pl.Tensor[[16], pl.INT32]]:
        return pl.add(x, x), pl.sub(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.multi, x)
        a, b = out
        return pl.add(a, b)


def test_multi_output_single_result_target_is_r2_lenient() -> None:
    """**F32 — accepted divergence under R2 (tracing is a superset); no fix.**

    ``out, tid = pl.submit(self.multi, x)`` binds the single result target to
    ``_SubmitResult``'s *nested outputs view* — the same Python tuple boundary
    the ergonomic ``(a, b), tid = pl.submit(...)`` target iterates — not to a
    dropped IR value: the two-target unpack consumes the view, so only the
    shared ``_submit_1`` temp and the TaskId projection reach the body. The
    parser's ``_parse_submit_assignment`` validates the result-target count
    against the callee's return arity and rejects the spelling
    (``pl.submit(...) unpacks 1 result value(s) but kernel 'multi' returns 2``),
    so there is no shared spelling: the tracing lane accepts *more*, never
    less.

    That extra acceptance is R2's legalised superset, not a silent drop. The
    view composes into a second-step unpack of the callee's results (the
    witness below), while every other use is loud at trace time
    (``pl.add(out, x)`` names ``_SubmitOutputs`` in an
    ``InvalidOperationError``; forwarding it to another ``pl.submit`` is the
    staged-operand ``TypeError``); the emitted IR keeps the Submit, all of its
    results, and the TaskId, and it print → reparses. Tightening the tracing
    lane to the parser's arity rule would delete the two-step unpack for a
    parity goal R2 does not create (``docs/fix-construct-parity.md`` §4:
    *'tracing may be a superset'*), so the assertions stay as the
    accepted-divergence witness — do not tighten a lane to make this test
    disappear."""

    traced = trace_program(SubmitMultiOutputSingleTarget)
    new = traced["main"]
    # The discarded name owns no IR binding, and nothing else is missing:
    # the whole body is the shared temp plus `tid`.
    assert "out" not in _bound_names(new)
    assert _bound_names(new) == ["_submit_1", "tid"]
    # The accepted shape has a printed form and re-parses (no round-trip break).
    ir.assert_structural_equal(pl.parse(str(traced)), traced, enable_auto_mapping=True)
    with pytest.raises(ParserError, match="unpacks 1 result value"):
        program_parser(SubmitMultiOutputSingleTarget)


def test_multi_output_result_view_composes_into_a_two_step_unpack() -> None:
    """The R2 capability behind the single-result-target leniency (F32).

    The single result target is ``_SubmitResult``'s nested outputs view, so a
    caller may name it and unpack it afterwards: the second step binds the
    callee's two results off the shared Submit — statements the parser cannot
    produce, because its surface demands the one-shot ``(a, b), tid = ...``
    target (the same arity rule the witness above pins). That is why F32's
    acceptance is a superset rather than a dropped binding; the traced IR is
    print → reparse stable."""

    traced = trace_program(SubmitMultiOutputTwoStepUnpack)
    new = traced["main"]
    assert _bound_names(new) == ["_submit_1", "tid", "a", "b"]
    lets = {s.var.name_hint: s for s in _body_stmts(new) if isinstance(s, ir.AssignStmt)}
    for name, index in (("a", 0), ("b", 1)):
        projection = lets[name].value
        assert isinstance(projection, ir.TupleGetItemExpr)
        assert projection.index == index
        base = projection.tuple
        assert isinstance(base, ir.Var) and base.name_hint == "_submit_1"
    # The Submit still carries every result plus the TaskId.
    submit_type = _first_submit(new).type
    assert isinstance(submit_type, ir.TupleType) and len(submit_type.types) == 3
    ir.assert_structural_equal(pl.parse(str(traced)), traced, enable_auto_mapping=True)
    with pytest.raises(ParserError, match="unpacks 1 result value"):
        program_parser(SubmitMultiOutputTwoStepUnpack)


class SubmitDeviceNonOrchestrator:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, device=0)
        return out


def test_submit_device_kwarg_role_check_is_parser_only() -> None:
    """**needs-decision** — `device=` on a submit whose callee is not an
    Orchestrator.

    The parser rejects it (`_validate_kernel_call_kwargs` adds `device` only for
    a `Role.Orchestrator` callee) and the tracing *plain-call* path does too
    (`Marker.__call__`; pinned in `test_call.py`), but the tracing submit path
    forwards `device=` unconditionally. The IR is identical when both lanes
    accept, so this is an acceptance-only gap; no doc decides whether a Submit's
    dispatch device may target a non-Orchestrator."""

    fn = trace_program(SubmitDeviceNonOrchestrator)
    assert "device" in _submit_attrs(fn)
    with pytest.raises(ParserError, match="does not accept keyword argument 'device'"):
        program_parser(SubmitDeviceNonOrchestrator)


def _submit_attrs(prog: ir.Program) -> dict:
    main = prog["main"]
    assert main is not None
    return dict(_first_submit(main).attrs or {})


class SpmdSubmitBoolCore:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=True)
        return out


class SpmdSubmitZeroCore:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=0)
        return out


class SpmdSubmitStagedZeroCore:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.spmd_submit(self.stage, x, core_num=pl.const(0, pl.INDEX))
        return out


def test_spmd_core_num_non_positive_rejected_by_both_lanes() -> None:
    """**F33 (fixed)** — ``core_num`` is a positive integer expression on both
    lanes. ``_parse_and_validate_core_num`` is the contract
    (`docs/en/dev/language/02-manual_dependencies.md`: the launch width is a
    "positive int expr"), and the tracing lane's own sibling launch heads
    already enforce it (``pl.spmd``'s ``_pin_spmd_core_num``,
    ``pl.intrin.spmd_direct``): the ``pl.spmd_submit`` kwarg was the one
    unchecked entry, and a zero-block launch whose Submit still claims a TaskId
    is a hole, not a superset spelling (R4).

    The rule reads the coerced expression, so the host literal and the staged
    ``pl.const(0, pl.INDEX)`` spelling reject alike — the parser's
    ``ConstInt`` check sees the same constant."""

    with pytest.raises(ValueError, match="must be a positive integer"):
        trace_program(SpmdSubmitZeroCore)
    with pytest.raises(ParserError, match="must be a positive integer"):
        program_parser(SpmdSubmitZeroCore)
    with pytest.raises(ValueError, match="must be a positive integer"):
        trace_program(SpmdSubmitStagedZeroCore)
    with pytest.raises(ParserError, match="must be a positive integer"):
        program_parser(SpmdSubmitStagedZeroCore)


def test_spmd_core_num_bool_rejected_by_both_lanes() -> None:
    """**F33 (fixed)** — a boolean ``core_num`` is not an integer expression.

    The tracing operand boundary's ``bool`` arm would otherwise coerce ``True``
    to an ``INT32`` ``1`` (the arm F41 pins) and launch one block; the parser
    rejects it because ``ConstBool`` is not integer-typed, and so do the
    tracing lane's sibling launch heads (``pl.spmd``, ``pl.intrin.spmd_direct``
    — the latter with the same ``TypeError``)."""

    with pytest.raises(TypeError, match="must be an integer expression"):
        trace_program(SpmdSubmitBoolCore)
    with pytest.raises(ParserError, match="must be an integer expression"):
        program_parser(SpmdSubmitBoolCore)


class SubmitPredicateShape:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, predicate=(x[0] % 8 == 0))
        return out


class SubmitDepsNone:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, deps=None, dumps=None)
        return out


class SubmitDumpsNonArgument:
    @pl.function
    def stage(self, x: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        return pl.add(x, x)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(self, x: pl.Tensor[[16], pl.INT32], y: pl.Tensor[[16], pl.INT32]) -> pl.Tensor[[16], pl.INT32]:
        out, tid = pl.submit(self.stage, x, dumps=[y])
        return out


def test_predicate_shape_is_codegen_owned() -> None:
    """The tracing lane records an arithmetic predicate for later codegen,
    while the parser rejects it at the frontend; the docs explicitly assign
    this expressiveness check to orchestration codegen."""
    new = trace_program(SubmitPredicateShape)
    assert isinstance(_first_submit(new["main"]).predicate, ir.Eq)
    with pytest.raises(ParserError, match="must compare one tensor element"):
        program_parser(SubmitPredicateShape)


def test_deps_none_and_non_argument_dumps_are_tracing_only() -> None:
    """**needs-decision** — two `deps=`/`dumps=` value-shape checks the parser
    owns and the tracing operand boundary does not.

    `deps=None` / `dumps=None` are a falsy no-op on the tracing side and a loud
    "must be a list / tuple / comprehension" on the parser side; a `dumps=`
    entry that is not a positional argument of the submit is rejected by the
    parser ("not an argument of this submit") and accepted (then recorded in
    `dump_vars`) by the tracing lane. Acceptance-only: no doc decides the
    boundary. The tracing lane *is* the strict one for a bare `deps=<var>`
    (rejected: "must be a list"), so the two lanes are inconsistent in both
    directions."""

    trace_program(SubmitDepsNone)
    with pytest.raises(ParserError, match="must be a list / tuple / comprehension"):
        program_parser(SubmitDepsNone)

    trace_program(SubmitDumpsNonArgument)
    with pytest.raises(ParserError, match="is not an argument of this submit"):
        program_parser(SubmitDumpsNonArgument)


class SubmitOutMidOmitted:
    """A *supplied* `In` param after an omitted mid-signature `Out`."""

    @pl.function
    def stage(
        self,
        x: pl.Tensor[[16, 16], pl.FP32],
        out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
        y: pl.Tensor[[16, 16], pl.FP32],
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        t = pl.load(x, [0, 0], [16, 16])
        out2 = pl.store(t, [0, 0], out)
        u = pl.load(y, [0, 0], [16, 16])
        return pl.store(u, [0, 0], out2)

    @pl.function(type=pl.FunctionType.Orchestration)
    def main(
        self, x: pl.Tensor[[16, 16], pl.FP32], y: pl.Tensor[[16, 16], pl.FP32]
    ) -> pl.Tensor[[16, 16], pl.FP32]:
        out, tid = pl.submit(self.stage, x, y)
        return out


def test_omitted_mid_signature_out_param_is_parser_only() -> None:
    """**needs-decision** — the two lanes' submit arity rules differ.

    The parser's window (`lo` = non-Out/InOut count, `hi` = all params) plus
    `_call_args_for_return_deduction`'s Out-skipping are general: it accepts
    `pl.submit(self.stage, x, y)` with `stage(x, out: Out, y)`. The tracing
    `_check_submit_callee` shifts by the *trailing* CommCtx suffix only, so
    without such a suffix it reads `y` as an omitted `In` param and raises
    (`missing positional argument 2`). The tracing suite's own
    `test_submit_omits_mid_signature_out_across_the_comm_ctx_suffix` covers the
    suffix-carrying shape; no doc decides the suffix-free one. The parser's
    acceptance is the *broader* rule, and whether the resulting Submit is
    well-formed is not settled here."""

    program_parser(SubmitOutMidOmitted)  # the parser accepts
    with pytest.raises(TypeError, match="missing positional argument 2"):
        trace_program(SubmitOutMidOmitted)
