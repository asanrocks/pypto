# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``pl.jit`` envelope + hardening.

The tracing-frontend jit (the ``pypto/jit/`` device-compile frontend is a
separate, coexisting subsystem): eager ⊂ lazy. The
eager gate — every param a facade annotation (concrete or DynVar-shaped,
both annotation-determined) — traces at decoration and equals
``function_parser``; the lazy gate — an unannotated/bare/Callable param —
defers to call, baking the real args (tensor facades → static-shape
params, host scalars → Const at the op boundary), re-tracing per call (no
cache). Device compile/cache/execute is out of scope here: the envelope
returns the traced IR.

Hardening: the out-of-scope construct gates (span-pointed
``NotImplementedError``/``RuntimeError`` naming the missing feature) and the
auto-unwind regression (a mid-trace raise LIFO-closes the open builders).
"""

import pypto.language as pl
import pytest
from pypto.language.parser import function_parser
from pypto.language.tracer.decorators.function import trace, trace_program
from pypto.language.tracer.decorators.jit import _JitSpecializer, jit_trace
from pypto.pypto_core import DataType, ir

from ._harness import canonicalize

# ===========================================================================
# The eager gate (annotation-determined → traced at decoration)
# ===========================================================================


def eager_kernel(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.mul(t, 2.0)
    pl.store(s, [0, 0], x)
    return x


def test_eager_gate_returns_function():
    """The annotation-determined kernel traces at decoration — `jit_trace(k)`
    returns the jit object whose ``.fn`` is the decoration trace (the host
    compile/dispatch lanes serve it; a bare ``ir.Function`` has no such
    surface)."""
    fn = jit_trace(eager_kernel)
    assert isinstance(fn.fn, ir.Function)
    assert fn.fn.name == "eager_kernel"


def test_eager_gate_conformance():
    """The eager gate's conformance: the decoration trace equals
    `function_parser(k)` raw and post-canonical (the traced-IR gate — only for
    the eager case)."""
    fn = jit_trace(eager_kernel).fn
    assert isinstance(fn, ir.Function)
    ast_ = function_parser(eager_kernel)
    ir.assert_structural_equal(fn, ast_, enable_auto_mapping=True)
    ir.assert_structural_equal(canonicalize(fn), canonicalize(ast_), enable_auto_mapping=True)


N = pl.dynamic("N")


def eager_dyn_kernel(x: pl.Tensor[[N, 64], pl.FP32]) -> pl.Tensor[[N, 64], pl.FP32]:
    return pl.add(x, 1.0)


def test_eager_gate_dynvar_shape():
    """A DynVar-shaped annotation IS annotation-determined — eager.
    The DynVar dim unwraps to its shared extent symbol (`_shape_dim`)."""
    fn = jit_trace(eager_dyn_kernel).fn
    assert isinstance(fn, ir.Function)
    assert isinstance(fn.params[0].type, ir.TensorType)


# ===========================================================================
# The lazy gate (call-time info → defer to call)
# ===========================================================================


def lazy_kernel(x: pl.Tensor, scale: float) -> pl.Tensor:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.mul(t, pl.const(scale, pl.FP32))
    pl.store(s, [0, 0], x)
    return x


def test_param_directions_survive_sourceless_kernels():
    """The direction source is the runtime annotation (``Out[T]`` carries
    its direction on the returned facade copy/subclass), not the kernel
    source — a generated / exec'd kernel keeps its ``Out`` directions. The
    retired source walk silently degraded to all-``In`` for such kernels.
    (The kernel still cannot be *traced* — the instrumentor is
    source-based — but the signature facts no longer depend on it.)"""
    from pypto.language.typing.arg_direction import param_directions  # noqa: PLC0415

    namespace = {"pl": pl}
    exec(  # noqa: S102 - the sourceless-kernel case under test
        "def generated(a, c: pl.Out[pl.Tensor]):\n    return c\n",
        namespace,
    )
    assert param_directions(namespace["generated"]) == {"c": ir.ParamDirection.Out}

    # the instance form (a shaped facade) carries its direction on the copy
    exec(
        "def shaped(a: pl.Tensor[[64], pl.FP32], c: pl.InOut[pl.Tensor[[64], pl.FP32]]):\n    return c\n",
        namespace,
    )
    assert param_directions(namespace["shaped"]) == {"c": ir.ParamDirection.InOut}


_M = pl.dynamic("M")


def k_bind_redundant(a: pl.Tensor[[_M, 64], pl.FP32]) -> pl.Tensor[[_M, 64], pl.FP32]:
    a.bind_dynamic(0, _M)  # already dynamic via the annotation — benign
    return a


def k_bind_override(a: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    a.bind_dynamic(0, _M)  # would promote a static dim — dropped semantics
    return a


def test_bind_dynamic_redundant_is_noop():
    """bind_dynamic on an already-dynamic dim (annotation-embedded
    pl.dynamic — pypto-lib's ~70-use pattern) is a benign no-op: the
    annotation already carries the DynVar."""
    fn = jit_trace(k_bind_redundant)
    assert isinstance(fn.fn, ir.Function)


def test_bind_dynamic_override_is_loud():
    """bind_dynamic on a static dim raises: the override form's semantics
    lived in the specializer's AST scan (retired with the tracing swap) —
    silent-drop is forbidden; the quarantined runtime still honors it."""
    with pytest.raises(NotImplementedError, match="static"):
        jit_trace(k_bind_override)


def test_lazy_gate_defers_to_call():
    """An unannotated-shape param (bare pl.Tensor) + a host scalar force
    lazy: `jit_trace` returns a specializer, not the IR."""
    spec = jit_trace(lazy_kernel)
    assert isinstance(spec, _JitSpecializer)


def test_lazy_call_bakes_and_traces():
    """At call with concrete args: the tensor facade becomes a static-shape
    param, the host scalar is baked (the ops lift it as Const at the
    operand boundary); the fresh ir.Function is returned."""
    spec = jit_trace(lazy_kernel)
    assert isinstance(spec, _JitSpecializer)
    x = pl.Tensor([64, 64], pl.FP32, _annotation_only=True)
    fn = spec(x, 2.0)
    assert isinstance(fn, ir.Function)
    assert [p.name_hint for p in fn.params] == ["x"]  # the scalar is host, no IR param
    assert "muls(t, 2.0)" in str(fn)  # the baked Const


def test_lazy_retraces_per_call_no_cache():
    """Re-trace per call (no cache): a different scalar value bakes a
    different Const — each call's specialization is fresh (the cache-key-
    includes-the-value concern is concept-3, out of scope)."""
    spec = jit_trace(lazy_kernel)
    assert isinstance(spec, _JitSpecializer)
    x = pl.Tensor([64, 64], pl.FP32, _annotation_only=True)
    fn2 = spec(x, 2.0)
    fn4 = spec(x, 4.0)
    assert "muls(t, 2.0)" in str(fn2)
    assert "muls(t, 4.0)" in str(fn4)


def test_lazy_call_guardrails():
    """The specializer's guards: arg-count mismatch, kwargs, expr-wrapping
    (shape-less) facades — all loud."""
    spec = jit_trace(lazy_kernel)
    assert isinstance(spec, _JitSpecializer)
    x = pl.Tensor([64, 64], pl.FP32, _annotation_only=True)
    with pytest.raises(TypeError, match="expects 2 argument"):
        spec(x)
    with pytest.raises(TypeError, match="keyword arguments"):
        spec(x, 2.0, extra=3)  # type: ignore[call-arg]
    runtime_x = pl.Tensor(expr=ir.Var("x", ir.ScalarType(DataType.FP32), ir.Span.unknown()))
    with pytest.raises(TypeError, match="annotation-mode"):
        spec(runtime_x, 2.0)


# ===========================================================================
# Out-of-scope construct gates (each names the missing feature)
# ===========================================================================


def k_pipeline(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.pipeline(2, stage=2):
        x = pl.add(x, 1.0)
    return x


def test_gate_pipeline():
    """``pl.pipeline(stage=)`` is a trace iterable: the loop traces with
    ``ForKind.Pipeline`` and the ``pipeline_stages`` attr
    (``lower_pipeline_loops`` consumes it, and the printer round-trips it)."""
    fn = trace(k_pipeline)
    assert fn is not None
    assert "pl.pipeline" in str(fn)


def k_parallel(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.parallel(2):
        x = pl.add(x, 1.0)
    return x


def test_gate_parallel():
    """``pl.parallel`` emits a ``ForStmt`` with ``ForKind.Parallel`` — the
    scheduler assigns cores/blocks at lowering (the kind is the IR-level
    contract; no silent Sequential-for divergence)."""
    fn = trace(k_parallel)
    fors = [s for s in _walk_stmts(fn.body) if isinstance(s, ir.ForStmt)]
    assert fors, "expected a ForStmt"
    assert fors[0].kind == ir.ForKind.Parallel


def k_unroll(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for i in pl.unroll(2):
        x = pl.add(x, 1.0)
    return x


def test_gate_unroll():
    """``pl.unroll`` is a trace iterable: the loop traces with
    ``ForKind.Unroll`` and the printer round-trips it."""
    fn = trace(k_unroll)
    assert fn is not None
    assert "pl.unroll" in str(fn)


def k_at_deps(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    # deps= takes TaskIds (no_dep_args= takes tensors): a producer scope's
    # `as tid` capture provides one.
    with pl.at(level=pl.Level.CORE_GROUP, name_hint="producer") as tid:
        t = pl.add(x, x)
    with pl.at(level=pl.Level.CORE_GROUP, deps=[tid]):
        y = pl.add(t, 1.0)
    return y


def test_gate_at_extended_kwargs():
    """``pl.at(deps=...)`` threads the deps onto the scope's
    ``manual_dep_edges`` attr (consumed by ``auto_derive_task_dependencies``
    + ``Arg::set_dependencies``); the scope body collects. The dep entry is
    a captured TaskId — the parser-accepted vocabulary (a tensor entry is
    parser-rejected and cannot resolve at codegen)."""
    fn = trace(k_at_deps)
    scopes = [s for s in _walk_stmts(fn.body) if isinstance(s, ir.InCoreScopeStmt)]
    assert len(scopes) == 2, "expected the producer and consumer InCore scopes"
    producer, consumer = scopes
    assert "task_id_var" in producer.attrs
    assert consumer.attrs.get("manual_dep_edges"), "deps did not thread onto manual_dep_edges"
    assert "tensor.add" in str(fn)  # the scope bodies collected


def k_at_slot_num(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(
        level=pl.Level.CORE_GROUP,
        optimizations=[pl.split(pl.SplitMode.UP_DOWN, slot_num=16)],
    ):
        x = pl.add(x, 1.0)
    return x


def test_gate_at_slot_num():
    """``pl.at(optimizations=[pl.split(mode, slot_num=N)])`` threads the
    slot onto the InCore scope's ``slot_num`` attr and matches the AST
    frontend (mirrors the parser's ``_append_split_slot_num_attr`` —
    ``ExpandMixedKernel`` sizes the cross-core ring from it)."""
    fn = trace(k_at_slot_num)
    scopes = [s for s in _walk_stmts(fn.body) if isinstance(s, ir.InCoreScopeStmt)]
    assert scopes, "expected an InCoreScopeStmt"
    assert scopes[0].split == ir.SplitMode.UP_DOWN
    assert scopes[0].attrs.get("slot_num") == 16
    # AST conformance (raw + post-canonical)
    ast_ = function_parser(k_at_slot_num)
    ir.assert_structural_equal(fn, ast_, enable_auto_mapping=True)
    ir.assert_structural_equal(canonicalize(fn), canonicalize(ast_), enable_auto_mapping=True)


def k_at_non_split(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP, optimizations=["bogus"]):  # type: ignore[list]
        x = pl.add(x, 1.0)
    return x


def test_gate_at_optimizations_rejects_non_split():
    """``pl.at(optimizations=[...])`` rejects non-split entries (mirrors the
    AST parser's ``Unsupported entry`` rejection — not a deferred semantic)."""
    with pytest.raises(TypeError, match="must be pl.split"):
        trace(k_at_non_split)


def k_split_aiv(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    for aiv in pl.split_aiv(2, mode=pl.SplitMode.UP_DOWN):
        x = pl.add(x, 1.0)
    return x


def test_gate_split_aiv():
    """``pl.split_aiv``'s loop form emits the first-class
    ``SplitAivScopeStmt`` region carrying the split mode, wrapped in a
    synthesized ``InCoreScopeStmt`` so OutlineIncoreScopes can outline it;
    the mode-inheritance channel is pushed in ``begin``."""
    fn = trace(k_split_aiv)
    stmts = _walk_stmts(fn.body)
    inc = [s for s in stmts if isinstance(s, ir.InCoreScopeStmt)]
    sa = [s for s in stmts if isinstance(s, ir.SplitAivScopeStmt)]
    assert sa, "expected a SplitAivScopeStmt"
    assert sa[0].split == ir.SplitMode.UP_DOWN
    assert inc, "bare-top-level split_aiv must wrap in an InCoreScopeStmt"


# ---------------------------------------------------------------------------
# pl.spmd(optimizations=[pl.split(mode, slot_num=N)]) — the split-optimization
# hint wraps the dispatch body in an InCoreScopeStmt(split=mode, slot_num attr),
# mirroring the AST parser's _parse_spmd_for_loop shape. AST-structural equality
# is blocked by the trace_program func_type propagation gap (main stays Opaque,
# not Orchestration) — a pre-existing issue orthogonal to this feature; assert
# the IR shape directly.
# ---------------------------------------------------------------------------


def _spmd_split_prog(*, with_form: bool, slot_num: int | None):
    opts = (
        [pl.split(pl.SplitMode.UP_DOWN, slot_num=slot_num)]
        if slot_num is not None
        else [pl.split(pl.SplitMode.UP_DOWN)]
    )

    class Prog:
        @pl.function
        def k(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            return pl.add(x, 1.0)

        @pl.function(type=pl.FunctionType.Orchestration)
        def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            if with_form:
                with pl.spmd(4, optimizations=opts) as _tid:
                    x = self.k(x)
            else:
                for _i in pl.spmd(4, optimizations=opts):
                    x = self.k(x)
            return x

    return trace_program(Prog)


def _assert_spmd_split_shape(prog, *, slot_num: int | None):
    main = next(f for f in prog.functions.values() if f.name == "main")
    stmts = _walk_stmts(main.body)
    spmds = [s for s in stmts if isinstance(s, ir.SpmdScopeStmt)]
    incs = [s for s in stmts if isinstance(s, ir.InCoreScopeStmt)]
    assert spmds, "expected a SpmdScopeStmt"
    assert incs, "expected an InCoreScopeStmt (the split-optimization wrap)"
    assert isinstance(spmds[0].body, ir.InCoreScopeStmt), "InCoreScopeStmt must nest inside the SpmdScopeStmt"
    assert incs[0].split == ir.SplitMode.UP_DOWN
    if slot_num is not None:
        assert incs[0].attrs.get("slot_num") == slot_num
    else:
        assert "slot_num" not in incs[0].attrs


def test_gate_spmd_split_loop_with_slot_num():
    _assert_spmd_split_shape(_spmd_split_prog(with_form=False, slot_num=16), slot_num=16)


def test_gate_spmd_split_loop_no_slot_num():
    _assert_spmd_split_shape(_spmd_split_prog(with_form=False, slot_num=None), slot_num=None)


def test_gate_spmd_split_with_form_slot_num():
    _assert_spmd_split_shape(_spmd_split_prog(with_form=True, slot_num=12), slot_num=12)


# ===========================================================================
# Auto-unwind regression: a mid-trace raise LIFO-closes builders
# ===========================================================================


def k_raise_in_scopes(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        for i in pl.range(2):
            pl.load(x, [0], [64])
            raise ValueError("mid-trace")
    return x


def test_auto_unwind_mid_trace_raise():
    """A host raise mid-trace (open at-scope + loop) must not crash the C++
    builder: the InstrumentedFunction.__call__ finally LIFO-closes the open scopes
    (Interpreter.close), then the ValueError propagates."""
    with pytest.raises(ValueError, match="mid-trace"):
        trace(k_raise_in_scopes)


def _walk_stmts(stmt) -> list:
    """Yield all sub-stmts (BFS) for structural assertions on a traced
    function's body."""
    out = [stmt]
    if hasattr(stmt, "body"):
        out.extend(_walk_stmts(stmt.body))
    if hasattr(stmt, "else_body") and stmt.else_body is not None:
        out.extend(_walk_stmts(stmt.else_body))
    if hasattr(stmt, "stmts"):
        for s in stmt.stmts:
            out.extend(_walk_stmts(s))
    return out
