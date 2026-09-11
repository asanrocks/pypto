# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The program-harness conformance suite: probes + corpus + retirements.

It covers:

- **Program-level structural equality**: ``assert_structural_equal`` works directly on ``ir.Program``
  with auto-mapping — two independent parses of the same class are
  structurally equal. This pins ``program_conformance``'s shape: a direct
  Program-level assertion, no per-function-by-name walk.
- **``program_conformance``**: ``build(cls)`` == ``program_parser(cls)``, raw
  or post-canonical; ``canonicalize_program`` runs the canonical pair over
  a whole Program.
- **The inline regime**: ``canonicalize(fn, inline=True, callees=[...])``
  packages [caller + callee] into a Program, runs ``inline_functions``
  (the first pipeline pass) standalone, extracts the caller back.
- **Multi-function build placeholders**: ``trace(kernel)``
  / ``trace_program(cls)`` delegate to the parser frontend.
- **The corpus** (AST-green first — the anchor authority side must parse):
  the ``softmax_tile`` helper, the gemm-with-calls caller, and the
  2-method program class with a forward-ref (entry defined before helper).
- **Retirements**: ``PtoFrame`` is a thin handle — no
  ``is_orchestration``/``set_function_type`` on the Sema or Frame; ``PtoSema``
  carries ``function_type``/``level``/``role``/``attrs`` →
  ``begin_function(type=…)``.
- **The host-param decision**: host-passthrough — a non-facade annotation
  passes through unchanged (no staged local, no IR param) instead of raising.
"""

import pypto.language as pl
from pypto import ir
from pypto.ir import IRBuilder
from pypto.language.parser import function_parser, program_parser
from pypto.language.tracer.decorators.function import function_trace, trace, trace_program
from pypto.language.tracer.semantics import PtoSema
from pypto.pypto_core import DataType, passes

from ._harness import canonicalize, canonicalize_program, program_conformance

# ===========================================================================
# The corpus (AST-green first — the anchor authority side must parse)
# ===========================================================================


# The standalone helper: the module-level name ``softmax_tile`` binds the
# AST-built ir.Function so the caller's ``function_parser`` resolves the call
# via closure_vars (the AST's _parse_external_function_call). Note the
# AST's GlobalVar uses the *definition* name (``softmax_tile_fn``), not the
# binding name — the tracing frontend's Marker must record the def name.
def softmax_tile_fn(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    s = pl.mul(t, 2.0)
    pl.store(s, [0, 0], x)
    return x


softmax_tile = function_parser(softmax_tile_fn)


# The gemm-with-calls standalone caller: ``softmax_tile(y)`` is a bare-name
# call to the external ir.Function.
def gemm_caller(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    t = pl.load(x, [0, 0], [64, 64])
    y = pl.store(t, [0, 0], x)
    return softmax_tile(y)


# The program-class corpus. Two classes: the main ``SoftmaxProgram`` is
# helper-first (``scale_tile`` before its caller) so it is canonical-regime
# safe — the AST's own roundtrip instrument (the tests/ut/conftest default)
# chokes on a forward-ref ``UnknownType`` call whose callee is in the same
# program (print → reparse recovers the derived type). The forward-ref case
# lives in ``ForwardRefProgram`` (entry defined before helper) and is
# exercised raw — the anchor records the tracing frontend's
# declared-type improvement over the AST's UnknownType.
class SoftmaxProgram:
    @pl.function
    def scale_tile(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        t = pl.load(x, [0, 0], [64, 64])
        s = pl.mul(t, 2.0)
        pl.store(s, [0, 0], x)
        return x

    @pl.function
    def entry(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.scale_tile(x)


class ForwardRefProgram:
    @pl.function
    def entry(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        return self.scale_tile(x)

    @pl.function
    def scale_tile(self, x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
        t = pl.load(x, [0, 0], [64, 64])
        s = pl.mul(t, 2.0)
        pl.store(s, [0, 0], x)
        return x


# ===========================================================================
# Program-level structural equality probe
# ===========================================================================


def test_program_structural_equal_o12():
    """``assert_structural_equal`` works directly on
    ``ir.Program`` with auto-mapping — two independent parses of the same
    class create entirely distinct Var/GlobalVar objects and still compare
    equal (shape + names, never identity). Pins ``program_conformance``'s
    shape: a direct Program-level assertion, no per-function walk. Probed
    on both corpus programs (incl. the forward-ref class)."""
    for cls in (SoftmaxProgram, ForwardRefProgram):
        p1 = program_parser(cls)
        p2 = program_parser(cls)
        ir.assert_structural_equal(p1, p2, enable_auto_mapping=True)
        assert p1.name == p2.name == cls.__name__


def test_program_structural_equal_cross_constructors():
    """Hardening: a program rebuilt from the same Functions via the
    ir.Program constructor equals the decorator-built program — the
    constructor path (which the harness uses for the inline regime) is
    structurally interchangeable."""
    p1 = program_parser(SoftmaxProgram)
    p2 = ir.Program(list(p1.functions.values()), p1.name, ir.Span.unknown())
    ir.assert_structural_equal(p1, p2, enable_auto_mapping=True)


# ===========================================================================
# The harness: program_conformance + canonicalize_program + the inline regime
# ===========================================================================


def test_program_conformance_identity():
    """``program_conformance`` with ``build=program_parser`` is an identity —
    the harness mechanics pinned on the real program. Raw and post-canonical (straight-line bodies are
    already canonical, so the pair is a no-op). The canonical regime uses
    the helper-first corpus (the forward-ref program is not roundtrip-
    stable under the conftest's roundtrip instrument — see the corpus
    note)."""
    program_conformance(SoftmaxProgram, build=program_parser)
    program_conformance(SoftmaxProgram, build=program_parser, canonical=True)


def test_canonicalize_program_runs_pair():
    """``canonicalize_program`` runs convert_to_ssa + simplify over the
    whole Program and returns a Program with every function."""
    prog = program_parser(SoftmaxProgram)
    canonical = canonicalize_program(prog)
    assert isinstance(canonical, ir.Program)
    assert {f.name for f in canonical.functions.values()} == {"entry", "scale_tile"}
    # straight-line corpus: the pair is a no-op, so canonical == raw
    ir.assert_structural_equal(prog, canonical, enable_auto_mapping=True)


def test_canonicalize_inline_splices_and_extracts():
    """The inline regime probe: ``canonicalize(fn, inline=True,
    callees=[...])`` packages [caller + callee] into a Program, runs
    ``inline_functions`` — the first pipeline pass — standalone over it
    (no full pipeline needed), and extracts the caller back with the
    callee's body spliced at the call site."""
    ib = IRBuilder()
    t64 = ir.TensorType([ir.ConstInt(64, DataType.INDEX, ir.Span.unknown())], DataType.FP32, None)
    one = ir.ConstFloat(1.0, DataType.FP32, ir.Span.unknown())

    with ib.program("inline_probe") as prog:
        g_inline = prog.declare_function("add_one")
        with ib.function("add_one", type=ir.FunctionType.Inline) as f:
            x = f.param("x", t64)
            f.return_type(t64)
            ib.return_stmt(ir.Call(ir.Op("tensor.adds"), [x, one], t64, ir.Span.unknown()))
        add_one = f.get_result()
        prog.add_function(add_one)

        with ib.function("caller") as f:
            x = f.param("x", t64)
            f.return_type(t64)
            ib.return_stmt(ir.Call(g_inline, [x], t64, ir.Span.unknown()))
        caller = f.get_result()
        prog.add_function(caller)

    spliced = canonicalize(caller, inline=True, callees=[add_one])
    assert isinstance(spliced, ir.Function)
    printed = str(spliced)
    # the Call to the inline gvar is gone; the callee's body is spliced in
    assert "add_one" not in printed
    assert "tensor.adds" in printed


def test_inline_functions_removes_inline_function():
    """The pass's contract: after splicing, the Inline callee is gone from
    the Program (only the caller remains)."""
    ib = IRBuilder()
    t64 = ir.TensorType([ir.ConstInt(64, DataType.INDEX, ir.Span.unknown())], DataType.FP32, None)
    with ib.program("p") as prog:
        g = prog.declare_function("inline_k")
        with ib.function("inline_k", type=ir.FunctionType.Inline) as f:
            x = f.param("x", t64)
            f.return_type(t64)
            ib.return_stmt(x)
        prog.add_function(f.get_result())
        with ib.function("caller") as f:
            x = f.param("x", t64)
            f.return_type(t64)
            ib.return_stmt(ir.Call(g, [x], t64, ir.Span.unknown()))
        prog.add_function(f.get_result())
    out = passes.inline_functions()(prog.get_result())
    assert {f.name for f in out.functions.values()} == {"caller"}


# ===========================================================================
# The multi-function build (placeholder mechanics)
# ===========================================================================


def test_trace_build_delegates_to_executed_path():
    """``trace`` (the multi-function build): the real executed trace — a
    call-free kernel builds the same IR as
    the parser frontend. Inline callees flow through the conformance
    harness's ``canonicalize(inline=True, callees=[...])`` regime (deduction
    reads the Marker only).

    A call-bearing kernel is exercised in ``test_call.py`` with a Marker in
    the caller's scope — the placeholder-era parse-time-resolution
    artifact (``function_parser``'s ``sys._getframe(1)`` through the trace
    indirection) is gone: the executed path resolves calls via the
    kernel's own closure cells (closure pass-through)."""
    new = trace(softmax_tile_fn)
    assert isinstance(new, ir.Function)
    ir.assert_structural_equal(new, function_parser(softmax_tile_fn), enable_auto_mapping=True)

    # the corpus call shape (definition-name gvar) is pinned on the direct
    # AST parse — the authority side of the standalone-call anchor
    class _V(ir.IRVisitor):
        names: list[str] = []

        def visit_call(self, op):
            if op.op.name == "softmax_tile_fn":
                _V.names.append(op.op.name)
            super().visit_call(op)

    _V.names.clear()
    _V().visit_function(function_parser(gemm_caller))
    assert _V.names == ["softmax_tile_fn"]


def test_trace_program_builds_both_functions():
    """``trace_program`` (the program build): builds an ``ir.Program``
    with both methods traced — the two-pass envelope verified
    against the AST side through the harness mechanics."""
    prog = trace_program(SoftmaxProgram)
    assert isinstance(prog, ir.Program)
    assert prog.name == "SoftmaxProgram"
    assert {f.name for f in prog.functions.values()} == {"entry", "scale_tile"}
    # each function is the parser frontend's build for that method
    ir.assert_structural_equal(prog, program_parser(SoftmaxProgram), enable_auto_mapping=True)


def test_trace_program_forward_ref():
    """The forward-ref program builds through the envelope too — both
    functions are traced (class-body order). Full structural equality
    with the AST does NOT hold here: the AST's forward-ref entry call
    carries UnknownType while the tracing frontend emits the helper's declared
    type (the documented improvement, asserted targeted in
    test_program.py)."""
    prog = trace_program(ForwardRefProgram)
    assert {f.name for f in prog.functions.values()} == {"entry", "scale_tile"}
    # the non-forward function matches the AST exactly
    ast_ = program_parser(ForwardRefProgram)
    scale_new = prog["scale_tile"]
    scale_ast = ast_["scale_tile"]
    assert scale_new is not None and scale_ast is not None  # Program.__getitem__ is typed Optional
    ir.assert_structural_equal(scale_new, scale_ast, enable_auto_mapping=True)


# ===========================================================================
# Corpus AST-green (the anchor authority side must parse)
# ===========================================================================


def test_corpus_ast_green():
    """Every corpus kernel parses on the AST side — the anchor authority.
    The forward-ref program (entry before helper) is included: it parses
    (its call carries UnknownType — the tracing frontend's declared-type
    improvement), but it is not canonical-
    regime safe (the conftest roundtrip instrument recovers the derived
    type on reparse)."""
    assert isinstance(function_parser(softmax_tile_fn), ir.Function)
    assert isinstance(function_parser(gemm_caller), ir.Function)
    for cls in (SoftmaxProgram, ForwardRefProgram):
        prog = program_parser(cls)
        assert {f.name for f in prog.functions.values()} == {"entry", "scale_tile"}


# ===========================================================================
# Retirements + the classification threading
# ===========================================================================


def test_ptoframe_is_thin_handle():
    """PtoFrame is a thin root handle — the
    function-classification surface (is_orchestration / set_function_type)
    retired from the Sema and the Frame. Verified by absence."""
    sema = PtoSema(name="probe")
    assert not hasattr(sema, "is_orchestration")
    assert not hasattr(sema, "set_function_type")
    frame = sema.begin_frame()
    assert not hasattr(frame, "is_orchestration")
    assert not hasattr(frame, "set_function_type")
    # the thin surface is finish() only (idempotent)
    frame.finish()
    frame.finish()


def test_sema_function_classification_threading():
    """PtoSema.__init__ grows function_type/level/role/attrs → the
    built Function carries them (the type is known at Marker creation and
    flows Marker → PtoSema.__init__ → begin_function(type=…) at construction;
    set_function_type is retired)."""
    sema = PtoSema(name="probe", function_type=ir.FunctionType.Inline)
    sema.begin_frame().finish()
    fn = sema.result()
    assert fn.func_type == ir.FunctionType.Inline

    sema2 = PtoSema(
        name="probe2",
        function_type=ir.FunctionType.Orchestration,
        level=ir.Level.CHIP,  # Orchestration implies CHIP (function.h:419's derived-level check)
        role=ir.Role.Orchestrator,  # and Orchestrator (function.h:427)
        attrs={"auto_scope": False},
    )
    sema2.begin_frame().finish()
    fn2 = sema2.result()
    assert fn2.func_type == ir.FunctionType.Orchestration
    assert fn2.level == ir.Level.CHIP
    assert fn2.role == ir.Role.Orchestrator
    assert fn2.attrs.get("auto_scope") is False


# ===========================================================================
# Host parameters: passthrough in ``bind_parameters``
# ===========================================================================


def test_bind_param_host_passthrough():
    """The host-param path must not raise. A non-facade
    annotation — the ``self``-proxy of a program method, a host container —
    passes through ``bind_parameters`` unchanged: stored as a host value
    (mirroring ``try_bind``'s False decline), no
    IR param created. Needed for the self-proxy."""
    sema = PtoSema(name="probe")
    proxy = object()
    (host,) = sema.bind_parameters({"self": proxy})
    assert host is proxy  # passed through unchanged, no raise

    sema2 = PtoSema(name="probe2")
    (n,) = sema2.bind_parameters({"n": 4})
    assert n == 4


def test_bind_param_host_mixes_with_staged():
    """A host param and a staged param coexist: the staged param gets
    its ``ir.Var`` (an IR param); the host param passes through. The typed
    staged initialization guardrail is untouched (the interpreter enforces
    it over the prior value of a rebind)."""
    sema = PtoSema(name="probe")
    ann = pl.Tensor([64, 64], pl.FP32, _annotation_only=True)
    host_param, x_param = sema.bind_parameters({"self": object(), "x": ann})
    assert not isinstance(host_param, pl.Tensor)  # host passthrough
    assert isinstance(x_param, pl.Tensor)
    assert isinstance(x_param.unwrap(), ir.Var)  # staged param has its Var
    sema.begin_frame().finish()
    fn = sema.result()
    assert [p.name_hint for p in fn.params] == ["x"]  # only the staged param


# ===========================================================================
# Direct-API probes (the existing function path stays green)
# ===========================================================================


def test_existing_function_trace_untouched():
    """The function path is unaffected by the program-level additions
    (function_type defaults to Opaque; bind_parameters is unchanged for
    facade annotations)."""
    fn = function_trace(softmax_tile_fn)
    assert fn.name == "softmax_tile_fn"
    assert fn.func_type == ir.FunctionType.Opaque
