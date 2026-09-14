# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Entrance-tier experiment anchors (design doc 03).

Each test is one verified claim behind ``docs/design/03-entrance.md`` —
the ADR's stages cite these as realized evidence. They are probes of
current behavior, not the stages themselves: where a probe pins a hazard
the ADR schedules a fix for (the loud ``DynVar`` collision, the silent
``$``-name printer), the docstring says so and the test flips when the
stage lands.

Mapping (experiment → ADR section):

* specialization-key composition (scalars, dtype, MemRef) → §4
* same-named ``DynVar`` extent conflation → §4, invariant 8
* resolution-assembly prototype vs the assembly machinery → §7
* read-through shape prototype (Stage A) → §8
* structural-equality compare on independent builds → §6
* silent ``$``-name printer (Stage C tripwire) → §5
* eager-lane inline-callee adaptation (Stage C corpus fix) → §8
"""

import pypto.language as pl
import pytest
from pypto.language.op.marker import Marker
from pypto.language.typing import Out, annotation_to_type
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import ir

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _trace_incore(kernel, annotations):
    """Trace one InCore body at the given annotation map (the construction
    primitive's current internal spelling)."""
    from pypto.language.tracer.decorators.function import (  # noqa: PLC0415
        _trace_with_annotations,
    )

    return _trace_with_annotations(kernel, annotations, None, function_type=ir.FunctionType.InCore)


def _renamed(fn: ir.Function, name: str) -> ir.Function:
    """Rebuild ``fn`` under a derived (``$``-variant) name."""
    return ir.Function(
        name,
        list(fn.params),
        list(fn.return_types),
        fn.body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
    )


def _structural_equal(left, right) -> bool:
    return ir.structural_equal(left, right, enable_auto_mapping=True)


# ---------------------------------------------------------------------------
# §4: the specialization key
# ---------------------------------------------------------------------------


class TestSpecializationKey:
    def _body(self, scale: float):
        def kernel(
            x: pl.Tensor[[64, 128], pl.FP32],
            o: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            t = pl.load(x, [0, 0], [64, 128])
            r = pl.mul(t, scale)
            pl.store(r, [0, 0], o)
            return o

        return kernel

    def test_baked_scalar_makes_bodies_unequal(self):
        """Scalar values must be part of the variant identity: bodies
        identical except for a baked constant are structurally unequal, so a
        type-only key would share one variant body carrying the wrong
        constant (ADR §4)."""
        ann = {
            "x": pl.Tensor[[64, 128], pl.FP32],
            "o": Out[pl.Tensor[[64, 128], pl.FP32]],
        }
        at_two = _trace_incore(self._body(2.0), ann)
        at_three = _trace_incore(self._body(3.0), ann)
        assert not _structural_equal(at_two, at_three)

    def test_same_key_retraces_are_structurally_equal(self):
        """Service determinism (ADR invariant 2): the same specialization key
        re-traced yields structurally equal IR. This is what variant sharing
        and the structural-equality cache policy rest on."""
        ann = {
            "x": pl.Tensor[[64, 128], pl.FP32],
            "o": Out[pl.Tensor[[64, 128], pl.FP32]],
        }
        first = _trace_incore(self._body(2.0), ann)
        second = _trace_incore(self._body(2.0), ann)
        assert _structural_equal(first, second)

    def test_dtype_is_distinguished(self):
        """The derived name must render dtype (ADR §5): same shape, FP32 vs
        BF16 are structurally distinct."""
        fp32 = annotation_to_type(pl.Tensor[[64, 128], pl.FP32])
        bf16 = annotation_to_type(pl.Tensor[[64, 128], pl.BF16])
        assert not _structural_equal(fp32, bf16)

    def test_memref_placement_is_invisible_to_structural_equality(self):
        """Structural equality ignores MemRef address/alignment entirely —
        plain, ``mem_ddr_0@0``, and ``mem_ddr_1@512`` all compare equal.
        Placement is excluded from the specialization key by decision, and
        structural equality can never be the key comparator (ADR §4, §9)."""
        plain = annotation_to_type(pl.Tensor[[64, 128], pl.FP32])
        m0 = annotation_to_type(
            pl.Tensor[[64, 128], pl.FP32, pl.MemRef("mem_ddr_0", pl.const(0, pl.INT64), 256)]
        )
        m1 = annotation_to_type(
            pl.Tensor[[64, 128], pl.FP32, pl.MemRef("mem_ddr_1", pl.const(1, pl.INT64), 512)]
        )
        assert _structural_equal(plain, m0)
        assert _structural_equal(m0, m1)


class TestDynVarExtentConflation:
    """FLIPPED probe: extent Vars are now shared by name (the specializer's
    per-program ``dyn_var_cache`` semantics, ported to tracing's ``unwrap``),
    dissolving the Stage-C conflation hazard by construction instead of
    detecting it. The variant identity rule is unchanged and still the Var
    object — same-named DynVars simply are one extent now, and a distinct
    name (``lm_head``'s ``LM_HEAD_T_DYN``) is the deliberate opt-out."""

    def test_same_named_dynvars_share_one_extent_var(self):
        """Two distinct ``DynVar("M")`` objects (the per-module corpus idiom)
        unwrap to the same extent ``Var``; the identity rule for variants
        remains the ``Var`` object, not structural equality (ADR §4,
        invariant 8)."""
        m1, m2 = pl.dynamic("M"), pl.dynamic("M")
        t1 = annotation_to_type(pl.Tensor[[m1, 128], pl.FP32])
        t2 = annotation_to_type(pl.Tensor[[m2, 128], pl.FP32])
        assert m1.unwrap() is m2.unwrap()
        assert _structural_equal(t1, t2)

    def test_same_named_extent_wiring_compiles(self, tmp_path, monkeypatch):
        """The former conflation hazard, dissolved: one module's ``M``-typed
        callee wired from another module's ``M``-typed entry is now genuinely
        one extent (the name-keyed registry), so the program is coherent and
        compiles — no silent cross-extent wiring exists to make loud."""
        from pypto.ir import compile as ir_compile  # noqa: PLC0415

        monkeypatch.chdir(tmp_path)  # ir_compile writes build_output under cwd

        m1, m2 = pl.dynamic("M"), pl.dynamic("M")

        @pl.function(type=ir.FunctionType.InCore)
        def bar_m1(x: pl.Tensor[[m1, 128], pl.FP32], o: pl.Out[pl.Tensor[[m1, 128], pl.FP32]]):
            t = pl.load(x, [0, 0], [64, 128])  # static sub-tile; M only in the signature
            pl.store(t, [0, 0], o)
            return o

        marker = Marker._from_ir_function(bar_m1)

        @pl.function
        def entry(x: pl.Tensor[[m2, 128], pl.FP32], o: pl.Out[pl.Tensor[[m2, 128], pl.FP32]]):
            r = marker(x, o)  # m2-typed args into an m1-typed callee
            return r

        prog = ir.Program([bar_m1, entry], "dynvar_conflate", entry.span)
        ir_compile(prog, skip_ptoas=True, platform="a2a3")  # no exception today


# ---------------------------------------------------------------------------
# §7: resolution assembly
# ---------------------------------------------------------------------------


class TestResolutionAssembly:
    def test_resolution_reproduces_assembly_machinery_output(self):
        """The §7 prototype: walking the entry's ``GlobalVar`` references and
        resolving transitively reproduces the assembly machinery's function
        set exactly (diamond: entry → two helpers → one shared leaf), and the
        resolution-built program is structurally equal to the
        machinery-built one."""
        torch = pytest.importorskip("torch")

        @pl.jit.incore
        def shared_leaf(x: pl.Tensor, o: pl.Out[pl.Tensor]):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        @pl.jit.incore
        def a_helper(x: pl.Tensor, o: pl.Out[pl.Tensor]):
            r = shared_leaf(x, o)
            return r

        @pl.jit.incore
        def b_helper(x: pl.Tensor, o: pl.Out[pl.Tensor]):
            r = shared_leaf(x, o)
            return r

        @pl.jit
        def entry(
            scale,
            x: pl.Tensor[[64, 128], pl.FP32],
            a: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
            b: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            ra = a_helper(x, a)
            b_helper(x, b)  # result unused: shape-only fixture
            return ra

        x = torch.randn(64, 128)
        a = torch.empty(64, 128)
        b = torch.empty(64, 128)
        prog = entry.compile_for_test(1, x, a, b)
        fns = {f.name: f for f in prog.functions.values()}
        assert set(fns) == {"entry", "a_helper", "b_helper", "shared_leaf"}

        memo = {n: f for n, f in fns.items() if n != "entry"}
        resolved = self._resolve(fns["entry"], memo)
        assert set(resolved) == set(memo)  # transitive + deduplicated
        resolved_prog = ir.Program([fns["entry"], *resolved.values()], "resolved", fns["entry"].span)
        assert _structural_equal(resolved_prog, prog)

    @staticmethod
    def _resolve(entry_fn: ir.Function, memo: dict[str, ir.Function]) -> dict[str, ir.Function]:
        """The §7 resolution walk: collect non-builtin (non-dotted) call
        targets from each body, resolve them from the memo, transitively."""

        class CalleeCollector(ir.IRVisitor):
            def __init__(self):
                super().__init__()
                self.names: set[str] = set()

            def visit_call(self, call: ir.Call) -> None:
                super().visit_call(call)
                if call.op is not None and "." not in call.op.name:
                    self.names.add(call.op.name)

        out: dict[str, ir.Function] = {}
        frontier = [entry_fn]
        while frontier:
            fn = frontier.pop()
            collector = CalleeCollector()
            collector.visit_stmt(fn.body)
            for name in collector.names:
                if name not in out:
                    callee = memo.get(name)
                    assert callee is not None, f"unresolved callee {name!r}"
                    out[name] = callee
                    frontier.append(callee)
        return out


# ---------------------------------------------------------------------------
# §8 Stage A: read-through shape
# ---------------------------------------------------------------------------


class TestReadThroughShapePrototype:
    def test_shape_read_through_equals_literal_shape_build(self):
        """The Stage A prototype: with a runtime-mode facade exposing the
        wrapped type's shape (populated from ``expr.type`` here via a scoped
        ``Tensor.__init__`` patch), the corpus idiom ``M, N = x.shape``
        traces and produces IR structurally equal to the literal-shape
        build. The production fix replaces the patch with native
        read-through in the facade."""

        def via_shape(
            x: pl.Tensor[[64, 128], pl.FP32],
            o: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            m, n = x.shape
            t = pl.load(x, [0, 0], [m, n])
            pl.store(t, [0, 0], o)
            return o

        def via_literal(
            x: pl.Tensor[[64, 128], pl.FP32],
            o: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        ann = {
            "x": pl.Tensor[[64, 128], pl.FP32],
            "o": Out[pl.Tensor[[64, 128], pl.FP32]],
        }
        original_init = Tensor.__init__

        def readthrough_init(self, *args, **kwargs):  # noqa: PLR0913
            original_init(self, *args, **kwargs)
            if self._expr is not None:
                ty = self._expr.type
                if hasattr(ty, "shape"):
                    self.shape = list(ty.shape)
                    self.dtype = ty.dtype

        Tensor.__init__ = readthrough_init  # type: ignore[method-assign,assignment]
        try:
            from_shape = _trace_incore(via_shape, ann)
        finally:
            Tensor.__init__ = original_init  # type: ignore[method-assign,assignment]
        from_literal = _trace_incore(via_literal, ann)
        assert _structural_equal(from_shape, from_literal)


# ---------------------------------------------------------------------------
# §6: structural-equality caching
# ---------------------------------------------------------------------------


class TestStructuralComparisonOnIndependentBuilds:
    def test_independent_builds_are_structurally_equal(self):
        """The §6 policy premise: two independently built programs with the
        same specialization key compare structurally equal, so a dispatcher
        may re-trace and compare instead of keying (measured cost on a small
        InCore kernel: ~10 ms/trace, ~0.02 ms/compare, ~215 ms/compile —
        timing deliberately not asserted)."""

        def make():
            @pl.function(type=ir.FunctionType.InCore)
            def b(x: pl.Tensor[[64, 128], pl.FP32], o: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
                t = pl.load(x, [0, 0], [64, 128])
                pl.store(t, [0, 0], o)
                return o

            return b

        @pl.function
        def entry(x: pl.Tensor[[64, 128], pl.FP32], o: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        p1 = ir.Program([entry, _renamed(make(), "bar$64x128")], "a", entry.span)
        p2 = ir.Program([entry, _renamed(make(), "bar$64x128")], "b", entry.span)
        assert _structural_equal(p1, p2)


# ---------------------------------------------------------------------------
# §5: the $-name printer tripwire
# ---------------------------------------------------------------------------


class TestDerivedNamePrinting:
    def test_printer_emits_dollar_names_and_reparse_fails(self):
        """Pins the §5 gap: the printer emits ``$``-names silently and
        re-parsing its output fails with a SyntaxError. The loud printer
        tripwire is a Stage C deliverable; when it lands, the emission
        assertion here flips to the loud error."""

        @pl.function(type=ir.FunctionType.InCore)
        def bar(x: pl.Tensor[[64, 128], pl.FP32], o: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        @pl.function
        def entry(x: pl.Tensor[[64, 128], pl.FP32], o: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        prog = ir.Program([entry, _renamed(bar, "bar$64x128")], "dollar", entry.span)
        text = prog.as_python()
        assert "bar$64x128" in text
        with pytest.raises(SyntaxError):
            pl.parse(text)


# ---------------------------------------------------------------------------
# §8 Stage C: the eager-lane inline-callee adaptation fix
# ---------------------------------------------------------------------------


class TestEagerLaneInlineCallee:
    def test_adapted_module_global_inline_callee_splices_in_passes(self):
        """The Stage C corpus fix (the ``rmsnorm.py`` pin's root cause): an
        eager entry calling a module-global ``@pl.jit.inline`` callee traces
        once the callee is adapted (the lazy lane's ``_LazyGlobalFn`` move),
        and the Default pipeline splices the detached call — the callee
        leaves the program and its body lands in the entry. The pin in
        ``test_corpus_gaps.py`` clears when the lane adopts this."""
        from pypto.ir import OptimizationStrategy, PassManager  # noqa: PLC0415
        from pypto.language.tracer.decorators.function import (  # noqa: PLC0415
            _trace_with_annotations,
        )
        from pypto.language.tracer.decorators.jit import _LazyGlobalFn  # noqa: PLC0415

        @pl.jit.inline
        def inline_copy(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            y[:] = x
            return y

        # The jit decoration's public result is the jit object; the frozen
        # callee (what this internal-lane test wraps) is its decoration trace.
        adapted = _LazyGlobalFn(inline_copy.fn)

        def entry(x: pl.Tensor[[64], pl.FP32], y: pl.Out[pl.Tensor[[64], pl.FP32]]):
            r = adapted(x, y)  # consumed through the return value (InOutUseDiscipline)
            return r

        fn = _trace_with_annotations(
            entry,
            {"x": pl.Tensor[[64], pl.FP32], "y": Out[pl.Tensor[[64], pl.FP32]]},
            None,
            function_type=ir.FunctionType.Orchestration,
        )
        assert "inline_copy" in fn.as_python()  # the detached call was emitted

        prog = ir.Program([fn, inline_copy.fn], "eager_inline", fn.span)
        post = PassManager.get_strategy(OptimizationStrategy.Default).run_passes(prog)
        text = post.as_python()
        assert "inline_copy(" not in text  # spliced by inline_functions
        assert [f.name for f in post.functions.values()] == ["entry"]


# ---------------------------------------------------------------------------
# §7: the forced per-build memo
# ---------------------------------------------------------------------------


class TestProgramNameUniqueness:
    def test_program_rejects_same_named_structurally_equal_functions(self):
        """The program constructor's name check is name-based, blind to
        structural equality — this is what forces the per-build derived-name
        memo in ADR §7 (the callee lane cannot re-trace per call site)."""

        def kernel(
            x: pl.Tensor[[64, 128], pl.FP32],
            o: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
        ):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        ann = {"x": pl.Tensor[[64, 128], pl.FP32], "o": Out[pl.Tensor[[64, 128], pl.FP32]]}
        v1 = _renamed(_trace_incore(kernel, ann), "bar$64x128")
        v2 = _renamed(_trace_incore(kernel, ann), "bar$64x128")
        assert _structural_equal(v1, v2)

        @pl.function
        def entry(x: pl.Tensor[[64, 128], pl.FP32], o: pl.Out[pl.Tensor[[64, 128], pl.FP32]]):
            t = pl.load(x, [0, 0], [64, 128])
            pl.store(t, [0, 0], o)
            return o

        with pytest.raises(ValueError, match="Duplicate function name"):
            ir.Program([entry, v1, v2], "dup", entry.span)
