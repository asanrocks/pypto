# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""What makes two calls to the same helper the same callee.

One program instantiates a helper once per distinct parameter type: the traced
variant's signature is what its second caller must match. Shape and dtype do
not decide that type alone — the same ``[64, 64]`` FP32 buffer is ND or NZ
depending on its ``TensorView``, and the TypeChecked verifier compares layouts
at the call boundary. Keying the variant on shape and dtype alone handed the
second caller the first caller's signature:

    parameter 'x' is declared ND but the argument is NZ

These tests therefore read the pre-pass program: the variants are the fact
under test, and the later NZ-load constraint (an NZ buffer is a cube operand,
loaded into ``pl.Mem.Mat``) is a different rule about a different layer.
"""

import pypto.language as pl
import pytest
import torch
from pypto.language.tracing.decorators.jit_trace import (  # noqa: SLF001 — the key under test
    _attrs_key_part,
    _variant_key,
    _variant_key_part,
)
from pypto.pypto_core import ir

F32 = torch.float32


@pl.jit_trace.incore
def scale_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """The helper both entries below call with different declared views."""
    tile = pl.load(x, [0, 0], [64, 64])
    pl.store(pl.mul(tile, 2.0), [0, 0], out)
    return out


@pl.jit_trace
def same_views(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
    out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    out_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """Two ND call sites: one variant serves both."""
    with pl.at(level=pl.Level.CORE_GROUP):
        scale_incore(a, out_a)
        scale_incore(b, out_b)
    return pl.Tuple(out_a, out_b)


@pl.jit_trace
def mixed_views(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32, pl.NZ],
    out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    out_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """An ND and an NZ call site: the callee needs one variant per type."""
    with pl.at(level=pl.Level.CORE_GROUP):
        scale_incore(a, out_a)
        scale_incore(b, out_b)
    return pl.Tuple(out_a, out_b)


def _sample(shape: tuple[int, int] = (64, 64)) -> torch.Tensor:
    return torch.empty(*shape, dtype=F32)


def _callee_layouts(program: ir.Program) -> dict[str, list[str | None]]:
    """The declared ``TensorView.layout`` of each callee parameter, per function.

    A parameter with no view reports ``None`` (the default ND reading), so the
    names — not the enum objects — are what the assertions compare.
    """

    def layout_name(param: ir.Var) -> str | None:
        layout = getattr(getattr(param.type, "tensor_view", None), "layout", None)
        return None if layout is None else layout.name

    return {
        fn.name: [layout_name(param) for param in fn.params]
        for fn in program.functions.values()
        if fn.name != program.name
    }


def _program(kernel, *samples: object) -> ir.Program:
    """The pre-pass program — ``specialize``, so no pipeline rule interferes."""
    return kernel.specialize(*samples)


_MIXED_VIEW_SAMPLES = (
    pl.Tensor[[64, 64], pl.FP32],
    pl.Tensor[[64, 64], pl.FP32, pl.NZ],
    pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    pl.Out[pl.Tensor[[64, 64], pl.FP32]],
)


class TestCalleeVariantIdentity:
    """A callee variant is keyed on the type its caller will be checked against."""

    def test_matching_views_keep_one_variant(self):
        layouts = _callee_layouts(_program(same_views, _sample(), _sample(), _sample(), _sample()))
        assert len(layouts) == 1, f"two identical call sites must share one variant: {layouts}"

    def test_differing_views_get_one_variant_each(self):
        layouts = _callee_layouts(_program(mixed_views, *_MIXED_VIEW_SAMPLES))
        assert len(layouts) == 2, f"an ND and an NZ call site are different types: {layouts}"

    def test_each_variant_declares_the_layout_it_was_traced_with(self):
        layouts = _callee_layouts(_program(mixed_views, *_MIXED_VIEW_SAMPLES))
        declared = {tuple(params) for params in layouts.values()}
        assert declared == {(None, None), (ir.TensorLayout.NZ.name, None)}, declared

    def test_dtype_still_separates_variants(self):
        """The pre-existing key dimensions keep working."""

        @pl.jit_trace
        def mixed_dtypes(
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP16],
            out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                scale_incore(a, out_a)
            return out_a

        program = mixed_dtypes.specialize(_sample(), torch.empty(64, 64, dtype=torch.float16), _sample())
        assert len(_callee_layouts(program)) == 1  # one call site, one helper variant

    def test_two_call_sites_at_different_shapes_get_distinct_variants(self):
        """Shape is still part of the key (the pre-existing behavior)."""

        @pl.jit_trace.incore
        def copy_incore(x: pl.Tensor, out: pl.Out[pl.Tensor]):
            tile = pl.load(x, [0, 0], [64, 64])
            pl.store(tile, [0, 0], out)
            return out

        @pl.jit_trace
        def two_shapes(
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[128, 64], pl.FP32],
            out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            out_b: pl.Out[pl.Tensor[[128, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                copy_incore(a, out_a)
                copy_incore(b, out_b)
            return pl.Tuple(out_a, out_b)

        program = two_shapes.specialize(_sample(), _sample((128, 64)), _sample(), _sample((128, 64)))
        assert len(_callee_layouts(program)) == 2


class TestRawExprVariantIdentity:
    """A raw ``ir.Expr`` argument keys on binding-aware identity, not its
    printed text.

    The printer only builds its var-rename map for function and bare-stmt
    roots — an ``ir.Expr`` root prints two distinct Vars sharing a
    ``name_hint`` identically — so a text key handed the second caller the
    first caller's variant, whose baked body embeds the first caller's Var
    while the second caller's value silently vanished from the call."""

    def test_same_named_distinct_vars_get_distinct_keys(self):
        ty = ir.ScalarType(pl.INDEX)
        v1 = ir.Var("idx", ty, ir.Span.unknown())
        v2 = ir.Var("idx", ty, ir.Span.unknown())
        assert _variant_key_part(v1) != _variant_key_part(v2)

    def test_same_var_object_dedupes(self):
        v = ir.Var("idx", ir.ScalarType(pl.INDEX), ir.Span.unknown())
        assert _variant_key_part(v) == _variant_key_part(v)

    def test_same_value_constants_still_dedupe(self):
        """Constants key on value: two callers baking the same constant share
        one variant (structural hash of equal constants is equal)."""
        c1 = ir.ConstInt(4, pl.INDEX, ir.Span.unknown())
        c2 = ir.ConstInt(4, pl.INDEX, ir.Span.unknown())
        c3 = ir.ConstInt(8, pl.INDEX, ir.Span.unknown())
        assert _variant_key_part(c1) == _variant_key_part(c2)
        assert _variant_key_part(c1) != _variant_key_part(c3)


class TestVariantKeyComposition:
    """The key itself: name-canonical parts, storage identity, hashable attrs."""

    def test_annotation_insertion_order_does_not_split_the_key(self):
        """The lanes build the annotation map in their own order.

        The callee lanes insert declared order; the parse-time import bridge
        inserts the runtime parameters then the sorted baked ones. An
        interspersed baked/staged helper must enter under one key whichever
        lane traced it — the same key is also the in-flight cycle guard's,
        and a cross-lane cycle that misses it re-traces instead of being
        refused.
        """

        def kernel(a, knob, b): ...  # noqa: ANN001 — never called, only keyed

        staged = pl.Tensor[[8, 8], pl.FP32]
        # Declared order (the callee lanes) against the import bridge's
        # runtime-then-sorted-baked order.
        declared_order = {"a": staged, "knob": 3, "b": pl.Tensor[[16, 8], pl.FP32]}
        bridged_order = {"a": staged, "b": pl.Tensor[[16, 8], pl.FP32], "knob": 3}
        first = _variant_key(kernel, declared_order, ir.FunctionType.Inline, None, None, None)
        second = _variant_key(kernel, bridged_order, ir.FunctionType.Inline, None, None, None)
        assert first == second
        # A name is part of the key: renaming a parameter is a different
        # variant even at identical parts.
        assert _variant_key(
            kernel, {"a": staged, "other": 3}, ir.FunctionType.Inline, None, None, None
        ) != _variant_key(kernel, {"a": staged, "knob": 3}, ir.FunctionType.Inline, None, None, None)

    def test_equivalent_but_distinct_memrefs_share_one_key_part(self):
        """Storage identity, not MemRef object identity.

        A re-subscripted ``slots=N`` declaration mints a fresh MemRef object
        per selection, all describing the same base/offset/size/slot window —
        keying on the wrapper's own identity split those into duplicate
        variants (sound but over-specialized). A different slot, or a
        different allocation, still splits.
        """
        l0c = pl.MemRef(slots=2)
        first = pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec]
        second = pl.Tile[[64, 64], pl.FP32, l0c[0], pl.Mem.Vec]
        assert first.memref is not second.memref  # distinct wrappers, one storage
        assert _variant_key_part(first) == _variant_key_part(second)
        other_slot = pl.Tile[[64, 64], pl.FP32, l0c[1], pl.Mem.Vec]
        other_alloc = pl.MemRef(slots=2)
        elsewhere = pl.Tile[[64, 64], pl.FP32, other_alloc[0], pl.Mem.Vec]
        assert _variant_key_part(first) != _variant_key_part(other_slot)
        assert _variant_key_part(first) != _variant_key_part(elsewhere)

    def test_unhashable_attrs_values_fall_back_to_identity(self):
        """An attrs payload that sorts but does not hash must not blow up at
        the first dict lookup — it falls back to object identity
        (under-dedupe: duplicate variants, still correct)."""

        def kernel(a): ...  # noqa: ANN001 — never called, only keyed

        attrs = {"payload": []}
        assert _attrs_key_part(attrs) == ("attrs-id", id(attrs))
        # The whole key stays hashable, and two such payloads never collide.
        hash(
            _variant_key(kernel, {"a": pl.Tensor[[8, 8], pl.FP32]}, ir.FunctionType.Opaque, None, None, attrs)
        )
        assert _attrs_key_part({"payload": []}) != _attrs_key_part({"payload": []})

    def test_compound_exprs_over_distinct_same_named_vars_differ(self):
        ty = ir.ScalarType(pl.INDEX)
        v1 = ir.Var("idx", ty, ir.Span.unknown())
        v2 = ir.Var("idx", ty, ir.Span.unknown())
        assert _variant_key_part(v1 + 1) != _variant_key_part(v2 + 1)

    def test_unsupported_payload_falls_back_to_object_identity(self, monkeypatch):
        """Attr payloads with no structural encoding key on object identity:
        that can under-dedupe (duplicate variants, still correct) but never
        hands one caller another caller's variant."""

        def unsupported(node):
            raise ir.StructuralComparisonUnsupported("test: no structural encoding")

        monkeypatch.setattr(ir, "structural_hash", unsupported)
        v = ir.Var("idx", ir.ScalarType(pl.INDEX), ir.Span.unknown())
        assert _variant_key_part(v) == ("expr-id", id(v))

    def test_same_named_distinct_raw_vars_get_one_variant_each(self):
        """End to end: two call sites passing distinct same-named raw Vars
        must not share a variant (the printed text alone would collide)."""

        @pl.jit_trace.incore
        def take_idx(i: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor]):
            t = pl.load(out, [0, 0], [8, 8])
            pl.store(t, [0, 0], out)

        @pl.jit_trace
        def two_idx(out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            idx_type = ir.ScalarType(pl.INDEX)
            take_idx(ir.Var("idx", idx_type, ir.Span.unknown()), out)
            take_idx(ir.Var("idx", idx_type, ir.Span.unknown()), out)
            return out

        program = two_idx.specialize(_sample())
        variants = [fn.name for fn in program.functions.values() if fn.name.startswith("take_idx")]
        assert len(variants) == 2, f"distinct same-named Vars must not share a variant: {variants}"


class TestWrapperKindVariantIdentity:
    """Two wrappers around one Python function are two callees.

    ``@pl.jit_trace`` and ``@pl.jit_trace.host`` on the same kernel trace
    different function kinds and levels; a variant key naming only the kernel
    and the annotations handed the second wrapper the first wrapper's
    variant, so a HOST-orchestration spelling ran an Orchestration body.
    """

    def test_two_wrappers_of_one_kernel_get_one_variant_each(self):
        def helper_kernel(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                pl.store(pl.mul(pl.load(x, [0, 0], [8, 8]), 2.0), [0, 0], out)
            return out

        device = pl.jit_trace(helper_kernel)
        host = pl.jit_trace.host(helper_kernel)

        @pl.jit_trace
        def entry(
            x: pl.Tensor[[8, 8], pl.FP32],
            out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        ):
            device(x, out)
            host(x, out)
            return out

        program = entry.specialize()
        kinds = sorted(
            (fn.func_type.name, fn.level.name if fn.level is not None else None)
            for fn in program.functions.values()
            if fn.name != program.name and fn.name.startswith("helper_kernel")
        )
        assert kinds == [
            (ir.FunctionType.Opaque.name, ir.Level.HOST.name),
            (ir.FunctionType.Orchestration.name, ir.Level.CHIP.name),
        ], kinds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
