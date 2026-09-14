# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host-side static shape introspection: ``pl.static_dim`` / ``pl.tensor.static_dim``.

Pins the contract established with the dsv4 corpus refactoring:

* the query reads the extent off the facade's own type — the shape fixed
  when the producing operator yielded the value — never ambient builder
  state, so it is value-directed observation, not a type lie;
* it returns a plain Python ``int`` (host arithmetic stays host
  arithmetic), unlike raw shape peeking whose ``ConstInt`` extents
  silently build IR expressions under ``//`` and flow symbolic bounds
  into loops;
* a kernel whose tiling derives from it is IR-neutral against the
  literal-bound kernel — the naive ``pl.tensor.dim`` spelling is not
  (symbolic loop bounds survive the Default pipeline);
* a dynamic extent is a loud error — the caller must choose
  ``pl.tensor.dim`` for a staged runtime extent.
"""

import pypto.language as pl
import pytest
from pypto import DataType, ir
from pypto.language.diagnostics import InvalidOperationError


class TestStaticDimIRNeutrality:
    """Tiling derived from ``static_dim`` matches the literal-bound kernel."""

    def test_tensor_kernel_matches_literal(self):
        @pl.jit
        def literal(
            x: pl.Tensor[[8, 128], pl.BF16], y: pl.Out[pl.Tensor[[8, 128], pl.BF16]]
        ) -> pl.Tensor[[8, 128], pl.BF16]:
            d = 128
            for db in pl.pipeline(d // 32, stage=2):
                d0 = db * 32
                y[0:8, d0 : d0 + 32] = x[0:8, d0 : d0 + 32]
            return y

        @pl.jit
        def inferred(
            x: pl.Tensor[[8, 128], pl.BF16], y: pl.Out[pl.Tensor[[8, 128], pl.BF16]]
        ) -> pl.Tensor[[8, 128], pl.BF16]:
            d = pl.tensor.static_dim(x, 1)
            for db in pl.pipeline(d // 32, stage=2):
                d0 = db * 32
                y[0:8, d0 : d0 + 32] = x[0:8, d0 : d0 + 32]
            return y

        ir.assert_structural_equal(
            getattr(literal, "fn", literal), getattr(inferred, "fn", inferred), enable_auto_mapping=True
        )

    def test_tile_extent_is_host_int(self):
        """A staged Tile's extents are static by construction; the query
        reads them as plain host ints (no IR emitted)."""
        captured: dict[str, object] = {}

        def probe(x):
            t = pl.load(x, [0, 0], [8, 32])
            captured["w"] = pl.tensor.static_dim(t, 1)
            captured["h"] = pl.tensor.static_dim(t, -2)
            return x

        @pl.function
        def k(x: pl.Tensor[[8, 128], pl.BF16]) -> pl.Tensor[[8, 128], pl.BF16]:
            return probe(x)

        assert captured["w"] == 32
        assert type(captured["w"]) is int
        assert captured["h"] == 8

    def test_expansion_body_geometry_from_operand(self):
        """The corpus usage shape: an ``@pl.expand`` block infers its geometry
        from the operands instead of capturing module-level constants."""

        @pl.expand
        def copy_wide(x, y):
            w = pl.tensor.static_dim(x, 1)
            t = pl.load(x, [0, 0], [8, w])
            pl.store(t, [0, 0], y)
            return y

        @pl.jit
        def kernel(
            x: pl.Tensor[[8, 128], pl.BF16], y: pl.Out[pl.Tensor[[8, 128], pl.BF16]]
        ) -> pl.Tensor[[8, 128], pl.BF16]:
            return copy_wide(x, y)

        @pl.jit
        def reference(
            x: pl.Tensor[[8, 128], pl.BF16], y: pl.Out[pl.Tensor[[8, 128], pl.BF16]]
        ) -> pl.Tensor[[8, 128], pl.BF16]:
            t = pl.load(x, [0, 0], [8, 128])
            pl.store(t, [0, 0], y)
            return y

        # An eagerly traced ``@pl.jit`` contributes its decoration trace (``.fn``).
        ir.assert_structural_equal(
            getattr(kernel, "fn", kernel), getattr(reference, "fn", reference), enable_auto_mapping=True
        )


class TestStaticDimHostSemantics:
    def test_returns_plain_int(self):
        captured: dict[str, object] = {}

        def probe(x):
            d = pl.static_dim(x, 1)
            captured["d_type"] = type(d)
            captured["quotient"] = d // 32
            return x

        @pl.function
        def k(x: pl.Tensor[[8, 128], pl.BF16]) -> pl.Tensor[[8, 128], pl.BF16]:
            return probe(x)

        assert captured["d_type"] is int
        assert captured["quotient"] == 4
        assert type(captured["quotient"]) is int

    def test_promoted_and_qualified_spelling_agree(self):
        assert pl.static_dim is pl.tensor.static_dim

    def test_negative_axis(self):
        span = ir.Span.unknown()
        t = pl.Tensor(expr=ir.Var("x", ir.TensorType([8, 128], DataType.BF16), span))
        assert pl.static_dim(t, -1) == 128
        assert pl.static_dim(t, -2) == 8

    def test_axis_out_of_range_is_loud(self):
        span = ir.Span.unknown()
        t = pl.Tensor(expr=ir.Var("x", ir.TensorType([8, 128], DataType.BF16), span))
        with pytest.raises(IndexError):
            pl.static_dim(t, 2)
        with pytest.raises(IndexError):
            pl.static_dim(t, -3)

    def test_non_shaped_value_is_loud(self):
        span = ir.Span.unknown()
        s = pl.Scalar(expr=ir.Var("n", ir.ScalarType(DataType.INT32), span))
        with pytest.raises(TypeError, match="tensor or tile"):
            pl.static_dim(s, 0)


class TestStaticShape:
    """``pl.static_shape``: the whole-tuple twin for fully-static operands."""

    def test_unpacks_into_locals(self):
        captured: dict[str, object] = {}

        def probe(q):
            t, h, head_dim = pl.tensor.static_shape(q)
            captured.update(t=t, h=h, head_dim=head_dim)
            return q

        @pl.function
        def k(q: pl.Tensor[[8, 16, 128], pl.BF16]) -> pl.Tensor[[8, 16, 128], pl.BF16]:
            return probe(q)

        assert captured == {"t": 8, "h": 16, "head_dim": 128}

    def test_agrees_with_static_dim(self):
        span = ir.Span.unknown()
        t = pl.Tensor(expr=ir.Var("x", ir.TensorType([8, 128], DataType.BF16), span))
        assert pl.static_shape(t) == (pl.static_dim(t, 0), pl.static_dim(t, 1))

    def test_any_dynamic_extent_is_loud(self):
        t_dyn = pl.dynamic("T_DYN")

        def make():
            @pl.jit
            def k(
                x: pl.Tensor[[t_dyn, 128], pl.BF16],
                y: pl.Out[pl.Tensor[[t_dyn, 128], pl.BF16]],
            ) -> pl.Tensor[[t_dyn, 128], pl.BF16]:
                x.bind_dynamic(0, t_dyn)
                y.bind_dynamic(0, t_dyn)
                _, d = pl.tensor.static_shape(x)  # dim 0 dynamic — loud
                return y

        with pytest.raises(InvalidOperationError, match="dynamic"):
            make()


class TestStaticDimDynamicIsLoud:
    def test_dynamic_extent_raises_at_trace_time(self):
        t_dyn = pl.dynamic("T_DYN")

        def make():
            @pl.jit
            def k(
                x: pl.Tensor[[t_dyn, 128], pl.BF16],
                y: pl.Out[pl.Tensor[[t_dyn, 128], pl.BF16]],
            ) -> pl.Tensor[[t_dyn, 128], pl.BF16]:
                x.bind_dynamic(0, t_dyn)
                y.bind_dynamic(0, t_dyn)
                d = pl.tensor.static_dim(x, 0)
                for db in pl.pipeline(d // 32, stage=2):
                    d0 = db * 32
                    y[0:8, d0 : d0 + 32] = x[0:8, d0 : d0 + 32]
                return y

        with pytest.raises(InvalidOperationError, match="dynamic"):
            make()

    def test_static_axis_of_dynamic_tensor_is_fine(self):
        t_dyn = pl.dynamic("T_DYN")

        @pl.jit
        def k(
            x: pl.Tensor[[t_dyn, 128], pl.BF16],
            y: pl.Out[pl.Tensor[[t_dyn, 128], pl.BF16]],
        ) -> pl.Tensor[[t_dyn, 128], pl.BF16]:
            x.bind_dynamic(0, t_dyn)
            y.bind_dynamic(0, t_dyn)
            d = pl.tensor.static_dim(x, 1)  # static axis of a dynamic tensor
            for db in pl.pipeline(d // 32, stage=2):
                d0 = db * 32
                y[0:8, d0 : d0 + 32] = x[0:8, d0 : d0 + 32]
            return y

        # the static query folded to a host int: the bound is the literal 4
        assert "pl.pipeline(4, stage=2)" in str(k.fn.as_python())
