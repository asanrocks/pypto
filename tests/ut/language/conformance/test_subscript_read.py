# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The numpy-style subscript reads on the ``Tensor`` / ``Tile``
facades.

Ports the AST's ``_parse_tensor_subscript`` / ``_parse_tile_subscript``
surface onto the existing ``tensor.slice`` / ``tensor.read`` /
``tile.slice`` / ``tile.read`` ops (``typing/_subscript.py``): all-scalar
full-rank → a scalar read; partial / sliced / rank-reducing → a slice view
carrying ``drop_dims``, with the tile valid-shape intersection and the 2D
clamp. Post-canonical anchors assert the tracing frontend's IR equals the AST
frontend's for the same kernels (the tracing frontend's "Tensor/Tile subscript syntax
is only available inside @pl.function" NotImplementedError lines die).

Guardrails mirror the AST's own rejections: slice ``step`` and tile dynamic
lower bounds are loud; too many indices for the rank is loud.
"""

import warnings

import pypto.language as pl
import pytest
from pypto.language.tracer.decorators.function import function_trace
from pypto.pypto_core import ir

from ._harness import conformance

# —— the anchored read kernels (post-canonical: raw equality vs function_parser) ——


def k_tensor_read(x: pl.Tensor[[64, 64], pl.INT32], i: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    v = x[i, 0]
    return v


def k_tensor_slice(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64], pl.INT32]:
    row = x[3]
    return row


def k_tensor_slice_range(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[32, 64], pl.INT32]:
    row = x[0:32]
    return row


def k_tile_read(x: pl.Tensor[[64, 64], pl.INT32], i: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.INT32]:
    t = pl.load(x, [0, 0], [64, 64])
    v = t[i, 0]
    return v


def k_tile_slice(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
    t = pl.load(x, [0, 0], [64, 64])
    s = t[0:32]
    pl.store(s, [0, 0], x)
    return x


def k_tile_valid_shape(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
    t = pl.load(x, [0, 0], [64, 64], [32, 64])
    s = t[0:16]
    pl.store(s, [0, 0], x)
    return x


def test_tensor_read_conformance():
    conformance(k_tensor_read, function_trace, canonical=True)


def test_tensor_slice_conformance():
    conformance(k_tensor_slice, function_trace, canonical=True)


def test_tensor_slice_range_conformance():
    conformance(k_tensor_slice_range, function_trace, canonical=True)


def test_tile_read_conformance():
    conformance(k_tile_read, function_trace, canonical=True)


def test_tile_slice_conformance():
    conformance(k_tile_slice, function_trace, canonical=True)


def test_tile_valid_shape_conformance():
    conformance(k_tile_valid_shape, function_trace, canonical=True)


# —— the 2D-promotion warning (the AST's non-fatal clamp notice) ——


def k_tile_promotion(x: pl.Tensor[[64], pl.INT32]) -> pl.Tensor[[64], pl.INT32]:
    t = pl.load(x, [0, 0], [64, 1])
    s = t[0]
    pl.store(s, [0, 0], x)
    return x


def test_tile_promotion_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        function_trace(k_tile_promotion, function_type=ir.FunctionType.InCore, level=pl.Level.CHIP_DIE)
    assert any("auto-promoting to 2D" in str(x.message) for x in w)


# —— guardrails (the AST's own loud rejections) ——


def test_slice_step_rejected():
    def kernel(x: pl.Tensor[[64, 64], pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
        row = x[0:32:2]
        return row

    with pytest.raises(TypeError, match="step"):
        function_trace(kernel)


def test_tile_dynamic_lower_rejected():
    def kernel(x: pl.Tensor[[64, 64], pl.INT32], i: pl.Scalar[pl.INT32]) -> pl.Tensor[[64, 64], pl.INT32]:
        t = pl.load(x, [0, 0], [64, 64])
        s = t[i:32]
        pl.store(s, [0, 0], x)
        return x

    with pytest.raises(TypeError, match="dynamic lower bounds"):
        function_trace(kernel, function_type=ir.FunctionType.InCore, level=pl.Level.CHIP_DIE)


def test_too_many_indices_rejected():
    def kernel(x: pl.Tensor[[64], pl.INT32]) -> pl.Scalar[pl.INT32]:
        v = x[0, 0]
        return v

    with pytest.raises(TypeError, match="indices but the operand"):
        function_trace(kernel)
