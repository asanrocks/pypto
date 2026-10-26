# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Unit tests for the mixed-extent shape overloads of ir.TensorType / ir.TileType.

``shape: Sequence[Expr | int]`` is the declared contract; the raw constructors
normalize a literal extent into a ``ConstInt``. These tests pin the acceptance
boundary: anything that is not an ``Expr`` or an ``int`` — ``bool`` included,
since ``PyNumber_Index`` would silently convert it to 0/1 — must be rejected
with a ``TypeError`` naming the offending element index.
"""

import pytest
from pypto import DataType, ir


@pytest.fixture
def symbolic_dim():
    span = ir.Span.unknown()
    return ir.Var("n", ir.ScalarType(DataType.INT32), span)


class TestTensorTypeMixedShape:
    def test_mixed_expr_and_constant_accepted(self, symbolic_dim):
        tensor_type = ir.TensorType([symbolic_dim, 16], DataType.FP32)
        assert len(tensor_type.shape) == 2
        assert isinstance(tensor_type.shape[0], ir.Var)
        assert isinstance(tensor_type.shape[1], ir.ConstInt)

    def test_homogeneous_constant_shape_still_accepted(self):
        tensor_type = ir.TensorType([32, 64], DataType.FP32)
        assert len(tensor_type.shape) == 2

    def test_homogeneous_expr_shape_still_accepted(self, symbolic_dim):
        span = ir.Span.unknown()
        shape = [ir.ConstInt(2, DataType.INT32, span), symbolic_dim]
        tensor_type = ir.TensorType(shape, DataType.FP32)
        assert isinstance(tensor_type.shape[0], ir.ConstInt)

    # `None` is deliberately absent: the homogeneous vector<ExprPtr> overload
    # casts it to a null pointer, and that pre-existing behaviour is out of
    # scope here (the mixed overload is never selected for it).
    @pytest.mark.parametrize("bad", [True, "16", 2.5])
    def test_non_int_element_rejected_with_index(self, symbolic_dim, bad):
        with pytest.raises(TypeError, match=r"index 1"):
            ir.TensorType([symbolic_dim, bad], DataType.FP32)

    def test_bool_element_reported_as_bool(self, symbolic_dim):
        with pytest.raises(TypeError, match="bool"):
            ir.TensorType([symbolic_dim, True], DataType.FP32)

    def test_out_of_range_int_element_rejected_with_index(self, symbolic_dim):
        # A Python int that does not fit int64 must not surface as a context-free
        # `RuntimeError: std::bad_cast` from nanobind's caster.
        with pytest.raises(ValueError, match=r"index 1.*int64"):
            ir.TensorType([symbolic_dim, 2**70], DataType.FP32)


class TestTileTypeMixedShape:
    def test_mixed_expr_and_constant_accepted(self, symbolic_dim):
        tile_type = ir.TileType([16, symbolic_dim], DataType.FP32)
        assert len(tile_type.shape) == 2
        assert isinstance(tile_type.shape[1], ir.Var)

    def test_homogeneous_constant_shape_still_accepted(self):
        tile_type = ir.TileType([16, 64], DataType.FP32)
        assert len(tile_type.shape) == 2

    @pytest.mark.parametrize("bad", [True, "16", 2.5])
    def test_non_int_element_rejected_with_index(self, bad):
        with pytest.raises(TypeError, match=r"index 1"):
            ir.TileType([16, bad], DataType.FP32)

    def test_out_of_range_int_element_rejected_with_index(self):
        with pytest.raises(ValueError, match=r"index 0.*int64"):
            ir.TileType([2**70, 16], DataType.FP32)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
