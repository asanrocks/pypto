# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""A traced subscript write is one contiguous window, so a step is refused.

``Tensor.__setitem__`` and ``Tile.__setitem__`` both lower to a contiguous
``assemble`` at the slice's start offset. A step would therefore write
consecutive elements where the source asked for every other one — a silently
different program. Reads already refuse a step; writes now do too.

The write also spells a window — ``out[0:2, :] = src`` writes two rows — so
the source is checked against the stop-derived extents exactly like the
parser's ``_parse_subscript_assignment``: a source larger than the window
would silently overwrite past it, a smaller one silently under-fill it, and
a rank-mismatched one has no rank-lifted spelling on the traced frontend.
"""

import pypto.language as pl
import pytest
import torch

F32 = torch.float32


def _tensor() -> torch.Tensor:
    return torch.randn(64, 64, dtype=F32)


def _out() -> torch.Tensor:
    return torch.empty(64, 64, dtype=F32)


class TestSliceStepOnWriteTargets:
    """A write lowers to a contiguous window; a step would move the rows."""

    def test_tensor_write_with_a_step_is_rejected(self):
        @pl.jit_trace
        def strided_write(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[::2, :] = x[:32, :]
            return out

        with pytest.raises(TypeError, match="step"):
            strided_write.compile_for_test(_tensor(), _out())

    def test_tensor_write_with_a_bounded_step_is_rejected(self):
        @pl.jit_trace
        def strided_write(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[0:16:2, :] = x[:8, :]
            return out

        with pytest.raises(TypeError, match="step"):
            strided_write.compile_for_test(_tensor(), _out())

    def test_tile_write_with_a_step_is_rejected(self):
        @pl.jit_trace
        def strided_tile_write(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                tile[::2, :] = pl.load(x, [0, 0], [32, 64])
                pl.store(tile, [0, 0], out)
            return out

        with pytest.raises(TypeError, match="step"):
            strided_tile_write.compile_for_test(_tensor(), _out())

    def test_contiguous_write_still_compiles(self):
        """The supported write keeps lowering to a contiguous assemble."""

        @pl.jit_trace
        def window_write(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[:32, :] = x[:32, :]
            return out

        program = window_write.compile_for_test(_tensor(), _out())
        assert "window_write_incore_0" in {fn.name for fn in program.functions.values()}


class TestWindowShapeValidation:
    """The source must cover exactly the window the key spelled."""

    def test_tensor_source_larger_than_the_window_is_rejected(self):
        @pl.jit_trace
        def over_window(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[0:2, :] = x[0:4, :]
            return out

        with pytest.raises(TypeError, match="window expects 2 elements, source has 4"):
            over_window.compile_for_test(_tensor(), _out())

    def test_tensor_source_smaller_than_the_window_is_rejected(self):
        @pl.jit_trace
        def under_window(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[0:32, :] = x[0:8, :]
            return out

        with pytest.raises(TypeError, match="shape mismatch on source axis 0"):
            under_window.compile_for_test(_tensor(), _out())

    def test_tensor_source_of_matching_window_still_compiles(self):
        @pl.jit_trace
        def exact_window(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[0:32, 0:16] = x[0:32, 0:16]
            return out

        program = exact_window.compile_for_test(_tensor(), _out())
        assert "exact_window_incore_0" in {fn.name for fn in program.functions.values()}

    def test_rank_mismatched_tensor_source_is_rejected(self):
        @pl.jit_trace
        def rank_mismatch(
            x: pl.Tensor[[64, 64], pl.FP32],
            v: pl.Tensor[[64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                out[0:32, :] = v
            return out

        with pytest.raises(TypeError, match="rhs rank must match the target"):
            rank_mismatch.compile_for_test(_tensor(), torch.randn(64, dtype=F32), _out())

    def test_tile_source_larger_than_the_window_is_rejected(self):
        @pl.jit_trace
        def tile_over_window(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [32, 64])
                tile[0:8, :] = pl.load(x, [0, 0], [32, 64])
                pl.store(tile, [0, 0], out)
            return out

        with pytest.raises(TypeError, match="window expects 8 elements, source has 32"):
            tile_over_window.compile_for_test(_tensor(), _out())

    def test_tile_source_of_matching_window_still_compiles(self):
        @pl.jit_trace
        def tile_exact_window(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [32, 64])
                tile[0:32, :] = pl.load(x, [0, 0], [32, 64])
                pl.store(tile, [0, 0], out)
            return out

        program = tile_exact_window.compile_for_test(_tensor(), _out())
        assert "tile_exact_window_incore_0" in {fn.name for fn in program.functions.values()}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
