# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Subscript-write sugar (``dst[...] = src`` → ``dst = pl.assemble(dst, src, offsets)``).

Regression (gap K7 / task F1): the written indices follow the AST parser's
numpy rule — a bare key is axis 0, and dims past the written indices are
implicit full slices — so the offsets are always full-rank. Before the fix a
bare ``t[:]`` wrapped to a one-axis subscript whatever the target rank: a
rank-2 ``out[:] = ...`` traced to ``assemble(out, src, [0])`` and ptoas
rejected the store (``'pto.partition_view' op offset count (1) must match
source rank (2)``), while a too-wide key (three indices on a 2D target)
lowered silently to a three-offset assemble.
"""

import pypto.language as pl
import pytest
import torch
from pypto import ir
from pypto.runtime import RunConfig


class TestTensorSubscriptWrite:
    """Tensor ``__setitem__`` offsets are full-rank (numpy-style)."""

    def test_bare_slice_expands_to_target_rank(self):
        """``out[:]`` on a rank-2 target lowers to offsets ``[0, 0]``."""

        @pl.function
        def sugared(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            out: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
            out[:] = pl.abs(x)
            return out

        @pl.function
        def explicit(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            out: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
            out = pl.tensor.assemble(out, pl.abs(x), [0, 0])
            return out

        ir.assert_structural_equal(sugared, explicit)

    def test_partial_indices_pad_trailing_axes(self):
        """``out[2:8]`` on a rank-2 target: axis 1 is an implicit full slice."""

        @pl.function
        def sugared(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            out: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
            out[2:8] = pl.slice(x, [6, 128], [0, 0])
            return out

        @pl.function
        def explicit(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            out: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
            out = pl.tensor.assemble(out, pl.slice(x, [6, 128], [0, 0]), [2, 0])
            return out

        ir.assert_structural_equal(sugared, explicit)

    def test_rank1_bare_slice_stays_one_axis(self):
        """A rank-1 target's bare slice is the whole tensor — unchanged."""

        @pl.function
        def sugared(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            out[:] = pl.abs(x)
            return out

        @pl.function
        def explicit(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
            out: pl.Tensor[[64], pl.FP32] = pl.create_tensor([64], dtype=pl.FP32)
            out = pl.tensor.assemble(out, pl.abs(x), [0])
            return out

        ir.assert_structural_equal(sugared, explicit)

    def test_too_many_indices_rejected_loudly(self):
        """More indices than the target's rank is a loud trace-time error."""

        with pytest.raises(TypeError, match="3 indices but the target is 2D"):

            @pl.function
            def kernel(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
                out: pl.Tensor[[16, 128], pl.FP32] = pl.create_tensor([16, 128], dtype=pl.FP32)
                out[0:4, 0:4, 0:4] = pl.abs(x)
                return out


class TestTileSubscriptWrite:
    """Tile ``__setitem__`` shares the same full-rank rule."""

    def test_bare_slice_expands_to_target_rank(self):
        """``t[:]`` on a rank-2 tile lowers to offsets ``[0, 0]``."""

        @pl.function
        def sugared(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            target[:] = pl.abs(target)
            return pl.store(target, [0, 0], x)

        @pl.function
        def explicit(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            target = pl.tile.assemble(target, pl.abs(target), [0, 0])
            return pl.store(target, [0, 0], x)

        ir.assert_structural_equal(sugared, explicit)

    def test_partial_indices_pad_trailing_axes(self):
        """``t[2:8]`` on a rank-2 tile: axis 1 is an implicit full slice."""

        @pl.function
        def sugared(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            source: pl.Tile[[6, 128], pl.FP32] = pl.load(x, [0, 0], [6, 128])
            target[2:8] = source
            return pl.store(target, [0, 0], x)

        @pl.function
        def explicit(x: pl.Tensor[[16, 128], pl.FP32]) -> pl.Tensor[[16, 128], pl.FP32]:
            target: pl.Tile[[16, 128], pl.FP32] = pl.load(x, [0, 0], [16, 128])
            source: pl.Tile[[6, 128], pl.FP32] = pl.load(x, [0, 0], [6, 128])
            target = pl.tile.assemble(target, source, [2, 0])
            return pl.store(target, [0, 0], x)

        ir.assert_structural_equal(sugared, explicit)


class TestSubscriptWriteReachesPtoas:
    """The end-to-end regression (gap K7's evidence): a rank-2 bare-slice
    write must survive the assembler, not just trace."""

    def test_rank2_bare_slice_write_compiles_through_ptoas(self):
        """A 2D ``scratch[:]`` / ``out[:]`` task-graph entry compiles on a2a3.

        Before the fix ptoas rejected both stages'
        ``pto.partition_view`` (offset count 1 vs source rank 2).
        """

        @pl.jit
        def inferred_edge(x: pl.Tensor, scratch: pl.Out[pl.Tensor], out: pl.Out[pl.Tensor]):
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage1"):
                scratch[:] = pl.add(x, x)
            with pl.at(level=pl.Level.CORE_GROUP, name_hint="stage2"):
                out[:] = pl.add(scratch, scratch)
            return scratch, out

        x = torch.randn(128, 128, dtype=torch.float32)
        scratch = torch.zeros(128, 128, dtype=torch.float32)
        out = torch.zeros(128, 128, dtype=torch.float32)
        compiled = inferred_edge.compile(x, scratch, out, config=RunConfig(platform="a2a3"))
        assert compiled is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
