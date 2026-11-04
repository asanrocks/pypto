# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Host scalars may feed a staged loop, but the loop may not advance them.

A traced body runs one representative iteration, so a host value rebound inside
a staged loop would be baked as the constant that iteration produced while every
untraced iteration expected it to advance. The binding guard refuses that
rebinding and points at the staged spelling, which carries across iterations.
"""

import pypto.language as pl
import pytest
import torch

F32 = torch.float32


def _tensor() -> torch.Tensor:
    return torch.randn(64, 64, dtype=F32)


def _out() -> torch.Tensor:
    return torch.empty(64, 64, dtype=F32)


class TestLoopCarriedHostState:
    """A host value may be read in a staged loop but not advanced by it."""

    def test_host_rebinding_inside_a_staged_loop_is_rejected(self):
        """``factor *= 2`` bakes the representative iteration's 2.0 forever."""

        @pl.jit_trace
        def advancing_scale(
            x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ):
            factor = 1.0
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                for _i in pl.range(n):
                    factor = factor * 2.0
                    pl.store(pl.mul(tile, factor), [0, 0], out)
            return out

        with pytest.raises(TypeError, match="loop-carried host"):
            advancing_scale.compile_for_test(_tensor(), 3, _out())

    def test_host_rebinding_inside_a_staged_while_is_rejected(self):
        """The same rule holds for the staged ``while`` spelling."""

        @pl.jit_trace
        def advancing_scale(
            x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ):
            factor = 1.0
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                while n > 0:
                    factor = factor * 2.0
                    pl.store(pl.mul(tile, factor), [0, 0], out)
            return out

        with pytest.raises(TypeError, match="loop-carried host"):
            advancing_scale.compile_for_test(_tensor(), 3, _out())

    def test_host_constant_read_inside_a_staged_loop_still_compiles(self):
        """A host knob the loop only *reads* is a bake, not carried state."""

        @pl.jit_trace
        def fixed_scale(
            x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ):
            factor = 2.0
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                for _i in pl.range(n):
                    pl.store(pl.mul(tile, factor), [0, 0], out)
            return out

        program = fixed_scale.compile_for_test(_tensor(), 3, _out())
        assert "fixed_scale_incore_0" in {fn.name for fn in program.functions.values()}

    def test_fresh_host_local_inside_a_staged_loop_still_compiles(self):
        """A host local first bound in the body is per-iteration, not carried."""

        @pl.jit_trace
        def per_iteration_scale(
            x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                for _i in pl.range(n):
                    local = 2.0
                    local = local * 3.0
                    pl.store(pl.mul(tile, local), [0, 0], out)
            return out

        program = per_iteration_scale.compile_for_test(_tensor(), 3, _out())
        assert "per_iteration_scale_incore_0" in {fn.name for fn in program.functions.values()}

    def test_staged_carry_is_the_supported_spelling(self):
        """The staged accumulator keeps its per-iteration update in the loop."""

        @pl.jit_trace
        def staged_scale(
            x: pl.Tensor[[64, 64], pl.FP32], n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                acc = pl.mul(tile, 1.0)
                for _i in pl.range(n):
                    acc = pl.mul(acc, 2.0)
                    pl.store(acc, [0, 0], out)
            return out

        program = staged_scale.compile_for_test(_tensor(), 3, _out())
        assert "staged_scale_incore_0" in {fn.name for fn in program.functions.values()}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
