# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tile arguments at a traced call boundary.

A loaded tile is a value in the traced vocabulary, so an ``.inline`` helper
over one is the natural way to share on-chip arithmetic:

    @pl.jit_trace.inline
    def twice(t: pl.Tile):
        return pl.add(t, t)

The call-site adapter has to describe that argument as an annotation before the
callee can be traced, and the tile's type carries more than a tensor's — a
memory space and a view. The adapter therefore rebuilds the ``pl.Tile``
annotation from the argument's whole ``TileType`` instead of refusing it, so
the callee's parameter is the argument's type rather than an approximation.
"""

import pypto.language as pl
import pytest
import torch
from pypto.pypto_core import DataType, ir

F32 = torch.float32


@pl.jit_trace.inline
def twice(t: pl.Tile):
    """The helper this suite is about: a tile in, a tile out."""
    return pl.add(t, t)


@pl.jit_trace.inline
def keep(t: pl.Tile):
    """Identity helper, for inspecting the type a callee was traced with."""
    return t


@pl.jit_trace
def tile_inline_entry(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        doubled = twice(tile)
        pl.store(doubled, [0, 0], out)
    return out


@pl.jit_trace
def mat_tile_entry(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """A cube tile: its memory space is the metadata that must survive."""
    with pl.at(level=pl.Level.CORE_GROUP):
        cube = pl.load(x, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
        keep(cube)
        tile = pl.load(x, [0, 0], [64, 64])
        pl.store(tile, [0, 0], out)
    return out


def _tensor() -> torch.Tensor:
    return torch.empty(64, 64, dtype=F32)


def _variant(program: ir.Program, name: str) -> ir.Function:
    fn = next((f for f in program.functions.values() if f.name == name), None)
    assert fn is not None, f"missing callee {name!r}; have {[f.name for f in program.functions.values()]}"
    return fn


class TestTileArguments:
    """A tile argument reaches the callee with its type intact."""

    def test_inline_helper_over_a_tile_compiles(self):
        program = tile_inline_entry.compile_for_test(_tensor(), _tensor())
        names = {fn.name for fn in program.functions.values()}
        assert "tile_inline_entry_incore_0" in names, names
        # The inline helper is spliced at the call site, so it is not a callee.
        assert "twice" not in names, names

    def test_callee_parameter_is_the_argument_tile_type(self):
        program = tile_inline_entry.specialize(_tensor(), _tensor())
        param = _variant(program, "twice").params[0]
        assert isinstance(param.type, ir.TileType)
        assert param.type.dtype == DataType.FP32
        assert [dim.value if isinstance(dim, ir.ConstInt) else None for dim in param.type.shape] == [64, 64]

    def test_callee_keeps_the_cube_memory_space(self):
        program = mat_tile_entry.specialize(_tensor(), _tensor())
        param = _variant(program, "keep").params[0]
        assert isinstance(param.type, ir.TileType)
        assert param.type.memory_space == ir.MemorySpace.Mat

    def test_tile_call_sites_at_different_types_get_distinct_variants(self):
        """A load's implicit space and ``tile.add``'s inferred ``Vec`` differ.

        The two arguments agree on shape and dtype, so a key that ignored the
        tile's metadata would hand the second call the first variant — whose
        parameter is a different type. Both are kept, each as traced.
        """

        @pl.jit_trace
        def two_calls(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [64, 64])
                first = twice(tile)
                second = twice(first)
                pl.store(second, [0, 0], out)
            return out

        program = two_calls.specialize(_tensor(), _tensor())
        variants = {
            fn.name: fn.params[0].type.memory_space
            for fn in program.functions.values()
            if fn.name != "two_calls"
        }
        assert variants == {"twice": None, "twice_2": ir.MemorySpace.Vec}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
