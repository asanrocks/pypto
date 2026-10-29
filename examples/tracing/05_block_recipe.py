# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
A tiling pattern as a library construct.

Examples 01-04 built up one kernel. This one is about kernels in the plural:
a model has dozens, and they keep repeating the same arbitration story —

    divide the rows into blocks, dispatch one block per core-group, decode
    this block's rows out of the block index, and slice accordingly.

Read this file in three steps:

1. **The pattern, twice (by hand).** The two kernels below write that story
   out in full — the grid count, the block index, the tile constants. The only
   thing that differs between them is the arithmetic on the tile.
2. **The construct a kernel author uses.** Two more kernels, and a third for
   the same reason, state the grid once:

       with row_blocks(ROWS, TILE_ROWS, name_hint="scale") as grid:
           row0 = grid.row_offset
           ...

   Nothing else about them changes. That is the whole point: the grid is
   *policy* (how many blocks, which core-group gets which), and policy belongs
   to whoever owns the tiling, not to every kernel that happens to tile.
3. **The implementation (framework-author territory).** The recipe is 20 lines
   of ordinary Python over ``pl.authoring``: a generator whose code before
   ``yield`` opens the region, whose ``yield`` hands the body a plain Python
   handle, and whose code after ``yield`` runs at construct exit. It adds no
   lowering of its own — the program it builds is the one the hand-written
   grid builds, byte for byte, which is the whole claim of an authoring tier
   ("the compiler has no magic").

The payoff is where an improvement goes when it lands: into the recipe, once,
for every kernel. A validation the policy owns (this example's divisibility
check) protects every consumer even after the consumer is written, and it is
the reason a kernel author never mentions ``get_block_idx``. At model scale the
same shape carries the DeepSeek block grids — see
``pypto-lib/models/deepseek_v4_flash_mtp/blocked_spmd.py``, the same construct
over shape/tile axes and dispatch dependencies.

Run:  python examples/tracing/05_block_recipe.py
Next: examples/tracing/advanced/06_recipe_policy.py
"""

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

ROWS = 64
COLS = 64
TILE_ROWS = 16

SIM = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM)


def _program_text(kernel, *args) -> str:
    return ir.python_print(kernel.compile_for_test(*args))


# ---- Step 1: the pattern, written by hand in two kernels -------------------


@pl.jit_trace
def scale_by_hand(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Kernel 1: scale every row, tiled by hand."""
    with pl.spmd(ROWS // TILE_ROWS, name_hint="scale") as grid:  # noqa: F841
        row0 = pl.get_block_idx() * TILE_ROWS
        tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
        pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def bias_by_hand(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Kernel 2: add a bias to every row. Same grid story, again."""
    with pl.spmd(ROWS // TILE_ROWS, name_hint="bias") as grid:  # noqa: F841
        row0 = pl.get_block_idx() * TILE_ROWS
        tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
        pl.store(pl.add(tile, 1.0), [row0, 0], out)
    return out


# ---- Step 2: the construct a kernel author uses ----------------------------


@pl.jit_trace
def scale_by_recipe(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Kernel 1 again: the grid is stated once, nothing else changed."""
    with row_blocks(ROWS, TILE_ROWS, name_hint="scale") as grid:
        row0 = grid.row_offset  # each access is a fresh block-index read: bind it once
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def bias_by_recipe(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Kernel 2 again: the same grid, no second copy of the policy."""
    with row_blocks(ROWS, TILE_ROWS, name_hint="bias") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.add(tile, 1.0), [row0, 0], out)
    return out


@pl.jit_trace
def relu_by_recipe(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A third consumer: the same grid, written after the policy already existed."""
    with row_blocks(ROWS, TILE_ROWS, name_hint="relu") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.relu(tile), [row0, 0], out)
    return out


@pl.jit_trace
def bad_grid(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A grid that does not divide the rows: refused by the recipe itself."""
    with row_blocks(ROWS, 20) as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(tile, [row0, 0], out)
    return out


# ---- Step 3: the implementation (framework-author territory) ---------------
#
# Everything below is written once per *policy*, not once per kernel. It uses
# ``pl.authoring``, which is the surface for people building constructs for
# others; a kernel author only ever writes the three ``with`` blocks above.


class RowGrid(pl.authoring.ConstructHandle):
    """The ``as`` target: this block's row offset and its tile capacity."""

    def __init__(self, tile_rows: int) -> None:
        self._tile_rows = tile_rows

    @property
    def row_offset(self):
        """Staged element offset of this block's first row.

        A property, not a value: every access reads the block index again, so
        a kernel binds it once when it needs the same offset twice.
        """
        return pl.get_block_idx() * self._tile_rows

    @property
    def tile_rows(self) -> int:
        """Static tile capacity of one block."""
        return self._tile_rows


@pl.authoring.construct
def row_blocks(ctx, rows: int, tile_rows: int, *, name_hint: str = ""):
    """Dispatch one block per group of ``tile_rows`` rows.

    Everything before ``yield`` is host code with the usual Python
    obligations: the arguments are checked here, and a bad grid raises at the
    call site instead of lowering.
    """
    if rows % tile_rows:
        raise ValueError(f"row_blocks: {rows} rows do not divide into {tile_rows}-row blocks")
    with ctx.region(
        pl.spmd(rows // tile_rows, name_hint=name_hint),
        single_call_body=ctx.single_call_body,
    ):
        yield RowGrid(tile_rows)


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. Each kernel keeps its own arithmetic; only the grid moved.
    for name, by_recipe, by_hand, expected in (
        ("scale", scale_by_recipe, scale_by_hand, a * 2.0),
        ("bias", bias_by_recipe, bias_by_hand, a + 1.0),
    ):
        recipe_out, hand_out = torch.zeros_like(a), torch.zeros_like(a)
        by_recipe(a, recipe_out, config=SIM)
        by_hand(a, hand_out, config=SIM)
        assert torch.allclose(recipe_out, expected), name
        assert torch.allclose(recipe_out, hand_out), name
    print("numerics : both kernels unchanged — scale == a * 2.0, bias == a + 1.0")

    # 2. ... and each one builds exactly the program its hand-written twin
    #    builds, byte for byte. The recipe is a Python object driving existing
    #    constructs; it leaves nothing of its own in the IR.
    for name, by_recipe, by_hand in (
        ("scale", scale_by_recipe, scale_by_hand),
        ("bias", bias_by_recipe, bias_by_hand),
    ):
        recipe_ir = _program_text(by_recipe, a, empty).replace(f"{name}_by_recipe", "P")
        hand_ir = _program_text(by_hand, a, empty).replace(f"{name}_by_hand", "P")
        assert recipe_ir == hand_ir, f"{name}: an authored construct must lower to the hand-written grid"
    print("program  : recipe == hand-written, byte for byte, for both kernels")

    # 3. A third consumer needs nothing from the policy's history: it states
    #    the grid and gets the same contract. This is what "the improvement
    #    landed once" means for the next kernel someone writes.
    relu_out = torch.zeros_like(a)
    relu_by_recipe(a, relu_out, config=SIM)
    assert torch.allclose(relu_out, torch.relu(a))
    print("third    : relu reuses the same grid — out == relu(a)")

    # 4. Host code in the recipe keeps host semantics: the grid is validated
    #    while the body runs, so the error is an ordinary Python one — and it
    #    protects every consumer, not just the one that existed when it landed.
    try:
        bad_grid.compile_for_test(a, empty)
    except ValueError as exc:
        print("refused  :", exc)
    else:  # pragma: no cover - the recipe's own check is the point
        raise AssertionError("a non-dividing grid must be refused by the recipe")

    print("OK")


if __name__ == "__main__":
    main()
