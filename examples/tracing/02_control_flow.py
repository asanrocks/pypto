# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Which loop is a loop: control flow follows the domain of its subject.

Example 01 gave one number two possible homes. The same question applies to a
loop, and the answer is the same rule applied to the loop's subject:

    for chunk in range(CHUNKS):     # host subject: really iterates now
    for chunk in pl.range(CHUNKS):  # staged subject: recorded once, repeats on device

Four tile updates, spelled two ways. The host loop runs four times while the
body executes, so the program holds four copies of the body. The staged loop
runs *once*, for one representative iteration, and the program holds a loop.
Both compute the same thing — the difference is in the code that ships.

This is the first abstraction the tracing frontend adds over the parser: there
is no loop *syntax* to learn, only the domain of the thing you iterate over.

Run:  python examples/tracing/02_control_flow.py
Next: examples/tracing/03_host_objects.py
"""

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

ROWS = 64
COLS = 64
TILE_ROWS = 16
CHUNKS = ROWS // TILE_ROWS

SIM = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM)


def _program_text(kernel, *args) -> str:
    return ir.python_print(kernel.compile_for_test(*args))


def _loads(program: str) -> int:
    """Load calls in a printed program: ``tile.load`` before the passes, ``buffer.load`` after."""
    return sum(1 for line in program.splitlines() if ".load" in line)


@pl.jit_trace
def unrolled(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A host loop: ``range`` iterates while the body runs."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def recorded(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A staged loop: ``pl.range`` records one body and repeats it on device."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def unrolled_wide(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The host spelling of a tile-size change: fewer, wider tiles."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(ROWS // 32):
            row0 = chunk * 32
            tile = pl.load(a, [row0, 0], [32, COLS])
            pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def recorded_wide(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The staged spelling of the same change: still one copy of the body."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(ROWS // 32):
            row0 = chunk * 32
            tile = pl.load(a, [row0, 0], [32, COLS])
            pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. Same answer either way — the loop spelling is not a semantics change.
    host_out, staged_out = torch.zeros_like(a), torch.zeros_like(a)
    unrolled(a, host_out, config=SIM)
    recorded(a, staged_out, config=SIM)
    assert torch.allclose(host_out, a * 2.0)
    assert torch.allclose(staged_out, a * 2.0)
    assert torch.allclose(host_out, staged_out)
    print("numerics : host loop == staged loop == a * 2.0")

    # 2. The programs differ exactly as the domains predict.
    host_ir = _program_text(unrolled, a, empty)
    staged_ir = _program_text(recorded, a, empty)
    print(f"unrolled : {_loads(host_ir)} loads, has a loop: {'for ' in host_ir}")
    print(f"recorded : {_loads(staged_ir)} load(s), has a loop: {'for ' in staged_ir}")
    assert _loads(host_ir) == CHUNKS, "a host loop copies the body per iteration"
    assert _loads(staged_ir) == 1, "a staged loop records the body once"
    assert "for " not in host_ir and "for " in staged_ir

    # 3. The payoff: the recorded program is size-invariant, the unrolled one
    #    is not. A tiling decision moves the pasted program's size — its body
    #    is copied once per source iteration — while the staged program still
    #    holds one body and reads its bounds from the loop.
    host_wide = _program_text(unrolled_wide, a, empty)
    staged_wide = _program_text(recorded_wide, a, empty)
    assert _loads(host_wide) == ROWS // 32, "the pasted body follows the tiling decision"
    assert _loads(staged_wide) == 1, "the recorded body does not"
    print(f"tile 32  : unrolled {_loads(host_wide)} loads, recorded {_loads(staged_wide)} load")
    print("           — the tiling decision shows up in one program, not in the other")

    print("OK")


if __name__ == "__main__":
    main()
