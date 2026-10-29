# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
One body, called like a function: statements need a declared boundary.

Example 03 ended on a rule: a helper that computes a staged *value* may be
called anywhere an expression is expected (``block.scaled(k).value``). Its
result is an operand, so nothing else is needed.

Statements are the other side of that boundary. A helper that emits
``pl.store`` is not an operand — it has to land somewhere in the trace, and a
plain Python call is not a place the trace expects statements. ``07_diagnostics``
shows what happens without a boundary: the statements are dropped, the kernel
compiles, and the output is quietly untouched.

So a reusable block of statements declares itself once, and is then called
like a function:

    @pl.jit_trace.body
    def scale_block(a, out, row0, k): ...

    scale_block(a, out, row0, k)      # the statements land here

That is the whole API a kernel author needs. The fragment expands at the call
site — no callee, no ``ir.Call``, nothing for a later pass to inline — which is
why a body has no signature to declare and no annotations to get right. Two
different argument types can pass through the same body.

With that, the kernel reads as the algorithm:

    with pl.at(...):                   # one on-chip compute region
        for chunk in pl.range(...):    # the grid, stated as a loop
            row0 = chunk * TILE
            scale_block(a, out, row0, k)

The framework-author view of the same thing — the placements a body can take,
and how ``@pl.jit_trace.body`` itself is built — is
``advanced/06_recipe_policy.py``.

Run:  python examples/tracing/04_body_placement.py
Next: examples/tracing/05_block_recipe.py
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


def _normalized(kernel, entry: str, *args) -> str:
    """The printed program with its entry and derived callee names folded to ``P``.

    Outlining names each derived callee after its entry (``scale_incore_0``), so
    the entry name has to go first for the comparison to be about statements
    rather than about naming.
    """
    return _program_text(kernel, *args).replace(f"{entry}_", "P_").replace(entry, "P")


# ---- The fragment, declared once -------------------------------------------


@pl.jit_trace.body
def scale_block(a, out, row0, k):
    """One tile of ``out = a * k``.

    No IR annotation: a body is expanded, so there is no callee to declare a
    signature for. ``row0`` is whatever the caller passes — a host constant or
    a staged offset.
    """
    tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
    pl.store(pl.mul(tile, k), [row0, 0], out)


# ---- The kernel it keeps readable ------------------------------------------


@pl.jit_trace
def scale(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The algorithm, with the tile step named once."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            scale_block(a, out, row0, k)
    return out


# ---- The same statements, pasted, for the structural claim -----------------


@pl.jit_trace
def scale_pasted(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The tile step written out in place — what a body must lower to."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


# ---- A body may deliver a value as well as emit statements -----------------


@pl.jit_trace.body
def scaled(x, k):
    """A value-only body: it returns an operand and emits nothing."""
    return pl.mul(x, k)


@pl.jit_trace
def scale_whole(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The return crosses back into the caller's expression position."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(scaled(tile, k), [0, 0], out)
    return out


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. Naming the step changes the source, not the result.
    for name, kernel in (("body", scale), ("pasted", scale_pasted)):
        out = torch.zeros_like(a)
        kernel(a, 2.0, out, config=SIM)
        assert torch.allclose(out, a * 2.0), name
    print("numerics : body == pasted == a * 2.0")

    # 2. ... nor the program. A body call *is* the pasted statements: the
    #    fragment leaves nothing behind, which is what "expands at the call
    #    site" means.
    body_ir = _normalized(scale, "scale", a, 2.0, empty)
    pasted_ir = _normalized(scale_pasted, "scale_pasted", a, 2.0, empty)
    assert body_ir == pasted_ir, "a body call lowers to the statements it contains"
    print("program  : body == pasted, byte for byte — no callee, nothing to inline")

    # 3. A body is not a function: it names no function in the program. The
    #    only callee here is the caller's own pl.at region.
    program = scale.compile_for_test(a, 2.0, empty)
    names = sorted(global_var.name for global_var in program.functions)
    assert "scale_block" not in names, names
    print("functions:", ", ".join(names), "— the body is not among them")

    # 4. The return path: a body that only computes a value needs no statement
    #    boundary at all, and its result is an ordinary operand.
    out = torch.zeros_like(a)
    scale_whole(a, 3.0, out, config=SIM)
    assert torch.allclose(out, a * 3.0)
    print("value    : scaled(tile, k) returns an operand — out == a * 3.0")

    print("OK")


if __name__ == "__main__":
    main()
