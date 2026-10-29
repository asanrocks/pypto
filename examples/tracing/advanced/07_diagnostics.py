# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Two ways a traced kernel goes wrong — and the diagnostics that catch them.

Tracing executes the kernel body, so a mistake does not have to be a syntax
error to be dangerous: the body can run happily and produce a program that
computes something else. The frontend closes those paths, but it closes them in
two different ways, and knowing which is which saves an afternoon.

Part 1 — the trace refuses. A host value rebound inside a staged loop would be
baked as the value the representative iteration produced, while every untraced
iteration expected it to advance. That is a *loud* error, and the repair is to
spell the local as staged state:

    factor = 0.0                      # host: refused inside a staged loop
    factor = pl.const(0.0, pl.FP32)   # staged local: the loop carries it

Part 2 — the trace stays silent. A plain helper function may compute staged
*values* anywhere an expression is expected (03), but statements only land where
the trace expects them. A helper that calls ``pl.store`` is therefore not a
body: its statements are dropped, the kernel still compiles, and the output is
quietly untouched. Nothing raises — which is exactly why the rule exists:

    def emit(...): pl.store(...)          # dropped, silently
    @pl.jit_trace.body
    def emit(...): pl.store(...)          # a declared boundary: call it

Part 1 is the good outcome: you cannot ship it. Part 2 is the one to watch for
when a kernel compiles, runs, and writes nothing.

Run:  python examples/tracing/advanced/07_diagnostics.py
Next: examples/models/qwen3_jit/qwen3_decode.py — the same frontend, whole model
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
    """Load calls in a printed program, whatever stage printed it.

    A pre-pass dump spells them ``tile.load``; a post-pass dump spells the same
    call ``buffer.load``. Counting the op, not the spelling, keeps the check
    meaningful either way.
    """
    return sum(1 for line in program.splitlines() if ".load" in line)


# ---- Part 1: a host local carried by a staged loop -------------------------


@pl.jit_trace
def carried_staged(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    n: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The repair: ``pl.const`` starts a staged local, so the loop can carry it."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        factor = pl.const(0.0, pl.FP32)
        for _i in pl.range(n):
            factor = factor + 1.0
        pl.store(pl.mul(tile, factor), [0, 0], out)
    return out


@pl.jit_trace
def carried_host(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    n: pl.Scalar[pl.INDEX],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The mistake: a bare Python float, refused before it can bake one iteration."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        factor = 0.0  # a host value: emits nothing, so the loop cannot carry it
        for _i in pl.range(n):
            factor = factor + 1.0
        pl.store(pl.mul(tile, factor), [0, 0], out)
    return out


# ---- Part 2: statements that never reach the caller ------------------------


def emit_block(a, out, row0, k) -> None:
    """Not a body: a plain function that emits statements, which are dropped."""
    pl.store(pl.mul(pl.load(a, [row0, 0], [TILE_ROWS, COLS]), k), [row0, 0], out)


@pl.jit_trace.body
def declared_block(a, out, row0, k) -> None:
    """The declared spelling: the same statements, with a boundary."""
    pl.store(pl.mul(pl.load(a, [row0, 0], [TILE_ROWS, COLS]), k), [row0, 0], out)


@pl.jit_trace
def silent(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Calls a plain helper: the statements are dropped, and nothing raises."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            emit_block(a, out, row0, k)
    return out


@pl.jit_trace
def declared(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The same kernel with the declared body, called like a function."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            declared_block(a, out, row0, k)
    return out


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. The staged local carries across iterations, as intended.
    carried = torch.zeros_like(a)
    carried_staged(a, 3, carried, config=SIM)
    assert torch.allclose(carried, a * 3.0), "the loop carried the staged factor"
    print("staged local : factor counted to 3.0 on device, out == a * 3.0")

    # 2. The host spelling of the same local is refused at trace time.
    try:
        carried_host.compile_for_test(a, 3, empty)
    except TypeError as exc:
        print("refused      :", str(exc).split(".")[0] + ".")
    else:  # pragma: no cover - the guard is the point of this half
        raise AssertionError("a loop-carried host value must be refused")

    # 3. The silent half: a plain helper's statements never reach the caller.
    #    The structural view comes first: dispatching specializes the kernel,
    #    and this check is about the trace, not the dispatch.
    silent_ir = _program_text(silent, a, 2.0, empty)
    quiet = torch.zeros_like(a)
    silent(a, 2.0, quiet, config=SIM)
    assert _loads(silent_ir) == 0, "a plain helper contributes no statements"
    assert not torch.allclose(quiet, a * 2.0), "and the output is left untouched"
    print("silent       : 0 loads in the program, output untouched — nothing raised")

    # 4. Declaring the boundary is the whole repair.
    declared_ir = _program_text(declared, a, 2.0, empty)
    loud = torch.zeros_like(a)
    declared(a, 2.0, loud, config=SIM)
    assert torch.allclose(loud, a * 2.0)
    assert _loads(declared_ir) == 1
    print("declared     : the same statements, called — out == a * 2.0")

    print("OK")


if __name__ == "__main__":
    main()
