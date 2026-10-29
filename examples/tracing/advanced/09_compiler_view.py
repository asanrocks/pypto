# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
The compiler view: one kernel, four readable stages.

Everything the tracing frontend claims is visible in one small kernel if you
know where to look. This file prints the four stages, in order, and asserts the
fact each one carries:

1. **Source.** What the author wrote. There is no special syntax: `with`, a
   `for`, a function call. The frontend cannot reinterpret these — this is the
   constraint the parser lane works around and the tracing lane removes.

2. **Instrumented source.** What actually runs. The rewriter keeps the kernel's
   shape and inserts the recorder: `rt.enter_frame(...)` opens the trace,
   `rt.on_begin_with(...)` turns a `with` into a construct, `rt.begin_for(...)`
   turns a staged loop into a loop, `rt.vars.X` is a name the trace tracks, and
   `rt._emit(...)` is the one call that records an IR statement.

3. **Collected IR** (`kernel.fn`). What the trace produced: the same kernel as
   IR, before any pass. Two things are already decided here — staged values are
   operands (`pl.tile.load`), and host values are gone (`ROWS // TILE` is the
   literal `4`).

4. **Lowered program** (`compile_for_test`). What ships: the pass pipeline has
   run, so the tile ops have become buffer ops with memory references, and the
   scopes have been outlined into device functions.

The step between 2 and 3 is the part no parser can do: the trace *executes* the
instrumented body, so Python's own semantics decide what the program is before
any IR exists. That is also why the author's `if`, `for` and objects are free
(`01`-`03`) and why a mistake can be silent (`07`).

Run:  python examples/tracing/advanced/09_compiler_view.py
Next: examples/tracing/advanced/08_migration.py — the parser lane, same kernel
"""

import inspect
import textwrap

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.language.tracing.collector.rewriter import emit_kernel
from pypto.runtime import ExecutionMode, RunConfig

ROWS = 64
COLS = 64
TILE_ROWS = 16
CHUNKS = ROWS // TILE_ROWS

SIM = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM)


@pl.jit_trace
def scale(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


def _source(kernel) -> str:
    """The kernel as the author wrote it, dedented."""
    lines, _start = inspect.getsourcelines(kernel)
    return textwrap.dedent("".join(lines)).rstrip()


def _instrumented(kernel) -> str:
    """The body the tracer runs — the rewriter's output, before compilation.

    ``emit_kernel`` is the collector's own entry; an example is allowed to look
    inside the frontend, a kernel is not.
    """
    lines, start = inspect.getsourcelines(kernel)
    source = textwrap.dedent("".join(lines))
    emitted, _prefix = emit_kernel(
        source,
        kernel.__name__,
        kernel.__code__.co_freevars,
        bool(kernel.__code__.co_flags & 0x0A0000),  # __future__.annotations
        start - 1,
    )
    return emitted.rstrip()


def _banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros_like(a)

    # 1. Source.
    _banner("1. source — no tracing syntax")
    print(_source(scale.kernel))

    # 2. Instrumented source. Read it against the source above: the statements
    #    are the same, each one now routed through the recorder.
    _banner("2. instrumented source — what the trace executes")
    instrumented = _instrumented(scale.kernel)
    for marker, meaning in (
        ("rt.enter_frame", "the trace starts (parameters become tracked names)"),
        ("rt.on_begin_with", "`with` -> a construct; the body runs inside it"),
        ("rt.begin_for", "`for` over a staged range -> one recorded loop"),
        ("rt.vars.", "a name the trace tracks (staged or host)"),
        ("rt._emit", "an IR-emitting statement"),
        ("rt.return_", "the function's return path"),
    ):
        assert marker in instrumented, marker
        print(f"  {marker:<18} {meaning}")
    print()
    print(instrumented)

    # 3. Collected IR: the trace's output, before the pass pipeline.
    _banner("3. collected IR — what the trace produced (kernel.fn)")
    collected = ir.python_print(scale.fn)
    print(collected)
    assert "rt." not in collected, "the recorder is a runtime detail: it never reaches the IR"
    assert "pl.range(4)" in collected, "the staged loop is a loop, and the host bound is folded"
    assert "pl.tile.load" in collected, "staged values are operands"

    # 4. Lowered program: the same function after the pipeline.
    _banner("4. lowered program — what ships (after the pass pipeline)")
    lowered = ir.python_print(scale.compile_for_test(a, 2.0, out))
    print("\n".join(lowered.splitlines()[:24]))
    assert lowered != collected, "the pipeline ran"
    assert "MemRef(" in lowered, "memory is planned and addressed"
    assert "pl.tile.alloc(pl.Mem.Vec" in lowered, "on-chip buffers are allocated"
    assert "_incore_0" in lowered, "the pl.at region was outlined into a device function"

    # 5. The claim that ties it together: all four stages are the one kernel.
    out = torch.zeros_like(a)
    scale(a, 2.0, out, config=SIM)
    assert torch.allclose(out, a * 2.0)
    print("\nOK — four views of one kernel, and the numerics it computes")

    print(
        "\nThe step from 2 to 3 is the one a parser cannot take: the instrumented body is\n"
        "*executed*, so Python decides the program before any IR exists."
    )


if __name__ == "__main__":
    main()
