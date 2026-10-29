# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Two domains: the numbers I write, and the numbers the device reads.

A traced kernel body is *executed*: collecting IR is a side effect of running
it. So every value in the body has a provenance, and the provenance decides
whether the value survives into the compiled program. Look at one scale factor:

    k = 2.0                       # host value: read now, gone from the program
    s: pl.Scalar[pl.FP32]         # staged value: a runtime parameter

The host factor is baked in as a literal — change it and you have a different
program. The staged factor stays a parameter — one program serves every value
you pass at call time. That single distinction is the whole idea; everything
else in this ladder builds on it.

Run:  python examples/tracing/01_two_domains.py
Next: examples/tracing/02_control_flow.py
"""

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

ROWS = 64
COLS = 64

# Same artifact, no device: the simulator is what makes these examples free to
# run. Swap to ``ExecutionMode.ONBOARD`` (plus a ``device_id``) for silicon.
SIM = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM)


@pl.jit_trace
def baked(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: float,
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A host factor: ``k`` is consumed while the body runs."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, k), [0, 0], out)
    return out


@pl.jit_trace
def staging(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    s: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A staged factor: ``s`` is a parameter of the program."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, s), [0, 0], out)
    return out


def _program_text(kernel, *args) -> str:
    """The program ``kernel`` builds for ``args`` — no codegen, no dispatch.

    ``compile_for_test`` is the structural view: the same build a real call
    makes, stopped before the backend. It is how this ladder shows what a
    concept predicts instead of asking you to take it on trust.
    """
    return ir.python_print(kernel.compile_for_test(*args))


def _mul_line(program: str) -> str:
    """The scalar-multiply line of a printed program."""
    return next(line.strip() for line in program.splitlines() if "buffer.muls(" in line)


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. Both kernels do the job. Nothing here tells them apart yet.
    out = torch.zeros_like(a)
    baked(a, 2.0, out, config=SIM)
    assert torch.allclose(out, a * 2.0)

    out = torch.zeros_like(a)
    staging(a, 2.0, out, config=SIM)
    assert torch.allclose(out, a * 2.0)
    print("numerics : both spellings compute out == a * 2.0")

    # 2. A host factor is baked: the literal is in the program, so a different
    #    factor is a different program.
    two = _program_text(baked, a, 2.0, empty)
    seven = _program_text(baked, a, 7.0, empty)
    assert two != seven, "a host value is baked into the program"
    print("host 2.0 :", _mul_line(two))
    print("host 7.0 :", _mul_line(seven))

    # 3. A staged factor is not: the program is textually identical, and only
    #    the value passed at call time moves. This is why a dynamic extent or a
    #    runtime scale costs nothing in code size.
    s3 = _program_text(staging, a, 3.0, empty)
    s9 = _program_text(staging, a, 9.0, empty)
    assert s3 == s9, "a staged value never changes the program"
    print("staged   :", _mul_line(s3), "— same program for 3.0 and 9.0")

    out = torch.zeros_like(a)
    staging(a, 9.0, out, config=SIM)
    assert torch.allclose(out, a * 9.0), "the same program served a new value"

    print("OK")


if __name__ == "__main__":
    main()
