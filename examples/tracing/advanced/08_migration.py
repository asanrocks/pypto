# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Migrating a kernel: same program, then a door that opens.

The claim this file tests is the one a migration plan lives or dies on:

    a kernel ported to the tracing frontend lowers to the program the parser
    built — so the port is a change of frontend, not a change of kernel.

That is checkable, not rhetorical: the parser lane's ``.lower()`` and the
tracing lane's ``compile_for_test()`` both stop after the pass pipeline, and
here the two printed programs are identical byte for byte once the entry name
is folded. Migrating is then a mechanical act — move the body, change the
decorator — which is why the sweep can be incremental rather than a rewrite.

What the equivalence does *not* say is that the frontends are the same. The
second half of the file shows the door that only opens on the tracing side: a
host loop over a plan of tiles. The parser frontend has no host domain — its
``for`` is structural, so it needs a loop descriptor — while the traced body is
executed, so an ordinary Python loop expands the plan before the IR is built.

Run:  python examples/tracing/advanced/08_migration.py
Next: examples/tracing/advanced/09_compiler_view.py
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


def _normalized(program, *names: str) -> str:
    """The printed program with entry/callee names folded to ``P``."""
    text = ir.python_print(program)
    for name in names:
        text = text.replace(f"{name}_", "P_").replace(name, "P")
    return text.replace("_jit_P", "P")


# ---- The kernel, on both frontends ----------------------------------------


@pl.jit
def scale_parser(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The parser lane: the source is parsed into IR."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


@pl.jit_trace
def scale_traced(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The tracing lane: the same body, executed instead of parsed."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


# ---- The same shape, one door further --------------------------------------


PLAN = ((0, 0), (32, 32))  # (row offset, column offset) per tile


@pl.jit_trace
def plan_traced(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """A host plan: an ordinary Python loop expands it while the body runs.

    The plan is not a parameter and never enters the program — what enters is
    one load/store pair per entry, with the offsets baked as literals.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        for row0, col0 in PLAN:
            tile = pl.load(a, [row0, col0], [32, 32])
            pl.store(pl.mul(tile, k), [row0, col0], out)
    return out


@pl.jit
def plan_parser(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The same idea on the parser lane — a ``for`` over host data is not IR."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for row0, col0 in PLAN:  # pyright: ignore[reportGeneralTypeIssues]
            tile = pl.load(a, [row0, col0], [32, 32])
            pl.store(pl.mul(tile, k), [row0, col0], out)
    return out


def _loads(program: str) -> int:
    return sum(1 for line in program.splitlines() if ".load" in line)


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)

    # 1. Both frontends execute the kernel: same numerics.
    parser_out, traced_out = torch.zeros_like(a), torch.zeros_like(a)
    scale_parser(a, 2.0, parser_out, config=SIM)
    scale_traced(a, 2.0, traced_out, config=SIM)
    assert torch.allclose(parser_out, a * 2.0)
    assert torch.allclose(traced_out, a * 2.0)
    assert torch.allclose(parser_out, traced_out)
    print("numerics : parser == traced == a * 2.0")

    # 2. Both lower to the same program — the migration claim, checked rather
    #    than asserted. `.lower()` and `compile_for_test()` are the same stop
    #    point: post-pass IR, no codegen.
    parser_ir = _normalized(scale_parser.lower(), "scale_parser")
    traced_ir = _normalized(scale_traced.compile_for_test(a, 2.0, a), "scale_traced")
    assert parser_ir == traced_ir, "a ported kernel must lower to the program the parser built"
    lines = len(parser_ir.splitlines())
    print(f"program  : identical after {lines} printed lines — the port is a frontend change")

    # 3. Where the equivalence stops: a host plan. The traced body expands it
    #    into two tile pairs; nothing about the plan survives into the program.
    #    Check the tiles the plan covers, and only those: the region it does
    #    not touch is written by nobody, so the runtime leaves it undefined.
    plan_out = torch.zeros_like(a)
    plan_traced(a, 2.0, plan_out, config=SIM)
    for row0, col0 in PLAN:
        covered = (slice(row0, row0 + 32), slice(col0, col0 + 32))
        assert torch.allclose(plan_out[covered], a[covered] * 2.0), (row0, col0)
    plan_ir = _normalized(plan_traced.compile_for_test(a, 2.0, a), "plan_traced")
    assert _loads(plan_ir) == len(PLAN), "the host plan is expanded, one load per entry"
    print(f"host plan: {len(PLAN)} entries -> {_loads(plan_ir)} loads, offsets baked")

    # 4. ... and the parser lane refuses the same source, loudly: its `for` is
    #    structural, so host data has no domain to run in. Porting such a kernel
    #    is therefore not optional drudgery — it is the only way to write it.
    try:
        plan_parser.lower()
    except Exception as exc:  # noqa: BLE001 - the refusal kind is frontend-owned
        print("parser   :", type(exc).__name__, "-", str(exc).splitlines()[0][:96])
    else:  # pragma: no cover - the refusal is the point of this step
        raise AssertionError("the parser lane must refuse a host-data loop")

    print("OK")


if __name__ == "__main__":
    main()
