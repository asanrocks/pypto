# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Host objects organize staged computation.

Examples 01 and 02 covered single values. Real kernels juggle a tile plan and
per-tile coordinates, and the obvious spelling — repeat the constants and pass
five arguments around — gets unreadable fast. A traced kernel is ordinary
Python, so the ordinary Python answer works: put the numbers in an object.

Three objects, in the order a kernel needs them:

    SHAPE.chunks            # configuration: host data, read while the body runs
    Block(tile, row0)       # a staged tile that remembers where it came from
    Plan(tile_rows, k)      # a bundle expanded into the kernel's parameters

None of them reach the program. ``Block`` *carries* a staged value (its
``value`` field is a tile, so it is IR); the object itself is Python and is
gone by the time the program is built. That is the rule worth taking away:

    a value helper may appear wherever an expression is expected

Its result is an operand, exactly as if you had written the expression inline.
So the object spelling and the spelled-out spelling build the same program —
which this example asserts, because that is the claim.

The last step is the one that pays off in a model: a bundle **expands into the
kernel's flat parameter list** at the call site, so one kernel source serves a
family of programs chosen by host data. The bundle is not a parameter, and the
IR never learns its name.

(Statements are the other side of that boundary: a plain helper that emits
``pl.store`` is *not* a body, and its statements are dropped. That is a
diagnostics story — see ``advanced/07_diagnostics.py``.)

Run:  python examples/tracing/03_host_objects.py
Next: examples/tracing/04_body_placement.py
"""

from dataclasses import dataclass, fields, replace

import pypto.language as pl
import torch
from pypto import ir
from pypto.backend import BackendType
from pypto.runtime import ExecutionMode, RunConfig

ROWS = 64
COLS = 64
TILE_ROWS = 16

SIM = RunConfig(backend_type=BackendType.Ascend910, execution_mode=ExecutionMode.SIM)


@dataclass(frozen=True)
class Shape:
    """Step 1 — configuration. Host data: read while the body runs."""

    rows: int = ROWS
    cols: int = COLS
    tile_rows: int = TILE_ROWS

    @property
    def chunks(self) -> int:
        return self.rows // self.tile_rows


@dataclass(frozen=True)
class Block:
    """Step 2 — a staged tile that remembers where it came from.

    The object is plain Python; the ``value`` it carries is a staged tile, so
    rebinding it below produces IR.
    """

    value: object
    row0: int

    def scaled(self, k) -> "Block":
        """A new record: host objects are only safe to *replace*, not mutate."""
        return replace(self, value=pl.mul(self.value, k))


SHAPE = Shape()


@dataclass(frozen=True)
class Plan:
    """Step 3 — a bundle the call site expands into the kernel's parameters."""

    tile_rows: int = TILE_ROWS
    k: float = 2.0


def expand(plan: Plan) -> tuple[object, ...]:
    """The expansion idiom: the fields, in declaration order, are the parameters.

    Order matters because this is a positional expansion — ``expanded(a,
    *expand(plan), out=out)`` reads as ``expanded(a, tile_rows, k, out=out)``.
    ``dataclasses.astuple`` would truncate a nested tuple field to its first
    element; ``getattr`` per field does not.
    """
    return tuple(getattr(plan, field.name) for field in fields(plan))


def _program_text(kernel, *args, **kwargs) -> str:
    return ir.python_print(kernel.compile_for_test(*args, **kwargs))


def _entry_params(program) -> list[str]:
    """The orchestration entry's parameter names."""
    for _global_var, function in program.functions.items():
        if function.func_type == pl.FunctionType.Orchestration:
            return [parameter.name_hint for parameter in function.params]
    raise AssertionError("the program has no orchestration entry")


@pl.jit_trace
def expanded(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    tile_rows: int,
    k: float,
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """One source, a family of programs: ``tile_rows`` and ``k`` are host values.

    Both are baked (``01_two_domains.py``), which is the point here: the tile
    shape is static in the program, so a different ``tile_rows`` is a different
    program from the same source. The bundle is how a caller carries the pair
    without spelling it out at every call site.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(ROWS // tile_rows):
            row0 = chunk * tile_rows
            tile = pl.load(a, [row0, 0], [tile_rows, COLS])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


def run_expanded(plan: Plan) -> str:
    """Follow one plan through a dispatch, and return the program it built."""
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    out = torch.zeros_like(a)
    expanded(a, *expand(plan), out=out, config=SIM)
    assert torch.allclose(out, a * plan.k), plan
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)
    return _program_text(expanded, a, *expand(plan), empty)


@pl.jit_trace
def via_objects(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """Configuration, records and value helpers — no new API at all."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(SHAPE.chunks):
            row0 = chunk * SHAPE.tile_rows
            block = Block(pl.load(a, [row0, 0], [SHAPE.tile_rows, SHAPE.cols]), row0)
            pl.store(block.scaled(k).value, [block.row0, 0], out)
    return out


@pl.jit_trace
def spelled_out(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The same work with no objects, for the structural comparison."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(SHAPE.chunks):
            row0 = chunk * SHAPE.tile_rows
            tile = pl.load(a, [row0, 0], [SHAPE.tile_rows, SHAPE.cols])
            pl.store(pl.mul(tile, k), [row0, 0], out)
    return out


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # 1. The object spelling computes the right answer ...
    out, reference = torch.zeros_like(a), torch.zeros_like(a)
    via_objects(a, 2.0, out, config=SIM)
    spelled_out(a, 2.0, reference, config=SIM)
    assert torch.allclose(out, a * 2.0)
    assert torch.allclose(out, reference)
    print("numerics : objects == spelled out == a * 2.0")

    # 2. ... and builds the same program: a value helper is an expression, so
    #    the record, its method and its field access leave nothing behind.
    objects_ir = _program_text(via_objects, a, 2.0, empty).replace("via_objects", "P")
    spelled_ir = _program_text(spelled_out, a, 2.0, empty).replace("spelled_out", "P")
    assert objects_ir == spelled_ir, "host objects organize the trace; they do not change it"
    print("program  : objects == spelled out, byte for byte")
    print("           — the dataclasses and methods are gone before the backend runs")

    # 3. A bundle expands into the kernel's parameter list at the call site:
    #    one source, two plans, two programs. Its host fields then bake, so the
    #    program keeps only the staged parameters — the bundle, its fields and
    #    the tile shape have all left the signature.
    narrow_ir = run_expanded(Plan(tile_rows=16))
    wide_ir = run_expanded(Plan(tile_rows=32, k=3.0))
    assert "Plan" not in narrow_ir, "a bundle is host data: it never enters the IR"
    assert narrow_ir != wide_ir, "the expanded host values are baked into the program"
    params = _entry_params(expanded.compile_for_test(a, *expand(Plan()), empty))
    assert len(params) == 2, params
    assert not any("tile_rows" in name or name.endswith("__k") for name in params), params
    print("expansion: Plan(16, 2.0) and Plan(32, 3.0) pick two programs from one source")
    print("           entry parameters:", ", ".join(params), "— the bundle and its host fields are gone")

    print("OK")


if __name__ == "__main__":
    main()
