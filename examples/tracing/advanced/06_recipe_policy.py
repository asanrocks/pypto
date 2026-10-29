# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
What a placement method is made of — and what the kernel-facing spelling costs.

This is the framework-author half of the ladder. ``04_body_placement.py`` shows
the spelling a kernel author writes:

    @pl.jit_trace.body
    def scale_block(a, out, row0, k): ...

    scale_block(a, out, row0, k)

Three things are worth knowing from this side of the fence, in order:

1. **A body has three placements**, and ``.splice`` / ``.inline`` / ``.incore``
   are the explicit spellings. A splice pastes the statements; an Inline unit
   is a checked function the inliner erases; an InCore unit keeps its own
   placement and the caller dispatches to it. The same 20-line fragment serves
   all three.
2. **A placement is not primitive.** Each is four tracing effects over an
   authoring context, and ``@pl.authoring.recipe`` is the tier that spells them
   out — ``ctx.push_function(contract)`` opens a function unit whose contract
   *is* the placement, ``collect_body`` derives its signature from the staged
   arguments, ``finalize`` mints the artifact, ``emit_call`` emits the call.
3. **The kernel-facing decorator is a client of that tier, not a privileged
   path.** ``@pl.jit_trace.body`` is ``@pl.authoring.recipe`` over
   ``ctx.splice_body``; this file *rebuilds it* from the public surface and
   asserts the two produce the same program. What the stdlib adds is policy:
   the decorator form, the kernel's metadata, `auto_scope=` on the declaration,
   and an error that speaks "body" rather than "splice".

If a kernel-facing spelling ever needs something this tier cannot express, the
fix belongs in the tier — never in a private shortcut past it.

Run:  python examples/tracing/advanced/06_recipe_policy.py
Next: examples/tracing/advanced/07_diagnostics.py
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


@pl.authoring.body
def scale_block(a, out, row0, k):
    """The body all three placements below compose."""
    tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
    pl.store(pl.mul(tile, k), [row0, 0], out)


# ---- Part 1: one body, three placements ------------------------------------


@pl.jit_trace
def by_splice(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """``.splice``: the statements land here. No callee is created."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            scale_block.splice(a, out, row0, k)
    return out


@pl.jit_trace
def by_inline(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """``.inline``: a typed Inline callee that the inlining pass then erases."""
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            scale_block.inline(a, out, row0, k)
    return out


@pl.jit_trace
def by_incore(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The body keeps its own placement, so the caller has no ``pl.at`` here."""
    for chunk in pl.range(CHUNKS):
        row0 = chunk * TILE_ROWS
        scale_block.incore(a, out, row0, k)
    return out


# ---- Part 2: the placements, spelled as their four effects -----------------


@pl.authoring.recipe
def place_inline(ctx, body, *args, name: str = "placed"):
    """``body.inline(*args)``, spelled as its effects."""
    unit = ctx.push_function(pl.authoring.FunctionContract(name=name, function_type=pl.FunctionType.Inline))
    artifact = unit.finalize(unit.collect_body(body, args=args))
    return ctx.emit_call(artifact, *args)


@pl.authoring.recipe
def place_on_core(ctx, body, *args, name: str = "placed"):
    """``body.incore(*args)``, spelled as its effects."""
    unit = ctx.push_function(
        pl.authoring.FunctionContract(
            name=name,
            function_type=pl.FunctionType.InCore,
            level=pl.Level.CHIP_DIE,
        )
    )
    artifact = unit.finalize(unit.collect_body(body, args=args))
    return ctx.emit_call(artifact, *args)


@pl.jit_trace
def inline_by_recipe(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(CHUNKS):
            row0 = chunk * TILE_ROWS
            place_inline(scale_block, a, out, row0, k, name="scale_block")
    return out


@pl.jit_trace
def incore_by_recipe(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    """The body keeps its own placement, so the caller has no ``pl.at`` here."""
    for chunk in pl.range(CHUNKS):
        row0 = chunk * TILE_ROWS
        place_on_core(scale_block, a, out, row0, k, name="scale_block")
    return out


# ---- Part 3: `@pl.jit_trace.body`, rebuilt from the public tier ------------


def kernel_body(kernel):
    """A ``@body`` decorator, written with the public ``pl.authoring`` surface only.

    This is the whole implementation. The stdlib's version differs in policy —
    the decorator form, metadata, `auto_scope=`, error wording — never in how it
    reaches the IR.
    """
    prepared = pl.authoring.body(kernel)

    @pl.authoring.recipe
    def expand(ctx, *args, **kwargs):
        return ctx.splice_body(prepared, *args, **kwargs)

    expand.__name__ = kernel.__name__
    expand.__doc__ = kernel.__doc__
    return expand


def _scale_tile(a, out, row0, k):
    tile = pl.load(a, [row0, 0], [TILE_ROWS, COLS])
    pl.store(pl.mul(tile, k), [row0, 0], out)


framework_body = kernel_body(_scale_tile)
stdlib_body = pl.jit_trace.body(_scale_tile)


def _kernel_using(step):
    @pl.jit_trace
    def entry(
        a: pl.Tensor[[ROWS, COLS], pl.FP32],
        k: pl.Scalar[pl.FP32],
        out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
    ):
        with pl.at(level=pl.Level.CORE_GROUP):
            for chunk in pl.range(CHUNKS):
                row0 = chunk * TILE_ROWS
                step(a, out, row0, k)
        return out

    entry.__name__ = "entry"
    return entry


framework_entry = _kernel_using(framework_body)
stdlib_entry = _kernel_using(stdlib_body)


def main() -> None:
    a = torch.randn(ROWS, COLS, dtype=torch.float32)
    empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

    # ---- Part 1: three placements, one body --------------------------------
    for placement, kernel in (("splice", by_splice), ("inline", by_inline), ("incore", by_incore)):
        out = torch.zeros_like(a)
        kernel(a, 2.0, out, config=SIM)
        assert torch.allclose(out, a * 2.0), placement
    print("numerics : splice == inline == incore == a * 2.0")

    # The placement decides what survives. A splice and an Inline unit dissolve
    # into the caller; an InCore unit stays a device function.
    for placement, kernel in (("splice", by_splice), ("inline", by_inline), ("incore", by_incore)):
        program = kernel.compile_for_test(a, 2.0, empty)
        kinds = sorted((gv.name, fn.func_type.name) for gv, fn in program.functions.items())
        survives = any("scale_block" in name for name, _kind in kinds)
        assert survives == (placement == "incore"), (placement, kinds)
        print(
            f"{placement:<9}: {'keeps' if survives else 'erases'} scale_block —",
            ", ".join(k for _n, k in kinds),
        )

    # ---- Part 2: the same placements, spelled as effects -------------------
    for placement, by_recipe, by_method in (
        ("inline", inline_by_recipe, by_inline),
        ("incore", incore_by_recipe, by_incore),
    ):
        out = torch.zeros_like(a)
        by_recipe(a, 2.0, out, config=SIM)
        assert torch.allclose(out, a * 2.0), placement
        recipe_ir = _program_text(by_recipe, a, 2.0, empty).replace(f"{placement}_by_recipe", "P")
        method_ir = _program_text(by_method, a, 2.0, empty).replace(f"by_{placement}", "P")
        assert recipe_ir == method_ir, f"the {placement} recipe must equal .{placement}()"
        print(f"{placement:<9}: four effects == the built-in method, byte for byte")

    # ---- Part 3: the kernel-facing decorator is a client of this tier ------
    for name, entry in (("framework", framework_entry), ("stdlib", stdlib_entry)):
        out = torch.zeros_like(a)
        entry(a, 2.0, out, config=SIM)
        assert torch.allclose(out, a * 2.0), name
    framework_ir = _program_text(framework_entry, a, 2.0, empty)
    stdlib_ir = _program_text(stdlib_entry, a, 2.0, empty)
    assert framework_ir == stdlib_ir, "the rebuilt decorator must reach the same IR"
    print("body     : rebuilt from pl.authoring == @pl.jit_trace.body, byte for byte")
    print("           — the stdlib adds policy (metadata, auto_scope, error wording), not reach")

    print("OK")


if __name__ == "__main__":
    main()
