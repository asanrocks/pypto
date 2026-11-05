# Metaprogramming a Kernel

One kernel, five steps: from a value two ways, through host loops and objects, to a tiling
policy three kernels share. Every step runs, and every step is the point where the next
abstraction becomes the obvious thing to reach for.

> **Prerequisites:** [Values and Staging](../language/07-staging.md) and
> [Functions and Programs § reusing code](../language/01-functions.md).
> **Companion files:** `examples/tracing/01_two_domains.py` … `05_block_recipe.py` — each
> page step is that example, trimmed to its idea.

## What you are building

A row-scaling kernel that gets *easier to maintain* without getting more clever: the
arithmetic never changes, and each step buys a property a model-sized kernel needs — a knob
the device can vary, a plan the compiler can expand, a shape the source can state once.

The reason all of this is possible in one language: a `@pl.jit_trace` kernel is executed
while its IR is collected, so ordinary Python — loops over host data, objects, functions —
runs at build time and leaves no trace in the program. What decides whether a construct is
executed or recorded is the *domain of its subject*, never its syntax.

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
ROWS = COLS = 64
a = torch.randn(ROWS, COLS, dtype=torch.float32)


def program_text(kernel, *args) -> str:
    """The lowered program a call would ship, without compiling for a device."""
    return ir.python_print(kernel.compile_for_test(*args))


def loads(kernel, *args) -> int:
    """Load ops in the program: the evidence that a statement landed."""
    return sum(1 for line in program_text(kernel, *args).splitlines() if ".load" in line)
```

Every later block prints nothing: it asserts the property the step claims, so "the page is
right" and "the code runs" are the same check. Each step also names its companion example,
which pushes the same claims further.

## Step 1: one value, two worlds

Start with the smallest possible distinction — the scale factor:

<!-- doctest: run -->
```python
@pl.jit_trace
def baked(a: pl.Tensor[[64, 64], pl.FP32], k: float, out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(a, [0, 0], [64, 64]), k), [0, 0], out)
    return out


out = torch.zeros_like(a)
baked(a, 2.0, out, config=CFG)
assert torch.allclose(out, a * 2.0)
```

`k: float` is a **host** value: read while the tracing runs, folded into the program. Change
it and you have a different program — the compiled artifact is keyed on the value. Declare
`s: pl.Scalar[pl.FP32]` instead and the same source produces one program that serves every
value, because `s` is a **staged** runtime parameter.

That is the whole idea; the rest of the page spends it on control flow, objects and reuse.
The mental test: *must the device be able to choose this?* → stage it. *Must the compiler
see this?* → keep it host.

Companion: `examples/tracing/01_two_domains.py`.

## Step 2: the same rule for control flow

The distinction scales to statements, and the tell is the subject. Two kernels, one
difference — `range` versus `pl.range`:

<!-- doctest: run -->
```python
@pl.jit_trace
def unrolled(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(4):                      # host subject: the body runs four times
            row0 = chunk * 16
            pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def recorded(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):                   # staged subject: one loop, traced once
            row0 = chunk * 16
            pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), 2.0), [row0, 0], out)
    return out


out = torch.zeros_like(a)
recorded(a, out, config=CFG)
assert torch.allclose(out, a * 2.0)

empty = torch.zeros_like(a)
assert loads(unrolled, a, empty) == 4          # one copy of the body per source iteration
assert loads(recorded, a, empty) == 1          # one body, repeated by the loop
assert "for " not in program_text(unrolled, a, empty)
assert "for " in program_text(recorded, a, empty)
```

Both compute the same answer; they ship different programs. The host loop holds four copies
of the body, so a *tiling decision changes the program's size* — halve the tile and the
pasted kernel doubles in size. The staged loop holds one body and reads its bound from the
loop, which is why the same decision costs nothing in the recorded version.

`if` follows the same rule: a host predicate runs one arm at trace time and leaves no branch
behind; a staged predicate records a branch, with both arms traced once and the join
threading each arm's value through. The guards that make a wrong guess loud — host state
the loop tries to advance, a host name bound in one arm only — are in
[Values and Staging](../language/07-staging.md#carried-state).

Companion: `examples/tracing/02_control_flow.py`.

## Step 3: let Python organize the staged work

Since the body runs, the ordinary Python answer to "too many constants and arguments" works.
A frozen dataclass carries the tile plan, and one call site expands it into the kernel's flat
parameter list:

<!-- doctest: run -->
```python
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Plan:
    """Host knobs for one tiling decision."""

    tile_rows: int = 16
    k: float = 2.0


def expand(plan: Plan) -> tuple[object, ...]:
    """The expansion idiom: the fields, in declaration order, are the parameters."""
    return tuple(getattr(plan, field.name) for field in fields(plan))


@pl.jit_trace
def scaled(
    a: pl.Tensor[[64, 64], pl.FP32],
    tile_rows: int,                                # host: the tiling is part of the program
    k: float,                                      # host: so is the factor
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(64 // tile_rows):
            row0 = chunk * tile_rows
            pl.store(pl.mul(pl.load(a, [row0, 0], [tile_rows, 64]), k), [row0, 0], out)
    return out


out = torch.zeros_like(a)
scaled(a, *expand(Plan(tile_rows=16, k=2.0)), out=out, config=CFG)
assert torch.allclose(out, a * 2.0)

empty = torch.zeros_like(a)
narrow = program_text(scaled, a, *expand(Plan(tile_rows=16, k=2.0)), empty)
wide = program_text(scaled, a, *expand(Plan(tile_rows=32, k=3.0)), empty)
assert narrow != wide                  # the host fields bake: two plans, two programs
assert "Plan" not in narrow            # ... and the bundle is never a parameter
```

Three shapes cover the ground, and this is the third:

```python
SHAPE.chunks                        # configuration: a dataclass of host data
Block(pl.load(a, ...), row0)        # a record carrying a staged tile
entry(a, *expand(plan), out=out)    # a bundle expanded into the parameter list
```

None of them reaches the program: the object is Python, and the staged value it *carries* is
IR. A record's method is a way to build operands — `Block(...).scaled(k)` builds a `pl.mul`
and leaves no trace of itself. An object passed *as* an argument is refused, which is why a
bundle is expanded first. When a field should survive into the signature instead of baking,
declare it `pl.Scalar[...]` and expand it exactly the same way.

Companion: `examples/tracing/03_host_objects.py` (all three shapes, and the program equality
between the object spelling and the spelled-out one).

## Step 4: statements need a boundary

Value helpers are free — a function returning `pl.mul(...)` is an expression and disappears.
Statements are not: a plain function that calls `pl.store` is not a body, so its statements
land nowhere, the kernel compiles, runs, and writes nothing. Nothing raises.

<!-- doctest: run -->
```python
@pl.jit_trace.body
def scale_block(a, out, row0, k):     # a declared fragment: expanded at the call site
    pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), k), [row0, 0], out)


@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):
            scale_block(a, out, chunk * 16, k)   # expanded here: no callee, no call
    return out


out = torch.zeros_like(a)
entry(a, 2.0, out, config=CFG)
assert torch.allclose(out, a * 2.0)
```

Declaring the boundary is the whole repair. The body is prepared once, expanded where it is
called, and leaves no function or call behind — it owns a statement boundary, not a
signature. If a fragment needs its own placement (it does tile work that must be erased into
the caller), that is the `.inline` callee instead; if it needs to survive as a device
function, `.incore`.

Companion: `examples/tracing/04_body_placement.py` (the body and the pasted statements,
byte-identical programs) and `examples/tracing/advanced/07_diagnostics.py` (the silent
trap).

## Step 5: state the policy once

Two kernels now repeat the same arbitration story — divide the rows into blocks, dispatch one
block per core-group, decode this block's rows out of the block index. That repetition is not
style: it is *policy*, and it belongs to whoever owns the tiling. Here it is stated once, as
a construct, and consumed by two kernels plus a third written afterwards:

<!-- doctest: run -->
```python
class RowBlock(pl.authoring.ConstructHandle):
    """The `as` target: this block's row offset and its tile capacity."""

    def __init__(self, tile_rows: int) -> None:
        self._tile_rows = tile_rows

    @property
    def row_offset(self):
        """Staged offset of this block's first row; each read re-reads the block index."""
        return pl.get_block_idx() * self._tile_rows

    @property
    def tile_rows(self) -> int:
        return self._tile_rows


@pl.authoring.construct
def row_blocks(ctx, rows: int, tile_rows: int, *, name_hint: str = ""):
    """One block of `tile_rows` rows per core-group: host checks, then a body boundary."""
    if rows % tile_rows:
        raise ValueError(f"row_blocks: {rows} rows do not divide into {tile_rows}-row blocks")
    with ctx.region(
        pl.spmd(rows // tile_rows, name_hint=name_hint),
        single_call_body=ctx.single_call_body,
    ):
        yield RowBlock(tile_rows)


@pl.jit_trace
def scale_blocks(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 16, name_hint="scale") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def relu_blocks(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 16, name_hint="relu") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.relu(tile), [row0, 0], out)
    return out


out = torch.zeros_like(a)
scale_blocks(a, out, config=CFG)
assert torch.allclose(out, a * 2.0)

out = torch.zeros_like(a)                      # a third consumer, written after the policy
relu_blocks(a, out, config=CFG)
assert torch.allclose(out, torch.relu(a))


@pl.jit_trace
def bad_grid(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 20) as grid:         # 20 does not divide 64
        row0 = grid.row_offset
        pl.store(pl.load(a, [row0, 0], [grid.tile_rows, COLS]), [row0, 0], out)
    return out


try:
    bad_grid.compile_for_test(a, torch.zeros_like(a))
except ValueError as exc:
    assert "do not divide" in str(exc)         # the policy's own check, at the call site
else:  # pragma: no cover - the refusal is the point
    raise AssertionError("a grid that does not divide the rows must be refused")
```

The consumer writes the grid once and never mentions the block index. The recipe behind it is
a generator over `pl.authoring`: code before `yield` opens the region, the `yield` hands the
body a plain Python handle, code after it finalizes. It adds no lowering of its own — the
program it builds is the one the hand-written grid builds, byte for byte, which is the only
claim that makes "the compiler has no magic" checkable.

The payoff is where an improvement goes: into the recipe, once, for every kernel. A
validation the policy owns protects kernels written after it — that is what `bad_grid` shows,
and it is why a policy improves every consumer at once instead of every consumer being fixed
separately.

Companion: `examples/tracing/05_block_recipe.py` (the hand-written grids, the consumers, the
recipe, and the IR equality), whose production counterpart is
`pypto-lib/models/deepseek_v4_flash_mtp/blocked_spmd.py` — the same shape over shape/tile axes
with dispatch dependencies, carrying the DeepSeek block grids.

## Where this ends, and what to read next

| You want | Read |
| -------- | ---- |
| the rules behind each step, with the guards | [Values and Staging](../language/07-staging.md) |
| the four reuse forms and their traps | [Functions and Programs](../language/01-functions.md) |
| to author a construct for other people | [Scopes § writing your own construct](../language/04-scopes.md), then [Construct Recipes](../../dev/language/06-construct-recipes.md) |
| to move an existing kernel onto this frontend | [Functions § moving a kernel](../language/01-functions.md) and `examples/tracing/advanced/08_migration.py` |
| the whole picture traced through one small kernel | `examples/tracing/advanced/09_compiler_view.py` |

`examples/tracing/` is the executable form of this page; the advanced track continues into
framework-author territory (`examples/tracing/advanced/06_recipe_policy.py` rebuilds the
kernel-facing decorator from the public authoring surface).
