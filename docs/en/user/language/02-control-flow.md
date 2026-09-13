# Control Flow

Loops and conditionals, the values they carry across iterations, and the SSA rule that
shapes both.

> **Prerequisites:** [Functions and Programs](01-functions.md).

## Concept

PyPTO's IR is **SSA**: every binding is written exactly once. That is invisible in
straight-line code — the tracer renames `x` for you as often as you rebind it — but at a
loop boundary or a branch merge it becomes visible, because there is no single place for
the compiler to put the second write.

Under the tracing frontend you never declare that dataflow. The kernel body is executed
once, and the **passes** derive the rest:

- In a loop, a name the body rebinds (`acc = pl.add(acc, x)`) *is* the carried value.
  Reading it after the loop reads the last iteration's result.
- In a staged `if` / `else`, both arms simply assign the same name; the SSA pass
  synthesizes the merge (a phi node).

The parser-era declarations are gone: `init_values=` (and the iter-tuple unpacking that
came with it), per-branch `pl.yield_`, and the `pl.while_` / `pl.cond()` spelling no
longer exist. You will still *read* them — printed IR shows `init_values=` and
`pl.yield_` — but that is the printer's round-trip dialect, its own grammar consumed by
`pl.parse`, not the tracing input. Do not copy it back into a kernel.

One rule underlies everything on this page: **staged state starts at an operator or
`pl.const`.** A host literal (`acc = 0.0`) is not staged — feeding it to an operator is a
loud error, not a silent zero.

Four loop constructs share one syntax and differ only in what the compiler is told:

```text
pl.range     sequential — the default
pl.parallel  iterations are independent and may overlap
pl.unroll    fully unrolled at compile time; bounds must be literals
pl.pipeline  body replicated `stage` times for ping-pong buffering
```

## Quickstart: a loop that accumulates

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(128, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def accumulate(
    a: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(pl.load(a, [0, 0], [128, 128]), pl.load(a, [0, 0], [128, 128]))
        for i in pl.range(3):
            t = pl.add(t, pl.load(a, [0, 0], [128, 128]))   # carried across iterations
        pl.store(pl.mul(t, t), [0, 0], out)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
accumulate(A, out, config=CFG)
torch.testing.assert_close(out, (5 * A) ** 2, rtol=1e-4, atol=1e-4)
```

| Line | What it does |
| ---- | ------------ |
| `t = pl.add(pl.load(...), pl.load(...))` | Establishes the value before the loop — an operator result, so it is staged |
| `for i in pl.range(3)` | Three iterations; `i` is the loop variable |
| `t = pl.add(t, ...)` | Looks like mutation; is a carried value |
| `pl.store(pl.mul(t, t), ...)` | Reading `t` after the loop reads the last iteration's result |

Rebinding `t` inside the loop is the idiomatic way to carry a value — there is no other
way to declare one. The IR is SSA, and the loop pass threads each iteration's `t` through
the loop as a carried value; you never name that plumbing.

## Mechanics

### The four loop forms

All four take `(stop)`, `(start, stop)`, or `(start, stop, step)`. Arguments may be `int`
literals or `pl.Scalar` values, except where noted.

| Form | Emits | Carried values | Bounds |
| ---- | ----- | -------------- | ------ |
| `pl.range(...)` | `ForKind.Sequential` | Derived from rebinding | int or `Scalar` |
| `pl.parallel(...)` | `ForKind.Parallel` | Derived from rebinding | int or `Scalar` |
| `pl.unroll(...)` | `ForKind.Unroll` | Derived only — the explicit carry channel is rejected | literals only |
| `pl.pipeline(..., stage=N)` | Sequential + pipelining | Derived from rebinding | int or `Scalar` |

```python
for i in pl.range(10): ...            # 0..9
for i in pl.range(2, 10): ...         # 2..9
for i in pl.range(0, 100, 4): ...     # 0, 4, ..., 96
for i in pl.parallel(0, nblocks): ...
for i in pl.unroll(4): ...            # no explicit carry channel here
```

`pl.parallel` is an assertion, not a request: you are telling the compiler the iterations
are independent. If they are not, the result is a race.

`pl.pipeline(N, stage=F)` replicates the body `F` times per outer iteration so buffers can
ping-pong. The outer loop advances in strides of `stage * step` and a tail dispatch covers
the remainder when the trip count does not divide evenly. `stage` is required and must be
a positive integer (typically 2–4). It is lowered at tile level by
[LowerPipelineLoops](../../dev/passes/31-lower_pipeline_loops.md) — or, under
`memory_planner=PTOAS`, by [LowerPipelineToSlots](../../dev/passes/30-lower_pipeline_to_slots.md),
which rotates one un-replicated body through the slots of a single allocation instead.

```python
for i in pl.pipeline(64, stage=4):
    t = pl.load(a, [i * 64, 0], [64, 64])
    pl.store(t, [i * 64, 0], out)
```

### How a carry is derived

The loop's carried values are exactly the names the body rebinds self-referentially:

```python
acc = pl.load(a, [0, 0], [64, 64])     # staged seed — an operator result
for i in pl.range(4):
    acc = pl.add(acc, tile_i)          # `acc` is carried; `tile_i` is not
```

The same loop may rebind several names; each self-referential name becomes a carried
value. Single-branch definitions inside a loop body are legal — the missing branch reads
the previous iteration's value; that is loop-carried behavior and the pass owns it.

The seed must be staged. A host literal seed (`acc = 0.0`) does not silently become a
carry: the first operator that consumes it rejects it (`pl.add: expected Tensor or Tile
operands, got (float, Tensor)`). Seed with an operator result, or with a typed constant —
`pl.const(0.0, pl.FP32)` — which is also the spelling for a counter or a boolean flag.

### While loops

A staged while-loop is a bare `while` over a **staged** predicate. `pl.while_` and
`pl.cond()` do not exist. Initialize the counter with `pl.const`, and the carry is
derived from rebinding exactly as above — the body traces one representative iteration:

<!-- doctest: run -->
```python
@pl.jit
def chunked_square(
    a: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pos = pl.const(0, pl.INDEX)    # typed staged constant — a host `pos = 0` stays host
        while pos < 2:                 # staged predicate -> one representative body iteration
            tile = pl.load(a, [pos * 64, 0], [64, 128])
            pl.store(pl.mul(tile, tile), [pos * 64, 0], out)
            pos = pos + 1              # the rebind is the carry
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
chunked_square(A, out, config=CFG)
torch.testing.assert_close(out, A * A, rtol=1e-3, atol=1e-3)
```

Staged or host is decided by the *values*, not the keyword: if the predicate (and the
names it reads) are host values — plain Python ints, tensor shapes, anything an operator
did not produce — the `while` executes at trace time and the body is emitted once per
host iteration. That is real metaprogramming (a shape-driven unroll), not a fallback.

### Conditionals

A staged branch publishes its value by **assignment**. Both arms assign the same name;
convergence is the pass's job:

```python
if i == last:
    valid_len = last_valid_len             # staged value: plain assignment
else:
    valid_len = pl.const(block_size, pl.INDEX)   # host value crossing the join: pl.const
```

A staged `if` traces **both** arms, whatever the predicate's trace-time value. A host
`if` (host predicate) takes one path at trace time, like the host `while` above.

Two errors guard the join: a name that is staged on one arm and a host literal on the
other is rejected (`... is staged on one branch of this if and a host value on the
other`), and the predicate itself must be BOOL-typed — `if k:` on an integer `Scalar` is
rejected; write `if k != 0:`.

### The assembly channels: `pl.intrin.carry` / `yield_` / `leak`

Ergonomics never states a carry. The one place an explicit channel appears is the
assembly dialect — compiler tests, and kernel code that must state dataflow the
derivation would obscure (the task-graph dependency loop is the user-facing case; see
[Directives](06-syntax.md)). The three spellings:

```python
(acc,) = pl.intrin.carry(c)        # open the iter args the loop consumes
for i in pl.range(4):
    acc = pl.store(tile_i, [i * 64, 0], acc)
    pl.intrin.yield_(acc)          # publish the branch/iteration value
(result,) = pl.intrin.leak()       # bind the loop's result after it closes
```

Rules: the carry must be consumed by the loop opener immediately following it;
`pl.unroll` rejects it outright; the leak-bound name (`result`, not `acc`) is the one to
read after the loop; and every carried value must keep one type across the whole loop.
Mixing levels is where this bites — a value seeded from a tensor-level expression and
then combined with a tile-level one inside an InCore scope does not type-check, and the
error names the operator rather than the carry.

`pl.yield_` still exists as a deprecated alias of `pl.intrin.yield_`; it warns. Printed
IR renders the same loop with `init_values=` — again, that is the printer's dialect, not
input spelling.

### SSA, and when you have to think about it

Write ordinary Python — rebind names freely. The tracer renames:

```python
result = pl.mul(x, 2.0)
result = pl.add(result, 1.0)      # fine; the tracer produces two bindings
```

`@pl.function(strict_ssa=True)` rejects a rebind instead, which is occasionally useful
for catching unintended shadowing. It is a round-trip-dialect discipline — the error's
help text points at `@pl.function_parser` — and it also disables the `dst[...] = src`
subscript-write sugar, which works by rebinding — see
[Directives § subscript sugar](06-syntax.md#subscript-sugar).

The pipeline runs [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) early, so
non-SSA source is normal input, not a compatibility mode.

## Edge Cases

> **Fatal pitfall (parser era, now loud):** a host-literal seed used to drop the carry —
> the loop compiled and every iteration read the initial value. Tracing rejects the seed
> at the first operator that consumes it. If the error below names a Python type where
> you expected a tensor, your state was never staged.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`pl.add: expected Tensor or Tile operands, got (float, Tensor)`** | A host literal seeding staged state (`acc = 0.0`) | Seed with an operator result or `pl.const(value, dtype)` |
| **`range() got an unexpected keyword argument 'init_values'`** | Parser-era carry declaration | Drop it — the rebind is the carry |
| **`... is staged on one branch of this if and a host value on the other`** | A host literal crossing a staged join | `pl.const(...)` on that arm, or stage both arms |
| **`a staged if predicate must be BOOL-typed`** | `if k:` on an integer `Scalar` | Write the explicit test (`if k != 0:`) |
| **`a staged for loop cannot have an 'else' suite`** | `else` would be collected unconditionally | Restructure, or keep the predicate a host value |
| **`pl.unroll() cannot be combined with init_values`** | An explicit `pl.intrin.carry` feeding an unrolled loop | Derive the carry from rebinding, or use `pl.range` |
| **`tensor.add requires ... but got TileType`** | A carried value changed level mid-loop — inside an InCore scope `pl.slice` yields a Tile | Keep one level across the carry; use `pl.load` / `pl.store` explicitly inside the scope |
| **`unroll()` rejects a bound** | Bounds must be compile-time literals | Use `pl.range` for a `Scalar` bound |
| **`pipeline()` stage error** | `stage` missing or not a positive int | Pass `stage=N`, N ≥ 1 |
| **Race under `pl.parallel`** | Iterations are not actually independent | Use `pl.range`, or remove the cross-iteration dependency |
| **Loop body appears unrolled in the IR** | The predicate or bounds were host values, so the loop executed at trace time | That is legal metaprogramming; stage the values (`pl.const`) if you wanted device-side control flow |
| **`SSAViolationError`** | Rebinding under `strict_ssa=True` | Use distinct names, or drop `strict_ssa` |

## Worked examples

| Example | Shows |
| ------- | ----- |
| `examples/beginner/02_elementwise.py` (`chunked_add`) | A plain `pl.range` over tile chunks |
| `examples/intermediate/04_matmul_acc.py` | Accumulation without a loop — host-sequenced K-chunks through `pl.matmul_acc` |
| `examples/models/03_flash_attention.py` | Derived multi-name carries plus nested staged `if` — the full form |

## See Also

- [Types](00-types.md) — what the carried values are.
- [Scopes and Placement](04-scopes.md) — the placement scopes these loops sit inside.
- [Migration guide](../../../design/04-migration-guide.md) — every parser-era spelling and its tracing replacement.
- [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) — the conversion this page's rules come from.
- [UnrollLoops](../../dev/passes/02-unroll_loops.md) — what `pl.unroll` becomes.
- [LowerPipelineToSlots](../../dev/passes/30-lower_pipeline_to_slots.md) — what `pl.pipeline` becomes under `memory_planner=PTOAS`.
- [LowerPipelineLoops](../../dev/passes/31-lower_pipeline_loops.md) — what `pl.pipeline` becomes otherwise.
