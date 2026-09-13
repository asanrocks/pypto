# Statements and Control Flow

Statement forms the tracing frontend collects, and how the SSA pass derives
phi nodes from rebinding. See [Python IR Syntax Specification](00-python_syntax.md)
for the type system and expression grammar, and the [migration
guide](../../../design/04-migration-guide.md) for every parser-era spelling
and its replacement.

The printed round-trip dialect is different text. The IR printer renders a
lowered loop with iter args as `init_values=` plus assignment-form
`pl.yield_`, and an SSA-form while as `pl.while_(init_values=...)` with a
leading `pl.cond(...)`. That text is the parser's grammar ([IR
Parser](../ir/07-parser.md)) — correct wherever it appears, never tracing
input. Do not copy it back into a kernel.

## Statements

### Assignment

```python
x = expr                             # annotation optional
y: pl.Tensor[[4], pl.FP32] = tensor_op(a)
```

Under tracing a local annotation never flows into the IR: the value keeps
the type its operator gave it. A restating annotation is documentation; a
disagreeing one is silently ignored dead text — audit them yourself during
migration (guide §4.4). A function's declared `->` is the exception: a
return contract the body does not honor is a loud decoration-time error.

Augmented assignments lower to a rebind through the operand's binary
operator, so they work where that operator exists: a staged `Scalar`
supports `acc += 1`. Tile and Tensor wrappers do not overload arithmetic,
so `acc += ...` and `acc[...] += ...` on them fail at trace time with
Python's own
`TypeError: unsupported operand type(s) for +=: 'Tile' and 'Tile'` —
write the explicit form (`acc = pl.add(acc, ...)`) instead. For matrix
reductions, use `matmul_acc` with `init_cond`:

```python
acc[t0 : t0 + R, :] = pl.matmul_acc(
    acc[t0 : t0 + R, :], x_k, w_k, b_trans=True, init_cond=(k0 == 0)
)
```

The first K step overwrites only that window; later steps accumulate into it.
Other windows retain their values. The compiler can pack equal-size row windows
of a local accumulator into contiguous L0C windows; see
[accumulator row windows](../passes/14-flatten_tile_nd_to_2d.md#logical-accumulator-row-windows).

### If Statement (staged)

A staged `if` traces **both** branches; each branch publishes its values by
plain assignment, and the SSA pass synthesizes the merge (a phi node):

```python
# If with both branches — same names, plain assignments
if condition:
    y1 = value1
else:
    y1 = value2

# Multiple values at once
if condition:
    y1, y2 = value1, value2
else:
    y1, y2 = value3, value4
```

**Key points:**

- Both branches assign the same names; the convergence is the pass's job.
- A staged value is a plain assignment; a **host** value crossing the join
  needs `pl.const(value, dtype)` — a name staged on one branch and a host
  literal on the other is rejected.
- Reading a name that only some branches bound is a loud error (the
  `UnboundLocalError` discipline): bind it before the `if`, or in every
  branch.
- The predicate must be BOOL-typed — `if k:` on an integer `Scalar` is
  rejected; write the explicit test (`if k != 0:`). A **host** predicate
  takes one path at trace time (real metaprogramming, not a fallback).

### For Loop

```python
# Simple loop (1-3 positional args, like Python's range())
for i in pl.range(stop):                    # start=0, step=1
for i in pl.range(start, stop):             # step=1
for i in pl.range(start, stop, step):       # explicit

# Loop-carried values are derived from rebinding — nothing is declared.
s = pl.const(0, pl.INT64)                   # typed staged seed
for i in pl.range(n):
    s = s + i                               # the rebind *is* the carry
s_final = s                                 # reads the last iteration's result

# Parallel for loop (same 1-3 arg forms)
for i in pl.parallel(stop):
for i in pl.parallel(start, stop, step):
    body_statements
```

**Key points:** `pl.range()` no longer takes `init_values=` (nor the
iter-tuple unpacking that came with it). A loop's carried values are
exactly the names the body rebinds self-referentially; the same loop may
rebind several names, and single-branch definitions inside a body are
legal — the missing branch reads the previous iteration's value, which is
loop-carried behavior the pass owns. The seed must be staged: a host
literal seed (`s = 0`) is rejected at trace time
(`typed staged initialization: ... bind a typed constant first:
s = pl.const(..., dtype)`) because it would inline the literal on every
iteration instead of deriving a carry. Seed with an operator result or
`pl.const(value, dtype)`. `pl.parallel()` produces a `ForKind.Parallel`
loop while `pl.range()` produces `ForKind.Sequential` (default).

### While Loop

A staged while-loop is a bare `while` over a **staged** predicate;
`pl.while_` and `pl.cond()` do not exist as input spellings. The body
traces one representative iteration, and the carry is derived from
rebinding exactly as in `pl.range`:

```python
# Natural while: the condition is the while-header expression
i = pl.const(0, pl.INT64)     # typed staged seed
while i < n:                  # staged predicate
    i = i + 1                 # the rebind is the carry
```

The seed must be staged: an annotated seed (`i: pl.Scalar[pl.INT64] = 0`)
does **not** stage under tracing — local annotations never flow into the
IR — so the counter stays a host value and the loop degenerates (the
predicate folds against the literal, and the body contributes nothing; a
host-bound loop instead executes at trace time and emits the body once
per host iteration, with no WhileStmt at all).

Staged or host is decided by the *values*, not the keyword: if the
predicate and the names it reads are host values — plain Python ints,
tensor shapes, anything an operator did not produce — the `while` runs at
trace time. That is real metaprogramming (a shape-driven unroll), not a
fallback.

### Scope Context Managers

| Form | Scope Kind | Notes |
| ---- | ---------- | ----- |
| `pl.at(level=pl.Level.CORE_GROUP)` | `InCore` | Fixed-boundary outline at CORE_GROUP |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.split(MODE)])` | `InCore` | InCore + cross-core split hint |
| `pl.at(level=pl.Level.CORE_GROUP, optimizations=[pl.cross_core_slot(slot_num=N)])` | `InCore` | InCore + cross-core pipe slot count |
| `pl.at(level=pl.Level.HOST)` *(or any non-`CORE_GROUP` level)* | `Hierarchy` | Distributed hierarchy scope |
| `pl.cluster()` | `Cluster` | Co-scheduled AIC+AIV group |
| `pl.graph("name")` | `Graph` | Recordable orchestration region; the name is required and becomes the outlined function's — see [OutlineGraphScopes](../passes/08-outline_graph_scopes.md) |
| `with pl.spmd(N)` / `for i in pl.spmd(N)` | `Spmd` (for-form wraps inner `InCore`) | SPMD multi-block dispatch — see [pl.spmd](#plspmd-multi-block-dispatch) |
| `pl.spmd(N, optimizations=[pl.split(MODE)])` | `Spmd(InCore(split=MODE))` | Split hint applies to the inner InCore (both forms) |
| `pl.spmd(N, optimizations=[pl.cross_core_slot(slot_num=N)])` | `Spmd(InCore(slot_num=N))` | Slot count applies to the inner InCore (both forms); combinable with `pl.split(MODE)` |
| `pl.scope(mode=pl.ScopeMode.MANUAL)` / `pl.manual_scope()` | `Runtime(manual=true)` | Orchestrator MANUAL scope — user manages task ordering. Allowed in either `auto_scope` mode (it is a dependency-semantics choice). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives) |
| `pl.scope()` | `Runtime(manual=false)` | Orchestrator AUTO scope (`SIMPLER_SCOPE()`). Hand-placing one requires `@pl.function(auto_scope=False)` (in the default `auto_scope=True` the compiler owns AUTO placement). See [MaterializeRuntimeScopes](../passes/49-materialize_runtime_scopes.md) |

See [Scopes and Placement](../../user/language/04-scopes.md) for examples.

#### `pl.spmd` multi-block dispatch

`pl.spmd(N)` dispatches a kernel across `N` blocks. Forms:

- `with pl.spmd(N): ...` — body is **either** a *dispatch* body calling a pre-defined InCore kernel (`SpmdScopeStmt(body=<stmts>)`, no inner InCore wrapper) **or** an *inline* block auto-outlined into a synthetic InCore region (like the for-form, minus the auto-bound index). Decided semantically, not by statement count: a body reading `pl.tile.get_block_idx()` is inline and gets wrapped; otherwise it is a dispatch body and stays unwrapped, however many statements it holds. A body that neither reads the index nor dispatches a `self.<kernel>(...)` call is rejected. Captures no producer TaskId.
  - An explicit `with pl.at(<CORE_GROUP level>, ...):` as the sole body statement *is* the InCore carrier: parsed as an ordinary nested scope, not wrapped a second time (positional or keyword `level`, with or without `as tid` / `name_hint=`). This is the form the printer emits for `Spmd(InCore(...))`, so it is what makes that IR round-trip. When the body provides a carrier, `optimizations=` must go on that `pl.at(...)` — putting it on the `pl.spmd(...)` line is rejected, whether or not the carrier also carries one.
  - A dispatch body may launch **one** kernel. It is lowered via `FindFirstInnerCall`, which stops at the first call, so a second dispatch would be silently dropped rather than launched; the parser rejects it instead. Hoisted temporaries and tuple projections are not dispatches and do not count.
- `for i in pl.spmd(N): ...` — loop variable binds the per-block index (`pl.tile.get_block_idx()`); the body is auto-outlined into a synthetic InCore region.
- `with pl.spmd(N, deps=[...]) as tid: ...` — **capture form**: mirrors `with pl.at(...) as tid:`. Same body shapes as the plain form above, and additionally captures the dispatch's grid-wide producer `pl.Scalar[pl.TASK_ID]` in `tid` (usable as a `deps=` edge, stored into a `pl.array.create(N, pl.TASK_ID)`, or crossed into `pl.manual_scope`). TaskId capture is orthogonal to the inline body — it is the only thing this form adds over the plain form. Lowers to an `ir.Submit` whose trailing tuple element is the grid TaskId; `core_num` / `sync_start` ride on that `Submit`'s own fields (the launch spec belongs to the launch site, not the outlined callee). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives).
- `out, tid = pl.spmd_submit(kernel, *args, core_num=N)` — **submit form**: dispatches the kernel across `N` blocks *and* captures the dispatch's producer `pl.Scalar[pl.TASK_ID]` (the `pl.submit` sibling for a pre-defined kernel). See [Manual dependency primitives](02-manual_dependencies.md#manual-dependency-primitives).

All three `pl.spmd(...)` scope forms also accept `allow_early_resolve=True` (a boolean literal; same early-dispatch opt-in as `pl.submit` / `pl.at`). It forces the dispatch to lower to an `ir.Submit` even without `as tid` and lowers to `Arg::set_allow_early_resolve(true)`. Rejected on a `pl.cluster()`-nested `pl.spmd` (such a scope is unwrapped into the Group function and never produces a Submit, so the hint would be lost).

Optional `optimizations=[...]`. The entries are orthogonal and may be combined
in one list (e.g. `[pl.split(MODE), pl.cross_core_slot(slot_num=4)]`):

| Entry | Form | Effect |
| ----- | ---- | ------ |
| `pl.split(MODE)` | both | Sets the inner InCore's `split_` field (cross-core transfer hint, consumed by `ExpandMixedKernel` / `MemoryReuse`). The with-form gains an inner `InCoreScopeStmt` wrapper around the call. |
| `pl.cross_core_slot(slot_num=N)` | both | Sets the inner InCore's `slot_num` attr — the slot count (ring depth) of the automatic cross-core pipe, consumed by `ExpandMixedKernel`. Sizes a data channel only; it does **not** partition work, so it coexists with `pl.split_aiv` regions where `pl.split(...)` does not. Omit to keep the default depth of 2 per active direction. |

> `pl.split(MODE, slot_num=N)` is a deprecated alias for the slot count and warns
> — see [ExpandMixedKernel](../passes/24-expand_mixed_kernel.md#overriding-the-slot-count-slot_num).

### Yield Statement

A bare `yield` is a rewrite-unsafe construct under tracing — the frontend
executes the body, and a generator's effect on the IR cannot be observed —
so it is rejected at rewrite time. The surfaces that publish a value are:

- `return` — leaving a function with its result(s);
- `pl.intrin.yield_(...)` — the assembly channel that states a staged
  branch/loop value where the derivation would obscure it (migration guide
  Appendix A). `pl.yield_` remains as a deprecated alias of it; it warns;
- printed IR renders merges and carries as assignment-form `pl.yield_` —
  round-trip text for the parser, not input.

### Break and Continue

```python
break              # Exit innermost loop
continue           # Skip to next iteration
```

**Restrictions:** Only valid when the **innermost** enclosing loop is sequential (`pl.range`) or `while`. Not supported when the innermost loop is `pl.parallel()` or `pl.unroll()` — both are rejected at trace time (`'break' not supported in parallel loops` / `... in unrolled loops`). A `break` in an inner `pl.range` loop nested inside an outer `pl.parallel` loop is valid. **Note:** Codegen backend support for `break`/`continue` is tracked in [#448](https://github.com/hw-native-sys/pypto/issues/448).

### Compile-Time Debugging

`pl.static_print()` and `pl.static_assert()` are diagnostics of the AST
parser: they produce **no IR** and run only where the parser owns the
frontend (round-trip text, `@pl.function_parser` fixtures). Under the
tracing frontend they are **silent no-ops** — the body executes, the call
has no effect, and a failing `static_assert` does not raise. Move the
assertion into a host `if` that raises, or into a verifier pass (guide §8).

```python
@pl.function
def func(x: pl.Tensor[[128, 64], pl.FP16]) -> pl.Tensor[[128, 64], pl.FP16]:
    pl.static_print("input:", x)          # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_print(f"input: {x}")        # → static_print [file:line]: input: x: pl.Tensor[[128, 64], pl.FP16]
    pl.static_assert(True)                # passes silently
    pl.static_assert(N > 32, "N too small")  # checks closure variable N at parse time
    return x
```

(The output comments above are the parser lane's; under tracing none of the
four lines emits anything.)

| Function | Purpose | On failure |
| -------- | ------- | ---------- |
| `pl.static_print(*args)` | Print variable types/values to stdout | Requires ≥1 argument |
| `pl.static_assert(cond, msg="")` | Assert compile-time condition | Raises `ParserError` — parser lane only; silent under tracing |
| `pl.dump_tag(tensor)` | Mark a tensor for selective runtime tensor dump — declarative per-tensor marker (valid in Orchestration scope, or in an Inline helper that the orch inlines — see [Runtime DFX](../03-runtime-dfx.md#selective-tensor-dump)) | Raises `TypeError` outside an Orchestration or Inline function, or for non-`Name` arguments |

**Key points:**

- All three are statement-only (cannot be used in expressions)
- `static_print` accepts variables, constants, string labels (printed as-is), and f-strings with plain `{expr}` placeholders (formatted as IR). Conversions (`!r`, `!s`, `!a`) and format specs (`:...`) are not supported.
- `static_assert` supports closure variable expressions (e.g. `N > 32`) and IR constants; message must be a string literal
- `dump_tag` takes one bare tensor variable name bound in the enclosing Orchestration (or Inline) scope; the tracing frontend honors it — the marker is consumed at trace time and tracked by Var identity (not name) all the way to codegen. At an explicit `self.kernel(...)` site it records the tensor in the consuming Call's `dump_vars` on every subsequent consuming call; in the `@pl.jit` / `with pl.at(level=...)` style (where the dispatch is synthesised by the outline passes) it instead seeds the enclosing scope's `dump_vars` and the outliner maps it onto the synthesised dispatch arg (see [Runtime DFX](../03-runtime-dfx.md#selective-tensor-dump)). To list dump targets explicitly at a single task launch, use the `dumps=[...]` kwarg on `pl.submit(...)` / `pl.at(...)` (symmetric with `deps=`)
- Parser-lane output appears even if parsing fails later — useful for debugging parse errors

### Statement Sequences

```python
stmt1            # Natural Python sequencing
stmt2
stmt3
```

## How SSA Is Derived

The tracer collects what the body executes and renames bindings freely, so
ordinary Python is normal input — [ConvertToSSA](../passes/04-convert_to_ssa.md)
runs early in the pipeline and owns all graph-wide dataflow. Nothing about
the merge or the carry is declared at the source:

```python
# If: the phi node at the merge point is synthesized from the two assignments
if condition:
    y1 = pl.add(x, 1.0)
else:
    y1 = pl.add(x, 2.0)
# y1 = phi(x + 1.0, x + 2.0) — after ConvertToSSA

# For: the loop-carried value is the self-referential rebind
s = pl.const(0, pl.INT64)
for i in pl.range(10):
    s = s + i
s_final = s  # captures the final value
```

The one place a carry is still stated explicitly is the assembly channel
(`pl.intrin.carry` / `yield_` / `leak`, migration guide Appendix A) — used
by compiler fixtures and by kernel code that prefers to name the loop's
iter args outright (the task-graph dependency loop can also derive its
carry from rebinding; guide §6.3).

The printer's round-trip rendering of the same control flow is the
explicit dialect: `for i, (s,) in pl.range(10, init_values=(...)):` with
assignment-form `pl.yield_` in the body, and `pl.while_(init_values=...)`
plus a leading `pl.cond(...)` for an SSA-form while. That is the text
`pl.parse` consumes (see [IR Parser](../ir/07-parser.md)); tracing input
never spells a carry that way.

## See Also

- [Python IR Syntax Specification](00-python_syntax.md) — types and expressions
- [Manual Dependency Primitives](02-manual_dependencies.md) — opting out of auto-dep tracking
- [Functions and Program Structure](03-functions.md) — function types and parameter directions
- [Control Flow](../../user/language/02-control-flow.md) — the user-facing treatment of the same rules
- [Migration guide](../../../design/04-migration-guide.md) — every parser-era spelling and its tracing replacement
