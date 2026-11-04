# Construct Recipes

`@pl.authoring.construct` — authoring new `with` and `for` constructs without touching
the tracer.

A **construct recipe** declares how one construct lowers: a generator
function whose code before the `yield` opens the construct's regions, whose
`yield` hands the body its value, and whose code after it finalizes. The
tracer keeps everything else — execution, stack discipline, IR construction.

```python
import pypto.language as pl


@pl.authoring.construct
def blocked_spmd(ctx, shape, tile, *, name_hint="", deps=None):
    blocks = tuple((s + t - 1) // t for s, t in zip(shape, tile))
    core_num = 1
    for count in blocks:
        core_num *= count
    with ctx.region(pl.spmd(core_num, name_hint=name_hint, deps=deps)) as tid:
        tid_var = ctx.resolve_binding(tid) if tid is not None else None
        yield BlockGrid(blocks, tile, tid_var)


with blocked_spmd((tokens, heads), (1, HEAD_TILE)) as block:
    token, head = block.axes
    q = block.region(query).load()
```

## The Layering

```text
kernel author          recipes (blocked_spmd), pl.* users
    │ composes
construct tier          @pl.authoring.construct + ConstructContext   ← this page
    │ drives
tracer                  rewriter, interpreter, Sema/Frame
    │
IR builder              closed statement/scope vocabulary
```

The construct tier is the only sanctioned client of the tracer's internals
(together with the authoring runtime beneath it); recipe code sees only
`ConstructContext`. The trace protocol (`__trace_enter__` /
`__trace_bind__` / `__trace_exit__`) remains the ABI between the tier and
the tracer — but recipes never implement it. The driver implements it *for*
the recipe, which is why the private conventions of a hand-written construct
(deferred-carrier flags, binding decline, close ordering) cannot be gotten
wrong.

The construct tier and the general authoring tier
([07-authoring-recipes.md](07-authoring-recipes.md)) share one **authoring
session** per recipe invocation — the one-invocation-scoped owner of the
region stack (below) and the function units. `ConstructContext` is the
region view over that session; `@pl.authoring.recipe`'s context is the same view
plus the function, body, and control effects, so a general recipe composes
construct regions (`ctx.region(pl.spmd(...))`) exactly as a construct
recipe does.

The vocabulary stays closed where it must: recipes configure and compose
**existing** IR constructs. A recipe that needs a new lowering shape is a
framework change first — new scope kind, new statement, new capability —
registered before any recipe can use it. `if`/`else` and Python `while` stay
native; authorable sugar is the `with` shape and the trace-iterable `for`
shape (a plain `pl.range`-style loop descriptor stays a framework concern).

## Both Spellings

One generator serves the two statement spellings, and they differ only in
when the generator starts and what its `yield` value is:

| | `with construct(...) as x:` | `for i in construct(...):` |
| --- | --- | --- |
| generator starts | construct entry (`__trace_enter__`) | the loop-target binding — the point the target's source name is known |
| `yield` value | the `as` target — a `ConstructHandle`, a capture binding, or `None` | the loop variable — an `ir.Var` from `ctx.loop_var()` |
| the body | runs inside the `with` | runs as the representative iteration |
| `ctx.binding_name` | the `as` name (or `None`) | the loop target's name |
| close | post-yield code at `__trace_exit__` | the same post-yield code |

A recipe serving both spellings branches on `ctx.spelling`. `pl.spmd` is the
reference: one `_spmd_scope` recipe lowers `with pl.spmd(...) as tid:` and
`for i in pl.spmd(...)` — the for-arm mints the loop variable, always wraps
the carrier, and sets the carrier's deferred prologue (`i = pl.tile.get_block_idx()`)
as its first statement.

The for spelling carries one hard contract: every region the recipe leaves
pending must be exposed by its structural surface — a carrier opened with
`ctx.open_carrier(..., primary=True)`. The loop variable's prologue and the
body's statements both land inside that carrier; a recipe whose primary
region is anything else would register nothing and silently lower the body
outside it, so the driver rejects it loudly at the target binding. A recipe
that opens every region immediately (`defer=False`) has nothing pending and
needs no structural surface at all — `pl.split_aiv` is that shape.

## The Recipe Contract

- **Generator function, exactly one `yield`.** Code before the `yield` runs at
  construct entry; the `yield` is the body boundary; code after it runs at
  construct exit (normal path only — an exception discards the whole trace and
  runs no staged finalization). For the `for` spelling, "entry" is the
  loop-target binding — the generator starts there, when the target's source
  name is known.
- **The `yield` value is the `as` target (`with`) or the loop variable (`for`).**
  A rich handle should subclass `pl.authoring.ConstructHandle`: it host-binds (the body
  reads it as an ordinary Python object) and stays out of every staged-value
  path. A recipe that yields a handle requires an `as` binding; a recipe used
  without `as` must yield `None`. The `for` spelling must yield the `ir.Var`
  from `ctx.loop_var()` — anything else is rejected (a handle-yielding recipe
  serves the `with` spelling only).
- **One use per instance.** `blocked_spmd(...)` returns a single-use construct,
  like `pl.spmd(...)`; the factory itself is stateless and reusable.
- **Regions close in reverse open order.** `ctx.region(cm)` closes itself at
  the generator's end; an explicit `ctx.enter(cm)` must be paired with
  `ctx.exit(cm)` before the generator finishes.

## `ConstructContext`

The capability object handed to the recipe as its first argument. Its surface
is closed and curated — no builder access, no emission helpers.

| Capability | Purpose |
| ---------- | ------- |
| `ctx.region(cm, as_name=..., single_call_body=...)` | Compose one construct (`pl.spmd`, `pl.at`, another recipe) inside the recipe; a context manager whose body holds the `yield`. |
| `ctx.enter(cm)` / `ctx.exit(cm)` | The explicit form of `region`. |
| `ctx.open_scope(kind, attrs=..., defer=True, primary=False, prologue=..., **begin_kwargs)` | Open one scope directly — the lowering primitive. `kind` is drawn from the framework's closed scope vocabulary; the recipe owns the attr order because it is the lowering policy. With `defer=True` (the declarable-scope contract) the Frame opens the region at the body's first IR-emitting statement, folding in leading `pl.set_cache_policy(...)` declarations, or at `exit`; `defer=False` opens it now, for a construct whose attrs are fixed at entry and which takes no cache-policy declaration. `prologue` statements are the region's own first emissions (a for-form's loop-variable binding) — emitted right after an immediate begin, or at a deferred open before any body statement lands. `primary=True` makes the region the recipe's structural surface. |
| `ctx.open_carrier(kind, attrs=..., flush_on_emit=..., prologue=..., primary=...)` | A deferred scope whose open the *body* decides. `flush_on_emit=True` opens at the first emission (a wrapped body); `False` waits for the block-index trigger — the body proves itself inline, a dispatch body never opens it — and `prologue` statements (a loop variable binding the block index) are the carrier's first emissions. |
| `ctx.capture_task_id(name)` | Mint the TaskId capture for an `as tid` form: creates the Var, emits the placeholder assign, and returns `(binding, task_id_var_attr)` — the recipe appends the attr at its canonical position and yields the binding. |
| `ctx.lower_deps(deps, owner)` | Lower a `deps=[...]` list to `manual_dep_edges` Vars — the one trace-lane `deps=` lowering (`coerce_deps`), shared with the dispatch effect so the scope and submit spellings cannot drift. |
| `ctx.resolve_binding(tid)` | Resolve an `as tid` capture proxy to its `ir.Var` — how a recipe exposes a dispatch TaskId on its handle (e.g. for a later `deps=` edge). |
| `ctx.binding_name` | The name the user bound the recipe under: the `as` name for `with` (`None` without `as`), the loop target's name for `for`. |
| `ctx.spelling` | The statement spelling this use entered through: `"with"` or `"for"` — what a recipe serving both spellings branches on. |
| `ctx.loop_var()` | Mint the loop variable of a `for`-spelling use — the recipe's `yield` value. The Var carries the loop target's name and `INDEX` type; the Frame registers it under the target name without a `LetStmt`. |
| `ctx.open_loop(start, stop, step, kind=..., attrs=..., var_name=...)` | Open one `ForStmt` region directly — the loop-tier lowering primitive (`pl.range`'s `begin` as a capability). The loop opens now and closes at the recipe's exit (`with ctx.open_loop(...) as i:` brackets the `yield`); `kind` is the `pl.range` family's vocabulary and carries its transfer rules (a `range` loop accepts `break`/`continue`, `parallel`/`unroll` reject them). The loop-carried channels are the SSA pass's inference: bind the initial value through `ctx.bind` before the open, rebind the name in the body, and `ConvertToSSA` creates the `iter_args` / `return_vars`. |
| `ctx.bind(name, value)` | Bind one staged value under `name` in the caller's trace — the Frame's own bind path. The recipe's channel/`as`-name discipline: the bound name is a caller-trace name and must not collide with the recipe's own `as` target or the user's locals. |
| `ctx.single_call_body` | The rewriter's body-shape fact; forward it when composing a construct whose IR shape depends on it (`pl.spmd`'s direct-dispatch vs InCore-wrapped form). |
| `ctx.sticky_dump_vars()` | The function's forward-sticky dump-tagged tensor Vars (the `dump_vars` attr merge). |
| `ctx.is_inside_incore()` / `ctx.is_inside_scope(kind)` / `ctx.function_type()` | Read-only trace facts. |
| `ctx.enclosing_user_split()` | The innermost open InCore scope's user-declared `pl.split(...)` mode — the fact behind the AIV-split mutual-exclusion rule (`pl.split_aiv` reads it). |

`as_name` defaults to the recipe's own `as` name — the composed construct's
TaskId capture mints its Var under the name the user bound.

The recipe's bound constructor arguments are readable on the instance before
it is entered (the **constructor surface**): `pl.at`'s `level` is the
pre-enter fact the Frame's InCore hint reads to decide whether an outer
deferred carrier absorbs instead of opening. A wrong call fails at the
factory — the constructor-surface `TypeError` a hand-written construct's
`__init__` raises.

## What Is Authored on the Tier Today

Every stdlib `with`-construct is a recipe: `pl.at`, `pl.scope`,
`pl.manual_scope`, `pl.graph`, `pl.cluster`, and `pl.spmd`. Each
one's pre-yield code is its original `__trace_enter__` body in the original
order — attr collection, TaskId capture, the parser-parity validations — with
no builder access. The conversions' acceptance is IR parity with the
hand-written shapes: asserted directly for a minimal `core_group` recipe
against `pl.at`, and by the full construct corpora (the tracer-vs-parser lane
parity suites).

Three shapes cover them:

- **Leaf scope** (`pl.at`, `pl.scope`, `pl.manual_scope`, `pl.graph`,
  `pl.cluster`) — one `ctx.open_scope` per construct. `pl.at` defers (it
  accepts cache-policy declarations); the other four open immediately
  (`defer=False`), exactly as before, so their cache-policy rejection is
  unchanged.
- **Carrier** (`pl.spmd`) — `ctx.open_carrier` for the auto-outlined InCore
  region nested in the dispatch scope, marked `primary=True` because the
  carrier — not the dispatch scope around it — owns the body-decided open.
  The plain form waits for the per-block-index read; a split hint, a
  cross-core slot, or a captured multi-statement body opens at the first
  emission.
- **Immediate-open for-region** (`pl.split_aiv`) — `ctx.open_scope(...,
  defer=False, prologue=[(loop_var, lane_idx)])` for the SplitAiv region
  itself, plus a synthesized InCore wrapper for the bare top-level form
  (the forward-sticky dump tags merge onto it; a form already inside InCore
  — or inside an InCore-typed function body — emits in place). Nothing is
  deferred, so the recipe needs no structural surface.

The two `for` spellings are recipes on the same tier. `pl.spmd` is one
recipe for **both spellings**: `with pl.spmd(...)` and
`for i in pl.spmd(...)`. The with-arm keeps the original hand-written
lowering (capture, wrap decision, name-split decision); the for-arm captures
nothing, always wraps, and mints the loop variable bound to
`pl.tile.get_block_idx()` as the carrier's deferred prologue. The conversion
deleted the for-form's separate descriptor path (its `begin`/`__trace_exit__`
arms and carrier state) from `constructs/scopes.py` — the class is a thin
façade that creates the recipe instance and forwards. Acceptance is IR parity:
the for-form traces byte-identically to its hand-written predecessor (plain,
empty, `deps=`, cache-policy, split/slot, and cluster-rejection variants), and
the tracer-vs-parser lane parity covers it through the `spmd_loop_kernel`
reference. `pl.split_aiv` is the immediate-open counterpart: one recipe
(`_split_aiv_scope`) synthesizing the InCore wrapper only for the bare
top-level form, with the RFC #1820 function-level-split rejection reading
`ctx.enclosing_user_split()` — and the lane parity covers it through the
`split_aiv_loop_kernel` reference.

`pl.range` / `parallel` / `unroll` / `pipeline` stay hand-written on purpose:
they already share one clean `RangeIterator`, and they carry host-execution
parity (host iteration of the same source) — a semantics a recipe rejects by
design. The tier's `ForStmt` primitive is `ctx.open_loop`: a recipe can
author a real loop region (the `counted` / `accumulate` acceptance recipes
trace IR-identical to the `pl.range` forms they replace, carried channels
included), but the stdlib entries themselves remain framework-owned so
`pl.range(n)` keeps its host behavior.

The first externally authored recipe is `blocked_spmd` — a block-grid
dispatch over `pl.spmd`: one representative block whose axes project onto
tensors, with the dispatch TaskId exposed for `deps=` wiring. Its acceptance
tests assert **IR parity** — the recipe and the
hand-written `pl.spmd` grid it replaces trace to structurally identical IR —
in `tests/ut/language/tracing/test_recipe.py`, alongside a `grid_loop`
recipe proving a kernel author can write the `for` spelling itself (IR-equal
to `for i in pl.spmd(n)`), with no import below `pypto.language`.

## See Also

- [Authoring Recipes](07-authoring-recipes.md) — the general tier beneath: `pypto.language.authoring` is the canonical import root for both protocols.
- [Statements and Control Flow](01-statements.md) — the scope context managers recipes compose.
- [Manual Dependency Primitives](02-manual_dependencies.md) — `deps=` edges and the TaskId capture channel.
