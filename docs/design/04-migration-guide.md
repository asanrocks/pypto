# Migrating Kernels to the Tracing Frontend

- **Audience:** PyPTO end users — kernel authors writing the ergonomics
  dialect (§4), jit users (§5), and advanced mixed-kernel / SPMD /
  task-graph authors (§6). Appendix A speaks to compiler developers who
  write the assembly dialect. The architecture itself is the subject of
  `00-tracing.md`; the language contract is `01-language.md`.
- **Scope:** what changed when the frontend moved from source parsing to
  tracing, why each change exists, and how to migrate existing kernels.
  Design rationale is stated once, briefly, as the reason for a change —
  the full reasoning lives in the design series, which this document
  points at and never restates as an obligation to sync.
- **Position:** the repo-internal migrations (`tests/ut`, corpus, and
  `examples/`) are complete; the `tests/st` and user/developer
  documentation migrations are pending and must land before this guide
  is released publicly — `docs/migration-todo.md` is the live tracker;
  Appendix B summarizes it.
- **Tiering:** read §1–§3, then the one tier that matches the code you
  write, then §7–§9. A kernel author who never writes cross-core pipes or
  SPMD scopes can stop after §4; nothing in the appendix is required
  reading for shipping a model.

## 1. Why tracing

A PyPTO kernel is ordinary Python. Collecting IR is a **side effect of
running the program**: each `pl.*` operation executes and yields a staged
value — a typed wrapper around an IR expression — as the kernel runs.
There is no dedicated DSL syntax — no `if_`, no `let`; the two spelling
crutches the parser era did have (`pl.while_` with `pl.cond()`) are gone
(§4.2). Under tracing your source text is no longer *interpreted*: it
is rewritten, type-blind, into observable events, and the semantics
come entirely from execution — your program is **executed**, and the
execution *is* the IR.

The previous frontend parsed the kernel's source text. That had a
structural consequence you have already felt in your code: everything the
parser could not see in the text had to be *declared* in it —
`init_values=` to say which values a loop carries, merge plumbing to say
how branches converge, annotations to say what type a view has after a
restamp. Tracing removes that entire class of declarations, because the
frontend now watches the actual bindings instead of guessing at them.

What you gain:

- **Metaprogramming becomes real.** Host control flow executes at trace
  time: a host `range` actually iterates and the loop body is emitted N
  times; a host `if` takes one path. Shapes, loop bounds, and tile layouts
  can be *computed inside the kernel* in ordinary Python instead of
  templated around it.
- **You declare less.** Loop-carried values are derived from rebinding —
  `acc = pl.add(acc, x)` inside the loop *is* the carry. Branch
  convergence needs no merge syntax: both branches assign the same name,
  and the SSA pass synthesizes the phi. `init_values=`, iter-arg tuple
  unpacking, and per-branch `yield_` plumbing disappear.
- **Silent wrong answers become loud errors.** Under the parser, an
  annotation could retype a value and the kernel still compiled; and
  execution-based collection opens silent paths of its own — a
  host-literal accumulator drops its loop carry and compiles fine
  (§4.1). Tracing closes every compile-but-wrong path with a diagnostic
  that names the operator and the site (§7).
- **Type honesty.** A value's type comes from the operator that produced
  it — never from an annotation and never from "are we tracing right
  now?". This is what makes the frontend predictable: the same kernel
  code means the same thing at every occurrence.

What runs, and when: a staged loop traces **one representative body
iteration**; a staged `if` records **both branches**; nothing touches a
device. Decoration collects IR; execution happens later, through the
normal compile and dispatch path.

What honestly got stricter: some constructs Python allows cannot be
observed soundly at trace time (`try/except`, `global`/`nonlocal`, `del`,
`eval`/`exec`, dynamic attribute names — §8 lists them all), and a few
diagnostics only the parser could run (`static_assert`, `static_print`)
are silent no-ops now.

## 2. The three principles behind every change

Every spelling change in this guide is an instance of one of three
principles. When you meet a change this guide does not list, derive it
from these; that is what they are for.

**P1 — Typing flows from values, not annotations.** Under tracing, a
value already exists when an annotation would be applied — it was
produced by an operator, with its real type already fixed. An annotation
that restates the type is documentation; an annotation that *changes* it
would be a lie. Everything the parser inferred from context — the type of
a restamped view, the memory space of a tile, the shape and dtype of a
cross-core pop — is now **stated at the operation**. `pl.const`,
`target_memory=`, `pl.intrin.transmute`, explicit tpop producers.

**P2 — Trace-time execution is the IR; the passes own the dataflow.**
Whatever the host does at trace time is exactly what the IR contains, and
all graph-wide dataflow (SSA promotion, phi synthesis, loop-carried
derivation, copy folding) belongs to the IR passes — the frontend is thin
on purpose. Dataflow you used to *declare* (`init_values=`, branch
yields, merge variables) is now *derived* from what your code actually
rebinds. Where a low-level test genuinely needs to state a channel, the
assembly dialect offers `pl.intrin.carry`/`yield_`/`leak` (Appendix A).

**P3 — Loud, not silent.** Every path that used to compile and then
compute the wrong result is closed by a loud error. When a construct you
wrote starts raising, the error message is the migration guide entry: it
is telling you which silent failure it just prevented (§7's table pairs
them).

## 3. Am I affected?

The dialects are distinguished by *frontend ownership*:

| Your code | Frontend | Action |
| --------- | -------- | ------ |
| Kernels via `@pl.function` / `@pl.program` / `@pl.jit` | tracing | Migrate — §4–§7 |
| IR printer output, consumed by `pl.parse` / `pl.loads` | parser, permanently | **Do not migrate** — `init_values=` in printed text is correct round-trip spelling |
| Fixtures under `tests/ut/language/parser/`, `tests/ut/ir/parser/` | parser, via `@pl.function_parser` / `@pl.program_parser` | **Do not migrate** — these are the parser's own tests |

Two mechanical points:

- The unsuffixed markers `@pl.function` / `@pl.program` collect through
  tracing everywhere outside the parser test directories. If your code
  was written for the parser and relied on parser-only behavior, it now
  runs under tracing and must obey this guide.
- `pl.yield_` still exists as a deprecated alias of
  `pl.intrin.yield_` (it warns). Most of its old uses disappear entirely
  under P2 — see §4.2.

## 4. Tier 1 — kernel authors (ergonomics dialect)

### 4.1 Staged initialization: `pl.const`

A host literal can no longer initialize staged state. `pl.const(value,
dtype)` is the one spelling for a concrete staged value:

```python
# Before: the accumulator is a bare Python int.
acc = 0
for i in pl.range(n):
    acc = pl.add(acc, tile[i, 0])

# After: a typed staged constant.
acc = pl.const(0, pl.FP32)
for i in pl.range(n):
    acc = pl.add(acc, tile[i, 0])
```

This is the P3 flagship, and the one change that fixes a real bug class
rather than a style. A host binding emits no IR, so the loop body would
inline the literal `0` on every iteration: the rebind is not
self-referential, no loop-carried value is derived, the kernel compiles —
and every iteration reads the initial value. `pl.const` creates the
staged local the derivation needs. The guard is symmetric: demoting a
staged local to a host literal under an open `if` or loop, or staging a
name on one branch and not the other, are both loud errors for the same
reason.

Booleans follow the same spelling: `flag = pl.const(False, pl.BOOL)`.

The init's **dtype must match the carried rebind's** — a rebind whose
operator produces a different dtype is a fresh definition, not a carry
(the pre-loop init is then read only by the representative iteration,
which is exactly the dropped-carry failure `pl.const` exists to close).
`ConvertToSSA` names the conflict: `loop-carry 'acc' rebinds as
Scalar[INDEX] inside the loop, but its pre-loop initialization has type
Scalar[INT32]`. The common corpus trigger is floordiv — `pl.read(x, [i])
// n` promotes its operand to `INDEX` — so an accumulator fed by it
initializes as `pl.const(0, pl.INDEX)`, not `INT32`.

### 4.2 Loop carries: derive, do not declare

`pl.range` (and `pl.parallel`, `pl.pipeline`) no longer takes
`init_values=`. The loop's carried values are derived from what the body
rebinds (P2):

```python
# Before:
for i, (acc,) in pl.range(n, init_values=(acc_init,)):
    acc = pl.yield_(pl.add(acc, x[i]))

# After: the rebind *is* the carry. Nothing is declared.
for i in pl.range(n):
    acc = pl.add(acc, x[i])
```

The same loop may rebind several names; each self-referential name
becomes a carried value. Single-branch definitions inside a loop body are
legal — the missing branch reads the previous iteration's value; that is
loop-carried behavior and the pass owns it.

Two shapes the example above does not show, both regular in real kernels:

- the initial value need not be `pl.const` — **any staged producer**
  (`pl.load`, a tile op, a callee result) already binds a staged name, and
  that name is a valid carry seed. `pl.const` (§4.1) is only for *host*
  literals.
- the parser era chained **nested-branch `yield_`s through intermediate
  names** (`m_valid = pl.yield_(...)` per branch, then
  `m_after = pl.yield_(m_valid, ...)`, then the carried rebind). Each
  `yield_` becomes a plain assignment of the same names; reads of the old
  *iter arg* (`m_iter`) become reads of the carried name (`running_m`).
  The alias chain itself can stay — copy folding is a pass concern.

**`pl.while_` and `pl.cond()` are gone.** The parser era spelled a staged
while-loop as `for (x,) in pl.while_(init_values=(x_init,)):` with
`pl.cond(...)` as the body's first statement. Write a bare `while` with a
staged predicate instead — the body traces one representative iteration,
and the carry is derived from rebinding exactly as above:

```python
pos = pl.const(0, pl.INT32)
while pos < seq_len:              # staged predicate → representative iteration
    chunk = pl.load(seq, [pos, 0], [16, 64])
    pos = pl.add(pos, pl.const(16, pl.INT32))
```

Task-graph dependency loops derive their carries the same way; the
explicit channel spelling for them is §6.3.

### 4.3 Branch values: plain assignment

A staged branch no longer publishes its value through `pl.yield_`. Assign
the name; both branches assign the same name; convergence is the pass's
job (P2):

```python
# Before:
if i == last:
    valid_len = pl.yield_(last_valid_len)
else:
    valid_len = pl.yield_(block_size)

# After:
if i == last:
    valid_len = last_valid_len
else:
    valid_len = pl.const(block_size, pl.INDEX)   # host value crossing a staged join — P1
```

Note the second branch: a *staged* value is a plain assignment; a *host*
value crossing the join needs `pl.const` (§4.1).

Reading a name that only some branches bound is a loud error — the same
`UnboundLocalError` discipline Python itself enforces. Bind the name
before the `if`, or in every branch.

### 4.4 Tile placement: `target_memory=`

A tile's memory space is stated on the operation that creates it (P1),
not inferred from the annotated view type:

```python
# Before: the annotation carries the placement.
x_tile: pl.Tile[[64], pl.FP32, pl.MemorySpace.Vec] = pl.load(x, [0], [64])

# After: the load states where the tile lands.
x_tile = pl.load(x, [0], [64], target_memory=pl.MemorySpace.Vec)
```

Annotations on locals remain legal — they are documentation, and they
never flow into IR. **Audit them yourself during migration: a stale local
annotation raises nothing.** A local annotation that disagrees with the
value's actual type is silently ignored — the value keeps the type its
operator gave it, and the printer restates the true type — so an
annotation that used to perform a restamp is now dead text, not an
error. Function *return* contracts are the exception: a declared `->`
that lies about what the body produces is a loud decoration-time error.

The one restamp with no operation kwarg — binding a tile to a declared
`pl.MemRef` slot — is stated at the operation through
`pl.intrin.transmute` (Appendix A.2), and it is loud there: selecting a
slot of a single-slot declaration, or a constant past the declared slot
count, is an immediate error, and a *runtime* rotation index is
expressed the same way (`pl.Tile[[M, N], dtype, buf[i % 2],
pl.Mem.Vec]` inside the transmute descriptor — the staged index may be
any index expression, exactly what the parser read out of the old
annotation).

### 4.5 Dynamic dimensions

`bind_dynamic` overriding a *static* dimension is now a loud error — its
old semantics lived in a source scan the tracing frontend does not have.
The supported spelling is the annotation-embedded `pl.dynamic(name)`:
declare the dimension dynamic where the parameter is annotated, and the
call site supplies the value.

One case where the error fires although the annotation already declares
the dimension dynamic: an **inline callee called in-trace with a
static-shaped argument**. The callee's in-trace parameter type follows the
*argument*, not the callee's annotation (type-soundness — the value really
is that shape), so a static slice reaching a `pl.dynamic`-annotated
parameter mints a static type and the body's `bind_dynamic` raises. Under
the specializer the AST scan read the annotation instead, so the call was
silently honored. The call is residue: delete it — the body reads the
minted (static) extent, which is the argument's true shape.

## 5. Tier 2 — jit users

`pl.jit` is the tracing jit. The AST-era runtime survives quarantined as
`pl.jit_specializer` for its own mechanics tests only — do not build on
it, and calling it from inside a trace raises.

### 5.1 Parameter worlds

A parameter's world is pinned by its declaration, and a contradicting
call site is a loud `TypeError` in both directions:

- a **`pl` facade annotation** (including the bare class, e.g. `pl.Tensor`)
  makes the parameter a **runtime parameter** — the value arrives at
  execution;
- a **plain Python annotation or none** makes it a **host knob**, baked
  into the IR at trace time — a Python `int`/`float` argument becomes a
  `Const`, and the cache key includes the value;
- `pl.runtime(value, dtype)` is the spelling for a **runtime scalar** —
  the dispatch cache keys on the dtype, never the value.

A `torch.Tensor` argument bridges to a static-shape facade; dynamic
extent dimensions appear only in annotations and are shared by name —
two parameters both annotated `N` share one extent symbol.

**Migrating an AST-era `pl.Scalar` parameter.** Under the specializer
every scalar specialized into the artifact by value, so a `pl.Scalar`
annotation did not pin a world — each one is an open question at
migration time, and the production call site is the ground truth:

- a host literal or module constant, the same on every dispatch and on
  every rank sharing the artifact, is a **host knob** — re-annotate it
  plainly (`done_epoch: int`), pass the value at the call or through
  `compile(name=value)`, and spell body constants `pl.const(...)` rather
  than `pl.cast(literal, ...)` (the cast-of-a-literal form was the parser
  era's coercion device into `Scalar`-typed params; the value now folds
  to a `Const`);
- a staged expression, a per-rank value, or a per-forward counter (an
  epoch advancing across dispatches of one artifact) is a **runtime
  parameter** — keep `pl.Scalar`, and dispatch runtime values
  (`pl.runtime(value, dtype)` at a host call; a test harness marks the
  scalar runtime in its spec).

**Which lane fixes a host int depends on where the call sits.** The
world check raises at both, but the two lanes take different spellings:

- at a **host call** (a fixture or driver dispatching the jit kernel):
  `pl.runtime(value, dtype)`, or mark the scalar runtime in the harness
  spec — the golden-harness spelling is
  `ScalarSpec(name, dtype, value, compile_runtime=True)`. Marking *any*
  scalar switches the compile to the annotation-driven signature path:
  every tensor parameter then needs a complete
  `pl.Tensor[[shape...], dtype]` annotation. A driver that deliberately
  keeps bare `pl.Tensor` params so per-case shapes bridge from the torch
  arguments cannot take this path — restructure its wrapper (full
  annotations with `pl.dynamic` extents) or bake the scalar instead;
- **inside a trace** (a kernel calling another kernel): pass a staged
  value. A trace-time constant is spelled `pl.const(value, dtype)` — a
  staged constant flowing into the runtime-scalar slot. `pl.runtime` is
  *not* valid here; it raises
  `RuntimeError: Cannot unwrap annotation-only Scalar`.

**The argument's dtype is the truth, too.** A `pl.Scalar[pl.INT32]`
parameter does not retype its argument: a staged value of another dtype
keeps its own. The corpus shape is an INDEX-typed loop product —
`layer = order * 2 + pl.const(2, pl.INT32)` promotes to INDEX because the
loop variable dominates — flowing into an INT32 parameter; it stays
INDEX, and the first dtype-checked consumer (`tensor.write` into an
INT32 tensor) fails loudly. The specializer read the annotation and
coerced; tracing reads the value. Cast at the source of the wrongly-typed
value (`layer = pl.cast(order * 2 + 2, pl.INT32)`), not at each consumer.

### 5.2 Directions

`pl.Out[T]` / `pl.InOut[T]` carry their direction on the facade copy, so
exec'd kernels keep their directions without source reconstruction. One
deliberate exception: on `@pl.jit.inline` helpers, `Out`/`InOut`
directions are dropped to `In` **with a deprecation warning** — the
old write-through-alias style (`hc_post(..., x_next)` then returning
`x_next`) trips the direction-use verifier, and the warning names it.
Return the value instead.

### 5.3 Timing and configuration

Decoration inspects the signature: every parameter facade-annotated →
eager, traced at decoration. Anything call-time-dependent (`Callable`
parameters, torch shapes) → lazy, traced at call. A missing parameter
annotation is a loud decoration-time error — there is no `UnknownType`
guessing under tracing (P1). Both timings return the same public jit
object: for an eager kernel, `compile`/dispatch serve the decoration
trace (`.fn`) — the declared dynamic extents stay dynamic, so dummy
compile args are dispatch vocabulary, never a re-specialization.

A host-side `compile` / `__call__` accepts `config=RunConfig` (compile
fields forward to `ir.compile`, runtime fields to the compiled program).
In-trace calls stay parser-aligned and reject it. `@pl.jit.extern` traces
a single-core stub as a header-only AIC/AIV function and a `mixed` stub
as an AIC + AIV member pair.

Harness and tooling code that introspected the specializer's private
`fn._func` (signature scans in spec builders) should read `fn.kernel` —
the original Python function — under the tracing jit.

Inline callees currently trace as sub-kernels and the backend inlining
pass splices them. `@pl.expand` is the additive tracing-native
alternative: the body executes and is emitted at *every* occurrence, with
normal Python call semantics — the manual-inline recipe from the
parser era maps onto it.

**`@pl.jit.inline` and `@pl.jit.incore` defer their trace to first use**
(`01-language.md` §10). The AST-era jit decorated lazily, so def order was free; under
tracing, an inline helper whose body referenced a later-defined callee
raised a raw `NameError` at import. Def order is free again — the trace
runs at the first structural read (`.fn`), host compile, or materialization
from a caller's trace. Loud errors that used to fire at import (the
`self.<sibling>` unknown, the annotationless-submit declared-return rule,
the Out-direction deprecation warning) now fire at that first use —
move them with the kernel if you split a module.

## 6. Tier 3 — advanced kernels

### 6.1 Cross-core pipes: explicit tpop producers

Cross-core `tpop` no longer infers its result shape and dtype from the
environment. Both are explicit value-producer inputs (P1):

```python
# Before: shape and dtype omitted — inferred from the environment.
tile = pl.tpop_from_aic(split=0)

# After: both producers are stated — shape positionally, dtype by keyword.
tile = pl.tpop_from_aic([64], dtype=pl.FP32, split=0)
```

### 6.2 SPMD scopes

The with-form `pl.spmd` body — including multi-statement bodies reading
`get_block_idx()` — is wrapped correctly by the tracer; that shape is
what the outline and codegen passes expect, and it now round-trips
through the conformance harness. Captured grid dispatch is stated
through the assembly intrinsic `pl.intrin.spmd_direct` /
`spmd_direct_call` (Appendix A.3) — task-id and output bindings are
stated at the intrinsic, not carried by the `with` form.

### 6.3 Task-graph carries: derive or state the channel

The orchestration loop that republishes a TASK_ID array as the next
iteration's submit dependencies can state its carry either way. Plain
rebinding (`tids = tids_next`, a fresh array per iteration) is accepted
at collection, the SSA pass derives the carry, and orchestration codegen
lowers the rebind onto the fresh array's backing storage with the
yield's slot-by-slot copy carrying state across iterations — the emitted
C++ is identical to the channel spelling's. The channel spelling —
`pl.intrin.carry` / `yield_` / `leak` — remains the explicit way to name
the loop's iter args (the assembly dialect's surface; Appendix A.1),
and the in-place shape (one array, `tids[branch] = tid` across
iterations) compiles as it always did:

```python
# The phase-fence pattern: two chained dependency loops.
# Derive the carry — the rebind is the carry, nothing is declared.
tids = pl.array.create(branches, pl.TASK_ID)
for a_phase in pl.range(2):
    tids_next = pl.array.create(branches, pl.TASK_ID)
    for branch in pl.parallel(branches):
        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids])
        tids_next[branch] = tid
    tids = tids_next
```

The channel spelling of the same loop:

```python
(tids_a,) = pl.intrin.carry(tids)
for a_phase in pl.range(2):
    tids_next = pl.array.create(branches, pl.TASK_ID)
    for branch in pl.parallel(branches):
        out, tid = pl.submit(self.kernel_stripe, data, row, 1.0, out, deps=[tids_a])
        tids_next[branch] = tid
    pl.intrin.yield_(tids_next)
```

This is the only `pl.intrin.*` spelling a kernel author normally meets.
Everything else in the `intrin` namespace is the assembly dialect —
Appendix A.

The `deps=` coercion accepts subscripted task-id arrays
(`deps=[tids[i]]`); the parser's desugaring is ported, so existing
dependency lists keep working.

## 7. What you will hit

The loud errors, each paired with the silent failure it prevents. Every
row is a P3 instance; the design-level statement is `01-language.md` §12.

| Error | The silent failure it prevents | Fix |
| ----- | ------------------------------ | --- |
| staged value coerced to bool (`if x:`, `assert x`, ternary, comprehension predicate) | one branch of a staged conditional silently collected | write the explicit test (`x != 0`) |
| missing typed staged initialization (§4.1) | loop reads the inlined literal every iteration — compiles, computes wrong | `pl.const(value, dtype)` |
| conditionally-bound name (§4.3) | silent undef where Python raises `UnboundLocalError` | bind before the `if` or in every branch |
| declared `->` return contract produces no matching traced return (§4.4) | a return contract the body never honors | fix the annotation or return the declared value |
| bare facade class in a return contract mismatches the traced kind or arity (`-> pl.Scalar` over a tensor body; one slot over a two-value return) | the AST-era jit tolerated bare `-> pl.Tensor` as dead text — an unchecked contract | keep the bare class (kind-checked, shape/dtype derived from the body — `01-language.md` §9) or annotate fully; extern stubs must be concrete |
| non-bool `Scalar` in `and`/`or`/`not` | host float truthiness carrying into staged logic | `x != pl.const(0, pl.FP32)` |
| `bind_dynamic` on a static dim (§4.5) | an override the frontend never saw | declare `pl.dynamic` in the annotation |
| world contradiction at a call (`parameter 'x' is declared Scalar[int32] ... but received a host int`, §5.1) | AST-era code whose `pl.Scalar` annotations were decorative — the specializer specialized every scalar by value, so a value-varying scalar silently rebaked the artifact per value | re-annotate by the call-site rule: constant → plain annotation, varying → keep `pl.Scalar` and pass `pl.runtime(value, dtype)` at a host call or `pl.const(value, dtype)` in-trace |
| `pl.runtime(...)` inside a trace (`RuntimeError: Cannot unwrap annotation-only Scalar`) | the host-dispatch spelling used in the wrong lane — the value never becomes staged | in-trace calls pass a staged value: `pl.const(value, dtype)` |
| staged loop with an `else` suite | `else` collected unconditionally, contradicting skipped-by-`break` | restructure, or keep the predicate a host value |
| callee recursion | device IR has no recursion | unroll on the host (`pl.unroll`) or restructure |
| staged-ness flip under open control flow (§4.1) | post-join reads inline the literal instead of the variable | keep the name staged on every path, or host on every path |

Beyond the table, the model-zoo corpus (the `deepseek_v4_pro` /
`qwen3_14b` import probes) pinned the failure modes of *real* kernels
when they first met tracing: module-global callees, `DistributedTensorType`
call-site args, `pl.Array` parameters, `self.<sibling>` submissions, and
annotationless `spmd_submit` callees all work now and are pinned as
regression tests (`tests/ut/jit/test_corpus_gaps.py`) — if you hit one
of them, the pin names the owning surface.

## 8. What does not work

**Forbidden constructs.** The tracing frontend executes the kernel body;
constructs whose effect on the IR cannot be determined from the source
are rejected at rewrite time: `eval` / `exec` / `compile`,
`getattr`/`setattr` with non-literal names, metaclasses, `__class__`
reassignment, dynamic imports, `global` / `nonlocal`, `try` / `except`,
`del`, and bare `yield`. (`try`/`finally` and `with` are fine; the
verdict system is `01-language.md` §11 — a construct is judged on
observability, determinism, and bool-coercion, never by a list you must
memorize.)

**Parser-only diagnostics.** `pl.static_assert` and `pl.static_print`
are compile-time checks only the AST parser runs. Under tracing they are
silent runtime no-ops — kernels using them stay importable, but the
check no longer fires. Move the assertion into a host `if` that raises,
or into a verifier pass.

**Tracked gap.** The prefetch-context corpus trace (the upstream cadence
surface this repo has not adopted) is the one known frontend gap; it is
tracked in `tests/ut/_migration_xfail.txt`. The fix is an upstream
adoption, not a workaround in your kernel.

## 9. Verifying a migration

- The **frontend-equivalence oracle**: the conformance harness
  (`tests/ut/language/conformance/`) compares unsuffixed-marker output
  against the parser's, program by program. It is the machine-checkable
  form of "my kernel means what it meant before."
- **Round-trip verification stays parser-owned**: printed IR text parses
  back through `pl.parse` / `pl.loads`. If you need to eyeball what
  tracing collected, print the function — the text you get is round-trip
  dialect, and its `init_values=` / `pl.yield_` spellings are the
  printer's, correct by design. Do not feed printer text back into
  tracing.
- The gate pins `PYPTO_COLLECT=tracing`; locally, the same variable
  selects the frontend for a focused run.

## Appendix A — the assembly dialect (compiler developers)

The assembly dialect is a strict low-level *extension* of ergonomics,
collected by the same tracing frontend, used to state explicit IR
constructs in compiler tests. It obeys the same sound-typed value model —
it is not the round-trip dialect, and it is never printer input.

**A.1 Loop channels.** `pl.intrin.carry(...)` publishes the iter args a
loop will consume; `pl.intrin.yield_(...)` publishes a staged branch's
value; `pl.intrin.leak()` binds a staged body's result. Carries are
otherwise derived from rebinding (P2); the channels exist to state
dataflow the derivation would obscure in a fixture.

**A.2 View restamps.** `pl.intrin.transmute(producer, type)` states a
lowered-form view's type explicitly (P1) — the escape hatch for
flattened shapes, halved lanes, and MemRef-view fixtures where the
restamp used to ride on a bare annotation.

**A.3 Captured grid dispatch.** `pl.intrin.spmd_direct(callee, ...,
core_num=...)` captures the SPMD grid dispatch — the task-id and output
bindings are stated at the intrinsic, the `with`-form's carrier is
deferred to the body. `spmd_direct_call` is the uncaptured sibling.

**A.4 Declared memrefs.** Declared-memref and multi-buffer fixtures are
expressed at the operation: the slot a tile binds is selected inside the
transmute tile type, by a constant *or* a staged runtime index
(`buf[i % 2]` — the rotation of every multi-buffer kernel). Their
round-trip content (the printed annotation spelling) stays parser-owned
under the parser suites and the `*_parser` markers. Tile placement in
assembly fixtures states `target_memory=` exactly as ergonomics does
(§4.4).

## Appendix B — internal migration tracker

`docs/migration-todo.md` is the live tracker (task states, red/green
checks, discovery protocol); this table is its summary, and the todo
file wins when the two disagree. Rows flip to done as the tracker's
groups close. **Pending** rows are release blockers.

| Area | Content | Owning note |
| ---- | ------- | ----------- |
| `tests/ut/` | all fixtures, the conformance harness, the jit oracle suites | gate-green; the xfail manifest is the remaining-gap record |
| `examples/` | models (`03`–`09`), utils, the phase-fence pair | **done** — `parse_from_text.py` stays parser-era by design (guide §3) |
| `tests/st/` | distributed, runtime, codegen suites | closed — runtime MemRef slot subscripts have a tracing spelling; the Case device-run pipeline has both kernel-source lanes covered on device; the discovery grep is clean (remaining hits are comments); invisible to the gate (hardware-gated) |
| `docs/en/user/` | control-flow, functions, directives, getting-started | **done** — control-flow, functions/types/scopes/quickstart, directives, syntax, tutorials, and the `lower()`/`@pl.jit.graph` and eager-lane blocks migrated; the doc runner is 40/40 with `--check-parity` OK |
| `docs/zh/user/` | the mirror of the above | **done** — every user page migrated with its en counterpart (marked code byte-identical, runner-checked); parity is lint-enforced (`check_docs_en_zh_parity.py`) |
| `docs/en/dev/language/` | python-syntax, statements, functions, manual-dependencies pages | **done** — loops, allocations, jit/constants sections, and the `pl.at(predicate=)` / parse-time contract-check claims migrated; the `pl.at(predicate=)` surface itself restored by its gap fix |
| `docs/en/dev/passes/` | DSL input examples vs printed-IR text | **triaged** — stale DSL examples migrated (12/14/20/50 + prose fixes in 23/31/02/18, en+zh); remaining grep hits are printed-IR blocks and round-trip prose; the `@pl.jit.graph` example in `48-legalize_graph_boundary.md` migrated too |
| pypto-lib corpus | `deepseek_v4_pro`, `qwen3_14b` import probes | 37/38 modules failed at the 2026-09 probe; all fixed and pinned in `test_corpus_gaps.py` |
