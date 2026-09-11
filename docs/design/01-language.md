# The Kernel Language

- **Audience:** language designers — this document discusses the *design*
  of the language surface, not its use. The tracing architecture behind
  it is `00-tracing.md`; the collector machinery is `02-collector.md`.
- **Scope:** the ergonomics dialect's language design — semantics, typing
  discipline, and guardrails. This is the surface contract faced jointly
  by kernel authors and stdlib authors. The source rewriter, trace
  interpreter, and Sema are implementation details, sketched only in the
  appendix.
- **Position:** this document states *target* semantics. Decided but
  unimplemented capabilities are stated at target semantics and marked
  deferred (§15 collects them).
- **Self-contained:** no other design document is required reading; every
  concept is defined at first use.

## §1 The model, in one sentence

A PyPTO kernel is ordinary Python. The stdlib `pl.*` yields IR nodes as
the kernel executes; **collecting IR is a side effect of running it**.
The language's entire responsibility is to make that side effect
observable while staying honest about types.

The one conceptual fork is between staged and host values:

- A **staged value** is a facade around an IR expression; its host value is
  unknown at trace time, and its value is the expression it wraps.
- A **host value** is an ordinary Python value, known at trace time.

From this fork, three ground rules (§2) derive every language decision in
this document. The reader needs only this: **what is kept is the host's
control flow, the host's typing rules, and the host semantics of host
values; what is added is value-directed staging and a small number of loud
errors.**

```python
import pypto.language as pl

# This is an ordinary Python function. Running it collects IR as a side effect.
@pl.jit
def tile_add_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [64, 64])   # pl.load yields a staged Tile
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.add(tile_a, tile_b)         # pl.add yields a staged Tile
        pl.store(tile_c, [0, 0], c)             # a bare side-effect statement
    return c
```

There is no dedicated DSL syntax — no `if_`/`while_`/`let`/`reassign`.
Bare `if`, bare `while`, and bare assignment serve both host and staged
execution.

## §2 Ground rules

**(1) A sound-typed subset.** PyPTO is a sound-typed subset of the host
language: no non-host syntax extensions, no type lies. Asked "is `5`
staged?", the answer may not be "yes, because we are tracing" — `5` is a
host int, and its properties are decided by its type. Deciding participation
in staging from execution context is a type lie.

**(2) Typing flows from values, not annotations.** A value's type is
decided by the operator that produced it. The result type of
`pl.add(t, 1)` comes from `pl.add`, regardless of any annotation on `t`.

**(3) Value-directed staging.** Whether an operation or construct executes
on the host or stages IR is determined by the recognized kind of its value.
"Is this value staged?" is answered by whether its type belongs to the
staged-value types. This rules out "we are tracing, so everything is
staged" — that is a type lie and implies a silent retype.

A design principle runs through everything: **loud, not silent.**
Wherever plain Python would fail silently in an observability context,
the dialect substitutes a loud error. Every guardrail in §4/§5/§7/§9 is
an instance; §12 collects them.

```python
# Ground rule (3) in action: participation in staging is read from the value kind.

n: pl.Scalar[pl.INT32]   # annotation is cosmetic (see §4); it does not make n staged

@pl.jit
def f(x: pl.Tensor, n: pl.Scalar[pl.INT32]) -> pl.Tensor:
    # n's type is Scalar — a staged-value type — so n is staged,
    # regardless of "are we tracing right now?".
    step = n * 2          # step is Scalar (operators preserve staging, §3)
    # A host int, by contrast, is never staged:
    tile_count = 4        # plain Python int — host value, obeys host rules
    return x
```

## §3 Values: staged and host

A staged value is produced by a stdlib operator: the operator builds an IR
expression, wraps it in a facade of a staged-value type, and returns it.
**The type is fixed when the operator produces the value** — the operator
knows its own output shape and dtype.

Operator chains preserve staging: operating on staged operands yields
staged results. If `a` is staged, `a + b` is staged. The marker survives
expression trees of any depth.

Host values strictly follow host rules — their arithmetic, comparisons,
and truthiness are exactly Python's. They produce no IR.

**The only automatic host-literal lifting happens at the stdlib operand
boundary**: when a host value is passed as an operand, the operator lifts
it to a typed constant IR expression at that boundary, following the
host's promotion rules. No other location converts values — not bindings,
control flow, or joins (see typed staged initialization, §4).

> **Stdlib author contract.** These rules assign operation semantics and
> result typing to the stdlib. When designing a `pl.*` operator, ensure:
> (a) it yields a typed staged value (the expression already carries the
> output type); (b) staging propagates through operator chains; (c) a value
> whose type is coerced to bool must *raise* (§5); (d) host-literal lifting
> happens *only* at the operand boundary. Break any one and the
> typing-from-values chain is broken.

```python
@pl.jit
def mul_by_n(x: pl.Tensor, n: pl.Scalar[pl.FP32]) -> pl.Tensor:
    t = pl.load(x, [0, 0], [64, 64])     # staged Tile
    out = pl.mul(t, n)                   # both operands staged → staged Tile

    # n * 2.0: n is Scalar, so the result is Scalar. The host 2.0 is
    # lifted to a typed Const inside Scalar.__mul__ (operand boundary);
    # nothing elsewhere converts it.
    two_n = n * 2.0

    pl.store(out, [0, 0], x)
    return x
```

To bind a boolean host value as staged state, use
`pl.const(False, pl.BOOL)` with an explicit dtype.

## §4 Typing from values, and typed staged initialization

The direct consequence of ground rule (2): **a value's type is fixed when
the operator yields it.** Annotations on locals are cosmetic — they serve
readability and never flow into IR.

From this follows the first loud error: a host literal cannot initialize
staged state that must survive a control-flow join. The user writes
`pl.const` instead. This is the rule for **typed staged initialization**.

```python
@pl.jit
def acc_loop(x: pl.Tensor, n: pl.Scalar[pl.INT32]) -> pl.Tensor:
    # Correct: pl.const yields a typed staged Scalar wrapping a Const node.
    acc = pl.const(0, pl.FP32)
    for i in pl.range(n):                    # pl.range is always staged (§8)
        tile = pl.load(x, [i, 0], [1, 64])
        acc = pl.add(acc, tile[0, 0])        # rebind reuses the mutable ir.Var (§7)
    return x
```

```python
# Same kernel, but the accumulator starts as a bare literal — a loud error.

@pl.jit
def acc_loop_bad(x: pl.Tensor, n: pl.Scalar[pl.INT32]) -> pl.Tensor:
    acc = 0                                  # → loud error: missing typed staged initialization
    for i in pl.range(n):
        tile = pl.load(x, [i, 0], [1, 64])
        acc = pl.add(acc, tile[0, 0])
    return x
```

This guardrail is a **correctness guardrail**, not mere typing
discipline, because of the binding model (§7): every assignment of a
staged value updates a mutable `ir.Var`, but a host assignment produces no
staged local. So `acc = 0` is a host binding with no staged local; inside
the staged loop, the read of `acc` sees the inlined literal `0`, the rebind
is not self-referential, and `ConvertToSSA` derives **no loop-carried
value**. The kernel compiles and every iteration reads the initial value
— **silently computing the wrong result**. `pl.const(0, pl.FP32)` creates
a staged local; the read and the rebind use the same `ir.Var`, and the pass
derives the loop-carried value. The loud error is what prevents this
compile-but-wrong result.

There is also a typing-level reason: automatically lifting `acc = 0` to a
typed zero would be a retype (an FP32 accumulator and an int zero are not
type-compatible). A scope-level opt-in lifting escape hatch is not
provided (deferred, §15).

The guard is symmetric: **staged-ness is stable while staged control flow
is open**. The tracer also rejects the flip in the other direction —
demoting a staged local to a host literal under an open `if`/loop — and a
name staged on one branch of an `if` but host on the other is rejected at
the merge. Both flips fail for the same reason: a host bind emits no IR,
so reads after the join or loop back-edge would inline the literal
instead of re-reading the mutable `ir.Var`, silently computing the wrong
result. Outside staged control flow the rule does not apply — sequential
host binds follow host rules exactly (§3).

## §5 Control flow: one syntax, two worlds

Bare `if` / `while` / `for` serve both host and staged execution.
**Value-directed staging uses the kind of the predicate or iterable** — not
dedicated syntax. This is why the dialect has no `if_`/`while_` constructs:
the value kind lets bare syntax carry both duties.

- **Staged control flow** collects *all paths*: both branches of an `if`
  are recorded; a staged `while`/`for` traces one representative body
  iteration while the loop structure is recorded as an IR node.
- **Host control flow** follows Python exactly: only the matching branch is
  entered, and a host range actually iterates N times.

**Boolean coercion** is both the staging decision entry point and the loud
error boundary: coercing a staged value to bool is a loud error. This
distinguishes universally safe constructs from constructs that need
rewriting to be trace-safe — it is the dividing line between rewrite-safe
and trace-safe (systematized in §11).

```python
@pl.jit
def k_split(x: pl.Tensor, k: pl.Scalar[pl.INT32]) -> pl.Tensor:
    # k is Scalar → (k == 0) is Scalar (§6) → staged if: BOTH branches recorded.
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [32, 32])

    # A host predicate (plain Python int/bool) → host if: one path only.
    if 4 > 2:
        b = pl.load(x, [0, 0], [8, 8])

    # A host range → host execution unrolls it at trace time (§8): no loop IR emitted.
    total = 0
    for j in range(3):
        total = total + j

    # A staged range (pl.range) → one representative iteration: a real loop IR node (§8).
    acc = pl.const(0, pl.FP32)
    for i in pl.range(k):
        acc = pl.add(acc, pl.const(1, pl.FP32))
    return x
```

The boolean-coercion guardrail covers every site that implicitly takes a
value's truthiness:

```python
# Every site below coerces a value to bool. A staged value reaching it is a loud error.

if some_scalar:           # → loud error: coerce staged value to bool
    ...
assert some_scalar        # → loud error: assert coerces to bool
while some_scalar:        # OK — 'while' is rewritten (predicate redirected, §8)
    ...
x = a if some_scalar else b   # → loud error: ternary predicate coerces to bool
```

The difference: `while` and `if` are *rewritten* into trace-safe form, so
a staged predicate routes through the trace interpreter's mode-aware resolver.
`assert`, ternary, and comprehension predicates *cannot* be rewritten
trace-safe, so a staged value reaching them is rejected.

## §6 `and` / `or` / `not`: values, not control flow

In stock Python, `and`/`or`/`not` short-circuit and return an operand. In
PyPTO, when the operands are staged they yield *first-class logical IR
nodes* — `And`/`Or`/`Not` — rather than being lowered to conditionals.
Ground rule (2) again: the operator decides the type, and a logical
operator over staged values yields a staged `Scalar`.

Two rules:

- Operands are normalized at the operand boundary (§3), so both sides of
  `a and b` are already in canonical staged form.
- A non-bool `Scalar` operand is a **verifier error**: the host's float
  truthiness must not carry over to staged values. To express "non-zero",
  write `x != 0`.

A resulting logical value feeds an outer `if` directly: `if (a < b) and
(c < d):` produces `IfStmt(cond = And(...))`, not nested `IfStmt`s.

```python
@pl.jit
def both(a: pl.Scalar[pl.FP32], b: pl.Scalar[pl.FP32]) -> pl.Tensor:
    x = pl.load(...)

    # Staged operands → a first-class And node (NOT lowered to a conditional).
    if (a < b) and (b < pl.const(10, pl.FP32)):
        pl.store(x, [0, 0], ...)

    # Non-bool Scalar operand → verifier error. Write the explicit test:
    if a != pl.const(0, pl.FP32):      # OK: a != 0 yields a bool Scalar
        ...
    # if a:                            # → verifier error: a is a non-bool Scalar
    #     ...
    return x
```

## §7 Binding, scopes, convergence

The binding model uses mutable locals: each staged local maps to one
mutable `ir.Var`; its first staged definition emits a `LetStmt`, and later
staged definitions emit an `AssignStmt` to the same variable. A rebind
reuses that variable by name within the frame — **unless the value's type
changed**, in which case the rebind starts a fresh `let` for the name (the
relaxed reassignment rule; #642's outright rejection is retired). The
split is not stylistic: assign-through pins the Var's declared type to
the first binding, so every later trace-time type query on the name
would read the stale type — a broadcast against an `[8,1]`-typed local
fed an `[1,8]` value silently widens. Both frontends compare through
the same typing seam, so they let and reject identically. Aliasing is
recovered by copy folding in the passes, not by a frontend special
case. This is ground rule (1)'s single-mechanism preference.

From this follows an extremely simple property: **convergence across
branches needs no merge syntax.**

```python
@pl.jit
def merge_example(x: pl.Tensor, k: pl.Scalar[pl.INT32]) -> pl.Tensor:
    # Both branches bind the SAME name 'a' → the SAME mutable ir.Var:
    # no merge construct, no return_vars, no let/reassign needed.
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])
    else:
        a = pl.load(x, [0, 0], [32, 32])

    # 'a' here is the converged value. ConvertToSSA synthesizes a phi
    # from the two branches' assignments to the shared variable. The user writes nothing.
    pl.store(a, [0, 0], x)
    return x
```

Both branches update the same variable; the `if` closes with no explicit
merge variables, and `ConvertToSSA` synthesizes the phi. This is
**convergence through mutable locals and SSA promotion**: per-branch locals
are merged into the enclosing mapping and convergence is the pass's job.
The dialect has **no merge channel at all** — including inline `return`
(§9), which is just a binding of a synthetic name converged by the same
mechanism.

One loud error lives in the convergence model — the **conditionally-bound
name**. A name bound on only *some* branches of a staged `if` and not
before it has no value on the missing branch. The host language rejects
the same program (`UnboundLocalError`); the dialect inherits that
discipline: reading a conditionally-bound name is a loud error. Fix: bind
the name before the `if`, or in every branch.

```python
@pl.jit
def bad_merge(x: pl.Tensor, k: pl.Scalar[pl.INT32]) -> pl.Tensor:
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])   # 'a' bound only on the then-branch
    # 'a' is absent on the else-branch (no else here) and unbound before the if.
    pl.store(a, [0, 0], x)                 # → loud error: a is conditionally bound
    return x
```

```python
# Fix: bind 'a' before the conditional (or in both branches).

@pl.jit
def good_merge(x: pl.Tensor, k: pl.Scalar[pl.INT32]) -> pl.Tensor:
    a = pl.load(x, [0, 0], [8, 8])         # bound before the if — has a value
    if k == 0:
        a = pl.load(x, [0, 0], [64, 64])   # rebind reuses the mutable ir.Var
    pl.store(a, [0, 0], x)
    return x
```

> **Single-branch definitions inside a loop are legal.** A branch that
> binds a name inside a loop body while another branch continues to the
> next iteration is not an error: the missing branch reads the previous
> iteration's value. That is loop-carried behavior, owned by `ConvertToSSA`,
> not a frontend guardrail.

Finally, **parameter passing is substitution, not assignment**. An inline
callee's parameters bind directly to the caller's argument values (no new
staged local, no `AssignStmt`). This rule constrains *user-written* bindings; call
semantics are separate (§9).

## §8 Loops: one representative iteration and derived loop-carried values

Two rules:

- `pl.range` / `pl.parallel` / `pl.pipeline` are *always* staged — even
  with host int bounds; they are trace iterables. Host `range` and host
  lists are host values — they actually unroll at trace time (trace-time
  execution determines the IR, §9 of the architecture document).
- A staged loop traces one representative body iteration. In the
  ergonomics dialect, **loop-carried values are never declared —
  `ConvertToSSA` derives them from assignments to mutable locals.**

```python
@pl.jit
def reduce_sum(x: pl.Tensor, n: pl.Scalar[pl.INT32]) -> pl.Scalar[pl.FP32]:
    # pl.const → a typed staged constant → a staged local (§4).
    acc = pl.const(0, pl.FP32)

    # pl.range(n) is staged → one representative iteration.
    for i in pl.range(n):
        t = pl.load(x, [i, 0], [1, 1])
        # acc = pl.add(acc, ...) — rebind the SAME mutable ir.Var. The
        # assignment reads and updates acc; ConvertToSSA derives the
        # loop-carried value from this dependency; the user declares nothing.
        acc = pl.add(acc, t[0, 0])

    return acc
```

This rule constrains the ergonomics dialect only. Compiler tests may opt
into the tracing-side assembly extension to state low-level IR channels
(`pl.intrin.carry`/`pl.intrin.yield_`/`pl.intrin.leak`, explicit phi
results, `strict_ssa`). Assembly is still executed by tracing and must
obey the sound-typed value model; it is not the round-trip dialect
(§14). The tracing surface of `pl.range`/`pl.parallel`/`pl.pipeline`
deliberately has no `init_values=` parameter — that spelling exists only
in printer/parser-owned round-trip text.

A staged `while` is analogous: a staged predicate (`pos < seq_len`) makes
the loop trace one representative body iteration. The predicate is
rewritten into a mode-aware resolver — host predicates iterate normally;
staged predicates return true once and false the second time (the body runs
exactly once). The predicate is an *expression over mutable locals* in the
IR — the device re-reads it every iteration; the loop-carried update is
still derived by the pass.

```python
@pl.jit
def while_kernel(seq: pl.Tensor, seq_len: pl.Scalar[pl.INT32]) -> pl.Tensor:
    pos = pl.const(0, pl.INT32)
    while pos < seq_len:              # staged predicate → one representative iteration
        chunk = pl.load(seq, [pos, 0], [16, 64])
        pos = pl.add(pos, pl.const(16, pl.INT32))   # self-referential store
    return seq
```

**Loop `else` belongs to host loops only.** Python's loop-`else` means
"skipped by `break`". A staged `break` suppresses the native keyword, and
a staged loop always exits through its predicate after one representative
iteration — the `else` would be collected unconditionally, contradicting
Python's semantics. Staged loops with an `else` suite are rejected at trace
time. Restructure the loop, or keep the predicate a host value.

## §9 Calls: sub-kernels and inline callees

A callee is either a *sub-kernel* or an *inline callee*. The distinction
is a property of the **callee**, decided at its definition at trace time —
not at the call site, not by a backend pass. Trace-time execution
 determines the IR.

- **Sub-kernel:** traced independently (its own frame, its own function);
  the call site yields a call node.
- **Inline callee:** its body executes in the caller's traced context —
  no call node, no separate function.

An inline callee's parameters are *substitution* (§7): parameter names
bind directly to the caller's argument values. This is what lets
operand-local folds observe the *caller's* values.

An inline `return e` ≡ *a bind of a synthetic name* plus a *dead flag* —
subsequent collection events on the current path are no-ops until the
current active construct exits. The value at the call site is that
synthetic name's final binding at frame exit, converged through mutable
locals and SSA promotion (§7). **Inline `return` therefore needs no merge
channel.**

**Recursion is a loud error** — device IR has no recursion.

```python
# A sub-kernel callee (default): traced separately, yields a call node.
@pl.function
def softmax_tile(t: pl.Tile) -> pl.Tile:
    m = pl.max(t)
    e = pl.exp(pl.sub(t, m))
    return pl.div(e, pl.sum(e))

@pl.jit
def parent(x: pl.Tensor) -> pl.Tensor:
    t = pl.load(x, [0, 0], [64, 64])
    # A plain Python call. The callee is dispatched at its definition:
    # softmax_tile is a sub-kernel → this yields a call node in the IR.
    out = softmax_tile(t)
    pl.store(out, [0, 0], x)
    return x
```

### Multi-value returns keep Python tuple semantics

Multi-value return keeps Python tuple semantics; it is not a special
value whose staging is determined by execution context. The tuple is a
host container whose staged elements are returned as separate IR values:

```python
@pl.function
def pair(x: pl.Tensor) -> tuple[pl.Tensor, pl.Tensor]:
    a = pl.add(x, x)
    b = pl.sub(x, x)
    return a, b
```

The source rewriter must not reinterpret CST shape as a different value
type. At the kernel boundary, the tracing frontend and the AST parser
lower the elements to the IR's multi-value `ReturnStmt`; neither frontend
creates a fictional staged-tuple value. They must agree on the IR
dataflow, while only the AST parser owns printer-output round trips.

> **Deferred.** Inline-callee trace-time dispatch is not implemented
> (§15). Until it lands, callees marked `inline=True` are traced as
> sub-kernels and spliced by the backend inlining pass; the target
> semantics above take effect when inline dispatch returns.

## §10 Trace timing: eager and lazy

The same trace driver runs eagerly (at decoration, with staged arguments
synthesized from annotations) or lazily (at call, with the real
arguments). **Eager ⊂ lazy** — the same driver; eager is a pre-execution
with synthesized arguments.

**The signature is the inspection contract:**

- **Parameters must be annotated.** Eager tracing and dispatch depend on
  signature inspection; a missing parameter annotation is a loud
  decoration-time error (no `UnknownType` guessing — that would be a
  type lie).
- **Locals are never annotated** (typing flows from values, §2).

Lazy is *forced* when the IR depends on call-time information —
`Callable` parameters, call-time shapes of torch arguments, dyn-dims that
need real arguments. `@pl.jit` *bakes* Python `int`/`float` arguments as
`Const` (the cache key includes the value); tensor arguments become
static-shape parameters; dyn-dims appear only in annotations and are
*shared by name* — two parameters both annotated `N` share one `N`
extent symbol.

```python
# Eager-collectable: IR fully determined by annotations. Traced at decoration.
@pl.function
def eager_kernel(x: pl.Tensor[[64, 64], pl.FP32]) -> pl.Tensor[[64, 64], pl.FP32]:
    return pl.add(x, x)

# Must-defer: a Callable param — IR depends on the call-time lambda.
@pl.function
def lazy_kernel(x: pl.Tensor, epilogue) -> pl.Tensor:
    y = pl.add(x, x)
    return epilogue(y)        # IR not determined until the actual lambda arrives

# @pl.jit specializes: Python scalars baked as Const; tensors → static shapes.
@pl.jit
def jit_kernel(x: pl.Tensor, scale: float) -> pl.Tensor:
    return pl.mul(x, pl.const(scale, pl.FP32))   # 'scale' baked per call
```

## §11 The static subset: three axes

A construct's *verdict* is not decided by an enumerated list — it is
derived from three orthogonal axes. Adding a construct computes its
verdict; it is never appended to a table.

- **Observability** — can the host observe the construct? If not
  (assignment, expression statements, block boundaries), it *must be
  rewritten*; if yes, it is a rewrite-safe candidate.
- **Determinism** — is the construct's effect on the IR fixed by the
  source? If not (exception paths, dynamic strings, dynamic attribute
  names, class-shape mutation, bindings destroyed or rebound outside the
  scope), it is *rewrite-unsafe* (forbidden).
- **Bool-coercion** — does the construct coerce a value to bool? If so
  and a staged value reaches it, it is either *rewritten* trace-safe
  (`if`/`while`/`and`/`or`/`not`) or it triggers a *loud error*
  (rewrite-safe but not trace-safe: comprehension predicates, `assert`,
  `match` scrutinees).

```python
# --- Must-rewrite (unobservable — the instrumentation boundary): assignment,
#     augmented/unpacking assignment, expression statements, block boundaries,
#     walrus :=. The host cannot hook frame-local assignment or block exits.

a = pl.add(x, y)          # assignment — rewritten (redirected)
a += pl.const(1, pl.FP32) # augmented assignment — rewritten
a, b = pair               # unpacking — rewritten
pl.store(a, [0, 0], x)    # expression-statement side effect — rewritten (§13)

# --- Must-rewrite (bool-coercing, trace-safe): if/else, while, and/or/not.
#     'for' is rewritten for target-routing (not bool-coercion).

if k == 0: ...            # if/else — rewritten
while pos < n: ...        # while — rewritten
if a and b: ...           # and/or/not — rewritten

# --- Rewrite-safe (observable + deterministic, pass through): functions,
#     if/for/while WHEN their predicate/iterable is a host value, with, attribute
#     access on literal names, calls, literals, return/break/continue,
#     pl.* calls, annotations (cosmetic for locals), comprehensions building
#     host lists, try/finally, match (host scrutinee), user lambda.

for j in range(3): ...    # host iterable → host unrolls, passes through
if 4 > 2: ...             # host predicate → one path, passes through

# --- Rewrite-unsafe (non-deterministic — rewrite-time error): eval/exec/
#     compile, getattr/setattr with non-literal names, metaclasses,
#     __class__ reassignment, dynamic imports, global/nonlocal, try/except,
#     del, bare yield.

# --- Rewrite-safe but NOT trace-safe (bool-coercing — loud error when a
#     staged value reaches it): a staged value in a comprehension predicate,
#     an assertion, or a match scrutinee. (Comprehensions whose elements are
#     staged values and which build a host list are fine.)
```

The bool-coercion axis is §5's guardrail generalized: inside a
rewrite-safe construct, a staged value that drives a decision is coerced
and rejected loudly — so rewrite-safe constructs never need defensive
rewriting.

## §12 Guardrail summary

The loud errors scattered through this document, collected, each with the
silent failure it prevents:

| Guardrail | Where | Silent failure prevented |
| --------- | ----- | ------------------------ |
| staged value coerced to bool | §5 | a staged value entering a host conditional → only one path is collected |
| missing typed staged initialization | §4 | host binding has no staged local → loop body reads the inlined literal → no loop-carried value derived → compiles but computes the wrong result |
| conditionally-bound name | §7 | a name with no value on some branch → silent undef instead of the host's `UnboundLocalError` |
| staged early `return` | §5/§9 | retired — per-branch `ReturnStmt` is recorded faithfully; the passes absorb the join (`normalize_return_order` + DCE) |
| non-bool `Scalar` as `and`/`or` operand | §6 | host float truthiness silently carrying into staged values |
| callee recursion | §9 | device IR has no recursion |
| staged loop `else` | §8 | `else` collected unconditionally, contradicting skipped-by-`break` semantics |

Meta-guardrail: **every silent path that could yield
"compiles but computes the wrong result" is closed by a loud error.**
When extending the dialect, the first question is not "is it
convenient?" but "which new silent-failure path does it open, and is that
path closed?"

## §13 Bare statements and side effects

An expression-statement side effect (e.g. `pl.store(...)` alone on a line) is
invisible to the trace interpreter — the host has no hook for an expression
statement. It is rewritten into an explicit emit; the collector wraps the
yielded call in an `EvalStmt`. A *pure* yielded call (its result threaded
into a binding) needs no `EvalStmt` — it is assigned to the staged local at
the binding boundary.

```python
pl.store(t, [0, 0], x)           # expression statement → rewritten, wrapped in EvalStmt

t = pl.load(x, [0, 0], [64, 64])   # pure yielded call threaded into a bind →
                                   # no EvalStmt (assigned to the staged local at bind)
```

## §14 The three dialects

PyPTO has three dialects with different audiences and frontend ownership:

- **ergonomics** — bare Python and ergonomic `pl.*` operations; the
  user-facing kernel language; collected by tracing.
- **assembly** — a strict low-level extension of ergonomics; compiler
  developers use it for explicit IR constructs (loop-carried channels,
  yields, strict-SSA fixtures); also collected by tracing and must remain
  type-sound.
- **round-trip** — the lossless Python representation emitted by the C++
  IR printer; consumed by the AST parser through `pl.parse`, `pl.loads`,
  `@pl.function_parser`, and `@pl.program_parser`; never a tracing input
  and not equivalent to assembly.

```text
ergonomics --tracing--> C++ IR
assembly   --tracing--> C++ IR
C++ IR --printer--> round-trip text --AST parser--> C++ IR
```

The dialect boundary is frontend ownership, not spelling. The loop-carried
channel spellings differ by owner: the assembly dialect states channels
through `pl.intrin.carry`/`pl.intrin.yield_`/`pl.intrin.leak` (executed
by tracing), while `init_values=`/`pl.yield_` on `pl.range`-style loops
exist only as printer output consumed by the AST parser (including the
`@pl.jit` path, which collects through `pl.parse`). The parser must not
learn tracing semantics, and the tracer must not consume printer output
as a round trip.

Unsuffixed `@pl.function`/`@pl.program` are tracing-owned;
`@pl.function_parser`/`@pl.program_parser` are permanent parser-owned
markers. Ergonomics deliberately leaves explicit SSA dataflow to the
passes; assembly may state that dataflow explicitly, but only through a
tracing-side extension. Round-trip text remains an internal serialization
format and the AST parser is its permanent assembler.

## §15 Deferred

Decided, not yet implemented. The rest of this document states them at
target semantics.

- **Inline-callee trace-time dispatch** (§9): currently inline callees
  trace as sub-kernels and the backend inlining pass splices them. On
  landing: substitution parameters, synthetic-name return bind + dead
  flag, recursion guard; the inlining pass retires.
- **`tensor.dim` extent-symbol fold going uniform**: depends on inline
  dispatch.
- **Scope-level opt-in host-literal lifting** (§4's typed staged
  initialization escape hatch).
- **Spans**: source-position capture for diagnostics (Tier 1 is realized
  — function/program spans; Tier 2 construct positions are deferred; see
  the architecture document).
- **Span Tier 2** construct-position propagation.

## Appendix A — the framework in one glance

This document covers the language contract. The three components that
implement it, one sentence each:

- **Rewriter** — a source-to-source rewriter over the decorated kernel
  body, converting bare statements into event-protocol calls. Type-blind;
  observation-only.
- **Trace interpreter** — drives normal Python execution, owns
  instrumented locals and the active-construct stack, and handles value-
  directed staging and exits. Not IR-aware.
- **Sema** — the plugin the trace interpreter consults; produces the
  structural IR (control-flow nodes, `LetStmt`/`AssignStmt`, `EvalStmt`)
  and answers whether values participate in staging at control-flow
  boundaries.

The consequence worth emphasizing: **the frontend is extremely thin — all
graph-wide dataflow (SSA promotion, phi synthesis, loop-carried
derivation, copy-folding, break/continue lowering) is absorbed by the IR
and its passes.** That is what makes the dialect possible: all it adds is
one marker and a few loud errors.
