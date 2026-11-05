# Tracing Architecture

How executing a kernel becomes IR: the components, their owners, the invariants that keep
a trace type-sound, and the failure model.

> **Audience:** contributors to `python/pypto/language/tracing/`. The user-facing contract is
> [Functions and Programs](../../user/language/01-functions.md); the value model is
> [Values and Staging](../../user/language/07-staging.md); how one statement becomes one IR
> statement is [Tracing Collector](10-tracing-collector.md).

## The model

A traced kernel is ordinary Python. Collecting IR is a **side effect of running the body
once**: `pl.*` returns typed facades wrapping IR expressions, and the frame inserts them
where the source bound or used them. No value's type depends on "are we tracing?" — it
comes from the operation that produced it.

Two consequences decide the whole component layout:

- **Value IR is detached; structural IR is stateful.** An `ir.Call` or constant is a closed
  expression: build it, hold it by reference, insert it later. An `if`, a loop or a scope is
  *open over time*: enter it, emit statements into it as the host executes the body, close it.
  So value ops never self-emit, and frame-owned machinery owns the open constructs.
- **The collector is language-agnostic.** Rewriting, driving and stack discipline know
  nothing about tiles, tensors or scopes; a language frontend supplies the meaning. Swapping
  either side leaves the other untouched.

## The components

```text
kernel source ─► rewriter ─► generated function ─► interpreter ─► Frame / Sema ─► IR
   (python)      (type-blind)  (rt.* calls)      (host semantics)  (PTO meaning)
```

| Component | Owns | Does not own |
| --------- | ---- | ------------ |
| rewriter (`collector/rewriter.py`) | statement structure, evaluation order, `rt.*` routing, unwind guards for `break`/`continue` | value types, IR, host-vs-staged decisions |
| interpreter (`collector/interpreter.py`) | instrumented locals, per-branch locals maps, the active-construct stack, branch selection, host loops, one representative staged iteration, transfer targets | language types, builder operations |
| Sema (`tracing/semantics.py`) | one function-trace compilation: the `IRBuilder`, parameters, storage interning, the trace result, one fresh Frame | active-construct order |
| Frame | source name → mutable `ir.Var`, bindings, expression statements, returns, pending `iter_args`, staged construct creation | any ordering mirror — it queries the interpreter |
| constructs (`tracing/pto_constructs.py`) | staged region data: deferred `for` construction, lazy `while` staging, yield tracking, result bindings | stack order and lifetimes |
| execution (`tracing/execution.py`) | the one run path every producer shares: active-frame install and restore, callee adaptation for one execution, declared-return pinning | the unit lifecycle and its guards (the session's), frontend policy |
| entry (`tracing/entry/`, `decorators/`) | annotations and worlds, lanes, assembly, dispatch; see [Tracing Frontend Boundaries](08-tracing-frontends.md) | anything below the collected `ir.Function` |

Two stacks coexist by design and must not mirror each other: the interpreter's construct
stack (logical host order) and the builder's stack (physical IR insertion). One logical
construct may own zero, one or several physical scopes; the interpreter never predicts that
shape, and the Frame never rebuilds the logical one.

## Invariants

1. **One fact, one owner.** Sema = one compilation, Frame = one function execution, construct
   = one active construct, interpreter = host order. A question about active structure is
   asked of the interpreter, never reconstructed from state kept elsewhere.
2. **No type observation.** A construct's behaviour follows from the *kind* of the value it
   received (a facade versus a host value), not from inspecting builder or execution state.
   Mutating the builder while emitting is expected; *reading* it to decide a type is the bug.
3. **A Sema is one-shot.** Reuse after success or failure is refused, never reset — a failed
   trace produced no IR to reset into.
4. **Representative execution.** A staged loop's body runs once. Anything the body would
   compute differently across iterations must be staged, or it is baked (hence the two guards
   below).
5. **Result-producing closes are pending.** Close returns `AwaitingResultBindings` and the
   construct stays on the stack until the result bindings finish it; ordinary operations
   refuse a pending result construct before mutation.
6. **Failure is atomic.** An escaping exception skips every finalization — interpreter close,
   frame finish, staged construct close, `builder.end_*` — the caller drops that one Sema and
   propagates the *original* exception unchanged.

## The failed-trace model

The trace cannot half-succeed, so a refusal must be loud at trace time rather than silent in
the program. Two guards carry that weight, both raised while the body runs:

| Guard | Refuses | Where |
| ----- | ------- | ----- |
| loop-carried host state | a host value advanced inside staged control flow; one iteration's value would bake into all | Frame binding |
| one-sided host bind | a host name bound in one arm of a staged `if` and read after the join | Frame join check |

Both surface as `TypeError` naming the repair ([Values and Staging](../../user/language/07-staging.md)).
For a *staged* value the same asymmetry is legal and delegated: the join threads the other
path's value through, so a one-sided rebind lowers to a merge rather than an error.

Not every silent failure is closable this way. A plain helper that emits statements is not a
body, so its `pl.store` lands nowhere and nothing raises — that is a language-shape boundary
([Functions and Programs § reusing code](../../user/language/01-functions.md)), not a guard the
architecture can add without inventing statement capture for arbitrary Python calls.

## Where a defect belongs

Decide by the fact the bug is about, then look at the owner's tests:

| Symptom | Layer |
| ------- | ----- |
| wrong evaluation order, a rewritten name, a missing unwind | rewriter |
| wrong branch chosen, a host loop run twice, transfer hitting the wrong target | interpreter |
| a parameter's extent or storage, one compilation leaking into the next | Sema |
| a binding, a carried value, a result merge, a construct created with the wrong shape | Frame / constructs |
| wrong world for an argument, a callee not joined, a cache hit that should be a miss | entry |

## See Also

- [Tracing Collector](10-tracing-collector.md) — instrumentation, tracked bindings, and IR emission in detail.
- [Tracing Frontend Boundaries](08-tracing-frontends.md) — lanes, assembly, and what migration parity measures.
- `tests/ut/language/tracing/test_semantics.py` — the trace-level facts: staged `if`/`for`/`while`, guards, joins, spans.
- `examples/tracing/advanced/09_compiler_view.py` — the same pipeline printed for one kernel.
