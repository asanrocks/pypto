# Tracing Frontend Boundaries

Where a traced kernel stops being Python and becomes an `ir.Program`, what crosses each
boundary, and what the tracing frontend deliberately does not own.

> **Audience:** contributors to `python/pypto/language/tracing/` and anyone debugging a
> kernel whose IR does not match the source. The user-facing contract is
> [Functions and Programs](../../user/language/01-functions.md); the value model is
> [Values and Staging](../../user/language/07-staging.md).

## The five boundaries

```text
declaration  ──► trace ──► collected IR ──► program ──► compilation ──► dispatch
   annotations     tracer     ir.Function    assembly     ir.compile      jit cache
```

| Boundary | What crosses | Owner |
| -------- | ------------ | ----- |
| declaration → trace | the parameter worlds the annotation declares; a bare `pl.Tensor` leaves the shape unknown, so the trace defers to the call | `tracing/entry/` |
| trace → collected IR | one `ir.Function` per kernel *variant*; host values already folded, staged values as operands | collector |
| collected IR → program | the entry plus every callee its body reached, deduped and named program-unique | `_ProgramAssembly` |
| program → compilation | an `ir.Program` plus `RunConfig.compile_kwargs()`, handed to the same `ir.compile` the parsing frontend uses | `ir.compile` |
| program → dispatch | the bridged signature: shapes, dtypes, layouts, host-knob values, compile identity | the jit cache |

Two properties follow, and they are why the frontend can be swapped without touching the
compiler: **the tracing frontend owns exactly the trace and the assembly** — everything
after `ir.Program` is shared — and **a host *knob* never crosses a boundary as a value**;
it is folded into the program, so the value is part of the artifact's identity instead.

## The lanes

One object (`_JitSpecializer`) serves four lanes, selected by what the call site passes.
The lanes are the frontend's whole public surface:

| Lane | Trigger | Result |
| ---- | ------- | ------ |
| dispatch | a call with `torch.Tensor` arguments | bridge each tensor to an annotation-mode facade, assemble, compile, dispatch, cache |
| specialize | a call with facades or host scalars | the traced `ir.Function`; no compilation |
| callee | a call while another trace is active | a detached `ir.Call`; the callee joins the caller's assembly |
| declaration-time | `.fn`, or any structural use | the annotation-determined trace, materialized once and reused |

`compile_for_test(*args)` is the assembly path with the pass pipeline and no codegen; it
exists so structural assertions (lane parity, recipe equivalence) can compare programs. A
kernel with an unannotated or bare parameter has no declaration-determined trace, so every
call re-traces: the lane is chosen by what the declaration can already answer.

## Assembly and identity

The assembly is per program build, not global. It appends in first-use order, dedupes on
object identity and on a variant key — kernel identity, the entry kind the wrapper names,
the derived annotations, and the baked host values — so a helper called twice at the same
shapes and constants is one function, while the same helper at a second shape is a second
variant with a `_2` suffix. Two refusals live here rather than in the tracer:

- **Cycles.** A kernel reachable from its own body while its variant is still being traced
  has no finite traced form; the assembly names the cycle instead of re-descending.
- **Name collisions.** Every name a variant could take is reserved before registration, so
  a minted variant cannot land on a name the program already holds.

## What the frontend does not own

| Concern | Owner | Why it stays there |
| ------- | ----- | ------------------ |
| SSA, loop-carried channels, outlining | the pass pipeline | the trace emits pre-SSA IR; `ConvertToSSA` derives carries exactly as it does for parsed kernels |
| memory planning, addresses, buffers | the pass pipeline | a traced kernel is not allowed a second allocation story |
| code generation, PTOAS, artifacts | `ir.compile` and the backend | the frontend's output is an `ir.Program`, nothing more |
| cache keys, persistence, warmup | [Persistent JIT Cache](../10-jit-cache.md) | the dispatch lane calls it; it does not reimplement it |
| ABI, launch, device selection | [Entry Points](../08-entry-points.md) | host knobs are consumed at compile; what remains is the runtime signature |

`compile()` shows the last boundary concretely: it returns a frozen materialization whose
calling signature *is* the runtime ABI, so passing a value for a consumed host knob is an
arity error naming the knob, not a silent recompile.

## Migration and parity evidence

The migration claim is checkable because both frontends stop at the same place: the parsing
lane's `.lower(*args)` and the tracing lane's `compile_for_test(*args)` both return the
post-pass `ir.Program` before codegen. Equality at that boundary means "the port changed
the frontend, not the kernel".

Read the evidence for what it measures:

- **In scope:** structural equality of the lowered program for one kernel on both
  frontends, entry and derived callee names folded
  (`examples/tracing/advanced/08_migration.py`).
- **Out of scope:** codegen output, runtime numerics, and artifact identity. Example 08
  checks the numerics of that one kernel too, but equality there is not a general numeric
  claim; cache reuse and specialization need their own evidence.
- **Where the equivalence ends:** host-domain metaprogramming has no parser spelling — a
  `for` over a host plan is not IR — so those kernels cannot be written on the parsing lane
  at all, which is the reason to migrate rather than a caveat about it.

## See Also

- [Functions and Program Structure](03-functions.md) — the user-facing contract: parameter worlds, spellings, supported surface.
- [Tracing Collector](10-tracing-collector.md) — the rewriter and interpreter: how one source statement becomes an IR statement.
- [Entry Points](../08-entry-points.md) — what the compiled artifact exposes.
- [Persistent JIT Cache](../10-jit-cache.md) — the dispatch lane's cache, keys, and persistence.
