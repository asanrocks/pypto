# Authoring Recipes

`@pl.authoring.recipe` — the general authoring tier: programs over an authoring context.

A construct recipe ([06-construct-recipes.md](06-construct-recipes.md)) is one
specialized protocol: a generator whose single `yield` is a body boundary. The
authoring runtime beneath it is more general — a **recipe** is an ordinary
function that receives an authoring context and composes *tracing effects*:
push a function, collect a body into it, finalize it, emit a call, splice a
body into the current trace, open regions. The tier's job is to make those
effects composable without exposing the tracer (no builder, no `PtoFrame`, no
interpreter):

```python
import pypto.language as pl


@pl.authoring.body
def squared(x):
    return pl.mul(x, x)


@pl.authoring.recipe
def scaled_square(ctx, body, x, k):
    fn = ctx.push_function(
        pl.authoring.FunctionContract(name="scaled_square", function_type=pl.FunctionType.Inline)
    )
    collected = fn.collect_body(body, args=(x,))
    artifact = fn.finalize(collected)
    return ctx.emit_call(artifact, x) if k == 1 else artifact
```

## The Layering

```text
kernel author          recipes, pl.authoring.body compositions, pl.* users
    │ composes
authoring tier          @pl.authoring.recipe / @pl.authoring.construct + the context views
    │ drives
tracer                  rewriter, interpreter, Sema/Frame
    │
IR builder              closed statement/scope vocabulary
```

`@pl.authoring.construct` is a *constrained recipe protocol* — one region, one body
boundary, driven by the trace protocol. `@pl.authoring.recipe` is the unconstrained
form: any sequence of effects, any number of bodies, no required `yield`. Both
receive a view over the same authoring session; a recipe context can do
everything a construct context can (region composition) plus function and body
effects.

The vocabulary stays closed where it must: recipes compose existing IR
constructs and existing function kinds. A recipe that needs a new lowering
shape is a framework change first.

## The Import Root

`pypto.language.authoring` is the tier's canonical namespace — the import
root for framework authors. The package splits by responsibility:

```text
authoring/
├── contracts.py    FunctionContract, ParamSpec, the artifacts
├── session.py      the authoring session and the function-unit lifecycle
├── construct.py    @pl.authoring.construct and the construct protocol
├── recipe.py       @pl.authoring.recipe, @pl.authoring.body, the composition entry points
└── dispatch.py     the dispatch effect (submit / deps lowering)
```

The package is the tier's **sole import root**: nothing of the authoring
surface is re-exported at the top-level `pl` namespace. The root stays
the kernel author's; framework authors (and the stdlib, which is the
tier's reference client, never a sibling with private access) import
here.

## The Surface Classification

The authoring layer's boundary is stated as a classification of the whole
public language surface, using one rule:

> **Classify by what a feature lowers to, not by what it returns.** If its
> meaning is *structure* — a region, loop, function, call, submit, binding,
> terminator — it belongs to the authoring family. If it lowers to a
> *dataflow op over staged values* — one op node consumed in expression
> position — it belongs to the value tier. Everything else is a frontend or
> a host-runtime API.

The capability families the authoring layer owns:

```text
region       scopes, carriers, with/for recipe lifecycle
body         collect, splice, inline, InCore placement
function     function units, contracts, finalization, calls
dispatch     submit, SPMD submit, dependencies, task captures
control      return, yield, break, continue
metadata     function attributes, cache policy, dump state
binding      task IDs, loop variables, storage, carried results
```

The classification of the public surface:

| Surface | Lowers to | Classification |
| ------- | --------- | -------------- |
| `pl.at`, `pl.scope`, `pl.manual_scope`, `pl.graph`, `pl.cluster` | scope | authoring — region effect (a recipe) |
| `pl.spmd` (both spellings), `pl.split_aiv` | dispatch scope + carrier / SplitAiv region | authoring — region effect (a recipe) |
| `pl.authoring.body(...)` `.splice(...)` | splice | authoring — body effect (the public splice spelling) |
| `pl.authoring.body` `.inline` / `.incore` | function unit + call | authoring — callee composition |
| `pl.submit`, `pl.spmd_submit` | `ir.Submit` — a structure statement | authoring — dispatch effect (the stdlib spellings are policy wrappers over the one effect; `ctx.submit` is the recipe spelling) |
| `pld.notify` / `wait` / `defer_wait` | sync statements | value tier — distributed ops (callable from any recipe body as ordinary ops; no dedicated authoring capability exists, and adding one is a framework change that needs a real recipe use case) |
| `pl.func_attr` | function attrs (prologue directive) | metadata effect (a `FunctionContract` carries attrs; the marker is the source spelling) |
| `pl.set_cache_policy` / `dump_tag` / `no_dep` | Var metadata riding values | metadata markers (the tier consumes their facts) |
| `pl.bind_storage` | storage declaration + base binding | binding effect |
| `pl.range` / `parallel` / `unroll` / `pipeline` | `ForStmt` primitives | framework-owned entries (host-execution parity); the tier's `ctx.open_loop` composes a `ForStmt` region, so recipes author real loops while the stdlib entries keep host behavior |
| explicit carry / result bindings | SSA mediation owned by the Frame | framework primitive — the pass-side inference lane |
| `if` / Python `while` | interpreter-native control flow | framework primitive — the control effects route into them but cannot create them |
| scope kinds, statement kinds, function types | closed IR vocabulary | framework-owned |
| `tensor` / `tile` / `unified` / `array` / `system` / `prefetch` ops (incl. allocations and block-identity queries) | op nodes in expression position | **value tier — never migrates** (the facade algebra and op emission own them) |
| `pld.remote_load` / `put` / `get` and distributed value ops | op nodes | value tier |
| `pl.jit` / `pl.jit_trace` / `pl.function` / `program` / `loads` | — | frontends (specialization, caching, ABI, dispatch; parse lane is a separate backend) |
| `while_`, `cond`, `static_print`, `static_assert` | — | parse-lane helpers / compile-time diagnostics |
| `split(...)`, `cross_core_slot(...)` | — | pure config descriptors — recipe arguments |
| `pld.CommCtx` sessions, `prefetch` async contexts, `AsyncEvent` / `AsyncSession` | no IR | host-runtime APIs — outside the IR authoring tier |

The two consequences the rule pins:

1. **The value tier needs its own internal cleanup, not migration.** Its
   problems (unified dispatch, op-emission helpers) belong to the facade
   algebra; conflating them with the authoring tier would blur the one clean
   boundary this design has.
2. **The stdlib is the reference client.** A stdlib feature should be
   implementable through public authoring capabilities with the tracer only
   providing the adapter. A recipe that needs `PtoFrame`, the interpreter
   stack, or a raw builder indicates either a genuinely new framework
   primitive or a missing semantic capability — never another private
   convention in the stdlib.

## The Session and Its Units

One recipe invocation owns one **authoring session**. The session is borrowed
(it wraps the active trace's frame) or root (it opens the entry function
itself). Everything the session owns is stack-shaped:

```text
session
    ├── the caller frame (borrowed) or the root function unit
    └── pushed function units, LIFO
```

A **function unit** is one pushed function: a one-shot Sema, its frame, and a
lifecycle:

```text
created → collecting → finalized
```

## The Effect Vocabulary

| Effect | Owner | Purpose |
| ------ | ----- | ------- |
| `ctx.push_function(contract)` | session | Open a child function unit from a `FunctionContract` (name, function type, level, role, attrs, optional params/returns). |
| `fn.collect_body(body, args=...)` | function unit | Execute one prepared body inside the unit: fresh lexical frame, parameters bound from the contract or derived from the staged arguments. Exactly one collection per unit. |
| `fn.finalize(collected)` | function unit | The contract step: check return paths, pin declared returns, mint the `FunctionArtifact`. Exactly once; LIFO among the session's units. |
| `ctx.emit_call(artifact, *args)` | caller frame | Emit the call site for a finalized artifact in the caller's trace (registration with the active program assembly included). |
| `ctx.materialize_entry(artifact)` | root session | Mark the root unit's artifact as the entry — the semantic step the jit envelope wraps. |
| `ctx.splice_body(body, *args)` | caller frame | The borrowed collection: execute a prepared body at the caller's current insertion point, with no function or call. |
| `ctx.region(...)` / `ctx.open_scope(...)` / ... | session | The construct-tier region composition, unchanged (see [06-construct-recipes.md](06-construct-recipes.md)). |
| `ctx.submit(callee, *args, ...)` | caller frame | The dispatch effect — the lowering `pl.submit` / `pl.spmd_submit` share (`core_num=` is the SPMD launch spec). Returns the structural Tuple facade; bind its projections with `ctx.bind`. |
| `ctx.bind(name, value)` | caller frame | The binding effect — recipe code is ordinary Python, so its unpackings never route through the rewriter. Lands in the Frame's own bind path: a projection carrying `__trace_bind__` names its shared node once, the name maps to one mutable `ir.Var`, and the returned facade wraps that Var. |
| `ctx.control.emit_return(...)` etc. | session | Semantic control capabilities (see below). |

## Invariants

1. **Every effect has an explicit owner.** A body is collected into a unit or
   into the borrowed caller frame — never into "whatever is active".
2. **Units finalize in LIFO order**, and a recipe cannot complete while it
   owns an unfinalized unit or an open region.
3. **One collection per unit.** A second body needs a second unit (the
   underlying Sema is one-shot by design).
4. **Collection never emits the call; finalization never does either.** The
   artifact is inert until `emit_call` / `materialize_entry` says otherwise.
   (Collection *does* close the unit's builder function — the runner's normal
   completion; finalization owns the *contract*: return paths, declared
   returns, artifact minting.)
5. **A finalized artifact is immutable** and safe to pass upward.
6. **A live unit cannot escape its recipe invocation** — a unit handle used
   after its session closed is a loud error.
7. **Body collection uses a fresh lexical frame**; definition-site globals and
   closures stay live; staged typing governs the body's interior.
8. **Failure is atomic.** An exception inside a recipe discards the enclosing
   trace (the tracer's one-shot doctrine); no partial state survives.
9. **The context exposes capabilities, never tracer internals.** No builder,
   no `PtoFrame`, no interpreter reaches recipe code.
10. **Splice nesting is refused.** One splice at a time per trace.

## Prepared Bodies (`pl.authoring.body`)

`@pl.authoring.body` prepares a Python function once (source rewrite and compile) and
returns a `TraceableBody`. Preparation caches code-shaped facts only. A
prepared body is placed by an explicit operation:

| Placement | Meaning |
| --------- | ------- |
| `body.splice(*args)` | Execute at the caller's insertion point, per occurrence — the borrowed collection. |
| `body.inline(*args)` | Push an `Inline` function unit, collect, finalize, emit the call. |
| `body.incore(*args, level=...)` | The same composition for an `InCore` function unit. |

The compositions dedupe per program build when an assembly is active: the
same body with structurally identical argument types is one function, not one
per call site (the jit envelope's per-call variant lane remains the
specializing, caching frontend; the authoring compositions are the explicit,
unspecializing spelling).

Derived signatures (a unit pushed without `params=`) come from the staged
arguments' types — every argument must be staged and statically shaped; a
host value or a dynamic extent needs an explicit `FunctionContract`.

### The Stdlib As A Reference Client: `@pl.jit_trace.body`

The kernel-facing spelling of the splice is built here, and it is the tier's
sharpest acceptance test: a policy that costs four lines of public surface,
not a privileged path. Kernel authors declare one meaning and call it as a
function:

```python
@pl.jit_trace.body
def scale_block(a, out, row0, k): ...

scale_block(a, out, row0, k)          # the kernel-facing spelling
```

An external framework reaches the same lowering from this page alone:

```python
prepared = pl.authoring.body(kernel)

@pl.authoring.recipe
def expand(ctx, *args, **kwargs):
    return ctx.splice_body(prepared, *args, **kwargs)
```

What the stdlib adds on top is policy, not capability: the decorator form, the
`__name__`/`__doc__` metadata a body keeps from its kernel, the placement
policy on the declaration (`auto_scope=`), and a refusal that speaks "body"
rather than "splice". If a kernel-facing spelling ever needs a capability this
surface does not have, the fix belongs in the tier — never in a private
shortcut past it.

## Control Capabilities

`ctx.control` exposes the semantic terminators and transfers as context
operations — the escape hatch for tricky user-defined hooks, integrated with
the framework rather than raw:

| Capability | Target selection |
| ---------- | ---------------- |
| `emit_return(*values)` | The active function (or the splice-local boundary — the splice's own return rules apply). |
| `emit_yield(*operands)` | The nearest result-producing region — the `pl.intrin.yield_` channel. |
| `emit_break()` / `emit_continue()` | The nearest legal loop, staged only — a recipe cannot transfer Python control, so a host-loop target is a loud error. |

Every knob routes through the Frame's own guards (pending result bindings,
pending carry groups, region acceptance). After an `emit_return` with no staged
control flow open, the frame's region is terminated: every further tier
effect into it — across recipe invocations, the flag rides the trace — is
rejected as dead code. A return emitted inside a staged branch is one
alternative return path (exactly as the parser treats a `return` inside an
`if`) and terminates nothing. Beyond the tier's boundary, statements after
an emitted terminator remain the author's contract (the tracer's own
dead-code discipline).

## Relation to the Entrances

The entrance family (`@pl.jit`, `@pl.jit_trace`, `.inline`, `.incore`) are
*frontends*: they own specialization, caching, and dispatch. The authoring
runtime is the effect-composition layer beneath them — the splice's public
spelling is `pl.authoring.body(...).splice(...)` and the composition methods above are
the explicit spellings of the callee placements. The parsing frontend remains
a separate backend over the same IR.

## See Also

- [Construct Recipes](06-construct-recipes.md) — the constrained region protocol over the same session.
- [Functions and Program Structure](03-functions.md) — the function kinds contracts describe.
- `examples/tracing/advanced/06_recipe_policy.py` — this page executed: three placements of one body, and the kernel-facing decorator rebuilt from these effects.
- [Functions and Programs § reusing code](../../user/language/01-functions.md) — the kernel-author spellings these recipes stand behind.
