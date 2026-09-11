# The Entrance Tier

- **Audience:** tracing-frontend contributors and authors of standard or custom
  PyPTO entrances.
- **Decision:** expose a small tracing foundation; implement decorators,
  derivation, variants, compilation, caching, and dispatch as libraries over it.
- **Position:** decided target architecture, partially landed. The foundation
  (`Param`, `Decl`, `TracedCallable`, `current_trace()` in
  `pypto.language.entrance`), the borrowing expansion with `pl.expand`, and
  the annotation deriver (`pypto.language.entrance.annotation`) are
  extracted; the framework tiers above them (typed calls, `BuildSession`,
  variants, compiler adapters, caches, dispatchers) remain to be extracted.
  The built-in entrances are not yet libraries over the foundation:
  `pl.function`'s eager path and the `pl.jit` envelope still compose the
  Sema directly through the decorator-adapter entry, reusing only the
  foundation's shared tail (annotation capture, return-contract pinning);
  the acceptance equivalence test is the drift guard until the framework
  tiers re-base them. The acceptance tests live in `tests/ut/language/entrance/`.

## 1. Decision

An entrance such as `pl.function`, `pl.program`, or compatibility `pl.jit` is
policy, not a privileged frontend. It combines four independent facts:

```text
IR function declaration
        +
Python invocation recipe
        +
body collection
        +
host-call and staged-call policy
```

The public entrance foundation has three classes and one function:

```text
Param | Decl | TracedCallable | current_trace()
```

`Param` and `Decl` are immutable data. `TracedCallable` is reusable prepared
instrumentation paired optionally with a declaration. `invoke(args, kwargs)`
performs one complete root trace atomically. `current_trace()` exposes an
opaque capability for tracing-time expansion; it does not expose the Sema,
Frame, interpreter, or builder.

The existing `pypto.ir.IRBuilder` remains the explicit IR-construction API. It
is not wrapped by an entrance `Builder` namespace.

Annotations, argument derivation, structured codecs, typed calls, variants,
program assembly, compiler adapters, caches, dispatchers, and decorators are
libraries above the foundation. The collector and concrete PyPTO Sema remain
implementation mechanisms below it.

The executable litmus test is that a user can implement equivalents of
`pl.function`, `pl.program`, compatibility `pl.jit`, and tracing-time expansion
without importing private collector, Sema, Frame, or decorator helpers.

## 2. Mental Model

### 2.1 Declaration and invocation are independent

`Decl` describes one IR function materialization. Its `Param` values are flat
IR parameters and may also be used as placeholders in a Python invocation
recipe:

```python
from pypto.language import entrance


a = entrance.Param("a", a_type)
b = entrance.Param("b", b_type)
out = entrance.Param(
    "out", out_type, direction=ir.ParamDirection.Out
)

decl = entrance.Decl(
    name="gemm",
    params=(a, b, out),
    returns=(out_type,),
    function_type=ir.FunctionType.InCore,
)

body = entrance.TracedCallable.from_callable(gemm, decl=decl)
fn = body.invoke(
    args=(a, b, out, selected_strategy),
    kwargs={},
)
```

For each invocation, `TracedCallable` creates fresh parameter Vars and facades,
then replaces direct `Param` values in `args` and `kwargs` by identity. Ordinary
host values pass through unchanged. This permits an invocation to reorder,
repeat, omit, or inject values without exposing the fresh Sema:

```python
body.invoke(args=(b, a, out), kwargs={})       # reorder
body.invoke(args=(a, a, out), kwargs={})       # repeat
body.invoke(args=(a, out, strategy), kwargs={})  # omit and inject
```

A `Param` not owned by the callable's `Decl` is a loud error. `args=None`
means all declared parameters in declaration order — a kwargs-only recipe
passes `args=()` explicitly, because `args=None` supplies every parameter
positionally and normal Python binding then rejects the duplicated names
loudly. Normal Python binding validates the resulting invocation in every
case.

Foundation substitution is deliberately flat: it does not inspect arbitrary
containers or dataclasses. Structured argument packing is codec policy above
the foundation.

### 2.2 Invocation owns the one-shot lifetime

```text
callable --prepare once--> TracedCallable
                               |
                 invoke(args, kwargs)
                               |
              fresh Interpreter + fresh one-shot Sema
                               |
                        IRBuilder -> ir.Function
```

`invoke` performs the complete root lifetime:

1. create a fresh PyPTO Sema from the declaration;
2. create the declared parameter Vars and matching facades;
3. resolve the flat invocation recipe;
4. execute with a fresh interpreter;
5. finalize only on success and return the `ir.Function`;
6. discard the one-shot state and propagate an escaping exception unchanged.

The internal collector still reports events to Sema/Frame and never constructs
or returns an `ir.Function`. `TracedCallable.invoke` is the public composition
which obtains `sema.result()` after successful collection. No public
trace-lifecycle object or manager is required.

Preparation may cache rewritten code, source maps, generated names, free-variable
layout, and instrumentor version/options. It must not cache execution, host
observations, arguments, Interpreter state, a Sema, or emitted IR.

A callable-backed instance retains a live definition environment, including
current globals, closure cells, defaults, and keyword defaults. A source-backed
instance requires an explicit environment because text has no globals,
closures, defaults, or identity.

### 2.3 Explicit construction is separate

Code that already knows its IR structure uses `pypto.ir.IRBuilder` directly.
Explicit users own valid ordering, scopes, terminators, and types. A bare
`pl.store(...)` expression is invisible without collector expression-statement
instrumentation, so explicit code emits it through `IRBuilder`. There is no
entrance wrapper or parallel shadow API such as `body.store`.

## 3. Layers

```text
standard/user entrances: pl.function | pl.program | jit compat | pl.expand
                              |
framework: derivation | calls | BuildSession | compiler | cache | dispatch
                              |
foundation: Param | Decl | TracedCallable | current_trace
                              |
collector + language: rewriter | Interpreter | PtoSema/Frame | stdlib
                              |
IRBuilder | C++ IR | passes
```

Dependencies point downward except for returned values. The collector imports
no annotation, runtime, compiler, cache, or entrance policy. Root collection
does not know how its declaration or invocation was derived. Standard entrances
have no private capabilities; the compiler accepts only `ir.Program`; dispatch
owns reuse and runtime state.

## 4. Foundation Contract

### 4.1 `Param` and `Decl`

`Param` contains a name, `ir.Type`, and direction. Names within one declaration
are unique. Parameter identity, not structural equality or name lookup, resolves
an invocation placeholder.

`Decl` is the complete immutable specification for one function trace:

- ordered parameters and return contract;
- stable symbol and display name;
- function type, level, role, attributes, and runtime-binding status;
- source span, automatic-scope policy, and strict-SSA mode.

It is broader than an ABI descriptor: source span and tracing controls configure
materialization. It owns no body, Python signature, collector, codec, cache
identity, runtime state, or Python invocation recipe.

If no symbol is supplied, `name` is both symbol and display name. Returns may
be explicit or `Infer`. A declaration referenced before materialization
requires explicit returns because a typed call cannot infer results from a
future body. Variant symbol allocation and mangling are upper-layer policy.

### 4.2 `TracedCallable`

```python
body = entrance.TracedCallable.from_callable(fn, decl=decl)
body = entrance.TracedCallable.from_string(
    source,
    name="fn",
    filename="generated.py",
    environment=environment,
    decl=decl,
)
fn_ir = body.invoke(args=(...), kwargs={...})
```

`decl` may be omitted for a body prepared only for expansion. Calling root
`invoke` without a declaration is a loud error.

`from_callable` and `from_string` perform expensive reusable preparation.
`invoke` always creates fresh execution and construction state. Internal
preparation may be shared by function identity even when upper layers pair the
same Python definition with multiple variant declarations.

`invoke` is not dispatch: its arguments are a tracing invocation recipe made of
owned `Param` placeholders and host values, not runtime tensors for an already
compiled program. It never compiles, caches an artifact, or executes on a
device.

### 4.3 Failure and completion

Invocation follows [The Collector](02-collector.md#6-failure-lifetime):

1. each invocation has fresh Interpreter state and a one-shot Sema/Frame;
2. normal finalization runs only on success;
3. an escaping exception propagates unchanged;
4. failed construction exposes no function result;
5. completed host side effects are not rolled back;
6. no IR rollback, abort callback, journal, or recovery walk exists.

### 4.4 `current_trace()` and expansion

`current_trace()` is the public projection of the existing single ambient trace
channel. It returns `None` on the host or an opaque, invocation-scoped capability
during collection. It does not introduce another ContextVar and must not be
retained after invocation.

The capability permits tracing-time expansion without exposing Sema ownership:

```python
body = entrance.TracedCallable.from_callable(scaled_add)


def expanded(*args, **kwargs):
    trace = entrance.current_trace()
    if trace is None:
        raise TypeError("expanded() requires an active trace")
    return trace.expand(body, args=args, kwargs=kwargs)
```

Expansion internally creates a fresh interpreter and borrowing expansion Sema:

- it borrows the parent's `IRBuilder` and current insertion point;
- it creates a fresh lexical Frame and no function, declaration, or call;
- expansion-local return supplies the value to the parent;
- control cannot target caller constructs;
- an item-assignment targeting a substituted parameter rebinds the caller's
  own binding — resolved by value identity against the owning frame (the
  expansion's locals shadow first) and routed through that frame's
  interpreter; plain rebinding and all name resolution stay in the fresh
  lexical frame, and no caller name is visible to the body (value-level
  write-through, not scope inheritance; ADR-0001);
- failure discards the complete parent trace.

This is the only foundation operation exposed by the opaque capability. Stdlib
trace-protocol hooks continue receiving their explicit language `TraceContext`;
`current_trace()` is not a replacement for that protocol.

Implementing nested expansion requires the corresponding amendment to the
current collector ownership record, which presently excludes overlapping
Frames. ADR-0001 sanctions one bounded exception today: an expansion frame
may resolve and route a rebind into the borrowed parent — one direction,
values only, through the parent's own interpreter — which is not the
general overlap nested expansion would need.

## 5. Framework Utilities

### 5.1 Derivation and structured arguments

A deriver produces a declaration and flat invocation recipe from any source:

```text
annotations              -> Decl + args/kwargs containing Params
runtime tensor metadata  -> specialized Decl + invocation recipe
in-trace facade values   -> variant Decl + invocation recipe
model schema/config      -> arbitrary Decl + invocation recipe
```

The standard annotation reader owns `typing.get_type_hints`, facade annotations,
returns, directions, and defaults. A torch bridge reads shape and dtype but
constructs no IR.

Structured values are not recursively inspected by foundation invocation. A
codec library may flatten the ABI and prepare an adapter which reconstructs the
logical object before expanding the instrumented user body. Directions belong
to flat leaves; dataclasses, tuples, and named tuples remain invisible to the
foundation.

### 5.2 Staged calls

A typed-call utility constructs a detached `ir.Call` from a declaration,
flattens arguments, and wraps result expressions as facades. It does not trace,
locate, or materialize a callee.

| Policy | In-trace behavior | Host behavior |
| ------ | ----------------- | ------------- |
| declared reference | emit call to declaration | reject |
| variant reference | intern variant and emit call | dispatch |
| tracing expansion | expand prepared body into caller | reject |
| external reference | emit call to external declaration | policy-defined |

A staged wrapper's own kind determines its behavior. It may obtain the current
trace capability needed to emit or expand, but foundation ambient state never
decides the value's type. Compatibility `pl.jit` is the deliberate exception:
it preserves its historical staged-reference versus host-dispatch split.

### 5.3 Build sessions and program assembly

Variant discovery and materialization belong to an explicit, build-local
`BuildSession`. It owns:

```text
VariantKey -> declaration, symbol, prepared materializer
symbol     -> materialized function
DFS state  -> unseen, visiting, complete
root set and deterministic symbol allocator
```

A staged variant call derives its `VariantKey`, interns a declaration and unique
symbol in the session, and emits a typed call. Variant wrappers close over the
session supplied by the upper materializer; they do not retrieve it from the
foundation ambient channel or recursively trace while the parent Sema is active.

After a root function completes, program assembly walks reachable non-builtin
`GlobalVar` calls. The session resolves and materializes each symbol with a
fresh root invocation, recursively discovers its calls, orders completed
functions, and creates `ir.Program`. A visiting symbol is recursion and is a
loud error unless recursive IR calls become explicit language semantics.

This deferred algorithm keeps independent root Semas non-overlapping and
removes ambient callee registration. A declaration that is callable before its
body is materialized must have explicit returns.

### 5.4 Variant and artifact identity

A build-local `VariantKey` includes definition identity, normalized IR argument
types, typed static values, stable callable keys, and dynamic-extent Var
identity. Same-key calls in one build share an instance. Different keys produce
distinct declarations and symbols, so one helper may be instantiated at
multiple shapes without duplicate function names.

An `ArtifactKey` is not `VariantKey` plus extra fields. It is a process-stable
encoding of the completed program and compilation inputs:

- canonical program structure and source/compiler version;
- symbolic extent equality classes encoded by ABI position, never object id;
- target, compiler strategy, and effective compilation options.

Canonical program identity permits compilation reuse after a fresh trace. Host
state does not need to enter this key because its observed effect is already in
the newly materialized IR.

A dispatcher that skips tracing must additionally validate every host
observation which affected the trace, or include an equivalent value in its
identity. Observation guards are an optional future optimization, not a
precondition for a sound compilation cache.

### 5.5 Compiler and dispatcher

The compiler boundary is only:

```text
ir.Program + target/options -> CompiledProgram
```

A sound dispatcher performs:

```text
runtime values
 -> metadata and invocation recipe
 -> fresh holistic trace
 -> canonical ArtifactKey
 -> compile-cache lookup
 -> CompiledProgram invocation
```

Host Python therefore executes on every dispatch. Arbitrary globals, closures,
randomness, environment reads, and side effects are observed again. Compilation
is reused only when the resulting canonical program and compilation inputs
match.

A future guarded dispatcher may bypass tracing when all observations from the
prior trace still validate. Compatibility `LegacyKeyPolicy` instead preserves
successful current `pl.jit` reuse by shapes, dtypes, and scalar arguments,
including its environment-sensitive unsoundness. Only the compatibility
envelope may select that policy.

Explicit compile-then-call intentionally freezes one materialization:

```python
impl = kernel.compile(*sample_args, target=target)
impl(*runtime_args)
```

Changing host state after `compile` does not mutate `impl`; requesting a new
compilation is explicit. This is the primary future runtime API. Direct
`kernel(runtime_args)` remains compatibility dispatcher sugar during migration.

## 6. User-Implementable Entrances

A standalone annotation entrance needs no private mechanism:

```python
def my_function(fn):
    plan = annotation_entrance.derive(fn)
    body = entrance.TracedCallable.from_callable(fn, decl=plan.decl)
    return body.invoke(args=plan.args, kwargs=plan.kwargs)
```

`plan.args` and `plan.kwargs` contain `Param` placeholders and host values. A
schema-driven entrance can derive the same three values without annotations.
A structured entrance uses a codec-generated adapter above the foundation. A
fully explicit entrance uses `pypto.ir.IRBuilder` directly.

Structural equality between user-defined and built-in `pl.function` is the
primary acceptance test. Equivalent declared-reference, variant-reference, and
expansion wrappers complete the litmus suite.

## 7. `pl.expand`

`pl.expand` is an additive tracing-native entrance:

> Values and control have function semantics; IR placement has expansion
> semantics.

```python
@pl.expand
def scaled_add(x, y, scale):
    return pl.add(x, pl.mul(y, scale))


@pl.function
def kernel(a: pl.Tensor[[64], pl.FP32], b: pl.Tensor[[64], pl.FP32]):
    return scaled_add(a, b, 2.0)
```

Its contract is:

- arguments evaluate once with normal Python binding; annotations play no
  role in expansion — they are not an IR ABI;
- facades, Tiles, host values, structured values, and callables may cross;
- locals use a fresh lexical Frame; definition-site globals and closures remain
  live; `return` is expansion-local;
- `break` and `continue` cannot target caller constructs;
- no declaration, function, `GlobalVar`, or `ir.Call` is emitted;
- the body and its host side effects execute once per occurrence; only
  preparation may be cached;
- an exception aborts the complete enclosing function trace.

The initial implementation requires one trailing return and rejects early
returns inside staged control flow. Recursion and any nested expansion get an
expansion-stack diagnostic. Existing inline
facilities remain unchanged. `FunctionType.Inline` collects
once, emits calls, and relies on the inlining pass; `pl.expand` executes and
emits its body at every occurrence. Their observable behavior differs for host
side effects, closure mutation, randomness, exceptions, and call-site types.

## 8. Compatibility and Migration

Migration proceeds by dependency order (✓ = landed, tracked by
`tests/ut/language/entrance/`):

1. ✓ publish immutable `Param` and `Decl`;
2. ✓ extract reusable preparation behind `TracedCallable.from_callable` and
   `from_string`;
3. ✓ implement atomic root `invoke(args, kwargs)` and prove user-defined
   `pl.function` equivalence;
4. ✓ publish opaque `current_trace()` over the existing ambient channel;
5. ✓ extract annotation derivation and flat invocation recipes
   (`entrance.annotation.derive`);
6. publish typed-call utilities and explicit `BuildSession` assembly;
7. extract compiler, artifact cache, runtime bridges, and dispatcher without
   changing successful compatibility behavior;
8. add sound build-local variants with distinct symbols;
9. ✓ implement borrowing expansion and add `pl.expand` (initial contract:
   one trailing return, no nested or recursive expansion);
10. make compile-then-call primary before deprecating compatibility `jit`.

The compatibility envelope remains a library over the same foundation. Known
failures remain explicit migration gaps; compatibility means preserving
successful behavior, not promoting failures or unsoundness into new contracts.
The AST parser remains the sole owner of round-trip syntax and is not taught
entrance or tracing semantics.

## 9. Invariants and Non-Goals

1. The entrance foundation exposes only `Param`, `Decl`, `TracedCallable`, and
   `current_trace()`.
2. `Param` and `Decl` are immutable data; direct placeholder resolution uses
   Param identity.
3. Root invocation is atomic and owns fresh one-shot construction state.
4. IR parameters and Python invocation arguments remain independently ordered.
5. Preparation is reusable; execution, host observations, and emitted IR are
   never cached by the foundation.
6. `current_trace()` is an opaque projection of the one ambient channel.
7. Root Semas own their builders; expansion may borrow one for one invocation.
8. Declared references, variants, and expansions are distinct call policies.
9. Variant materialization is deferred through an explicit `BuildSession`.
10. Different variant keys receive distinct symbols.
11. Build-local and persistent identities use different equality domains.
12. A sound dispatcher retraces before artifact lookup unless guards validate.
13. Compiler and dispatch policy never enter IR construction.
14. Explicit construction uses `pypto.ir.IRBuilder`, not an entrance wrapper.
15. Parser and tracing ownership remain disjoint.

This record does not prescribe annotation syntax, cache storage, runtime
internals below `CompiledProgram`, recursive calls, parser-side call type
checking, observation-guard implementation, or eventual removal of existing
inline facilities.
