# The Tracing Architecture

- **Audience:** contributors to the tracing frontend and the collector.
- **Scope:** the end-to-end architecture that turns a Python kernel into
  C++ IR by *executing* it. The kernel language surface is
  `01-language.md`; the collector's internal design is `02-collector.md`.
  This document states the model, the components, and why they are shaped
  this way. Contracts live in module docstrings; this document points at
  them but never restates them as an obligation to sync.

## 1. The model: execution is collection

A PyPTO kernel is ordinary Python. The stdlib `pl.*` yields IR nodes as
the kernel executes; **collecting IR is a side effect of running the
program**. There is no dedicated DSL syntax — no `if_`, no `while_`, no
`let`. Bare `if`, bare `while`, and bare assignment serve both host and
staged execution because value-directed staging determines which behavior
a construct follows:

- A **staged value** is a facade — a typed Python wrapper around an IR
  expression (`Scalar`, `Tensor`, ...). Its host value is unknown at trace
  time; its value is the IR expression it wraps.
- A **host value** is an ordinary Python value, fully known at trace time.

Everything else in this architecture exists to make that side effect
observable without lying about types.

```python
import pypto.language as pl

@pl.jit
def tile_add_64(a: pl.Tensor, b: pl.Tensor, c: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile_a = pl.load(a, [0, 0], [64, 64])   # yields a staged Tile
        tile_b = pl.load(b, [0, 0], [64, 64])
        tile_c = pl.add(tile_a, tile_b)         # yields a staged Tile
        pl.store(tile_c, [0, 0], c)             # a bare side-effect statement
    return c
```

## 2. Two axioms

Every design decision below derives from two axioms.

**Axiom A — a sound-typed subset of Python.** No non-host syntax
extensions, no type lies. A value's type comes from the operator that
yielded it — never from an annotation and never from execution context.
"Is `5` staged?" is answered by the value's kind, not by "are we tracing?".

The forbidden thing is **observing** builder or execution state to decide
a value's type or a construct's behavior — that is the type lie.
Mutating builder state as a side effect, with the type fixed by
construction, is not a lie: it is what `pl.store`'s collection already
is. Observation is side-effect free; mutation is an explicit part of
emitting IR. The old blunt rule "the stdlib never touches a builder"
conflated the forbidden with the permitted and is retired.

**Axiom B — value IR is detached; structural IR is stateful.** A value IR
node (an `ir.Call`, a constant) is a closed expression: build it, hold it
by a host reference, insert it into a region later. A structural IR node
(an `if`, a loop, or `ScopeStmt`) is *open over time*: you enter it, emit
statements into it as the host executes the body, then close it — its
body is not known until the body runs.

This asymmetry is forced by the IR's nature, and it decides who builds
what:

- **A value operation** (`pl.add`, `pl.load`) owns its detached expression:
  it builds the expression without a builder and wraps it in a typed
  facade. It never self-emits; `Frame` inserts the facade at binding or
  expression-statement time.
- **A trace-protocol object** such as the trace context manager `pl.at` or
  the trace iterable `pl.range` owns the state needed to construct its
  active construct. Its construction mutates the builder through the
  `Sema` (write-only: begin/end, never builder reads). The host `with`/`for`
  protocol delivers the object to the trace interpreter, which uses it as
  construction input for the active construct.
- **`if` and `while` have no trace-protocol object** — the predicate is a
  `Scalar` value, and Python statements do not deliver an object for these
  constructs. `Frame` constructs them directly. This seam is forced, not
  an oversight: artificial `pl.If`/`pl.While` objects would have no
  host-protocol mapping.

## 3. The components

```text
user kernel (ergonomics or assembly)
        │  @pl.function / @pl.program — the decorator adapter
        ▼
Source rewriter (type-blind LibCST rewriter)   ── language/tracer/collector/rewriter.py
        │  rewrites the kernel body into calls on the trace interpreter
        ▼
Trace interpreter (`Interpreter`)              ── language/tracer/collector/interpreter.py
        │  executes, owns instrumented locals + per-branch locals mappings
        │  and the active-construct stack; consults `Frame` at each binding
        │  and control-flow head
        ▼
`Frame` → `Sema` (the language plugin)         ── language/tracer/semantics.py
        │  owns the `IRBuilder`, staged locals, and staged constructs
        ▼
C++ IR
```

- **Stdlib** (`pl.*`) — operation semantics and result typing. It provides
  value operations and trace-protocol objects, with observation side effects
  kept out of value classification. Decorator adapters connect the source
  rewriter to the trace driver and establish the tracing context. The
  assembly operations (`pl.intrin.*`) are a controlled low-level tracing
  extension for compiler tests; they may state explicit IR channels but
  must obey the sound-typed value model.
- **Compat** (`language/compat/`) — `static_print`/`static_assert`: compile-time
  diagnostics that only the parser frontend runs. Under the tracing frontend
  they are silent runtime no-ops, so kernels written for the parser stay
  importable in a tracing environment.
- **Source rewriter** — a type-blind, lexically aware source rewriter. It
  rewrites only the decorated kernel body into calls on the trace
  interpreter's event protocol. What the host language can observe with
  its own protocols (calls, attribute access, `__enter__`/`__exit__`,
  `__iter__`/`__next__`) passes through; what it cannot observe (function
  local assignment, expression statements, block boundaries) must be
  redirected. This boundary is the **instrumentation boundary**.
- **Trace interpreter** — drives normal host execution and owns
  instrumented locals, per-branch locals mappings, and the active-construct
  stack. It is IR-blind: it never builds IR, never inspects a facade, never
  correlates or promotes. It consults `Frame` for language semantics at
  bindings and control-flow heads.
- **Sema** — one function-trace compilation. It owns the stateful
  `IRBuilder` (which trace-protocol objects use for construction), the
  compilation result, and the creation of one fresh `Frame` per trace. It
  keeps no ordering mirror of the interpreter's active-construct stack —
  structural questions are trace-interpreter queries.

The shared language layer owns the seams both frontends meet it at: the
ambient channel (`active_frame`, `language/ambient.py`) through which
shared ops reach the active Frame, the trace protocol
(`language/protocol.py`) that shared constructs implement hooks from —
including the recognition query `is_staged` — declared-signature direction
recovery (`typing/arg_direction.py`), and the frontend-common diagnostic
exceptions (`language/diagnostics.py`). Dependency direction is one-way:
the parser and tracer frontends import shared language; shared language
imports neither frontend at runtime (`if TYPE_CHECKING` imports are
typing-only), and the sibling frontends do not import each other
(`tests/ut/language/test_collector_contract.py`). The collector remains
below both frontends and imports only the Python standard library and
LibCST.

The division of state, in one table:

| State | Owner |
| ----- | ----- |
| Source transformation, generated names | Source rewriter |
| Instrumented locals, per-branch mappings, active-construct order | Trace interpreter |
| Host routing and construct mode | Host-kind Construct base classes |
| Staged region data | Frame-created Construct subclasses |
| `IRBuilder` and trace result | `Sema` |
| Mutable `ir.Var`s and function-local semantic state | `Frame` |
| Trace-protocol object parameters and marker-local state | Stdlib values |

## 4. The IR and passes contract

The frontend is thin because the IR and its passes absorb *all* graph-wide
dataflow:

- The IR is permissive: variables are mutable named `ir.Var`s. A first
  staged definition emits a `LetStmt`; later staged definitions emit an
  `AssignStmt` to the same variable. There is no frontend hold-inline
  special case, and aliasing is recovered by the passes' copy folding —
  never by the frontend.
- The passes do the dataflow: `ConvertToSSA` promotes mutable locals to SSA
  names, synthesizes phi nodes, derives loop-carried values, and lowers
  `break`/`continue`; copy folding remains a pass concern.

Two consequences shape the language:

- **Convergence through mutable locals and SSA promotion.** Branch bindings
  reuse one mutable `ir.Var` per name, so a staged `if` closes with no
  explicit merge variables and `ConvertToSSA` synthesizes the phi from the
  branches' assignments. There is no merge construct, no `return_vars`, and
  no `let`/`reassign` surface in the language.
- **Loop-carried values are derived, not declared.** The ergonomics
  frontend emits a loop with no loop-carried placeholders and assignments
  to mutable locals in the body; `ConvertToSSA` derives the loop-carried
  values, including conditional loop-carried values. The assembly dialect
  may state explicit channels (`pl.intrin.carry`/`pl.intrin.yield_`) — a
  controlled exception for compiler tests, not a second frontend. The
  tracing surface of `pl.range`-style factories has no `init_values=`
  parameter; that spelling is printer/parser-owned round-trip text.

**Trace-time execution determines the IR.** Whatever the host does at
trace time *is* the IR: an inline callee's body runs in the caller's
tracing context (the IR is the body, inlined), and a host range unrolls
(N copies). Post-hoc transforms that duplicate this retire; when a
mechanism cannot be expressed within execute-and-collect, the language
spec changes — the frontend does not special-case.

## 5. Two execution modes

The trace interpreter handles a duality the host cannot:

- **Staged execution (all paths):** both branches of a staged `if` are
  recorded (the generated guards return True on both paths); a staged
  loop traces one representative body iteration while the loop operation
  is recorded. This is how all IR is collected.
- **Host execution (one path):** host execution follows Python exactly —
  the non-matching branch is elided, and a host range actually iterates.
  This is what makes metaprogramming work: the same kernel can compute
  shapes and loop bounds at trace time.

Value-directed staging uses the kind of the predicate or iterable (axiom
A), never an ambient execution mode. The boundary guardrail: coercing a
staged value to a boolean is a loud error wherever the construct cannot be
rewritten to be trace-safe (a ternary, an `assert`, or a comprehension
predicate) — see `01-language.md` for the full three-axis derivation of
the static subset.

## 6. Dialects and pipelines

PyPTO has three dialects, distinguished by *frontend ownership*, not
necessarily spelling:

| Dialect | Audience | Frontend |
| ------- | -------- | -------- |
| **ergonomics** | kernel authors — bare Python + ergonomic `pl.*` | tracing |
| **assembly** | compiler developers — explicit IR constructs | tracing |
| **round-trip** | none (it is printer output) | AST parser |

```text
ergonomics --tracing--> C++ IR
assembly   --tracing--> C++ IR
C++ IR --IR printer--> round-trip text --AST parser--> C++ IR
```

The boundary is ownership: the parser must not learn tracing semantics,
and tracing must not consume printer output as a round trip. The loop-
carried channel spellings differ by owner: the assembly dialect states
channels through `pl.intrin.carry`/`pl.intrin.yield_`/`pl.intrin.leak`
(executed by tracing), while `init_values=`/`pl.yield_` on
`pl.range`-style loops exist only as printer output consumed by the AST
parser (including the `@pl.jit` path, which collects through `pl.parse`).
A round-trip printer form and an assembly program may therefore look
similar without sharing a frontend. Round-trip verification is always
parser-owned.

Frontend markers make this explicit: `@pl.function` / `@pl.program`
collect through tracing; `@pl.function_parser` / `@pl.program_parser` are
the permanent AST-parser entrypoints, used only by parser tests and
printer round-trip fixtures.

## 7. Trace timing: eager and lazy

The same trace driver (compile → inject the interpreter → execute) serves both
timings. **Eager ⊂ lazy**: eager tracing is the same driver pre-executed
at decoration time with staged arguments synthesized from annotations;
lazy tracing runs at call time with the real arguments. The decorator
inspects the signature to decide: annotation-determined IR → eager;
call-time-dependent IR (`Callable` parameters, call-time shapes, dyn-dims)
→ lazy. The signature is the inspection contract — parameters must be
annotated (a missing parameter annotation is a loud decoration-time
error, not a guessed `UnknownType`); locals are never annotated, because
typing flows from values. What happens downstream of IR production —
compile, cache, device execution — is a separate concern, deliberately
out of scope for the frontend.

## 8. Source spans

Tier 1 (realized) gives every traced function and program an explicit
coarse source span — filename + definition line — derived from
`co_firstlineno` / inspected source at the decorator, passed through the
Sema into `IRBuilder.begin_function`/`end_function`. Explicit metadata,
never ambient: no span is pinned while a body executes, and nested
operation calls keep their own resolution.

Operation wrappers resolve spans in a fixed order (the contract lives in
`python/pypto/ir/utils.py`): explicit span → parser-pinned ambient span →
user-frame capture → unknown. Function spans matter because the verifier
deduplicates diagnostics by source site — an unknown span silently
conflates unrelated operations.

The generic collector protocol stays span-free: spans are a
language-frontend concern and enter at the language Sema boundary, which
is already IR-aware. (The protocol's `span=` parameters are reserved
surface, not a live channel.)

Deferred (Tier 2): coarse libcst positions for loops, branches, and
yields — requires routing positions through generated calls, interpreter
forwarding, and protocol implementations; and deciding the position
representation (`(line, column)` vs an opaque marker). Exact operation
columns also remain deferred.

## 9. Rejected alternatives

Each rejected because it violates an axiom or adds a redundant
representation:

- **Blunt "stdlib never touches a builder"** — conflates the forbidden
  (observation, the type-lie vector) with the permitted (mutation);
  forces all structural construction into the Sema and blocks
  extensibility.
- **Sema owns all structural construction** — a god-class Sema; every new
  construct adds a Sema method; the trace-protocol object degrades to a
  data bag.
- **Stdlib owns the builder** — a stateful builder with no single owner
  is incoherent.
- **Construct-object layers (one `Pto*Construct` per `<X>Builder`)** — track one IR
  region three times; the trace-protocol object should provide construction
  input, not become the interpreter's active construct.
- **Artificial `if`/`while` trace-protocol objects** — no host-protocol
  mapping exists
  for them.
- **Frontend merge machinery** — branch bindings share mutable `ir.Var`s;
  `ConvertToSSA` synthesizes phis; a merge channel is redundant.
- **Frontend-declared loop-carried values (ergonomics)** — violates the
  IR and passes contract.
- **Context-recognized participation in staging** — a type lie by
  definition.
- **Automatic host-literal lifting at staged joins** — silently computes
  the wrong result (see typed staged initialization in `01-language.md`).

## 10. Deferred

- Inline-callee trace-time dispatch (currently inline callees trace as
  sub-kernels and the backend inlining pass splices them).
- `tensor.dim` extent-symbol fold in *Inline* callee bodies (depends on
  inline dispatch; Orchestration and Graph bodies fold — the
  `is_orchestration_like` gate).
- Scope-level opt-in host-literal lifting at staged joins.
- Span Tier 2 (§8).
