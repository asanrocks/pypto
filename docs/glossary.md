# Tracing Frontend Glossary

This glossary defines the repository-wide vocabulary for the PyPTO language
frontends. It is the permanent source of terminology for design documents,
code documentation, diagnostics, and tests.

Use standard Python, Clang, LLVM/MLIR, and PTO terms directly. The tracing
frontend introduces only two concepts that require project-specific names:
**value-directed staging** and the **trace protocol**.

Code identifiers are written literally in backticks. Prose names the concept,
not the spelling of an implementation symbol.

## Architecture

### Language package

`pypto.language`, conventionally imported as `pl`, owns the language semantics
and every language frontend.

The shared language layer contains operation semantics, types, facades, and
structural language constructs. A frontend may use that shared logic, but does
not redefine it.

### Parser frontend

The **parser frontend** reads Python syntax and constructs PTO IR from its AST.
It is the sole frontend for the round-trip dialect emitted by the IR printer.

### Tracing frontend

The **tracing frontend** executes instrumented Python and constructs PTO IR
from the observed operations and control flow. It is the frontend for the
ergonomics and assembly dialects.

### Collector

The **collector** is the language-independent machinery used by the tracing
frontend. It rewrites Python into an event stream and executes that stream
against a language-provided semantic interface. Language independence means
that collector modules do not import PTO IR or `pypto.language` implementation
modules.

The collector contains four parts:

- The **source rewriter** transforms Python syntax that cannot be observed
  through the Python data model into collector events.
- The **collector interface** defines the `Sema` and `Frame` protocols and the
  typed outcomes returned to the interpreter.
- The **trace interpreter** (`Interpreter`) executes the instrumented function
  and owns Python invocation state, instrumented locals, and active-construct
  order.
- The **trace driver** prepares an instrumented callable, creates one fresh
  trace, and finalizes it on successful execution.

### Sema and Frame

**Sema** is the Clang-inspired semantic-analysis object for one trace. A PTO
`Sema` owns the IR builder, trace configuration, and resulting function. A
mutable `Sema` is one-shot.

**Frame** is the per-function semantic callback object created by `Sema` and
used by the trace interpreter. It owns function-local PTO binding state and
handles bindings, expression statements, returns, staged control-flow
construction, and logical operations.

`Sema` and `Frame` have distinct responsibilities even when one trace contains
one function frame. Do not confuse `Frame` with the interpreter's private
Python invocation record, which owns host locals.

### Active construct

An **active construct** (`Construct`) is the collector state for one active
Python `if`, `for`, `while`, or `with`. The trace interpreter is the sole owner
of the active-construct stack and of construct installation and removal.

An active construct is not a **PTO runtime scope**. PTO runtime scopes are the
user-visible `pl.scope` resources represented by `ScopeStmt` in IR.

### Tracing context

A **tracing context** (`TraceContext`) is passed explicitly to trace-protocol
hooks. It exposes the PTO builder operations and semantic queries required by
shared language objects. It is not `Sema`, `Frame`, the trace interpreter, or
ambient trace state.

### Entrance tier

An **entrance foundation** is the minimal public surface for building
functions by tracing: the immutable declaration data (`Param`, `Decl`), the
prepared body (`TracedCallable`) whose `invoke` performs one complete root
trace, and the ambient trace capability (`current_trace()`). It speaks
`ir.Type` and carries no policy. Its design is the subject of
`design/03-entrance.md`.

An **envelope** is a decorator composed over the entrance foundation:
signature acquisition, trace scheduling, and policy checks. A *compat*
envelope preserves existing behavior and retires with its last user; an
*evolved* envelope is a new surface over the same foundation.

A **variant** is one specialization of a body for a distinct
specialization key: the symbolic argument-type signature plus the baked
scalar values. Dynamic dimensions compare by their extent symbol (extent
`Var` object identity), not their runtime bound; memory placement is not
part of the key; the variant name is derived from the key.

The **tracing service** is the stateless library role of the jit: signature
metadata in, IR out. No cache, no ambient state.

The **dispatcher** is the stateful role over the tracing service:
argument-to-metadata binding, caching policy, and runtime invocation.

## Staging

### Host and staged execution

A **host value** is an ordinary Python value evaluated immediately. A **staged
value** is a language facade wrapping a typed IR expression.

A **host construct** follows ordinary Python behavior. A host `if` executes one
branch, and a host loop executes its actual number of iterations.

A **staged construct** constructs PTO IR. A staged `if` records both branches,
and a staged loop traces one representative body iteration.

Use **host** and **staged** when discussing frontend behavior. Reserve
**concrete** and **symbolic** for domains where those established compiler
meanings are intended, such as concrete dimensions or symbolic expressions.

### Value-directed staging

**Value-directed staging** is the rule that a Python object's recognized kind
determines whether an operation or construct executes on the host or stages
PTO IR. The decision comes from the value, not from an ambient execution mode.

A raw `ir.Expr` is an IR implementation object, not automatically a staged
language value. At the language boundary, a facade supplies the value's type
and behavior.

### Facade

A **facade** is a typed Python wrapper around an IR expression. Use **wrap** and
**unwrap** for conversion between a facade and its expression. Use **lift**
only for conversion of an accepted host literal to a typed IR constant at an
operand boundary.

## Trace Protocol

The **trace protocol** is the Python-style hook protocol through which shared
language objects participate in tracing. Protocol membership is structural;
there is no central registry of object classes.

Name a protocol object by its role:

| Role | Meaning | Hook |
| ---- | ------- | ---- |
| trace iterable | Describes a staged loop | `__trace_iter__` |
| trace context manager | Opens or closes a staged context | `__trace_enter__`, `__trace_exit__` |
| trace-bindable object | Customizes binding to a Python source name | `__trace_bind__` |
| binding proxy | A one-use trace-bindable object that receives a source target name | `__trace_bind__` |
| expression-statement marker | Performs structural behavior when used as a statement | `__trace_stmt__` |

A loop-target binding proxy receives the target name of a staged `for`. An
operation-result binding proxy associates one result position with a source
name. These are roles within the trace protocol, not independent frontend
abstractions.

## Locals and SSA

### Instrumented locals

**Instrumented locals** are Python function locals whose reads and writes are
redirected through the trace interpreter. Reads and writes still obey Python
name-binding behavior.

A staged `if` records each branch in a separate **per-branch locals mapping**.
Reads search the current branch mappings before the function's locals. When
the construct closes, the interpreter either merges those mappings into the
enclosing locals or discards them when results are exported explicitly.

### Mutable locals and SSA promotion

Each staged Python local maps to one mutable `ir.Var`. Its first staged
definition emits a `LetStmt`; a later staged definition emits an `AssignStmt`
to the same variable. `ConvertToSSA` later promotes these mutable locals,
creates SSA names, and derives phi nodes and loop-carried values.

Describe this mechanism directly. A PTO `ir.Var` is not a CPython closure cell
and need not represent physical memory.

A local that must carry staged state through staged control flow requires a
typed staged initialization before the control-flow join. The frontend does
not implicitly lift an existing host value at that join.

### Binding

A **binding** associates a Python source name with a host value or with the
facade for a staged local. A trace-bindable object may instead register an
already-created IR variable or defer an operation until the source name is
known.

An **expression statement** is a Python expression used as a statement. The
`Frame` may emit an `EvalStmt`, consume an expression-statement marker, or do
nothing for a host-only value.

## Structured Control Flow

Use LLVM/MLIR vocabulary for structured control flow:

- A **region argument** is an argument supplied to an IR region.
- An **iteration argument** (`iter_arg`) carries a value from one loop
  iteration to the next.
- A **loop-carried value** is the source-language view of an `iter_arg`.
- A **yield terminator** ends a structured region and supplies operands to its
  parent operation.
- An **operation result** is produced by the parent operation from values
  yielded by its regions.
- A **result binding** associates an operation result with a Python source
  name.

`pl.intrin.carry` declares explicit loop-carried values. A loop without
explicit `iter_args` relies on SSA promotion to infer its loop-carried values.

`pl.intrin.yield_` emits a yield terminator. Its zero-argument form is a
**zero-operand yield terminator**.

`pl.intrin.leak` is the assembly-dialect spelling for binding operation
results after a region body. Keep `carry`, `yield_`, and `leak` only when
referring to those exact APIs.

### Pending result construct

Some assembly-dialect constructs bind operation results after their region
body closes lexically. During the **result-binding phase**:

1. the result-producing active construct remains at the top of the
   active-construct stack;
2. only bindings for that construct's operation results may proceed;
3. the final result binding performs deferred region finalization and removes
   the construct.

The object in this state is a **pending result construct**. A value supplied
only to establish an expected result type is a **result-type witness**.

### Interface tags

An **interface tag** is an opaque key used to find the innermost active
construct supporting a PTO behavior, such as accepting a yield terminator or
exposing a split mode. The trace interpreter orders the search; `Frame`
validates and uses the corresponding typed interface.

The tag is an implementation selector, not the interface itself.

## Execution and Lifetime

A staged loop traces one representative body iteration. State this behavior
directly rather than naming a separate execution mode.

A staged `while` makes a **lazy staging decision**: it begins with host
predicate evaluation and may transition to staged execution once, in place.

An **unwind scope guard** closes active constructs when native Python
`break`, `continue`, or `return` would skip a trailing collector event. Python
context-manager unwinding determines its lexical cleanup order.

Normal IR finalization occurs only after successful traced execution. An
exception escaping the instrumented function discards the one-shot `Sema` and
its unfinished builder state.

## Verb Conventions

Use these verbs consistently:

| Verb | Meaning |
| ---- | ------- |
| execute | Perform ordinary host Python behavior |
| trace | Run the complete tracing frontend |
| stage | Select or construct behavior represented in IR |
| emit | Append an IR operation or statement |
| bind | Associate a source name with a host value or staged local |
| wrap / unwrap | Convert between a facade and an IR expression |
| lift | Convert an accepted host literal to a typed IR constant |
| finalize | Complete an IR region or function |
| lower | Translate to a more explicit or lower-level representation |

Use direct verbs such as **create**, **produce**, **emit**, **wrap**, and
**bind**. There is no separate value-creation phase requiring its own verb.
