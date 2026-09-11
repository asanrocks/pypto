# The Collector

- **Audience:** contributors to `python/pypto/language/tracer/collector/` and the
  language frontend that plugs into it.
- **Construct:** the design of the language-agnostic execution collector —
  ownership, boundaries, invariants, and why. The **contracts live in the
  module docstrings** (`interface.py`, `interpreter.py`, `rewriter.py`,
  `driver.py`); this document records the architecture behind them and the
  alternatives that were rejected. When this document and a docstring
  disagree about a surface, the docstring wins.
- **Position:** current state. The rejected-alternatives section is the
  historical record of *why not*; everything else describes what is.

## 1. What the collector is

The collector turns a Python kernel into a stream of semantic events by
*executing* it:

```text
Python kernel → source rewriter → generated function → trace interpreter (`Interpreter`) → `Frame`/`Sema` → IR
```

It is language-agnostic: the collector defines the mechanics of
observation (rewriting, driving, ownership), and a language frontend
(the PyPTO Sema/Frame/Constructs in `language/tracer/`) supplies the meaning.
The tracing architecture and its axioms are the subject of
`00-tracing.md`; the collector is their realization as a reusable package.

The package has four layers:

| Layer | Module | Role |
| ----- | ------ | ---- |
| source rewriter | `collector/rewriter.py` | rewrites kernel syntax into calls on the trace interpreter |
| trace interpreter | `collector/interpreter.py` | executes the rewritten function; owns invocation state, instrumented locals, per-branch locals mappings, and one active-construct stack |
| protocol | `collector/interface.py` | the boundary to a language frontend — `Sema`, `Frame`, and typed lifecycle results |
| trace driver | `collector/driver.py` | compiles the rewritten function, restores its closure, and injects a fresh interpreter per call |

Two litmus tests define the boundary:

- **Swap the frontend.** The stdlib-ast source rewriter could be replaced
  by a bytecode interpreter — different raw events, same interpreter
  semantics — and the language plugin would be untouched. (It was itself a
  swap: an earlier libcst implementation was retired after a differential
  bake; see ``rewriter.py``'s module docstring.)
- **Swap the plugin.** The source rewriter can host a different language (a
  `TritonSema` + `tl.*` stdlib) with no rewrite-rule or interpreter changes.
  Each plugin co-designs its trace-protocol objects with its own builder; the
  source rewriter is IR-blind.

## 2. Ownership: one fact, one owner

PTO state has three natural lifetimes, and each has exactly one owner:

```text
Sema   one function-trace compilation
Frame  one function execution
Construct  one active construct
```

- **Trace interpreter** owns instrumented locals, per-branch locals mappings,
  and the active-construct stack. It selects branches, executes host loops,
  traces staged loops for one representative body iteration, and chooses
  transfer targets. Active-construct order is host semantics, and the trace
  interpreter is the host driver — so the order belongs to it. It knows host
  kinds, shared Construct interfaces, the `Frame` protocol, and typed callback
  results. It knows no language type, trace-protocol object role, or builder
  operation.
- **Sema** is one function-trace compilation: one IRBuilder and its
  physical construction stack, the compilation configuration and result,
  and the creation of one fresh Frame per trace. The trace driver retains
  a factory (or immutable configuration) and acquires a fresh Sema per
  independent trace. A mutable Sema is one-shot: reuse after success or
  failure is rejected, never reset. A failed trace produces no IR, so
  there is nothing to reset *into*.
- **Frame** owns the source-name-to-`ir.Var` map, typed staged initialization
  policy, staged Construct construction, and mediation of stdlib trace
  protocols. It keeps **no ordering mirror** — no open-region stack, no loop
  list, no control-flow depth counter. When semantics depend on active
  structure, it queries the trace interpreter.
- **Construct** is one active construct. The trace interpreter owns
  stack position and logical lifetime; the host-kind base classes
  (`IfConstruct`/`ForConstruct`/`WhileConstruct`/`WithConstruct`) own host
  behavior and public routing; language subclasses (e.g.
  `PtoWhileConstruct`) inherit the matching class and implement staged
  hooks. A stdlib trace-protocol object is *construction input* to a
  Frame-created Construct — never a Construct itself.

Two stacks coexist by design: the trace interpreter's Construct stack
(logical host order) and the `IRBuilder`'s stack (physical IR insertion).
They are different structures with different jobs; the exact PtoConstruct on
the interpreter's stack connects them. One logical Construct may own zero,
one, or several physical builder scopes — the interpreter never mirrors
that shape.

## 3. The trust contract

The trace interpreter trusts the Sema, the Frame, and Frame-created
Constructs. Trusted callbacks receive the interpreter directly and use its
controlled query methods. There is no view object, cursor snapshot, handle
id, or generation counter.

Trust does not transfer ownership:

- The trace interpreter alone mutates Construct order and lifetime.
- Frame and Construct retain no ordered mirror.
- The trace interpreter exposes no mutable stack container.
- Trusted callbacks request host mutation through interpreter methods or
  typed return values.

The trace interpreter does *not* trust stdlib objects as structural
authorities. `Frame` mediates their trace-protocol hooks
(`__trace_iter__`, `__trace_enter__`, `__trace_bind__`, `__trace_stmt__`),
validates their outputs, and converts them into binding proxies, trusted
Constructs, or typed results. This is an interface-tag rule, not a sandbox
against arbitrary Python.

This trust posture is the deliberate rejection of a defensive design
(ADR-lineage "region machine"): when a trace fails, the one-shot Sema is
discarded whole. Recovery machinery — transactions, cleanup journals,
recovery walks — cannot make failed tracing produce valid IR, so none of
it exists.

## 4. Boundary contracts

### 4.1 Source rewriter → Trace interpreter

Generated code reports function entry, name access, comments, expression
statements, returns, structural boundaries, transfers, and lazy logical
operations through a fixed event surface (`enter_frame`, `eval_if`,
`eval_then`, `eval_else`, `begin_for`, `eval_bool`, `on_begin_with`,
`exit_construct`, `break_`, `continue_`, `vars.*`, `bind`, `return_`,
`_emit`, `_leading_comments`). The source rewriter preserves evaluation
order and supplies values, stable thunks, target names, and static syntax
facts such as `has_else`. Generated code never sees Construct, Frame, Sema,
or `IRBuilder`.

Host constructs are **kept, not replaced** — the source rewriter wraps and
drives; the host `for`/`while`/`if`/`with` remain. Host constructs that
cannot be observed pass through the instrumentation boundary; bool-coercing
constructs route through mode-aware resolvers.

**The one generated-shape exception: the unwind scope guard.** A native
`break`/`continue` skips a plain trailing `exit_construct()` call, so every
construct whose body holds a transfer keyword bound to an enclosing loop
is wrapped in `with rt.unwind_guard():` instead. The keyword's own unwind
runs the guard exit at the construct's exact lexical boundary — after inner
source context managers and `finally` bodies, before outer ones — so
active-construct cleanup preserves Python's unwind order. `with` constructs
are not wrapped: their generated context manager already runs during native
unwinding. Only the source rewriter knows, lexically, which constructs a
native transfer crosses (a `continue`
never crosses its own loop's close; a transfer in a loop's `else` suite
binds to the next enclosing loop, so it crosses the loop's *enclosing*
constructs but not the loop's own close).

Span parameters on the Frame protocol are reserved surface; the
instrumentation does not carry source spans into interpreter events (spans
enter at the language Sema boundary instead — see `00-tracing.md` §8).

### 4.2 Interpreter → Frame

Every interpreter event that needs language semantics is a Frame method:
bind, expression_statement, return_, the `create_*_construct` constructors, and
the logical value algebra. For eagerly classified constructs (`if`/`for`/
`with`), a `None` return from a constructor means the interpreter installs
its host-kind Construct; a returned Construct is trusted and installed
exactly. `while` makes a lazy staging decision: `Frame` may return a
Construct capable of staging at lexical entry even before a staged predicate
appears.

Lifecycle-sensitive `None`/boolean combinations are replaced by closed
result types — bind results distinguish decline, ordinary binding, result
binding progress, and exact completion; close results distinguish finalized
closes from pending result constructs. A multi-meaning `None` is how the
previous contract generation lost information; the typed results are the
fix.

The Frame may query the trace interpreter during construction but cannot install
or remove Constructs.

### 4.3 Interpreter → Construct

The trace interpreter routes every region-local callback to the exact
selected Construct; it never asks `Frame` to rediscover a target. Public
host-kind methods are interpreter-owned Template Methods; language
subclasses override the staged hook surface and never touch routing, stack
discipline, or Construct mode. `WhileConstruct.test` is the one *strict*
Template Method: the transition algorithm and staging flag are base-owned,
and a subclass implements only the protected activation hook.

Target selection is family-specific and never skips a nearer eligible
target: close and else address the exact lexical Construct; break/continue
address the nearest loop; yield addresses the nearest permitted Construct
accepting a yield terminator (the zero-operand yield terminator has its own typed
selection: each Construct answers accept/block/cross, and the trace
interpreter owns the ordered walk); result binding addresses the exact top
pending result construct.

### 4.4 Frame → Stdlib, and the ambient entry

Stdlib values are extension inputs, not execution authorities. Traceable
objects are recognized **by protocol membership** — the presence of a trace
hook identifies the role; there is no central registry. Hooks receive a
controlled tracing context from `Frame`, never a raw builder, the `Sema`, or
`Frame` itself.

The trace driver installs the current Frame for one function execution
through exactly one ambient entry: `active_frame()`. It resets that
ContextVar in an unconditional `finally`. Ambient operations call
`active_frame()`; the Frame exposes explicit named operations for them
and routes host-state changes through the trace interpreter. Construct-
spanning ambient state is forbidden — properties such as split mode live on
the exact Construct, obtained by querying the interpreter.

## 5. Construct semantics

### 5.1 Modes

For `if`, `for`, and `with`, the execution kind is fixed when the trace
interpreter installs the Construct: host-kind, or staged and Frame-created.
For `while`, one Construct starts in host execution and may activate staging
**in place** — stable identity for the whole lexical lifetime, at most one
host-to-staged transition, never the reverse, and the predicate thunk is
never reevaluated after activation. The base class flips the flag only after
the subclass's activation hook succeeds.

### 5.2 Branches and mutable locals

A staged `if` records *both* branches (all paths): each branch writes to
its own per-branch locals mapping, the trace interpreter switches mappings
at the else boundary, and at close it commits the union into the enclosing
mapping (or discards it for a result-producing region). The merge is
unchecked by design — a name bound on only some paths is rejected by the IR
verifier's use-after-def check at compile time, which matches Python's
path-sensitive `UnboundLocalError` semantics; the loop-body single-branch
definition (the passthrough is the previous iteration's value) is
loop-carried behavior owned by `ConvertToSSA`.

`Frame` owns the mapping from source names to mutable `ir.Var`s. Branch
bindings reuse one variable per name, so convergence needs no frontend
merge machinery: `ConvertToSSA` synthesizes the phi from the assignments
to that variable.

### 5.3 Carry, yield, and completion

- **Carry**: a completed group of pending `iter_args` is detached function
  state before any Construct exists, so `Frame` owns it. Adjacency is a
  literal invariant: after the carry bindings complete, the next semantic
  event must be the claiming loop opener; every other Frame-reached event
  rejects the pending group (comments remain exempt because they are
  annotation state, not mutations).
- **Yield**: arrives as an expression-statement marker; `Frame` asks the
  trace interpreter for the nearest permitted Construct accepting a yield
  terminator. Policy, arity, types, and crossing rules belong to that
  Construct's contract.
- **Result binding**: a result-producing close leaves the exact Construct at
  the top of the interpreter's stack as a **pending result construct**.
  Ordinary operations reject it before mutation; only bindings for its
  operation results may address it, and only by its exact object. The
  interpreter removes it when a binding reports exact completion. A
  structural boundary close (a `with` exit or unwind scope guard exit)
  follows the same rule against *inner* pending result constructs: the
  boundary's own result-producing close may defer to the result bindings
  that follow it, but a construct stacked above the boundary's own pending
  result construct is rejected — those bindings must finish inside the
  boundary.

### 5.4 Transfers and context managers

The trace interpreter chooses the nearest break/continue target from its
kind-preserving stack and applies exactly one rule: a staged target receives
the exact callback and the native keyword is suppressed; a host target with
no intervening staged Construct executes the native keyword; a host target
*with* an intervening staged Construct is rejected — the alternate staged
path was not collected. The interpreter never skips the selected target for
a more convenient outer loop. Staged loop `else` is rejected at the loop's
close, when the static `has_else` fact arrives.

A host `WithConstruct` preserves the user context manager's entry, exit,
exception triple, and suppression exactly once. A staged `WithConstruct`
follows the same public host contract through its staged hooks.

## 6. Failure lifetime

Finalization runs **only on the success path**: the trace interpreter
LIFO-closes every open Construct, then finishes the Frame, and the trace
driver calls `rt.close()` only there. An exception escaping the instrumented
function skips every normal finalization — no interpreter close, no Frame
finish, no staged Construct close, no builder `end_*` — and the outside
stdlib discards the one-shot Sema, propagating the original exception
unchanged.

Python itself still unwinds every successfully entered `with` through its
own context-manager protocol. A context manager that *suppresses* an
exception is trusted to own the IR consequences of that suppression: if
suppression leaves half-built IR, normal IR verification rejects the
result. There are no abort callbacks, sticky failure state, cleanup
journals, recovery walks, or builder rollback.

## 7. Invariants

1. The trace interpreter is the only ordered Construct owner and stack mutator.
2. Every instrumented control-flow structure installs exactly one host-kind or staged Construct.
3. Generated control-flow shapes are stable; the unwind scope guard around
   transfer-affected constructs is the one deliberate exception.
4. Interpreter host-kind classes define public routing and own Construct kind.
5. Language subclasses implement staged hooks only.
6. Only the while Construct transitions from host execution to staging, once,
   in place.
7. Frame and Sema retain no active-Construct ordering mirror.
8. Region-local callbacks target exact interpreter-selected Constructs.
9. Target selection is family-specific and never skips a nearer eligible
   target.
10. The trace interpreter interprets no language type, trace-protocol object
    role, or builder operation.
11. `Frame` mediates stdlib trace hooks; trace-protocol objects are never
    installed as Constructs.
12. Construct-spanning semantic properties reside on exact Constructs.
13. `Frame` queries the interpreter directly without caching active order.
14. Every function trace acquires a fresh one-shot Sema and Frame.
15. A pending result construct remains the interpreter's exact top Construct;
    ordinary methods reject it, and result bindings require its exact object.
16. Native break/continue never crosses an intervening staged Construct.
17. Rebinding a staged local through the stdlib routes through interpreter
    binding.
18. `active_frame()` is the only ambient tracing entry.
19. An exception escaping the instrumented function skips all normal
    finalization.
20. The outside stdlib discards that one-shot Sema and propagates the
    original exception.
21. Suppressing context managers own the IR consequences of suppression.
22. Success leaves no open Construct, no pending result construct, and no
    unconsumed pending `iter_args`.

## 8. Rejected alternatives

- **ScopeEntry with a Behavior object** — adds delegation and parent
  coordination to every Construct to solve a while-only transition problem.
- **Replacing the host while Construct on activation** — breaks stable
  lexical references.
- **Sema-owned active ordering** — duplicates the interpreter's host order;
  the mirrors were the defect this design removed.
- **Interpreter-dispatched stdlib trace hooks** — bypasses `Frame` as the
  language interface.
- **Stdlib trace-protocol object as Construct** — hands an unmediated
  extension value lifecycle authority.
- **Construct-spanning ambient state** — duplicates Construct lifetime and
  requires failure cleanup.
- **Region recovery machinery** (transactions, journals, recovery walks)
  — cannot make failed tracing produce valid IR.
- **A separate assembly frontend** — explicit loop-carried arguments,
  yield terminators, and result binding fit the `Frame` and Construct
  lifetimes.
- **The `try_*` decision channel** (decision folded into one
  construction call returning a handle or `None`) — the `None` carried
  two meanings; the typed result types and the `Frame` protocol replace it.
- **Trace-protocol-object-as-handle Construct entries** (the object itself
  on the stack) — the object is construction input; the Frame-created
  Construct owns the region data and validates the protocol at construction.
- **Auto-unwind finalization on the exception path** — a failed trace
  produces no IR; there is nothing to finalize into validity.
- **Multi-meaning `None`/bool protocol returns** — replaced by the typed
  lifecycle results throughout.

## 9. Non-goals

This design does not: change the round-trip parser or its ownership; add
staged loop `else`; change one-representative-iteration behavior after
while activation; recover partial IR; recover arbitrary external state from
malformed trusted callbacks; support overlapping Frames; create a second
assembly frontend; make the trace interpreter language-aware; sandbox
Python stdlib code;
require immutable snapshots; evaluate tracing performance; change
decorator argument injection or program assembly.

If healthy tracing ever requires the Frame to reconstruct active order,
the trace interpreter to inspect language semantics, the stdlib to mutate the
Construct stack, or a failed Frame to continue using the builder, this design
must be revisited rather than patched with another hidden owner.

## 10. Where this lives

| Concern | Module |
| ------- | ------ |
| Event protocol, generated shapes, static subset policy | `python/pypto/language/tracer/collector/rewriter.py` |
| Frame/Sema protocol, typed lifecycle results, NullSema | `python/pypto/language/tracer/collector/interface.py` |
| Interpreter, Construct hierarchy, per-branch locals mappings, pending result constructs, transfers | `python/pypto/language/tracer/collector/interpreter.py` |
| Compilation, closure restoration, per-call injection | `python/pypto/language/tracer/collector/driver.py` |
| Package map and the emitted event sequence | `python/pypto/language/tracer/collector/__init__.py` |
| Language realization (PtoSema, PtoFrame, PtoConstructs) | `python/pypto/language/tracer/` |
| Ambient entry (`active_frame`) | `python/pypto/language/ambient.py` |
| Conformance suites | `tests/ut/language/tracer/`, `tests/ut/language/test_collector_contract.py` |
