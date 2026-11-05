# Tracing Collector

One source statement, from the rewrite that instruments it to the IR statement it becomes —
and where to look when it becomes the wrong one.

> **Audience:** contributors to `python/pypto/language/tracing/collector/`. The component map
> and invariants are [Tracing Architecture](09-tracing-architecture.md); this page is the
> mechanics.

## Four layers

```text
rewriter.py    kernel source ─► instrumented source (rt.* calls on the interpreter)
driver.py      prepare once ─► compile ─► restore closure ─► inject a fresh interpreter
interpreter.py run the instrumented function; own locals, branches, the construct stack
interface.py   the protocol to a language frontend: Sema, Frame, typed lifecycle results
```

`emit_kernel(source, name, freevars, future_annotations, first_line)` is the rewriter's entry
and the one function an example may call to display instrumentation
(`examples/tracing/advanced/09_compiler_view.py`). The rewriter works on the AST, not on text,
so string contents cannot be rewritten accidentally; comments, which `ast.parse` drops, are
recovered from the token stream and attached as explicit metadata events.

## What the rewrite emits

| Instrumented call | Meaning |
| ----------------- | ------- |
| `rt.enter_frame(...)` | the trace starts; the kernel's names become instrumented locals |
| `rt.vars.x` | a tracked name — a staged facade or a host value; the interpreter decides which |
| `rt.bind("x", value)` | one binding, routed through the Frame (assignment, walrus, unpacking) |
| `rt.on_begin_with(...)` / `rt.begin_for(...)` | a `with` / `for` site becomes a construct the interpreter drives |
| `rt._emit(...)` | an IR-emitting statement |
| `rt.return_(...)` | the function's return path |
| `rt.unwind_guard()` | wraps a construct whose body holds `break` / `continue` aimed outside it |

The rewrite is type-blind: it preserves native evaluation order, short-circuiting, grouping
and exception behaviour, and it never decides whether a construct is host or staged. Nested
functions, classes, lambdas and comprehensions keep their own binders; reads of kernel locals
still redirect into `rt.vars`.

## Execution

Per invocation the driver injects a fresh interpreter into a copy of the function globals and
runs the instrumented function. The interpreter owns:

- **instrumented locals** and per-branch locals mappings (a `if` arm may bind names the other
  arm does not — for staged values that is a merge, for host values it is the error in
  [Tracing Architecture](09-tracing-architecture.md#the-failed-trace-model));
- **one active-construct stack** — the only logical representation of active structure;
- **host semantics** — which arm runs, how many times a host loop runs, and which construct a
  `break` / `continue` targets;
- **representative staged iteration** — the body runs once; the loop itself is IR.

A result-producing close leaves its construct on the stack as a *pending result construct*.
Ordinary operations refuse it, and only a result binding may address it; the binding that
reports completion removes it.

## Where the IR comes from

Two paths, and mixing them up is the most common local design error:

| Path | Example | Mechanism |
| ---- | ------- | --------- |
| value | `pl.mul(tile, k)` | builds a detached `ir.Call` and wraps it in a facade; the Frame inserts it when the name is bound or the expression is a statement |
| structure | `with pl.at(...)`, `if staged:`, `for i in pl.range(n)` | the construct mutates the builder: open the region, emit the body's statements as they run, close it |

Stdlib trace-protocol hooks (`__trace_enter__`, `__trace_bind__`, `__trace_iter__`,
`__trace_exit__`, `__trace_stmt__`) receive an explicit `TraceContext`, never the builder, the
Sema, or the Frame. That object is the seam the authoring tier is built on
([Construct Recipes](06-construct-recipes.md),
[Authoring Recipes](07-authoring-recipes.md)).

## Failure paths

An escaping exception skips **every** normal finalization: interpreter close, Frame finish,
staged construct close, and the builder's `end_*` calls. The caller drops that one-shot Sema
and re-raises the original exception, so there is no partially-built function to inspect and
nothing to reset. This is why a refusal inside a trace must be raised at trace time: the
alternative is a program that compiles and computes something else.

Prompt-level cleanup on the success path has two other obligations: a pending result
construct must be completed by its bindings, and a pending `iter_args` group must be consumed
by the immediately following loop opener (the Frame enforces that adjacency, so no extra
interface tag exists for it).

## Debugging a wrong statement

1. **Print the instrumented source** (`emit_kernel`, or `09_compiler_view.py`) and read it
   against the kernel. A statement that is not routed through `rt.*` will not be collected.
2. **Print the collected IR** (`kernel.fn`) and check the statement is there *before* passes.
   Present before, absent after, is a pass question, not a collector one.
3. **Locate the owner:** evaluation order and routing are the rewriter's; branch selection,
   host iteration and transfer targets are the interpreter's; bindings, carried values and
   construct shape are the Frame's and the constructs'.
4. **Check the guards' side first** when the symptom is a value that changed with the *number*
   of iterations: that is the representative-iteration model, not a codegen bug.

## See Also

- [Tracing Architecture](09-tracing-architecture.md) — components, invariants, failure model.
- [Tracing Frontend Boundaries](08-tracing-frontends.md) — what happens to the collected function afterwards.
- `tests/ut/language/tracing/collector/` — rewriter, interpreter, driver and portability tests.
- `examples/tracing/advanced/09_compiler_view.py` — source, instrumented source, collected IR and lowered program for one kernel.
