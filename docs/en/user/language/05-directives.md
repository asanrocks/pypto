# Compile-Time Directives

What "compile time" means under the tracing frontend: `pl.const` is the live
typed-literal construct, while `pl.static_print` / `pl.static_assert` are
parser-owned diagnostics that do nothing when a kernel is traced.

> **Prerequisites:** [Functions and Programs](01-functions.md).

## Concept

The tracing frontend **executes** your function as Python and collects IR as a
side effect — there is no parse step in which source text could be inspected.
That splits the old "compile-time directives" in two:

- `pl.const(value, dtype)` is the tracing-native construct: a typed literal,
  and the one spelling for a concrete staged value — a bare host literal
  cannot seed staged state ([Control Flow](02-control-flow.md)).
- `pl.static_print` and `pl.static_assert` are diagnostics of the **AST
  parser**: they produce no IR and run only where the parser owns the
  frontend (round-trip text, `@pl.function_parser` fixtures). Under tracing
  they are **silent no-ops** — the call executes, has no effect, prints
  nothing, and a failing `static_assert` does not raise.

Knowing this is the difference between a debugging aid and a puzzle: a
`static_print` that shows nothing is not evidence that the function "was never
parsed" — under tracing it never prints at all.

## Quickstart: what actually happens

<!-- doctest: setup -->
```python
import pypto.language as pl
```

<!-- doctest: run -->
```python
@pl.function
def probe(x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP32]:
    pl.static_print("x =", x)
    pl.static_assert(False, "expected 128 columns")
    return pl.tensor.muls(x, 2.0)


text = probe.as_python()
assert "static_print" not in text          # nothing printed, and nothing in the IR
assert "static_assert" not in text         # False did not raise either
```

Both statements vanished. The direct replacement for an assertion over host
values is a host `if` that raises — the condition executes at trace time, and
the untaken branch leaves no IR:

<!-- doctest: run -->
```python
COLS = 128


@pl.function
def checked(x: pl.Tensor[[64, COLS], pl.FP32]) -> pl.Tensor[[64, COLS], pl.FP32]:
    if COLS % 16 != 0:
        raise ValueError(f"{COLS} is not 16-aligned")
    return pl.tensor.muls(x, 2.0)


text = checked.as_python()
assert "ValueError" not in text            # the guard is host-side: no IR, no raise
```

## Mechanics

### Parser-owned diagnostics

| Construct | Under the parser | Under tracing |
| --------- | ---------------- | ------------- |
| `pl.static_print(*args)` | prints type/value info at parse time | silent no-op, returns `None` |
| `pl.static_assert(cond, msg)` | raises `ParserError` if false | silent no-op — even `False` does not raise |

Both stay importable everywhere: a kernel using them collects fine under
tracing, only the check never fires. When you still need the check:

- **a host `if` that raises** — for conditions over host values, as in the
  quickstart;
- **a verifier pass** — for checks over collected IR (see the passes
  documentation).

One trap worth naming: **never bind the result**. `s = pl.static_print(...)`
binds the no-op's `None`, and `None` is the TaskId sentinel — the binder
reifies it into a `pl.Scalar[pl.TASK_ID]` holding `pl.system.task_invalid()`,
and that op lands in your IR. Statement-position calls are safe.

The parser lane also used `static_print` to show what it *inferred* about a
value. Under tracing there is nothing to infer — the type is already fixed by
the producing operator — and the printed IR restates every type, so
`kernel.as_python()` is the replacement when you need one fact.

The parser's entry points live behind their own decorators: `pl.function_parser`
and `pl.program_parser` collect through the AST parser instead of tracing. They
are the permanent spelling for parser fixtures and round-trip source
(printer-emitted text), not a second way to write kernels — the unsuffixed
`pl.function` / `pl.program` are the user surface, and their collection frontend
is tracing.

### Typed constants

`pl.const(value, dtype)` builds a constant with an explicit dtype rather than
the default one inferred from the literal. It exists so the printer can
round-trip non-default constant types, and it is what you want when a
literal's width matters — or when a literal must *stage* at all:

```python
step = pl.const(1, pl.INT32)
```

It works in and out of a trace; inside a kernel it is the staged-initialization
spelling ([Control Flow](02-control-flow.md) carries the full rule).

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`static_print` prints nothing** | Under tracing it never does | Read the printed IR: `kernel.as_python()` restates every type |
| **`static_assert` did not catch a bad value** | It is a no-op under tracing | Host `if` + `raise`, or a verifier pass |
| **A `Scalar[pl.TASK_ID]` / `system.task_invalid` appeared in the IR** | You bound the no-op call's `None` result | Never bind `static_print` / `static_assert` results |
| **`TypeError: 'NoneType' object is not subscriptable` at trace time** | `x.shape` / `x.dtype` on a traced value are `None` — a staged value carries no shape to Python | Check the host constant, or read the shape from the printed IR |
| **A constant came out with the wrong width** | The dtype was inferred from the Python literal | Pin it with `pl.const(value, dtype)` |

## See Also

- [Language Syntax](06-syntax.md) — the execution-time behaviour of subscripts, operators, and names.
- [Functions and Programs](01-functions.md) — when each decorator traces.
- [Control Flow](02-control-flow.md) — `pl.const` as the staged-initialization spelling.
- [Python IR Syntax Specification](../../dev/language/00-python_syntax.md) — the parser's full surface, where the `static_*` checks still run.
