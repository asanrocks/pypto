# Values and Staging

Which values are decided while the kernel is being built, and which are decided when it
runs — the distinction every other tracing rule follows from.

> **Prerequisites:** [Types](00-types.md). This page explains the runtime side of the
> annotation; [Functions and Programs § the tracing frontend](01-functions.md) explains
> which decorator selects which frontend.

## Concept

A `@pl.jit_trace` kernel is **executed**. Collecting IR is a side effect of running the
body once, so every value in it has a provenance, and the provenance decides whether the
value survives into the compiled program:

| World | Where the value comes from | What the program gets |
| ----- | -------------------------- | --------------------- |
| **host** | Python: a literal, arithmetic on literals, a configuration object, a plain function's return value | nothing — the *value* is inlined where it was used, and the value itself is gone |
| **staged** | the signature (`pl.Scalar[dtype]`, `pl.Tensor`, tiles), `pl.const(...)`, a staged loop variable, or the result of a `pl.*` op | an operand or a parameter; the device supplies or computes it at run time |

The rule that follows from it: **a host value is visible to the compiler, a staged value is
visible to the hardware.** Tiling, tile shapes and which statements exist at all must be
host; anything the device may need to vary between calls must be staged.

Host data is not a second-class citizen. Because the body runs, ordinary Python —
`if`, `for`, dataclasses, list comprehensions, helper functions — is available to
*organize* the staged computation, and it leaves no trace in the program. This page covers
what the two worlds mean and where they are guarded; [Metaprogramming a Kernel](../tutorials/06-metaprogramming.md)
builds that up one step at a time.

## Quickstart: one factor, two worlds

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
ROWS = COLS = 64


def program_text(kernel, *args) -> str:
    """The lowered program a call would ship, without compiling for a device."""
    return ir.python_print(kernel.compile_for_test(*args))
```

<!-- doctest: run -->
```python
@pl.jit_trace
def baked(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: float,                                    # host: a compile-time knob
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, k), [0, 0], out)
    return out


@pl.jit_trace
def staged(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    s: pl.Scalar[pl.FP32],                       # staged: a runtime parameter
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [ROWS, COLS])
        pl.store(pl.mul(tile, s), [0, 0], out)
    return out


a = torch.randn(ROWS, COLS, dtype=torch.float32)
empty = torch.zeros(ROWS, COLS, dtype=torch.float32)

# Both compute the same thing for one value...
out = torch.zeros_like(a)
staged(a, 3.0, out, config=CFG)
assert torch.allclose(out, a * 3.0)

# ...but only the host knob changes the program, and only the staged factor
# survives into the signature.
assert program_text(baked, a, 2.0, empty) != program_text(baked, a, 7.0, empty)
assert program_text(staged, a, 3.0, empty) == program_text(staged, a, 9.0, empty)
assert "pl.Scalar[pl.FP32]" in program_text(staged, a, 3.0, empty)
assert "pl.Scalar[pl.FP32]" not in program_text(baked, a, 2.0, empty)
```

A different `k` is a **different program**: the literal is baked, so the dispatch cache
keys on the value. A different `s` is the **same program** with a different argument.

## Mechanics

### Reading a value's world

| You wrote | World | Why |
| --------- | ----- | --- |
| `k: float`, `n: int`, `flag: bool` in the signature | host | a plain Python annotation is a compile-time knob |
| `s: pl.Scalar[pl.FP32]` in the signature | staged | the annotation names a runtime parameter |
| `tile = pl.load(a, ...)`, `pl.add(tile, 1.0)` | staged | an op's result is an operand |
| `chunk * 16` inside `pl.range(...)` | staged | `chunk` is a staged loop variable, so the product is an operand |
| `CHUNKS = ROWS // TILE` in the body | host | arithmetic on host values runs now |
| `factor = pl.const(1.0, pl.FP32)` | staged | an explicit staged constant (a local with a runtime identity) |
| `cfg = TilePlan(...)`; `cfg.chunks` | host | an object is Python; its fields are host until one carries a staged value |

A host value passed where the signature declares a staged slot **lifts**: call
`staged(a, 3.0, out)` and `3.0` becomes a `pl.const` operand, because the declaration owns
the world. The reverse is refused with a `TypeError` naming the parameter: a staged value
cannot be baked into the program.

An object is only a container. A dataclass whose fields are host data is configuration; one
that carries a staged tile is a record the trace can pass around; neither is a kernel
parameter, and a bundle is expanded into the parameter list at the call site. See
[Functions and Programs](01-functions.md) for the four ways to reuse code, and
`examples/tracing/03_host_objects.py` for the three shapes.

### Control flow follows the subject

The same distinction decides what an `if` or a loop *is*. Nothing about the syntax decides
it — the domain of the thing being tested or iterated over does:

| Subject | `if` | `for` | `while` |
| ------- | ---- | ----- | ------- |
| **host** | one arm runs; the program holds no branch | `range` / list / tuple: the body runs once per entry, so the program holds one copy per iteration | runs now (a host condition ends it) |
| **staged** | a branch in the program; both arms are traced once, and the join threads the values each arm left behind | `pl.range` / `pl.parallel` / `pl.pipeline`: one loop in the program, its body traced once | an ordinary `while` over a staged condition: one `WhileStmt`, carried values rebound in the body |

Because a staged loop's body runs **once**, host code inside it is not repeated on the
device: the trace keeps one body, and the loop repeats it. `pl.unroll` is the exception
that proves the rule — it is a staged loop at trace time and a fully expanded body after
passes, which is why its bound must be a literal.

The parser frontend's loop spellings do not apply here: `pl.while_` and `pl.cond` are
parse-time syntax and raise `TypeError` inside a trace. Write `while i < n:` instead; the
predicate and the rebindings are staged, so the loop is recorded.

### Carried state

A value that changes across iterations must live somewhere. The two worlds have different
answers, and the trace enforces the difference rather than letting one silently bake:

```python
factor = pl.const(1.0, pl.FP32)             # staged state: a real carried value
for _i in pl.range(n):
    factor = factor * 2.0                   # rebind: the trace derives the carry
```

A **staged** local carries as above — bind it before the loop, rebind it in the body; the
printed program shows the derived `init_values` / `pl.yield_` pair. A **host** value may be
read inside staged control flow, but not advanced by it: `factor = 1.0` followed by
`factor = factor * 2.0` under `pl.range` raises `TypeError` ("loop-carried host state"),
because the body is traced once and that one iteration's value would be baked into every
iteration and into every read after the loop. Per-iteration host scratch under a fresh
name stays legal.

A staged branch joins the same way: rebinding a staged name in one arm only is legal —
the join threads the other path's value through — but a **host** name bound in one arm and
read after the join raises `TypeError` ("bound on only one branch"), since the read would
silently inline that branch's literal. Bind such a name in both arms, or make it staged.

## Edge Cases

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| A knob's value changed the program, or a second artifact appeared | The value is host and baked into the trace | Declare it `pl.Scalar[dtype]` to make it a runtime parameter |
| **`'Scalar' object cannot be interpreted as an integer`** | Python `range()` needs a host bound | Iterate `pl.range(n)`; a host `range` cannot express a runtime trip count |
| **`loop-carried host state`** | A host value is updated inside staged control flow | Seed it with `pl.const(...)` and rebind, or use a fresh name per iteration |
| **`bound on only one branch`** | A host name is bound in one arm of a staged `if` and read after the join | Bind it in both arms, or make it staged |
| **`pl.while_ is parser-frontend syntax …`** | Parse-time loop spelling in a traced kernel | Write an ordinary `while` over a staged condition |
| **A staged value rejected at a host knob** | The signature declares a plain Python type | Annotate the parameter as `pl.Scalar[dtype]` |
| Host `print` in a staged loop fires once | The body is traced once; only the staged structure repeats | Print from a host loop, or read the printed program instead |
| The program does not contain a branch the source spells | The predicate was host | Stage the predicate (read a `pl.Scalar`) if the device must choose |

## Worked examples

| Example | Shows |
| ------- | ----- |
| `examples/tracing/01_two_domains.py` | A host factor versus a staged factor, and the programs each builds |
| `examples/tracing/02_control_flow.py` | `range` expanding versus `pl.range` recording, and why the recorded one is size-invariant |
| `examples/tracing/advanced/07_diagnostics.py` | The loop-carried-host guard, its repair, and the statement-boundary trap that bodies fix |

Every example runs on the simulator and asserts the structural claim it makes, so the page
and the examples cannot drift apart silently.

## See Also

- [Types](00-types.md) — the annotations that declare a parameter's world.
- [Functions and Programs](01-functions.md) — `@pl.jit` versus `@pl.jit_trace`, and the spellings each frontend requires.
- [Control Flow](02-control-flow.md) — the staged loop forms and the SSA rules they come from.
- [Scopes and Placement](04-scopes.md) — where the staged statements run.
