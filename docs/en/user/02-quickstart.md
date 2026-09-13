# Quickstart

Write, compile, and inspect your first PyPTO kernels — at tensor level, where the
compiler places data for you.

> **Prerequisites:** PyPTO installed and importable — see [Installation](01-installation.md).
> Everything except the last section runs on a plain `pip install` — no NPU, and no ptoas
> either: `@pl.jit` detects whether ptoas is present and adjusts.

## Concept

A PyPTO kernel is ordinary Python that `@pl.jit` **traces**: the body executes as
Python at trace time, and every `pl.*` operation it performs is recorded as IR. Nothing
touches a device — you compile that IR and dispatch it later.

This page stays entirely at **tensor level**: you name whole arrays, apply operators to
them, and let the compiler decide what lands on chip and when. There is no `pl.load` or
`pl.store` anywhere below. Tile-level authoring — naming on-chip buffers and moving data
yourself — is a separate topic; see [Programming Model](03-programming-model.md) for what
it is and why you would reach for it.

Two structural facts shape every example:

- `@pl.jit` marks a **chip-level entry point**, which is control-plane code. Computation
  belongs on the execution plane, so the operators go inside
  `with pl.at(level=pl.Level.CORE_GROUP):` — the scope that says "this runs on chip".
  Omitting it fails with *"Misplaced tensor op ... should be inside InCore block"*.
- Outputs are written through a `pl.Out[...]` parameter, not returned as fresh arrays.

```python
import pypto.language as pl
import torch
```

## Quickstart: element-wise add

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(128, 128, dtype=torch.float32)
B = torch.randn(128, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def add(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[0:128, 0:128] = pl.add(a, b)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
compiled = add.compile(A, B, out, config=CFG)
print(f"Generated code in: {compiled.output_dir}")

add(A, B, out, config=CFG)
torch.testing.assert_close(out, A + B, rtol=1e-4, atol=1e-4)
```

| Line | What it does |
| ---- | ------------ |
| `@pl.jit` | Traces the body into an Orchestration entry point; a call with tensors specializes, compiles, and dispatches |
| `a: pl.Tensor` | A DDR tensor. Shape and dtype are read from the argument at the call; `In` is the default direction |
| `out: pl.Out[pl.Tensor]` | **Direction**: this parameter is written, not read |
| `with pl.at(level=pl.Level.CORE_GROUP)` | Marks the on-chip block. Operators are only legal inside one |
| `out[0:128, 0:128] = pl.add(a, b)` | Element-wise add over whole tensors, written back through an explicit slice. No offsets, no buffers, no data movement |
| `return out` | Returns the written tensor |
| `add.compile(A, B, out, config=CFG)` | Runs the pipeline for this specialization and returns a `CompiledProgram` |

Note what is *not* in that kernel: no tile type, no `pl.load`, no `pl.store`, no memory
space. The compiler's `ConvertTensorToTileOps` pass inserts all of it — you can see the
result in the pass dumps under `compiled.output_dir/passes_dump/`.

`pl.Out[...]` is load-bearing rather than decorative: it tells the compiler the buffer is
written, which decides whether the runtime uploads it before the call and downloads it
after. Every tensor parameter has a direction — `In` by default, or an explicit
`pl.Out[...]` / `pl.InOut[...]`.

`add`'s parameters are deliberately bare — `pl.Tensor`, no shape. That is the **lazy
lane**: the call itself specializes the kernel against its arguments. Fully annotated
parameters (`a: pl.Tensor[[128, 128], pl.FP32]`) pin the shape in the signature instead;
a kernel where *every* parameter is annotated is traced once, at decoration — see
[Compiling](#compiling).

> **Why `pl.at` is not optional.** Drop that line and keep the same body, and compilation
> fails at orchestration codegen: *"Misplaced tensor op 'tensor.add' in Orchestration
> function (should be inside InCore block)"*. A `@pl.jit` entry is control-plane code, and
> the scope is what moves the computation onto the execution plane.

## Mechanics

### Chaining operators

Intermediate values are ordinary Python names. They need no annotation, and no buffer is
declared for them — the compiler allocates whatever the chain requires:

<!-- doctest: run -->
```python
@pl.jit
def add_then_square(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        s = pl.add(a, b)
        out[0:128, 0:128] = pl.mul(s, s)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
add_then_square(A, B, out, config=CFG)
torch.testing.assert_close(out, (A + B) * (A + B), rtol=1e-4, atol=1e-4)
```

A shape annotation is an ordinary Python value, so a module-level alias
(`T = pl.Tensor[[128, 128], pl.FP32]`, then `a: T`) resolves like any other name — the
parser-era restriction to inline text is gone.

### Loops

`pl.range()` builds a loop in the IR, inside the on-chip scope. A value carried across
iterations is written as ordinary reassignment:

```python
@pl.jit
def accumulate(
    a: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(a, a)
        for i in pl.range(3):
            t = pl.add(t, a)      # carried across iterations
        out[0:128, 0:128] = pl.mul(t, t)
    return out
```

Rebinding `t` looks like mutation but is not: the IR is SSA, and the loop pass threads each
iteration's value through the loop as a carried value, giving each one its own name.
Reading `t` after the loop reads the last iteration's result.

Loop forms:

```python
for i in pl.range(10):        # 0 .. 9
for i in pl.range(0, 100, 2): # 0, 2, 4, ... 98
```

### Splitting work across functions

Beyond one kernel, put the computation in a `@pl.jit.incore` sub-function and let the
entry dispatch it. An `.incore` function is already on the execution plane, so it needs no
`pl.at`:

```python
@pl.jit.incore
def add_kernel(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    out[0:128, 0:128] = pl.add(a, b)
    return out

@pl.jit
def add_program(
    a: pl.Tensor,
    b: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    return add_kernel(a, b, out)      # discovered automatically — no registration
```

```text
add_program   (@pl.jit, Orchestration)  — control plane: dispatches
  └── add_kernel (@pl.jit.incore)       — execution plane: computes
```

The `@pl.jit` family, one decorator per IR function kind:

| Decorator | Becomes | Use for |
| --------- | ------- | ------- |
| `@pl.jit` | Orchestration | The chip-level entry point |
| `@pl.jit.incore` | InCore | A device kernel, outlined into its own file |
| `@pl.jit.inline` | Inline | A helper spliced into every call site |
| `@pl.jit.opaque` | Opaque | A separate IR function that may wrap loops and `pl.at` scopes |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | The HOST entry of a distributed (multi-card) program |

Sub-functions are discovered from the entry's body, so you just call them by name. One
deliberate exception: a plain `@pl.jit` entry does **not** discover other `@pl.jit`
entries — only `.host` reaches across the chip boundary, which keeps two unrelated
top-level kernels from silently folding into one program.

### Compiling

```python
compiled = add.compile(A, B, out, config=CFG)
print(f"Generated code in: {compiled.output_dir}")
```

`compile()` returns a **`CompiledProgram`** — not a path. `compiled.output_dir` is a
`pathlib.Path` holding:

```text
kernels/       generated device kernels, one per InCore function
orchestration/ generated host-side C++
ptoas/         the .pto (MLIR) and its assembled output, when ptoas is available
report/        compile-time reports, including perf hints
debug/         a runnable `run.py` harness
passes_dump/   per-pass IR snapshots
```

**`compile()` needs no ptoas flag.** `@pl.jit` checks for the binary itself — `$PTOAS_ROOT/ptoas`,
or `ptoas` on `PATH` — and skips the assembly step when it is absent. A machine with only
the Python package still gets IR and generated C++.

Where `compile()` gets its shapes:

| Signature style | When it traces | Compile it as |
| --------------- | -------------- | ------------- |
| Bare `pl.Tensor` (at least one parameter) | At the call — the lazy lane | `kernel.compile(a, b, out)` with sample tensors |
| Every parameter fully annotated | At decoration — the eager lane | The decorated name *is* the `ir.Function`; wrap it in a program: `ir.compile(ir.Program([kernel], kernel.name, kernel.span), platform=...)` |

Sample tensors are read for shape and dtype only; contents are never touched, so
`torch.empty(...)` is enough. The eager lane has no `compile()` of its own — see
[Running on Device](00-getting_started.md#compiling-from-the-signature-no-sample-tensors)
for the full eager route.

> **`compile()`'s arguments are the kernel's, not the compiler's.** `compile(*args, **kwargs)`
> binds the decorated function's own parameters. Passing an `ir.compile()` option there —
> `compile(skip_ptoas=True)` — is either rejected as an unexpected kernel argument or
> silently ignored. Compile-side options travel through `config=RunConfig(...)`, whose
> compile knobs are forwarded to `ir.compile()`.

To inspect a kernel without producing code, read its IR. The lazy lane — `add` above —
traces per call, and `compile_for_test(*sample_args)` is the spelling that produces its
IR: it runs the Default pass pipeline and returns the post-pass `ir.Program`:

```python
import torch

x = torch.zeros((128, 128), dtype=torch.float32)
program = add.compile_for_test(x, x, x)
```

It performs no code generation and writes no artifacts. This makes it fast, but also
means it does **not** catch codegen-stage errors such as the misplaced-tensor-op failure
above. Use `compile()` to verify code generation.

A fully annotated kernel is traced at decoration, so the decorated name already *is* the
`ir.Function` — `print(kernel.as_python())` reads the traced IR before any pass, with no
call and no sample tensors at all.

### Reading the IR

A lazy-lane kernel has no `as_python()` of its own — the IR does not exist until a call
traces it. Read the `ir.Program` returned by `compile_for_test()` directly, or read the
`program` stored in the `CompiledProgram` returned by `compile()`:

```python
print(program.as_python())
print(compiled.program.as_python())
```

What comes back is the specialized `@pl.program` class your jit functions turned into,
which is also the clearest way to see what `@pl.jit` actually does — and, at tensor level,
what the compiler filled in on your behalf. Compare the pass dumps before and after
`ConvertTensorToTileOps` to watch `pl.tensor.add` become tile loads, a tile add, and a
store.

## Running it on hardware

> **Needs the runtime and a device or simulator platform.** Nothing above this section does.

```python
import torch
from pypto.runtime import RunConfig

a = torch.full((128, 128), 2.0, dtype=torch.float32)
b = torch.full((128, 128), 3.0, dtype=torch.float32)
out = torch.zeros((128, 128), dtype=torch.float32)

add(a, b, out, config=RunConfig())          # compiles, caches, dispatches
assert torch.allclose(out, a + b, rtol=1e-5, atol=1e-5)
```

Calling a `@pl.jit` function directly does the whole thing: specialize on the argument
shapes and dtypes, compile, cache, dispatch. Later calls with the same shapes reuse the
cached compilation. `examples/beginner/01_hello_world.py` is this pattern, at tile level.

## Edge Cases

> **Fatal pitfall:** a `@pl.jit` body is **executed as Python at trace time** — it is not
> parsed, and it does not run on device. A `print()` or `assert` in the body runs once,
> when the kernel is traced (at decoration for a fully annotated kernel, at the first
> call otherwise) — never per launch. Stepping through it in a debugger shows you the
> trace, not the computation. Debug by reading `compiled.program.as_python()`.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **`Misplaced tensor op ... should be inside InCore block`** | Operators directly in the `@pl.jit` body | Wrap them in `with pl.at(level=pl.Level.CORE_GROUP):`, or move them into a `@pl.jit.incore` sub-function |
| **`TypeError: 'ir.Function' object is not callable`** (or `has no attribute 'compile'`) | Every parameter is annotated, so the kernel was traced at decoration and the name is the `ir.Function` | Compile it with `ir.compile(ir.Program([...]), ...)`, or leave one parameter bare to get the call/compile surface |
| **`Cannot reassign 'out' with a different type`** | The expression's dtype differs from the declared `Out` dtype | Match them, or bind the result to a new name |
| **`got an unexpected keyword argument 'skip_ptoas'`** | An `ir.compile()` option passed to `compile()` | Pass compile options via `config=RunConfig(...)` |
| **Output tensor comes back unchanged** | Result written to a parameter not declared `pl.Out[...]` | Add the direction |
| **`compile_for_test()` succeeds but `compile()` fails** | `compile_for_test()` does not run code generation | Expected — use `compile()` as the codegen check |
| **`AttributeError: as_python`** | Called on the jit function | It lives on the IR: `compiled.program.as_python()` |

`PYPTO_PROG_BUILD_DIR` is a **runtime environment variable** —
`PYPTO_PROG_BUILD_DIR=/tmp/out python kernel.py` relocates every compile output.
Distinguish it from `SIMPLER_HOST_STRACE` and `SIMPLER_DFX`, which are **compile-time
macros** of the runtime (`-DXXX=1` at build time); setting those in the shell has no
effect.

## See Also

- [Installation](01-installation.md) — getting to the point where these examples import.
- [Programming Model](03-programming-model.md) — tensor vs. tile vs. block level, the two planes, the memory hierarchy, and the execution model.
- [Language Guide](language/index.md) — the full surface: tile-level authoring, `pl.load` / `pl.store`, memory spaces, and the `@pl.function` / `@pl.program` form `@pl.jit` specializes into.
- [Operations](ops/index.md) — the operator surface across `pl.*`, `pl.tensor.*`, and `pl.tile.*`.
- [Running on Device](00-getting_started.md) — resident device tensors, explicit dispatch, benchmarking, distributed execution.
- `examples/beginner/` and `examples/intermediate/` — tile-level kernels in the same `@pl.jit` idiom.
