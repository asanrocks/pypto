# Functions and Program Structure

Function declaration forms, parameter directions, cross-module reuse, and how to
print IR back to Python syntax.

## Constants and compilation reuse

`@pl.jit` traces the callable: the body runs as ordinary Python while the trace
is taken, so a name it inherits from its own module or an enclosing function is
read from that namespace at trace time — the value itself is used, never a
source form of it. Literals (`int`, `float`, `bool`, `str`, `None`), the `pl`
dtype and enum constants (`pl.INT8`, `pl.Mem.Vec`, `pl.PadValue.zero`, `pl.NZ`),
and lists/tuples nested from those all work, because they are just Python values
the trace observes. (The old rule that a constant "must have a source form"
belonged to the AST-era runtime — quarantined as `pl.jit_specializer` — which
generated `@pl.program` source and re-parsed it in a namespace holding only
`pl` and `pld`.)

When the read happens depends on the lane. The **eager lane** — every parameter
facade-annotated — traces at decoration and returns a bare `ir.Function`, so
every inherited name is baked into that function at decoration time; rebinding
it afterwards does not change the artifact (compile it with the
one-function-program route described in
[Functions](../../user/language/01-functions.md)). The **lazy lane** — a bare
`pl.Tensor` parameter, or anything call-time dependent — defers the trace to
each call or `compile()`, so a rebind is observed by the next trace:

```python
BLOCK = 32

@pl.jit
def slice_kernel(x: pl.Tensor) -> pl.Tensor:
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.slice(x, [BLOCK, 128], [0, 0])
    return result

first = slice_kernel.compile(x)    # traced with BLOCK = 32
BLOCK = 64
second = slice_kernel.compile(x)   # re-traced with 64 rows
```

Host knobs and the cache key:

- A parameter with a plain Python annotation (or none) is a **host knob**: a
  Python scalar argument is baked into the trace as a `Const`, and its *value*
  is part of the dispatch cache key — two calls with different values never
  share an artifact.
- `pl.runtime(value, dtype)` is the **runtime scalar** spelling for a
  `pl.Scalar[...]` parameter: the key carries the dtype only, never the value.
- Dispatch (`kernel(*args, config=...)`) caches on the bridged signature — each
  tensor's shape, dtype, and layout, each host scalar's value, each runtime
  scalar's dtype — plus the compile identity described below. Names the body
  inherits from its module are *not* tracked: once an entry is warm it keeps
  serving the artifact traced under the old value, so rebind a referenced
  constant before its first dispatch, or use `compile()` for a fresh trace.
- `kernel.compile(*args, config=...)` never reads or writes that cache — each
  call builds a fresh artifact; explicit compile-then-call freezes one
  materialization (03-entrance.md §5.5).

Each trace observes the enclosing namespaces as they are while it runs;
mutating a captured object during a trace is not tracked by any key.

### Compile options and diagnostic requests

Both dispatch and `kernel.compile(*args, config=...)` resolve `config` the same
way: the compile-side fields of `RunConfig` (strategy, dumps, diagnostics)
forward to `ir.compile`, while runtime-side controls such as `device_id` and
`codegen_only` apply at dispatch. Omitting `config` uses the same defaults as
`RunConfig()`: the `a2a3sim` platform, the default optimization strategy, and
pass dumps disabled.

The two surfaces differ in caching. The dispatch cache keys on the bridged
signature *plus the compile identity* — platform, optimization strategy,
distributed config, `analyze_auto_scopes_for_deps`, the effective memory
planner, and the L0C double-buffer flag, each resolved from the `RunConfig` and
any active `PassContext` — so two compile configurations never share one
artifact. `kernel.compile()` has no cache at all: every call builds a fresh
artifact.

Diagnostic requests therefore route through `compile()`. `dump_passes=True` (or
a dump level other than `PassDumpLevel.NONE`), `dump_ptoas_passes=True`, and an
explicit `save_kernels_dir` are all honored there by construction, and repeating
a request regenerates the output. The dispatch cache deliberately does not key
on these knobs — they do not change the artifact — so a warm entry is reused
and the requested dumps are *not* emitted; ask for diagnostics through
`compile()` when you need the output. Diagnostic controls never split the
dispatch cache.

```python
from pypto.runtime import RunConfig

first = slice_kernel.compile(x)
second = slice_kernel.compile(x)  # a fresh build — compile() has no cache
dumped = slice_kernel.compile(
    x, config=RunConfig(dump_passes=True, save_kernels_dir="debug_kernel")
)
```

### Prepare binaries without executing

`kernel.compile(*args, config=...)` is the prepare-without-execute surface: it
traces, runs the pass pipeline, and builds every kernel and orchestration
binary, then returns the `CompiledProgram` without dispatching it. No NPU is
initialized and no runtime worker is created; the build host still needs the
target compiler, SDK, and runtime Python/native dependencies.

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def add_three(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, 3.0), [0, 0], out)
    return out


import torch

x = torch.randn(16, 16, dtype=torch.float32)
out = torch.empty(16, 16, dtype=torch.float32)
config = RunConfig(platform="a2a3")
prepared = add_three.compile(x, out, config=config)  # shapes and dtypes only
# Later, on a host with an available NPU:
# prepared(x, out, config=config)
```

Sample tensors are bridged by metadata — shape and dtype — never their
contents. Scalar arguments follow the parameter worlds above: a plain-annotated
parameter is a host knob whose value is baked at trace time, and a
`pl.Scalar[...]` parameter takes `pl.runtime(value, dtype)` to stay a runtime
argument. Calling the returned object dispatches with the full argument list.
Compilation errors propagate to the caller; a failed build can be retried by
calling `compile()` again.

The persistent artifact store, the `warmup()` method, and the metadata-only
warmup CLI documented in [Persistent JIT Cache](../10-jit-cache.md) belong to
the quarantined `pl.jit_specializer` runtime (`python/pypto/jit/`); the tracing
jit has only the in-process dispatch cache described above.

## Functions

```python
# Single return type
def function_name(param1: pl.INT64, param2: pl.FP32) -> pl.INT64:
    x: pl.INT64 = param1 + 1
    return x

# Multiple return types
def function_name(x: pl.INT64) -> tuple[pl.INT64, pl.INT64]:
    y: pl.INT64 = x + 1
    z: pl.INT64 = x * 2
    return y, z

# No return types
def function_name(x: pl.INT64):
    y: pl.INT64 = x + 1

# With function type
@pl.function(type=pl.FunctionType.Orchestration)
def orchestrator(n: pl.INT64) -> pl.INT64:
    return n + 1

@pl.function(type=pl.FunctionType.InCore)
def aicore_kernel(x: pl.INT64) -> pl.INT64:
    return x * 2
```

### Function Types

| Type | Usage | Description |
| ---- | ----- | ----------- |
| `pl.FunctionType.Opaque` | Default | Unspecified function type |
| `pl.FunctionType.Orchestration` | Host/AICPU | Control flow and dependency analysis |
| `pl.FunctionType.InCore` | AICore | Sub-graph on specific AICore (unspecialized) |
| `pl.FunctionType.AIC` | Cube core | Cube core kernel (specialized InCore) |
| `pl.FunctionType.AIV` | Vector core | Vector core kernel (specialized InCore) |
| `pl.FunctionType.Group` | Multi-core | Co-scheduled group of AIC + AIV kernels |
| `pl.FunctionType.Graph` | Host/AICPU | Recordable orchestration fragment, replayed by the `host_build_graph` runtime (see below) |

When no type is specified, functions default to `Opaque`.

### Graph Fragments

`pl.FunctionType.Graph` marks a function as a recordable orchestration fragment.
Under the `host_build_graph` runtime each call site becomes one task launch that
the runtime records on the first call and replays afterwards, so N calls cost one
graph build instead of N:

```python
@pl.program
class Decoder:
    @pl.function(type=pl.FunctionType.Graph)
    def layer(self, cur, normed, next_hidden, wq, layer_base: pl.Scalar[pl.INDEX]):
        ...

    @pl.function
    def decode(self, cur, normed, next_hidden, wq):
        for i in pl.range(40):
            self.layer(cur, normed, next_hidden, wq, i * 5120)
```

One Graph function is one recorded topology: the runtime identifies the recording
by the address of the emitted C++ function, so there is no cache key to name or
keep unique.

### Parameter Directions

Parameters can have `In` (default), `Out`, or `InOut` directions using wrapper types:

```python
@pl.function(type=pl.FunctionType.InCore)
def kernel(
    qi: pl.Tensor[[16, 128], pl.BF16],                   # In (default)
    output: pl.InOut[pl.Tensor[[16, 128], pl.FP32]],      # InOut
    result: pl.Out[pl.Tensor[[16, 128], pl.FP32]],        # Out
    scale: pl.Scalar[pl.FP32],                             # In (default)
) -> pl.Tensor[[16, 128], pl.FP32]:
    ...
```

| Direction | Wrapper | Description |
| --------- | ------- | ----------- |
| `In` | None (default) | Read-only input parameter |
| `Out` | `pl.Out[type]` | Write-only output parameter |
| `InOut` | `pl.InOut[type]` | Read-write input/output parameter |

**Constraint:** `Scalar` parameters cannot have `InOut` direction (raises `ParserTypeError`).

#### Writing an `Out` / `InOut` parameter

**A bare assignment does not write the parameter.** It rebinds the Python name:
the parameter Var is re-pointed at a freshly computed tensor and the caller's
buffer is left untouched. The program still compiles and runs. What the caller
gets back depends on the direction: an `Out` buffer is freshly allocated and
never initialised, so it reads as garbage; an `InOut` buffer still holds the
input the caller passed in, so the result is silently stale.

```python
out = pl.add(a, b)          # ❌ writes nothing
out[:] = pl.add(a, b)       # ✅ writes the whole tensor
```

Use the subscript form to write through the parameter:

| Spelling | Writes | Use when |
| -------- | ------ | -------- |
| `out[:] = <expr>` | the whole tensor | the result is the entire output |
| `out[<slices>] = <expr>` | that sub-window | writing part of the output |
| `out = pl.assemble(out, <expr>, <offset>)` | a window at `<offset>` | the explicit form the subscript sugar builds |
| `out = <expr>` | **nothing** | never — see the warning below |

Only the first and third are equivalent, and only when the slice covers the full
extent and `<offset>` is all zeros.

##### The `OutParamWriteDropped` warning

The compiler reports a bare assignment that drops the write:

```text
[warning] [OutParamWriteDropped] (pipeline_input) Assigning to Out parameter 'out'
in function 'main' rebinds the name only — the caller's buffer is never written.
Use 'out[:] = <expr>' to write the whole tensor, or 'out[<slices>] = <expr>' for
a sub-window. at repro.py:12:9
```

The check is data-flow based, not syntactic. A value can reach the parameter
through a loop carry — `pl.store` writes through the parameter, and the carried
binding keeps that alias — and that is a genuine write-through, so it stays
silent:

```python
for col in pl.range(0, n, chunk):
    data = pl.store(local, [0, col], data)   # the rebind is the loop's derived carry
data = pld.tensor.allreduce(data, signal, ...)   # the carried `data` *is* the parameter; no warning
```

The check is deliberately conservative: a value that merely *reads* the
parameter, such as `out = pl.add(out, b)`, is not reported even though it also
drops the write. Telling "reads the parameter" from "writes through the
parameter" needs per-op write semantics the operator registry does not record, and
a false warning on correct code costs more than a missed one. Write
`out[:] = pl.add(out, b)` when the whole tensor is meant.

Disable the check with `disabled_diagnostics` if it is not useful for your
program:

```python
disabled = passes.DiagnosticCheckSet()
disabled.insert(passes.DiagnosticCheck.OutParamWriteDropped)
ir.compile(program, disabled_diagnostics=disabled)
```

## `@pl.program` Classes Defined in Branches

Under tracing, `@pl.program` executes the class body as Python at decoration —
the decorator receives the class object the `class` statement just built, so
there is nothing to locate and no ambiguity to resolve. One function may define
the same class name in several branches (every one of them carries the same
`__qualname__`); each definition is its own decoration and produces its own
`Program`:

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # traced from *this* body
            @pl.function
            def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
                return pl.add(x, 1.0)
        return Prog

    @pl.program
    class Prog:                                    # and this one from *this* body
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, 3.0)
    return Prog
```

A closure variable parameterizes one definition the same way — `scale` below is
read from the enclosing scope while the trace runs:

```python
def make(scale):
    @pl.program
    class Prog:                                    # one definition, parameterised
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

The source-location machinery this section used to describe — finding the
`class` statement from the line numbers of its methods, and rejecting genuinely
ambiguous definitions with a `ParserSyntaxError` — is real only for the
parser-owned `@pl.program_parser` entrypoint (round-trip fixtures), which still
collects the class body from source. Tracing never reads source.

## Cross-Module Function Reuse

Functions defined outside a `@pl.program` class can be reused via two mechanisms.

### External `@pl.function` Calls

An externally-defined `@pl.function` can be called by name inside `@pl.program`. The function is automatically added to the Program and an `ir.Call(GlobalVar, args)` is emitted.

```python
@pl.function
def softmax(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    ...

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = softmax(x)   # ir.Call(GlobalVar("softmax"), [x])
        return y
```

**Rules:**

- Uses the function's `.name` as GlobalVar (aliases are transparent)
- External and internal function names must not conflict
- Two different externals with the same `.name` is an error
- Same external called from multiple methods is added once

### `@pl.inline` Decorator

`@pl.inline` captures a function for statement-level inlining. No function is added to the Program — the body is expanded at each call site.

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    result: pl.Tensor[[64], pl.FP32] = pl.mul(x, 2.0)
    return result

@pl.program
class MyModel:
    @pl.function
    def main(self, x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
        y: pl.Tensor[[64], pl.FP32] = normalize(x)  # statements inlined in-place
        return y
```

**Rules:**

- Argument count must match parameter list exactly
- Closure variables from the inline definition site are available
- Inline functions can be called multiple times (each expansion is independent)
- Nested inline calls are supported

## Complete Example

### Tensor Operations (Loop with a derived carry)

```python
import pypto.language as pl

@pl.function
def loop_sum(n: pl.Scalar[pl.INT64]) -> pl.Scalar[pl.INT64]:
    total = pl.const(0, pl.INT64)
    for i in pl.range(n):
        total = total + i
    return total
```

The accumulator is a typed staged constant — a bare `0` would inline the literal
every iteration — and the rebind `total = total + i` is itself the loop's carry;
nothing is declared. Printing this function after the SSA pass yields round-trip
text that renders the carry as `init_values=`/`pl.yield_()`; that text is the
printer's dialect, parsed back by `pl.parse` — never tracing input.

### Tile Operations (Tile-based computation)

```python
import pypto.language as pl

@pl.program
class BlockExample:
    @pl.function
    def tile_add(
        self,
        input_a: pl.Tensor[[64, 64], pl.FP32],
        input_b: pl.Tensor[[64, 64], pl.FP32],
        output: pl.Tensor[[64, 64], pl.FP32],
    ) -> pl.Tensor[[64, 64], pl.FP32]:
        tile_a: pl.Tile[[64, 64], pl.FP32] = pl.load(input_a, [0, 0], [64, 64])
        tile_b: pl.Tile[[64, 64], pl.FP32] = pl.load(input_b, [0, 0], [64, 64])
        tile_c: pl.Tile[[64, 64], pl.FP32] = pl.add(tile_a, tile_b)
        result: pl.Tensor[[64, 64], pl.FP32] = pl.store(tile_c, [0, 0], output)
        return result
```

## Printing IR Nodes

Use `as_python()` on any IR node to get its Python representation:

```python
print(stmt.as_python())          # "x: pl.Scalar[pl.INT64] = a + b" (default "pl" prefix)
print(stmt.as_python("ir"))      # "x: ir.Scalar[ir.INT64] = a + b" (custom prefix)
```

### Concise Mode

Pass `concise=True` to omit intermediate type annotations. Function signature types (parameters and return) are always preserved:

```python
print(func.as_python())                  # verbose (default): type on every assignment
print(func.as_python(concise=True))      # concise: omits intermediate type annotations
```

Verbose output:

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y: pl.Tensor[[64, 128], pl.FP32] = pl.some_op(x)
    result: pl.Tensor[[64, 128], pl.FP16] = pl.cast(y, pl.FP16)
    return result
```

Concise output:

```python
def main(self, x: pl.Tensor[[64, 128], pl.FP32]) -> pl.Tensor[[64, 128], pl.FP16]:
    y = pl.some_op(x)
    result = pl.cast(y, pl.FP16)
    return result
```

The free function `ir.python_print(node)` is also available and supports the same parameters.

## See Also

- [Python IR Syntax Specification](00-python_syntax.md) — types and expressions
- [Statements and Control Flow](01-statements.md) — statement forms inside a function body
- [Integrating Hand-Written C++ Kernels](04-external-kernels.md) — calling external kernels
