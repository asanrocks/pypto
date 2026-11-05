# Functions and Programs

How a Python function becomes an IR function, which decorator to reach for, and how
functions call each other.

> **Prerequisites:** [Types](00-types.md).

## Concept

A decorator turns your function into an IR function, and there are two frontends that get
there differently:

| Frontend | How the body becomes IR | Reach for it when |
| -------- | ----------------------- | ----------------- |
| `@pl.jit` (**parsing**) | the decorator **parses the source**; the body never runs as Python | the kernel is expressible as syntax the parser understands, and you want its parse-time errors and its full IR vocabulary |
| `@pl.jit_trace` (**tracing**) | the decorator **executes an instrumented copy**; collecting IR is a side effect of running it | ordinary Python — host loops, dataclasses, helper functions — has to *organize* the staged computation |

Both produce IR functions of the same kinds and lower through the same pipeline, so one
repository — or one program — may hold both; the choice is per definition. What they share
is what a staged operation means; what differs is when it is produced, and therefore what
an annotation means:

```python
def scale_parsed(x: pl.Tensor, k: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor]): ...
def scale_traced(x: pl.Tensor, k: float, out: pl.Out[pl.Tensor]): ...
#                the same first parameter is a staged tensor in both frontends,
#                but `k` is a runtime parameter above and a baked knob below
```

That difference is the whole of [Values and Staging](07-staging.md): which values are
decided while the kernel is built, and how the domain of a predicate or iterable then
decides what the program contains. This page covers the decorators themselves, the
reuse forms, and how to move a kernel between the frontends; the sections below say which
one they describe.

**`@pl.jit` is how the rest of this manual writes kernels.** Types come from the arguments
at the first call, the function specializes, and sub-functions are discovered
automatically — you call them by name and the decorator finds them. Its rules are the ones
stated on this page; where the tracing frontend differs, the section says so.

You will also meet `@pl.function` inside `@pl.program`: a class where each method is one
IR function and calls between them are written `self.other(...)`. That form is a
one-to-one transcription of the IR, and it exists mainly for writing compiler test cases,
where a test needs to state a program's exact shape without ever running it. As a user you
do not need it — [the section below](#plfunction-and-plprogram) is there for when you
read a compiler test or a piece of printed IR.

## Quickstart: an entry point and a device kernel

```python
import pypto.language as pl

@pl.jit.incore
def add_kernel(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out[:] = pl.add(a, b)
    return out

@pl.jit
def entry(
    a: pl.Tensor[[128, 128], pl.FP32],
    b: pl.Tensor[[128, 128], pl.FP32],
    out: pl.Out[pl.Tensor[[128, 128], pl.FP32]],
):
    out = add_kernel(a, b, out)      # sub-function discovered automatically
    return out
```

| Line | What it does |
| ---- | ------------ |
| `@pl.jit.incore` | Marks a device kernel — the execution plane, where operators live |
| `@pl.jit` | Marks the chip-level entry point — the control plane, which dispatches |
| `add_kernel(a, b, out)` | A plain call; the decorator discovers the callee and wires it in |
| `out: pl.Out[...]` | Declares the direction, which is how the compiler orders this task |

To read the IR this becomes, call `entry.lower(*args)` for the post-pass `ir.Program`, or
`entry.compile(*args)` and print `compiled.program.as_python()`.

## Mechanics

### The `@pl.jit` family

Five variants, one per IR function kind, so a single program can span host, chip, and
core levels:

| Decorator | IR target | Use for |
| --------- | --------- | ------- |
| `@pl.jit` | `Orchestration` | Chip-level entry point that dispatches InCore work |
| `@pl.jit.host` | `level=HOST, role=Orchestrator` | HOST entry — allocates window buffers, dispatches chip orchestrators per rank |
| `@pl.jit.incore` | `InCore` | A device kernel (accepts `level=` to target a specific hierarchy level) |
| `@pl.jit.inline` | `Inline` | Helper spliced into every call site by `InlineFunctions` |
| `@pl.jit.opaque` | `Opaque` | A separate IR function that may hold orchestration loops and `pl.at` scopes |
| `@pl.jit.graph` | `Graph` | A recordable orchestration fragment — the `host_build_graph` runtime records its task topology on the first call and replays it after, so N calls cost one graph build rather than N. Requires compiling under `RuntimeKind.HOST_BUILD_GRAPH` |

`@pl.jit.graph` has a scope form: `with pl.graph("name"):` marks a region *in place*
instead of splitting it into a function. The two compile to the same thing — the region is
outlined into a Graph function named after `name` — so the choice is ergonomic. Use the
decorator when the layer is already its own function; use the scope when the region is a
slice of a larger orchestration body you would rather not split up:

```python
@pl.jit
def decode(w: pl.Tensor, hidden: pl.InOut[pl.Tensor]):
    for layer in pl.range(40):
        with pl.graph("decoder_layer"):        # recorded once, replayed 39 times
            ...
    return hidden
```

The name is required and becomes the recorded graph's identity, so keep it stable. A Graph
region cannot nest inside another Graph region, nor inside `pl.at` / `pl.cluster` /
`pl.spmd` — those become a single device task, whereas a Graph region records a topology of
them. All three are compile errors.

Sub-function dependencies (`.incore` / `.inline` / `.opaque` / `.graph`) are auto-discovered from
the entry's body — call them by name. The name you call is resolved in the entry's own
namespace, so an aliased import (`from kernels import matmul as mm`, or a plain
`mm = matmul` rebinding) is discovered like any other binding; the generated program still
names the function after its `def`. When two distinct sub-functions share that name — two
modules each defining `helper`, or two kernels from one factory — the second one generated
is suffixed (`helper`, `helper__2`), so both specializations survive; the entry keeps its
own name. A `@pl.jit.host` entry additionally discovers
`@pl.jit` chip-orchestration dependencies, so a full distributed program needs no
`@pl.program` class.

The fragment below shows only the discovery structure — the kernel bodies are elided, and
the distributed types it names are covered in the distributed chapter, which is not
written yet:

```python
import pypto.language.distributed as pld

@pl.jit.inline
def reduce_step(local, peer, out): ...

@pl.jit
def chip_orch(inp: pl.Tensor, out: pl.Out[pl.Tensor],
              data: pl.InOut[pld.DistributedTensor], peer: pl.Scalar[pl.INT32]):
    return reduce_step(inp, peer, out)      # auto-discovered sub-function

@pl.jit.host
def host_orch(
    inputs: pl.Tensor[[2, 1, 256], pl.FP32],
    outputs: pl.Out[pl.Tensor[[2, 1, 256], pl.FP32]],
):
    data_buf = pld.alloc_window_buffer(256 * pl.FP32.get_byte())
    for r in pl.range(pld.world_size()):
        data = pld.window(data_buf, [1, 256], dtype=pl.FP32)
        chip_orch(inputs[r], outputs[r], data, (r + 1) % pld.world_size(), device=r)
    return outputs
```

Plain `@pl.jit` entries do **not** discover other `@pl.jit` entries — only `.host`
reaches across the chip boundary. That keeps two unrelated top-level kernels from
silently folding into one program.

`@pl.jit.host` rejects `level=` (HOST is implicit).

### The tracing frontend: `@pl.jit_trace`

`@pl.jit` parses a kernel body; `@pl.jit_trace` records it by executing an
instrumented copy instead. The choice is per definition, so both can coexist in
a program -- but not across a call: a helper must use the same decorator family
as the entry that calls it. What the value worlds mean is
[Values and Staging](07-staging.md); this section is the decorator and the
artifact it builds.

The observable difference is what an annotation means. Under `@pl.jit_trace`, a
plain Python annotation (`scale: float`) is a compile-time knob baked into the
kernel, while `pl.Scalar[dtype]` is a runtime parameter whose value arrives at
dispatch ([Values and Staging](07-staging.md) covers the rule and its guards,
including what happens when a loop tries to advance a baked value):

```python
@pl.jit_trace
def scale_relu(
    x: pl.Tensor,
    scale: float,                 # host knob: baked; changing it recompiles
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        pl.store(pl.relu(pl.mul(tile, scale)), [0, 0], out)
    return out
```

Decoration runs nothing. The body is collected once at the first `.fn`,
`.compile()`, `.lower()`, or host call and reused after; a bare `pl.Tensor`
leaves the shape unknown, so that kernel re-traces per call. A body may read
`x.shape` — the extents it sees are the call site's, static dims as constants
and dynamic dims as `pl.tensor.dim` reads. Tensor contents and
a `pl.Scalar` parameter's value never split the compiled artifact; a shape,
dtype, or host-knob change does. A `config=RunConfig(...)` call honors the same
compile knobs as `@pl.jit`; asking it to dump passes or profile a compilation
always runs the compiler rather than reusing a cached artifact.

A host knob is read-only inside a staged loop: the body is traced once, so
rebinding it there (`factor = factor * 2.0` under `for _ in pl.range(n)`) would
bake one iteration's value into all of them. Carry the state as a staged value
instead — bind it with `pl.const(...)` and let the loop update it.

This frontend covers explicit-output entries, their `incore` / `inline`
callees, and the `host` / `graph` orchestrators built over them. Outputs created
by the return value, and `@pl.function` / `@pl.program` declarations, remain on
the `@pl.jit` lane; the `@pl.jit_trace` spellings for them (`opaque`, `extern`)
raise at decoration. An external kernel is a declaration with no Python body to
trace: call a `@pl.jit.extern` stub from a traced body and it joins the entry's
assembly.

### Moving a kernel to the tracing frontend

A port changes the frontend, not the kernel. Both lanes lower through the same pipeline and
stop at the same place — `.lower(*args)` on the parsing lane, `compile_for_test(*args)` on
the tracing lane — so the ported kernel's program can be compared against the original's
before anything is generated. `examples/tracing/advanced/08_migration.py` does exactly that
for one kernel and prints the equality.

What usually changes at the port are the spellings the tracing dialect requires:

| Parsing lane | Tracing lane |
| ------------ | ------------ |
| `vlen: pl.INDEX`, `scale: pl.FP32` | `pl.Scalar[pl.INDEX]`, `pl.Scalar[pl.FP32]` — a bare dtype names no world |
| `for i, (acc,) in pl.range(n, init_values=(acc0,)):` + `pl.yield_` | bind `acc = acc0` before the loop, rebind in the body; the trace derives the carry |
| `pl.while_` / `pl.cond` | an ordinary `while` over a staged condition |
| `pl.Out[pl.Tensor]` on an `.inline` helper parameter | a bare `pl.Tensor` — the helper is spliced into the caller, so the parameter is already an alias |

A host literal used as loop-carried state is the fourth: declare it with `pl.const(...)`
([Values and Staging](07-staging.md)). Everything else — scopes, tile ops, task dependencies,
`pl.split`, valid shapes — carries over unchanged, which is why the sweep can be
incremental: port one kernel, compare its program, keep both lanes in the same program.

Equality at that boundary is not a claim about code generation, runtime numerics, or cache
identity; the developer page
([Tracing Frontend Boundaries](../../dev/language/08-tracing-frontends.md)) states what each
piece of evidence measures.

### Reusing code in a traced kernel

Four forms reuse code, and they differ in what the program ends up holding. Pick by
what the reuse *is*, not by what reads best at the call site:

| You want | Write | What the program gets |
| -------- | ----- | --------------------- |
| an expression — a helper that computes a staged value | a plain Python function returning `pl.*` results | the ops, in place; the helper itself leaves nothing |
| statements you call like a function | `@pl.jit_trace.body` | the statements at the call site — no function, no `GlobalVar`, no `ir.Call` |
| a typed helper the inlining pass erases | `@pl.jit_trace.inline` | a call that `InlineFunctions` removes, leaving its statements |
| a device kernel that stays a function | `@pl.jit_trace.incore` | a call plus an AIV/AIC function in the program |

The distinction that bites is the second row against the first: a **plain function that
emits statements is not a body**. Its value return is fine (it is an expression), but a
`pl.store` inside it lands nowhere — the kernel compiles, runs, and writes nothing:

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
ROWS = COLS = 64
TILE_ROWS = 16


def loads(kernel, *args) -> int:
    """Load ops in the lowered program: the evidence a statement landed."""
    text = ir.python_print(kernel.compile_for_test(*args))
    return sum(1 for line in text.splitlines() if ".load" in line)
```

<!-- doctest: run -->
```python
def emit_block(a, out, row0, k):            # a plain function: an expression helper
    pl.store(pl.mul(pl.load(a, [row0, 0], [TILE_ROWS, COLS]), k), [row0, 0], out)


@pl.jit_trace.body
def body_block(a, out, row0, k):            # a declared statement fragment
    pl.store(pl.mul(pl.load(a, [row0, 0], [TILE_ROWS, COLS]), k), [row0, 0], out)


@pl.jit_trace
def via_plain(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(ROWS // TILE_ROWS):
            emit_block(a, out, chunk * TILE_ROWS, k)
    return out


@pl.jit_trace
def via_body(
    a: pl.Tensor[[ROWS, COLS], pl.FP32],
    k: pl.Scalar[pl.FP32],
    out: pl.Out[pl.Tensor[[ROWS, COLS], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(ROWS // TILE_ROWS):
            body_block(a, out, chunk * TILE_ROWS, k)
    return out


a = torch.ones(ROWS, COLS, dtype=torch.float32)
z = torch.zeros(ROWS, COLS, dtype=torch.float32)

# The plain helper contributes nothing — no load, no store — and nothing raises.
plain_program = ir.python_print(via_plain.compile_for_test(a, 2.0, z))
assert loads(via_plain, a, 2.0, z) == 0
assert ".store" not in plain_program
via_plain(a, 2.0, z, config=CFG)          # runs; the output is simply never written

# The declared body contributes its statements, and the kernel computes.
assert loads(via_body, a, 2.0, z) == 1
out = torch.zeros_like(a)
via_body(a, 2.0, out, config=CFG)
assert torch.allclose(out, a * 2.0)
```

`@pl.jit_trace.body` states the boundary. A body is prepared once and expanded where it is
called, so the fragment needs no parameter or result annotations — the arguments are the
call's staged values, and a value it returns crosses back into the caller's expression
position:

```python
@pl.jit_trace.body
def scale_block(a, out, row0, k):
    pl.store(pl.mul(pl.load(a, [row0, 0], [TILE_ROWS, COLS]), k), [row0, 0], out)

@pl.jit_trace
def entry(a: pl.Tensor, k: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):
            scale_block(a, out, chunk * TILE_ROWS, k)   # a call, not a callee
    return out
```

Three rules follow from "a body is expanded, not called". It runs only inside a trace (a
host call names the expansion rule and refuses); it cannot call another body — a body's
parameters *are* the caller's names, so reuse between fragments is written in the calling
body, or expressed as a `@pl.jit_trace.inline` callee; and its placement policy is
declared, not passed — `@pl.jit_trace.body(auto_scope=False)` for a fragment that keeps
its own `pl.at` scopes, while `auto_scope=` at a call site is refused.

An `.inline` helper is the other way to reuse statements: it declares a typed `Inline`
function *and* a call, and the inlining pass then erases the function. The catch is what
survives inlining — an inline helper's body lands in its caller, so tile work needs its
own `with pl.at(...)`, and its output parameter is the caller's alias (a bare
`pl.Tensor`, not `pl.Out[...]`), since there is no callee signature to declare directions
on. An `.incore` helper keeps its own placement by definition and stays in the program.

### Objects that organize staged values

Because the body runs, the ordinary Python answer to "too many constants and arguments"
works: put them in an object. A dataclass whose fields are host data is configuration; a
dataclass that *carries* a staged tile is a record — the object is Python and disappears,
the staged value it holds is IR. Neither is a parameter: a bundle is expanded at the call
site (`kernel(a, *expand(plan), out=out)`), and with a frozen dataclass and `fields()` in
declaration order that expansion is the kernel's flat parameter list.

```python
@dataclass(frozen=True)
class Plan:
    tile_rows: int = TILE_ROWS
    k: float = 2.0

def expand(plan: Plan) -> tuple[object, ...]:
    return tuple(getattr(plan, f.name) for f in fields(plan))

entry(a, *expand(Plan(tile_rows=32, k=3.0)), out=out)   # two baked host knobs
```

An object passed *as* an argument is refused — the boundary accepts tensors, facades and
host scalars — which is why the spelling above expands it first. When the fields should
survive into the signature instead of baking, make them `pl.Scalar[...]` and expand
exactly the same way.

### Shared snippets: `pl.authoring.body(...).splice(...)`

`pl.authoring.body` prepares a snippet you want pasted into its caller rather than
called; `.splice(...)` places it at every occurrence. Its statements join the
caller's IR — no function, no call — so it needs no parameter or result
annotations at all:

```python
@pl.authoring.body
def twice(x):
    return pl.add(x, x)

@pl.jit_trace
def entry(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [64, 64])
        pl.store(twice.splice(tile), [0, 0], out)   # a Tile here ...
        out[:, :] = twice.splice(a)                 # ... a Tensor there
    return out
```

Use it when a helper is a snippet rather than a contract: it skips the IR
signature, so one body serves several argument types, and a value it creates can
be returned without a parameter backing it. Two rules still apply — it only runs
inside a trace (a host call is refused), and staged typing still governs the
body's interior, exactly as it does in the caller. Reach for
`@pl.jit_trace.inline` when you want the body collected once and its signature
declared; reach for `pl.authoring.body(...).splice(...)` when there is nothing to declare.

### What a sub-function hands back keeps its shape and dtype

Specialization stamps every generated parameter with a concrete shape and dtype, so a
tensor a sub-function returns must be traceable to one. Both conventions work, and you can
mix them in the same entry:

```python
@pl.jit.inline
def make_pair(x: pl.Tensor[[1, 8], pl.FP32]):
    a = pl.create_tensor([1, 8], dtype=pl.FP32)   # helper allocates its own results
    b = pl.create_tensor([1, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        a[:, :] = x[:, :]
        b[:, :] = pl.mul(x[:, :], 2.0)
    return a, b

@pl.jit.incore
def relu_kernel(x: pl.Tensor, out: pl.Out[pl.Tensor]):   # caller allocates, kernel fills
    ...

@pl.jit
def entry(x: pl.Tensor[[1, 8], pl.FP32], out: pl.Out[pl.Tensor[[1, 8], pl.FP32]]):
    a, b = make_pair(x)          # metas read out of make_pair's own body
    buf = pl.create_tensor([1, 8], dtype=pl.FP32)
    mid = relu_kernel(a, buf)    # mid aliases buf, so it inherits buf's meta
    ...
```

A device kernel (`@pl.jit.incore`) cannot allocate, so it always takes the second form —
`pl.create_tensor` belongs on the control plane. An `@pl.jit.inline` helper is spliced into
the caller, so either form is available to it.

A complete one-dimensional view of an NZ tensor also keeps inferable metadata. For
`w: pl.Tensor[[256, 512], pl.INT8, pl.NZ]`, `w_flat = pl.reshape(w, [131072])`
can be passed to a helper declared with `src: pl.Tensor`; specialization supplies
`src: pl.Tensor[[131072], pl.INT8]` (ND). The reshape creates a view of the existing physical
bytes without layout conversion or copying. Every source dimension and the target length must be
static, and the target length must equal the source element count. Other NZ reshape
targets and symbolic NZ flattening remain unsupported.

An extent the specializer cannot compute statically is *not* by itself a problem. A
`pl.create_tensor` sized from a value only the device knows — `pl.tensor.read(cfg, [0])`,
`pld.world_size()` — becomes a dynamic dimension and keeps flowing, and the shared pass
pipeline judges whatever the program then does with it (loading the whole tensor as a tile,
say, gets you `InitMemRef requires static shape` — the same error the equivalent
`@pl.program` earns).

`pl.reshape` also preserves metadata when its target dimensions use known tensor
dimensions and integer arithmetic (`+`, `-`, `*`, `//`, and unary `-`). For example,
`pl.reshape(src, [src.shape[0] * src.shape[1]])` keeps a dynamic extent when either
source dimension is dynamic. The consuming JIT function reads the actual extent from
the reshaped tensor's descriptor. The original reshape expression remains in IR,
and existing reshape validation is unchanged.
Calls to `pl.tensor.dim` support positional or keyword `tensor`/`axis` arguments
and static constant or negative axes. Unknown runtime axes remain unresolved.
Shape subscripts use nonnegative integer literal indices; use `pl.tensor.dim(src, -1)` for
a negative axis. Dynamic unary `+` remains unsupported.

Intermediate scalar aliases work as well: `total = src.shape[0] * src.shape[1]`
followed by `flat = pl.reshape(src, [total])` preserves the dynamic extent.
Scalar calculations stay scalar: inference records immutable expression snapshots
and creates dynamic-axis symbols when recording a tensor shape. Rebinding a source
name does not change an already computed alias. Chains of simple names such as
`total = half = total // 2` evaluate the right-hand side once and are lowered
to individual assignments before parsing. Static integer aliases fold normally;
unknown runtime values and invalid arithmetic remain unresolved. Aliases rebound
under conditionals or loops are handled conservatively across control-flow
boundaries, rather than assuming the value from one syntactic execution.
After a tensor variable is rebound, shape reads use its current value, including
whole-shape unpacking and results of tensor operations. Shape and dtype arguments
such as `shape=src.shape` and `dtype=src.dtype` also use the current tensor type.
Unrelated branches and loops keep existing aliases; a loop-carried alias becomes
unknown until a definite assignment establishes its value in the current iteration.
Mutually exclusive `if`/`elif`/`else` arms start from the same incoming scalar
bindings and tensor metadata. Conditional scalar writes remain unknown after
the branches join. Tensor metadata established identically in both arms remains
usable; conflicting or missing tensor metadata remains unknown.
Metadata keeps independent derived dimensions separate when a Python variable
is reused or a helper is called repeatedly. Use explicit assignments such as
`n = n + step`; augmented assignment (`n += step`) remains unsupported.

What does not resolve is a returned tensor whose shape the specializer cannot *reach* at
all: a `pl.reshape` sized from an unknown runtime scalar, such as `pl.tensor.read(cfg, [0])`,
or a result rebound through an operation the specializer does not model. That surfaces as
`missing inferred tensor metadata for parameter '<name>'` at the *next* call that consumes
it — the error names the consumer, but the fix belongs at the producer: give the producing
statement a statically inferable shape, or pass the buffer in as a `pl.Out[...]` parameter.

### Three constraints that decide whether a jit kernel compiles

These are the failures new `@pl.jit` code hits, in the order it hits them.

**1. A `@pl.jit` entry body cannot hold operators.** It is an Orchestration function —
the control plane. Put the operators inside `with pl.at(level=pl.Level.CORE_GROUP):`, or
move them into a `@pl.jit.incore` sub-function.

```python
@pl.jit
def bad(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    out[:] = pl.add(x, x)        # ✗ Misplaced tensor op ... should be inside InCore block
    return out

@pl.jit
def good(x: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        out[:] = pl.add(x, x)    # ✓
    return out
```

**2. `JITFunction` has no `as_python()`.** The IR does not exist until a specialization
does. Call `lower(*args)` for the post-pass `ir.Program`, or `compile(*args)` and read
`compiled.program.as_python()` for the specialized, pre-pass IR.

**3. `compile()` takes the kernel's own arguments, not compile options.** Compile-time
knobs go through `config=RunConfig(...)`. A stray `compile(skip_ptoas=True)` is bound
against the kernel's signature and raises `TypeError: got an unexpected keyword argument`.
`@pl.jit` detects whether `ptoas` is available on its own, so `skip_ptoas` is not something
you need to pass.

**A device kernel cannot return a scalar.** The runtime's two task-argument channels are
disjoint: scalars go *in* by value, and only tensors come back. So a value computed on
device and needed by the caller — the orchestration function that launched the kernel, and
the host beyond it — has to travel in a tensor. That covers a scalar tucked inside a
`pl.Tuple[...]` return, too.

```python
@pl.jit.incore
def bad(x: pl.Tensor[[64], pl.FP32]) -> pl.Scalar[pl.INDEX]:   # ✗ no carrier for the return
    ...

@pl.jit.incore                                                  # ✓ hand it back in a [1] tensor
def good(x: pl.Tensor[[64], pl.FP32], n_out: pl.Out[pl.Tensor[[1], pl.INT32]]):
    ...
# then, in the entry body, after the launch:
n = pl.tensor.read(n_out, [0])
```

The same rule applies to a scalar you assign *inside* `with pl.at(...)` and read after it —
that is a kernel return in disguise. When the value only depends on things the entry body
already has (a loop variable, a scalar parameter), the compiler moves the computation out
of the scope for you; when it depends on device data, it asks you to route it through a
tensor as above.

A helper that computes a scalar *for* the kernel is fine — write it `@pl.jit.inline`
(`FunctionType.Inline`). It is spliced at the call site, so it is not a task and the rule
does not apply to it.

### `@pl.function` and `@pl.program`

You reach for this form when writing a compiler test case, not when writing a kernel. It
describes the IR one-to-one: the class is the program, each method is a function, and the
call graph is written out rather than discovered. `@pl.jit` specializes into exactly this
shape — printing a compiled program shows you `@pl.program` source.

```python
@pl.program
class Adder:
    @pl.function(type=pl.FunctionType.InCore)
    def add_kernel(self, a, b, out): ...

    @pl.function(type=pl.FunctionType.Orchestration)
    def entry(self, a, b, out):
        out = self.add_kernel(a, b, out)     # explicit cross-function call
        return out
```

Every method takes `self` (it is stripped from the IR), and `Adder` becomes an
`ir.Program` — not a Python class you can instantiate. `Adder.as_python()` prints it.

`type=` names the plane each function belongs to:

| Function type | Plane | Typical use |
| ------------- | ----- | ----------- |
| `Opaque` (default) | none yet | Standalone building block; takes its plane from where it is used |
| `InCore` | Execution | Load / compute / store kernel |
| `Orchestration` | Control | Creates tensors, dispatches InCore tasks |
| `Inline` | none | Spliced at every call site; leaves no function behind |

A standalone `@pl.function` called from inside a `@pl.program` is added to the program as
a separate function. `@pl.inline` (and `@pl.jit.inline`) instead expand at the call site
and leave no function behind.

```python
@pl.inline
def normalize(x: pl.Tensor[[64], pl.FP32]) -> pl.Tensor[[64], pl.FP32]:
    return pl.mul(x, 2.0)
```

The decorated object is a `pl.InlineFunction` — a template the parser splices, not a
function you can call from Python.

### Function attributes: `pl.func_attr`

Metadata about the function as a whole is declared with `pl.func_attr({...})` as the
**first statement** of the body:

```python
@pl.program
class Kernels:
    @pl.function(type=pl.FunctionType.InCore)
    def kernel(self, x: pl.Tensor[[64, 64], pl.FP32], w: pl.Tensor[[64, 64], pl.FP32],
               out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
        pl.func_attr({"stationary": w, "split": pl.SplitMode.UP_DOWN})
        ...
```

It reads oddly for a *function*-level declaration to sit inside the body, so it is worth
saying why. A decorator is evaluated before the signature binds any name, so
`@pl.function(attrs={"stationary": w})` cannot be written at all — `w` does not exist
yet. Body position places the declaration after the parameters are bound, which is what
makes an attribute that *references a parameter* expressible. The alternatives are a
positional index (`{"stationary_param": 1}`, which breaks the moment a pass reorders
parameters) or a naming convention nothing enforces.

Rules worth knowing:

| Rule | Why |
| ---- | --- |
| Must precede every other statement | An attribute describes the whole function; it must not appear to start applying partway down a body. This also bounds what it can reference to the parameters. |
| A bare name is always a parameter | `pl.func_attr({"n": k})` records the parameter `k`, never a same-named Python variable from the enclosing scope. Write Python constants as literals. |
| Multiple calls merge | A key declared twice is an error naming the key, so which value wins is never a matter of parse order. |
| `auto_scope=` and `external_source=` stay on the decorator | The parser reads them *before* it walks the body, so a body-position declaration would arrive too late to take effect. |

`@pl.function(attrs={...})` is **deprecated** and emits a `DeprecationWarning`. It still
parses and behaves identically, but it can only ever carry values that reference nothing.
Printed IR always uses a non-deprecated spelling — the `pl.func_attr` prologue, or the
dedicated `auto_scope=` / `external_source=` keywords — so reparsing compiler output never
warns.

### Splitting compile from dispatch

`@pl.jit` kernels normally fuse specialize + compile + dispatch into one `kernel(*args)`
call. `JITFunction.compile(*sample_args)` stops after compilation and hands back the
`CompiledProgram` — for driving `ChipWorker` yourself, inspecting artifacts under
`compiled.output_dir`, or validating codegen ahead of time.

```python
compiled = my_kernel.compile(sample_x, sample_w, sample_out)
print("artifacts in:", compiled.output_dir)
```

The returned object is the same one the JIT cache holds, so a later call with the same
specialization key returns the identical instance.

`lower(*sample_args)` stops one stage earlier: it runs the passes and returns the
post-pass `ir.Program`, with no code generation, no `ptoas`, no artifacts, and no cache
write. Use it to read lowered IR; use `compile()` when codegen itself is what you want to
check. Both accept `config=RunConfig(...)`, but `lower()` ignores the runtime and artifact
fields. Compile options are in [Compiling](../execution/00-compile.md) and the runtime surface in
[Running](../execution/01-run.md).

`specialize(*sample_args)` stops one stage earlier still: entry and deps are specialized
into `@pl.program` source and parsed, and the **pre-pass** `ir.Program` comes back
untransformed. Reach for it only when something downstream runs the pass pipeline itself
— most of all `ir.compile(program, output_dir=...)`, which runs passes *and* code
generation, so handing it `lower()`'s output would run the pipeline twice.

```python
program = my_kernel.specialize(sample_x, sample_w, sample_out)
ir.compile(program, output_dir="build/out", backend_type=BackendType.Ascend910)
```

It takes no `config=`: no pass runs here, so a `RunConfig` would have nothing to
configure. Sample arguments may be omitted only when every tensor parameter is annotated
with full shapes — a bare `pl.Tensor` has none to read.

> Two kernels that specialize to the same program compare equal **after** passes, not
> before: the specializer renames SSA-rebound locals (`out` becomes `out_v1`), and
> canonicalization removes the difference. Compare `lower()` output when asserting
> equivalence against a hand-written `@pl.program`.

Three accessors describe the signature without specializing anything: `param_names` (in
declaration order), `output_param_names` (the `pl.Out[...]` and `pl.InOut[...]` params,
also in declaration order), and `__name__`.

### External C++ kernels

A hand-written C++ kernel can be called like any other function. See
[Integrating Hand-Written C++ Kernels](../../dev/language/04-external-kernels.md).

## Edge Cases

> **Fatal pitfall:** verify a new `@pl.jit` example with a full `compile()`, never with
> `lower()` alone. `lower()` stops after the passes, so the "operators in an Orchestration
> body" error above never fires — the kernel appears to pass and fails only when someone
> runs it for real.

| Symptom | Likely cause | Fix |
| ------- | ------------ | --- |
| **A kernel that emits statements through a plain helper writes nothing** | A plain function is an expression helper, not a body: its `pl.store` lands nowhere and nothing raises | Declare the fragment `@pl.jit_trace.body` and call it |
| **`... cannot be called from inside another body`** | A body calls another body, or a body calls itself | Write the statements in the calling body, or declare a `@pl.jit_trace.inline` callee |
| **`... cannot run as a host call`** | A `@pl.jit_trace.body` was called outside a trace | Call it from inside a traced kernel |
| **`auto_scope is declared on the body, not passed at the call site`** | The policy was passed at a call | Move `auto_scope=` to the `@pl.jit_trace.body(...)` declaration |
| **`The tile '...' lives in a Orchestration function`** | An `.inline` helper did tile work without its own `pl.at` | Place the work: `with pl.at(level=pl.Level.CORE_GROUP):` inside the helper |
| **`argument 'plan' (...) is not a tensor, facade, or host scalar`** | A host object was passed as a kernel parameter | Expand it at the call site (`*expand(plan)`), or pass its fields |
| **`Misplaced tensor op ... should be inside InCore block`** | Operators directly in a `@pl.jit` body | Wrap in `with pl.at(level=pl.Level.CORE_GROUP):` or move to `@pl.jit.incore` |
| **`AttributeError: 'JITFunction' object has no attribute 'as_python'`** | Printing IR that does not exist yet | `f.lower(*args)`, or `f.compile(*args)` then `compiled.program.as_python()` |
| **`lower()` passes but `compile()` fails** | `lower()` runs no code generation | Expected — use `compile()` to check codegen |
| **`TypeError: got an unexpected keyword argument`** | A compile option was passed to `compile()`, which binds against the kernel's signature | Pass `config=RunConfig(...)` |
| **A second top-level kernel is missing from the program** | Plain `@pl.jit` does not discover other `@pl.jit` entries | Use `@pl.jit.host`, or make the callee `.incore` / `.opaque` |
| **`auto_scope=False` rejected** | Used on `.incore` / `.opaque` | Put it on the entry or on an `.inline` helper |
| **`self` missing from a `@pl.program` method** | Every method needs it | Add `self`; it is stripped from the IR |
| **`A task cannot return a scalar`** | A device kernel declares a `pl.Scalar` return | Write it into a `[1]` tensor output and `pl.tensor.read(t, [0])` after the launch |
| **`cannot return a scalar` naming a variable in a scope** | A scalar assigned inside `pl.at(...)` is read after it, and depends on device data | Route it through a `[1]` tensor, or move the computation out of the scope |

## Worked example

`examples/utils/cross_function_calls.py` — `@pl.jit.inline` helpers auto-discovered as
deps of a `@pl.jit` entry and spliced at the call site.

| Example | Shows |
| ------- | ----- |
| `examples/tracing/03_host_objects.py` | Configuration, a record carrying a staged tile, and a bundle expanded into the parameter list |
| `examples/tracing/04_body_placement.py` | `@pl.jit_trace.body` as a call, its value-return form, and the same statements pasted |
| `examples/tracing/advanced/07_diagnostics.py` | The silent plain-helper trap and the declared repair |

## See Also

- [Control Flow](02-control-flow.md) — loops and conditionals inside these bodies.
- [Values and Staging](07-staging.md) — host versus staged values, and which control flow each produces.
- [Scopes and Placement](04-scopes.md) — `pl.at` and the other placement scopes.
- [Quickstart](../02-quickstart.md) — the same decorators in a worked example.
- [InlineFunctions](../../dev/passes/01-inline_functions.md) — how `Inline` bodies are spliced.
- [Integrating Hand-Written C++ Kernels](../../dev/language/04-external-kernels.md) — calling external kernels.
