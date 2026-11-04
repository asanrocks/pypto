# Functions and Program Structure

> **Legacy Python API examples:** `platform=` calls on this page are retained for
> compatibility. They emit `DeprecationWarning`; new code should use
> `backend_type` and `execution_mode`. See the [target mapping](../08-entry-points.md#legacy-platform-examples).

Function declaration forms, parameter directions, cross-module reuse, and how to
print IR back to Python syntax.

## JIT implementation structure

The JIT implementation lives in `python/pypto/jit/`. The public `jit` object and
`JITFunction` remain in `decorator.py`; private analysis modules supply metadata
without owning compiled-object caches or dispatching kernels.

| Module | Responsibility |
| ------ | -------------- |
| `decorator.py` | Decorators, `JITFunction`, dependency discovery, cache ownership, and compile/execute entry points |
| `_arguments.py` | Optional torch loading, tensor argument metadata, annotations, and signature/constexpr validation |
| `_ast_analysis.py` | Shared read-only AST cache, source names, and declared dynamic dimensions |
| `_call_binding.py` | Call-site argument mapping, constexpr variants, and dynamic-dimension propagation across dependencies |
| `_tensor_meta.py` | Source-ordered local tensor metadata, branch/loop bindings, and recursive dependency-return inference |
| `specializer.py` | Specialization context, annotation handling, program source assembly, and diagnostic source maps |
| `_specialize_body.py` | Stateful AST body rewriting, variable renaming, and lexical scope handling |
| `_specialize_values.py` | Shared constant and dtype/layout source rendering for specialization and cache keys |
| `_specialize_types.py` | Shared `DynDim`, `ShapeDim`, and `TensorMeta` definitions; existing imports from `specializer` remain valid |
| `_source.py`, `cache.py`, `_persistent.py` | Request namespace snapshots, cache keys, and persistent artifact resolution |

`JITFunction` owns the dependency-graph state, compiled-object caches, and their
lock. Key construction and specialization continue to share one request's
namespace snapshot. Metadata inference receives dependency discovery as an
explicit callback and carries it through recursive return analysis, so it does
not import the decorator at runtime. Shared helper lookups use their owning
modules, keeping AST caches and test patch targets in one place.

## JIT constants and compilation reuse

A `@pl.jit` body is parsed into `@pl.program` source and re-parsed in a namespace
holding only `pl` and `pld`, so any name it inherits from its own module or an
enclosing function must be replaced, at its use site, by source text that
evaluates back to the same value. That covers literals (`int`, `float`, `bool`,
`str`, `None`), the `pl` dtype and enum constants (`pl.INT8`, `pl.Mem.Vec`,
`pl.PadValue.zero`, `pl.NZ`), distributed enum constants (`pld.ReduceOp.Max`,
`pld.NotifyOp.AtomicAdd`, `pld.WaitCmp.Ge`), and lists/tuples nested from those —
a shape, rounding mode, or collective reduction selected by a factory, for instance.
A value with no source form is left alone, so the name survives and the parser
reports it.

Every name that folds is also in the compilation key, including constants used by
transitive JIT helpers and source annotations. The key hashes the *emitted text*
for each name, straight from the function the specializer folds with, so the two
cannot drift: a constant that changes the generated source changes the key by
construction, and one that does not fold contributes nothing. Rebinding a
referenced constant causes a new specialization; changing an unrelated global or a
name shadowed by a body-local variable does not invalidate the body dependency
key.

```python
BLOCK = 32

@pl.jit
def slice_kernel(x: pl.Tensor[[128, 128], pl.FP32]) -> pl.Tensor[[BLOCK, 128], pl.FP32]:
    with pl.at(level=pl.Level.CORE_GROUP):
        result = pl.slice(x, [BLOCK, 128], [0, 0])
    return result

first = slice_kernel.compile()
BLOCK = 64
second = slice_kernel.compile()  # A distinct specialization with 64 rows.
```

Name resolution, key construction, and specialization share a per-call namespace
snapshot. Constants rebound after capture affect the next call, not the artifact
being compiled. Each helper retains its own namespace, even when constants have
the same name. Rebinding a referenced JIT helper also refreshes the dependency
graph. Each concurrent call retains its captured graph throughout key construction
and specialization. The hash also includes each helper's function type, level, and
`auto_scope` setting, so rebinding an identically sourced function with different
compilation attributes cannot reuse the old artifact.

Each module's globals are copied once per request; helpers use closure overlays
on that shared snapshot. An empty closure cell still shadows a same-named global.
Validated graphs retain their Python source hashes, and declared layouts are
reused while their annotation bindings stay unchanged. External source files are
still checked on each request. Referenced closure constants are covered by the
source dependency hash without a separate closure-key component.

The snapshot copies bindings only: mutating arbitrary configuration objects
or editing compiler/source files during compilation is not supported by this
constant-tracking mechanism. Persistent reuse additionally requires the full
[artifact identity and cache policy](../10-jit-cache.md).

### Dynamic dimension metadata

Before parsing the specialized program, the JIT infers local `TensorMeta` values
to type arguments passed to bare-annotated helpers. Scalar bindings use `int`
constants, existing-axis `DynDim` references, immutable expression tuples, or
`None` (unknown). For `width = n * 8`, the snapshot is `(ast.Mult, (n_dim, 8))`.
Assignment aliases retain that tuple; subsequent arithmetic creates a different
snapshot. Rebinding an operand's source name does not change an earlier snapshot.

Reshape inference gives scalar expressions result-specific `DynDim` symbols
when recording `TensorMeta.shape`; scalar snapshots stay outside tensor types. Repeated
uses of the same snapshot within one result shape share a symbol; separate
results get independent bindings. The callee reads the dimension from its actual
tensor argument's descriptor. The original arithmetic and reshape validation
remain in the program; placeholder bounds do not replace runtime extents.

`synthesized` records an automatically generated placeholder symbol and retains
its existing annotation-overlay and helper-return behavior.

Subscript views also retain a static width when their runtime bounds have a
provably constant difference. For example, `src[base:base + 128, :]` carries
`[128, src.shape[1]]` metadata, allowing a join with a same-typed `[128, N]`
allocation when the second dimensions agree. The JIT translates integer
addition/subtraction and constant multiplication into temporary expressions
and reuses the parser's arithmetic analyzer to prove the width. Each binary
operation must have a constant operand to promote its arithmetic to INDEX. Since
scalar dtypes are unavailable in this inference, operations on two runtime
operands and runtime negation are conservatively rejected. For example,
`base + other:base + (other + 128)` cannot establish width 128: the lower sum
may overflow in INT32 before the upper expression promotes to INDEX.
The proof resolves literals, unshadowed global/closure constants, constexpr
parameters, direct static tensor-shape reads from the current tensor metadata,
and shape aliases that the body transformer actually replaces with literals.
Direct shape reads also work for locally allocated tensors whose dimensions
are folded by the parser. Parent extents retained as fallbacks for unresolved
slices are marked inexact and cannot prove widths through later shape reads,
tensor aliases, branch joins, or helper returns. A helper's declared dynamic
axis replaces that fallback with its actual descriptor-bound dimension.
Local scalar assignments such as
`width = 128` or `width: pl.Scalar[pl.INT32] = 128` remain symbolic even when their values are
known to dimension inference: they survive specialization as variables, not
INDEX literals. Runtime dimensions never use sample bounds as proof.
Symbols are shared only within one pair of slice bounds, so
rebinding and aliases from other statements do not establish new equalities.
Unresolved calls, reads, dynamic division, and nonlinear products are not
cancelled as symbols. Unproved widths keep the existing parent-extent fallback,
and step slices remain unsupported by this metadata inference. Branch joins
still require compatible shapes, dtypes, layouts, and dynamic-axis identities.
This inference does not change the slice operation or its storage lifetime.

### Tile parameters in JIT inline helpers

An inline helper can take an explicitly typed on-chip Tile:

```python
@pl.jit.inline(auto_scope=False)
def keep_tile(tile: pl.Tile[[8, 64], pl.FP32, pl.MemorySpace.Vec]) -> pl.Tile:
    return tile
```

Specialization preserves the declared shape, dtype, and memory space, folding
module and closure constants in the annotation. When a postponed annotation
uses the helper's own `pl.constexpr` parameter, that call's constexpr binding is
substituted before the annotation is evaluated; it takes precedence over a
same-named module or closure constant. Different bindings produce separate
helper specializations. DSL aliases such as `p.Tile` or a directly imported
`Tile` are rendered as canonical `pl` paths, preserving `TileView` and `MemRef`
metadata. An unrelated user variable named `pl` does not affect this rendering.
For an eager Tile annotation whose factory-local constants are absent from the
function closure, specialization can retain the already evaluated annotation.
Independent shape axes and dtype survive even when other axes depend on helper
parameters; those dependent components are substituted per call. The merged
contract also contributes to the cache identity. Constexpr memory space
parameters are substituted while independent shapes evaluated in the factory
remain intact. Metadata replacements must retain their original kind; ambiguous
replacements produce a diagnostic.
If one dependent expression
mixes a helper parameter with an unavailable factory local, pass that local as
another constexpr parameter or capture it in the helper body.
Dynamic symbols referenced
only by Tile parameter types receive their own module-level `pl.dynamic()`
declarations, even when no Tensor parameter uses them.
This also applies when the emitted symbol name matches a Tile or Tensor
parameter name; actual Scalar dimension parameters remain local.
If a free dynamic symbol and a distinct Scalar parameter share a label,
specialization reports the ambiguity; give the free symbol a unique
`pl.dynamic()` label.

Before substituting arguments, `InlineFunctions` checks Tile dtype, physical
rank and shape, and conflicting explicitly resolved memory spaces. Generic
physical dimensions bind to the actual Tile's physical extents, including
integer affine bindings such as `2*M` to `16` (`M=8`); valid extents
do not constrain that check. Known Scalar values are substituted first:
`n=8` and `n+M=16` infer `M=8`. Separate Tile arguments can bind their generic
physical dimensions independently, including when the helper returns its Tile parameters.
Returned-parameter types also follow simple immutable aliases such as
`tmp = second; return tmp`.
Dimensions shared with Tensor types, explicit Tile views, or Scalar parameters
retain their shared constraints, including in helpers with no return value.
Actual Scalar values also constrain view-only dimensions: `n=8` cannot match
an actual valid extent of `3` when the declared valid extent is `n`.
Scalar-dependent view expressions such as `n - 1` are compared after argument
substitution, including valid extents, strides, and offsets. This includes indirect
dependencies through shared dimensions such as `rows = 2*n`. An omitted actual
valid shape uses its physical shape, and an omitted offset is zero. A
Scalar-dependent stride contract requires explicit actual stride metadata.
JIT specialization also retains nested `pl.Tuple` parameter annotations, including
the complete Tile types, their dynamic symbols, and per-call constexpr bindings.
The same physical and view checks apply recursively to Tiles in tuple parameters.
Dynamic dimension names emitted from Tile annotations must use assignable Python
identifiers in normalized spelling, must not be Python keywords, and must not
shadow the DSL aliases `pl` or `pld`.
Inlining substitutes the caller's Tile without
copying it, preserving its valid shape. Specialization also retains evaluable
Tile annotations on non-inline functions. This preserves their declarations
without adding Tile arguments to device dispatch interfaces; ordinary InCore
entry points still reject Tile parameters.

### Compile options and diagnostic requests

JIT calls and `kernel.compile()` resolve compile options before looking up an
in-process artifact. Omitting `config` uses the same defaults as `RunConfig()`:
`a2a3sim`, the default optimization strategy, and pass dumps disabled. Semantic
options (including the effective planner, runtime ABI, and source-location
emission controlled by `PYPTO_EMIT_PTO_LOC`) participate in the key. Runtime
controls such as `device_id`, `codegen_only`, and `save_kernels` alone do not.

The following requests compile afresh on every call, without reading or adding
an entry in the artifact cache:

- `dump_passes=True` or a dump level other than `PassDumpLevel.NONE`, and
  `dump_ptoas_passes=True`.
- `compile_profiling=True`, an active `CompileProfiler`, or profiling enabled
  through `PYPTO_COMPILE_PROFILING`.
- An explicit `save_kernels_dir`.
- Explicitly enabled `PYPTO_EMIT_DEBUG_RUNNER` or `PYPTO_REBUILD_FROM_PTO`.
  Values `0`, `false` and `no` disable the request (case-insensitive, ignoring
  surrounding whitespace); empty-but-set retains the enabled/default behavior.
- Explicit diagnostic settings, or an active `PassContext` with instruments
  or verification/diagnostic settings different from the pipeline defaults.

This ensures a warm kernel still emits requested dumps and reports. Repeating
a request regenerates the output; a failed diagnostic compile leaves ordinary
cached entries intact. Automatically allocated JIT output directories are unique
per compilation, so fresh diagnostics cannot overwrite a cached artifact.
Explicit output directories are used as requested. Diagnostic controls do not
split compilation keys.
Conflicting explicit settings and an active `PassContext` raise the same error
on cache hits as on fresh compilation. A plain planner/runtime context can
still reuse a matching cached artifact.

`PYPTO_PROG_BUILD_DIR` only selects the base for new private build and runtime
directories; it does not request fresh compilation. Existing cached objects
retain their directories, and shared artifacts stay in the cache. Use
`save_kernels_dir` when each call must generate a fresh, inspectable output tree.

```python
from pypto.runtime import RunConfig

cached = slice_kernel.compile()
slice_kernel.compile(config=RunConfig(dump_passes=True, save_kernels_dir="debug_kernel"))
assert slice_kernel.compile() is cached
```

### Prepare binaries without executing

`kernel.warmup()` uses the same specialization, configuration, and in-process
object cache as `kernel.compile()`, and also prepares every kernel and
orchestration binary before returning. It initializes no NPU and creates no
runtime worker. The build host still needs the target compiler, SDK, and runtime
Python/native dependencies.

```python
import pypto.language as pl
from pypto.runtime import RunConfig

@pl.jit
def add_three(
    x: pl.Tensor[[16, 16], pl.FP32],
    out: pl.Out[pl.Tensor[[16, 16], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [16, 16])
        pl.store(pl.add(tile, 3.0), [0, 0], out)
    return out

config = RunConfig(platform="a2a3")
prepared = add_three.warmup(config=config)  # No sample tensor allocation.
# Later, on a host with an available NPU:
# prepared(x, out, config=config)
```

You can supply sample tensors as with `compile()`; warmup reads their metadata,
not their contents. With complete tensor annotations, omit the tensors entirely:
scalar parameters are runtime values, so they need no value here, and dynamic
extents retain the existing compile rules. A `pl.constexpr` parameter is the
exception: its value selects the artifact, so it must be supplied as a keyword
or a signature default. When calling the returned compiled object, supply its
full parameter list, including scalar arguments — that is where each scalar's
value is read. A constexpr parameter is absent from that list; the artifact
already resolved it.
Execution-only settings such as `codegen_only` do not suppress binary preparation.

The result is the same compiled object selected by `compile()`, with live IR
retained for fresh compilation. Warmup prepares every chip-level child of a
`DistributedCompiledProgram` and every orchestration sub-build of a
multi-orchestration `CompiledProgram`. It does not call the distributed object's
`prepare()`: that method creates a live worker for execution. Compilation errors
propagate to the caller; a failed binary build can be retried by calling warmup
again. Diagnostic/output requests still compile afresh as described above.

With [persistent caching](../10-jit-cache.md) enabled, warmup automatically
publishes or reuses READY artifacts through the [runtime protocol](../09-artifact-store.md).
Persistence is disabled by default. Read-only misses, unsupported inputs and
storage failures can prepare private results. A restored result has
`.program is None`; disable persistence or use `specialize()`/`lower()` to require IR.
The public cache policy, statistics and metadata-only warmup CLI are documented
in [Persistent JIT Cache](../10-jit-cache.md).

## The tracing frontend: `@pl.jit_trace`

Two frontends can produce the IR of a JIT kernel, and the decorator selects
one per definition. `@pl.jit` parses the body, as described above.
`@pl.jit_trace` executes an instrumented copy of the body and records the
operations that run. Importing `pypto.language` changes neither: a definition
keeps the lane its own decorator names.

```python
@pl.jit_trace
def scale_relu(
    x: pl.Tensor,
    scale: float,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [64, 64])
        pl.store(pl.relu(pl.mul(tile, scale)), [0, 0], out)
    return out
```

`@pl.jit_trace` shares its entry points with `@pl.jit` (`specialize`,
`lower`, `compile`, and a host call with torch tensors). Of the family
spellings it verifies `incore` and `inline` callees, plus the `host` and
`graph` entries — each an alias of its parsing-lane spelling, keeping the same
function kind, level, role and kwarg surface (see
[Supported surface](#supported-surface) below). `opaque` and `extern` raise a
`TypeError` at decoration: `@pl.jit.extern` is a declaration with no Python body
to trace, and `opaque` admits bodies outside the tracer's vocabulary. A
helper must be declared in the lane of the entry that calls it: a traced
body that calls an AST-lane `@pl.jit` kernel raises instead of silently
leaving the frontend, and the parser lane's loop spellings (`pl.while_`,
`pl.cond`) raise inside a trace — the tracing dialect writes ordinary
Python `while`.

### Collection timing and reuse

Decoration traces nothing. The first structural use — `.fn`, `.specialize()`,
`.lower()`, `.compile()`, or a host call with torch tensors — runs the body
once and keeps the record, and later compatible uses reuse it. A kernel whose
annotations leave a shape unknown (a bare `pl.Tensor`) cannot be traced from
the declaration alone, so it re-traces per call. Reading that shape in the
body works from the first call: `x.shape` lowers like the parser frontend's
`x.shape` — a static extent is its constant, a dynamic one is a runtime
`pl.tensor.dim` read that names the parameter's declared extent symbol inside
an orchestration body. A bare annotation therefore does not prevent
shape-dependent bodies; it only removes the declaration-time trace.

The dispatch cache is keyed on the artifact contract: tensor shape, dtype and
layout, the host-knob values, and the compile identity (platform, strategy,
distributed config, memory planner, dbC flag, runtime ABI, Buffer IR, source
locations). Tensor *contents* and the *value* of a `pl.Scalar[...]` parameter
are not in the key — the scalar stays a runtime parameter of the generated
program, so one artifact serves every value.

A `config=RunConfig(...)` call forwards the compile-side fields through the
same `RunConfig.compile_kwargs()` mapping every jit lane uses. Requests that
run the compiler — pass dumps, PTOAS dumps, compile profiling, an explicit
output directory, or an outer `PassContext` with instruments — are never
served from the cache: each one compiles fresh, and an ordinary compatible
call afterwards still reuses the warm artifact.


### Parameter worlds

The annotation decides whether an argument is runtime data or a compile-time
constant:

| Annotation | World | Behaviour |
| ---------- | ----- | --------- |
| `pl.Tensor`, `pl.Tile`, `pl.Out[...]`, `pl.Scalar[dtype]` | runtime parameter | Part of the generated signature; the value arrives at dispatch |
| a plain Python annotation (`float`, `int`) or none | host knob | Baked into the trace; changing the value specializes a new artifact |

A host scalar passed to a runtime-declared slot lifts into a `Scalar`.
A host scalar passed to a tensor slot, or a staged value passed to a host knob,
is a `TypeError` naming the parameter: the world is a property of the
declaration, not of the call.

A facade declaration owns the parameter's *type*, not only its world. The
lane derives a callee's parameter from the call site — that derivation is what
specializes a call — but the derivation is bounded by the declaration, and an
integer argument inside the declared slot's lattice is *converted* to the
declared dtype: a `pl.Scalar[pl.INT32]` parameter takes an INDEX value as INT32,
in either direction, and the conversion is bound to a temporary
(`_rank_int32 = pl.cast(r, pl.INT32)`) so the emitted call carries the declared
type. Both frontends emit the same statement, from the same admission rule
(`pypto.language.scalar_conversion`), so the callee has one signature for every
integer caller and printed IR is the hand-written spelling. Everything outside
the lattice keeps its refusal — two fixed-width dtypes, floats, `pl.BOOL`,
`pl.TASK_ID`, packed FP4, and every tensor/tile dtype, shape or layout
mismatch — as a `TypeError` naming both types, with the repair at the call site
(`pl.cast`) or in the declaration. The rule, and why it is narrow, is the user
document's "Declared scalar dtypes convert"
(`docs/en/user/language/00-types.md`). A bare Python integer is not converted
but *checked*: it carries no dtype, so an out-of-range value is refused at every
dispatch (a scalar value never splits the artifact cache) while a typed value
converts. A
declared dynamic extent (`pl.Tensor[[N, 8], ...]`) is not a contradiction: it
states "whatever the call site passes", so any concrete extent specializes it.
One symbol names one dimension: if `N` appears in several parameters, every
concrete extent supplied for it must agree. Passing `[16, 8]` and `[32, 8]` to
two `N`-shaped parameters is refused; declare a second symbol when the two may
differ. Torch has no INDEX dtype; its `int64` storage is the host carrier, so a
torch `int64` sample into a declared INDEX slot bridges to INDEX; an explicit
`pl.INT64` facade into that slot is an admitted conversion too, recorded as a
value-preserving type change rather than elided. A bare marker is
refused for the same reason (`vlen: pl.INDEX` states a dtype but no world, so it
would have become a host knob): the error names the `pl.Scalar[pl.INDEX]`
spelling. The declared **layout** bridges the same way: a torch tensor carries
no layout, so `pl.Tensor[[N, K], pl.FP16, pl.NZ]` takes its `NZ` packing from the
declaration while shape and dtype still answer to the sample — a sample that
contradicts the declared extent is still refused.

**A keyword names a declared parameter.** A call may bind any parameter by
name — `helper(x, out=out, factor=2.0)` — and the tracer folds that binding into
the declaration order before any lane reads it, so the specialize lane's
annotations, the callee lane's variant key, the dispatch bridge and the printed
program all see one positional call. The spelling is therefore never part of the
artifact: the two forms above specialize to the same program, which is what lets
shared code keep the clearer of the two. `device=` and `attrs=` stay call-site
metadata rather than parameters (a detached call dispatches to that device),
and `config=RunConfig(...)` selects the compile knobs at a top-level call. Every
malformed spelling is refused by name instead of being silently dropped: an
unknown keyword, a parameter bound both positionally and by keyword, a parameter
bound nowhere, and a surplus positional argument.

A host knob is read-only inside staged control flow. The body is traced once,
so a host value the loop advances — `factor = factor * 2.0` under
`for _ in pl.range(n)` — would bake that representative iteration's result into
every iteration and into whatever reads the name after the loop. The binding
guard raises `TypeError` instead and names the staged spelling: bind the state
with `pl.const(...)` and let the loop carry it. Reading a host knob (or binding
fresh host scratch inside the body) stays legal — nothing is carried.

### Supported surface

Every decorated definition under `examples/` is on this lane: explicit-output
(`pl.Out` / `pl.InOut`) entries with their `incore` and `inline` callees,
staged `for` / `if` / `while` over runtime scalars, host knobs, runtime scalar
parameters, tile assembly with loop-carried tiles, `pl.parallel` with atomic
accumulation, scope optimizations (`pl.split`, `deps=`), valid-shape loops,
the `host` / `graph` entrances with their per-rank dispatch and recorded
replay, and the dispatch-cache contract.

Three spellings the lane requires, each a deliberate difference from the
parser lane:

| Parser lane | Tracing lane | Why |
| ----------- | ------------ | --- |
| `vlen: pl.INDEX`, `scale: pl.FP32` | `pl.Scalar[pl.INDEX]`, `pl.Scalar[pl.FP32]` | A bare data-type marker states a dtype but no world, so it is refused at decoration with a `TypeError` naming this spelling |
| `for i, (acc,) in pl.range(n, init_values=(acc0,)):` + `pl.yield_` | bind `acc = acc0` before the loop, rebind at the end of the body | The carried channels are derived from the rebinding; `init_values=` is the printer/parser round-trip spelling of the same channels, and the parser still prints it |
| `pl.Out[pl.Tensor]` on an `inline` helper parameter | bare `pl.Tensor` | The helper body is spliced at the call site, so the parameter is already an in-place alias of the caller's variable |

A host literal carried into staged control flow is a fourth: `factor = 1.0`
rebound under `pl.range` raises, and the typed spelling `pl.const(1.0, pl.FP32)`
is what a carried scalar needs.

#### Body splice: `pl.jit_trace.body`

`@pl.jit_trace.body` declares a reusable **statement fragment** and calls it
like a function. The fragment is placed where it is called, so the kernel
reads as `scale_block(a, out, row0, k)` rather than as a placement method:
**no function, `GlobalVar`, or `ir.Call` is emitted** — the program gains the
body's statements and nothing else. A host call is refused; the body only
expands inside an active trace.

```python
@pl.jit_trace.body
def scale_block(a, out, row0, k):     # no IR annotation: there is no callee
    tile = pl.load(a, [row0, 0], [16, 64])
    pl.store(pl.mul(tile, k), [row0, 0], out)

@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.FP32],
          out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):
            row0 = chunk * 16
            scale_block(a, out, row0, k)
    return out
```

The declaration carries the placement policy: `@pl.jit_trace.body(auto_scope=False)`
keeps a hand-scoped fragment's explicit regions legal while the caller's own
policy governs everything else, and passing `auto_scope=` at a call site is
refused — one meaning per fragment. This is the kernel-author spelling; the
framework-author spelling below keeps the placement explicit.

#### Body splice: `pl.authoring.body(...).splice(...)`

`pl.authoring.body` prepares a Python function once (source rewrite and compile) and
returns a reusable body; `.splice(...)` places it at the caller's insertion
point at every occurrence inside a traced kernel: **no function, `GlobalVar`,
or `ir.Call` is emitted** — the program gains the body's statements and
nothing else. A host call is refused; the placement only executes inside an
active trace.

This is the framework-author spelling: it keeps the placement visible
(`.splice` / `.inline` / `.incore`) for a body that serves several placements.
`@pl.jit_trace.body` is one policy over it — a `@pl.authoring.recipe` whose
body is `ctx.splice_body` — so a framework outside this repository can build
the same kernel-facing spelling from the public authoring surface.

Because there is no callee there is no ABI, so **annotations play no role**:
`def twice(x)` needs no `pl.Tensor[[...]]` declaration, and one body serves
several argument types (a `Tensor` at one occurrence, a `Tile` at the next).
This is the difference from `@pl.jit_trace.inline`, which collects one body,
emits calls, and therefore must declare its parameter and result types.

```python
@pl.jit_trace.body
def twice(x):                 # no IR annotation: there is no callee
    return pl.add(x, x)

@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [64, 64])
        pl.store(twice(tile), [0, 0], out)
        out[:, :] = twice(a)            # the same body, a second argument type
    return out
```

`return` is splice-local: the value crosses to the caller's expression
position, so a body may deliver a value it created without any parameter
backing it. The interior is still staged, though: the splice is pasted into
the caller's control flow, so a value crossing `pl.range` / `pl.if` *inside*
the body obeys the same staged rules (and the same `pl.const` spelling) as the
enclosing function.

This wave allows one splice at a time per trace: a body that splices another
body — nesting or host recursion — is refused loudly rather than mis-spliced.
A `@pl.jit_trace.body` reports that refusal in its own vocabulary (a body
cannot be called from inside another body), and the statements it reuses
belong in the body that calls them or in a `@pl.jit_trace.inline` callee.
The body's authored scope policy travels with it:
`body.splice(..., auto_scope=False)` keeps a hand-scoped body's explicit
regions legal while the caller's own policy still governs everything else,
which is the splice spelling of `@pl.jit_trace.inline(auto_scope=False)`.

A splice's parameters are **in-place aliases** of the caller's values, the
same rule the inline splice follows. A body that rebinds one — including a
rebind carried by a loop inside a `pl.at` scope, the shape every Qwen3 helper
uses — writes through the caller's variable, so the caller sees the update and
the enclosing loop derives its carried channel. Handing a name back to itself
(`out = splice_into(x, out)`, the write-through idiom) emits nothing.

A return delivers a value through a parameter-backed buffer or a fresh,
static-shape entry allocation. Existing `pl.Out` buffers and values that trace
back to them keep their write-through behavior. The check follows the value
(`return_lineage`), not the variable name, so a callee result that returns the
buffer it wrote is delivered:

```python
@pl.jit_trace.inline
def project(x: pl.Tensor[[64, 64], pl.FP32], y: pl.Tensor[[64, 64], pl.FP32]):
    y[:, :] = pl.add(x, 1.0)
    return y

@pl.jit_trace
def tuple_of_results(
    x: pl.Tensor[[64, 64], pl.FP32],
    a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    qa = project(x, a)
    qb = project(x, b)
    return qa, qb                # two delivered results, spliced flat
```

Multiple results are one staged `pl.Tuple` — the operator that yields a
`TupleType`. A Python tuple literal at the return position is not a host
container there: it splices into the parser lane's flat contract (`return a, b`
is N results), while a list literal and `pl.Tuple(a, b)` are the one structural
value. Both derive the same flat signature
(`-> tuple[pl.Tensor[...], pl.Tensor[...]]`), so the two lanes agree and the
materialized `ReturnStmt` is the only difference; spelling a return as
`pl.Tuple(...)` is never required. In PROGRAM mode, a fresh top-level allocation
returned by the entry is
lowered to a hidden `Out` argument. The host allocates its buffer and returns it
without adding a parameter to the public signature. Both frontends use this
lowering. Mixed results retain source return order, and repeated results retain
their aliasing. Dynamic created outputs, returned views without a delivery
mapping, and allocations whose lifetime cannot be promoted are rejected.
KERNEL mode continues to require results backed by external parameters.

A tuple-valued callee call can be returned directly: `return callee(x, a, b)`
has the same delivery lineage as binding its results and returning
`pl.Tuple(first, second)`. Classification follows the tuple value's elements,
including through multiple forwarding calls.

A declared `->` is checked against the values, never stamped over them. If the
body returns a `[8, 8]` buffer under a declared `[16, 8]`, or returns one value
under a two-result declaration, tracing stops with a `TypeError` naming the
position and both types. Alternative staged paths must also agree: two branches
returning different result types are refused instead of leaving the signature
empty, because an empty signature would hide the disagreement from every later
caller. A body with no value-bearing return keeps its declared type — a jit entry
describes the output buffer it writes through `pl.Out`.

`@pl.function` / `@pl.program` declarations continue to use the AST lane. An
external kernel has no Python body to trace, so it stays a declaration: calling
a `@pl.jit.extern` stub from a traced body joins the entry's assembly, while
`pl.jit_trace` has no `extern` form of its own. The remaining family spellings
beyond the verified surface (`opaque`) raise a `TypeError` at decoration
instead of tracing an unverified lane.

That list is the boundary: a kernel that needs a capability it leaves out stays
on the parsing lane, and a traced body that calls a parser-lane kernel raises
instead of silently leaving the frontend.
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
without naming it — through a loop carry, for instance — and that is a genuine
write-through, so it stays silent:

```python
for col, (d,) in pl.range(0, n, chunk, init_values=(data,)):
    d = pl.store(local, [0, col], d)
    staged = pl.yield_(d)
data = pld.tensor.allreduce(staged, signal, ...)   # `staged` *is* `data`; no warning
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

## How a `@pl.program` Class Is Located

`@pl.program` parses the class *body from source*, so it first has to find the
`class` statement that produced the decorated object. The class name alone does
not identify it: one function may define the same name in several branches, and
every one of them carries the same `__qualname__`.

The decorator resolves this from the line numbers of the methods in the class
body, so each branch is parsed from its own source:

```python
def make(case):
    if case == "add":
        @pl.program
        class Prog:                                # parsed from *this* body
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

When the definitions genuinely cannot be told apart, the decorator raises a
`ParserSyntaxError` naming every candidate line rather than picking one — a
wrong pick would compile a body you never wrote. Give the classes distinct
names, or define the class once and vary it through a closure variable:

```python
def make(scale):
    @pl.program
    class Prog:                                    # one definition, parameterised
        @pl.function
        def main(self, x: pl.Tensor[[8], pl.FP32]) -> pl.Tensor[[8], pl.FP32]:
            return pl.mul(x, scale)
    return Prog
```

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

### Tensor Operations (Loop with iter_args)

```python
# pypto.program: my_program
import pypto.language as pl

def loop_sum(n: pl.INT64) -> pl.INT64:
    sum_init: pl.INT64 = 0
    for i, (sum,) in pl.range(n, init_values=(sum_init,)):
        sum = pl.yield_(sum + i)
    return sum
```

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

Generic Function attrs print as `pl.func_attr({...})` directives in the body
prologue. They remain Function metadata when parsed, including on HOST
SubWorkers: the captured Python `InlineStmt.body` excludes these directives
while preserving the implementation and its docstring. Parameter-valued attrs
resolve to the function's own parameters. The prologue must precede executable
statements (a leading docstring is allowed); misplaced directives and duplicate
keys are errors, also for SubWorkers.

### Printer implementation

The private `IRPythonPrinter` declaration lives in
`src/ir/transforms/python_printer.h`. Its implementations share one visitor
instance, so variable renaming, signature context, indentation, and scope state
remain consistent across recursive expression, type, and statement printing.

| Source file in `src/ir/transforms/` | Responsibility |
| ----------------------------------- | -------------- |
| `python_printer.cpp` | Public entry points and format callback, name disambiguation, function/program printing and dependency order |
| `python_printer_expr.cpp` | Expression precedence, constants, calls/submits, and attribute values |
| `python_printer_stmt.cpp` | Control flow, scopes, comments, SSA yields, and window-buffer declarations |
| `python_printer_type.cpp` | Type annotations, shapes, memory references, and tensor/tile views |

The public declarations remain in `include/pypto/ir/transforms/printer.h`.
Sub-expression printers inherit the enclosing variable-name maps, preserving
bindings in shape expressions, memory offsets, and slot indices.

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
