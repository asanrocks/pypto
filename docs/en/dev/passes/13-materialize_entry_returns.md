# MaterializeEntryReturns Pass

Exports an entry's freshly allocated, statically shaped result buffers as
hidden host-allocated `Out` parameters during PROGRAM (artifact) compilation.
The public signature stays what the author declared; only the runtime ABI
gains the parameters it needs to hand the results back.

## Overview

The runtime's external result ABI is whole-buffer: every value an entry
delivers must live in a parameter the host can pass in and read back. An
entry that *returns a buffer it allocated* —

```python
out = pl.create_tensor([8, 8], dtype=pl.FP32)
...
return out
```

— hands the caller nothing to read: the host holds no handle on the
allocation. This pass is the boundary adapter that fixes that, and it is
frontend-neutral: a `@pl.jit` entry and a `@pl.jit_trace` entry that both
return a fresh allocation deliver through the same transformation.

The pass is a Python program pass (`python/pypto/ir/_entry_outputs.py`)
over the existing pass interface — no new core IR node — because what it
changes is not IR structure but the entry's *ABI*: the parameter list the
runtime dispatches against. That is also why it is a boundary adapter
rather than an ordinary optimization: it must run exactly once, in the
compile mode that owns that ABI.

## Example

Author-facing kernel (tracing lane; the parsing lane spells the same
shape):

```python
@pl.jit_trace
def add_one(x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out = pl.create_tensor([8, 8], dtype=pl.FP32)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.add(pl.load(x, [0, 0], [8, 8]), 1.0), [0, 0], out)
    return out
```

Before the pass, the entry owns the allocation and the host has nothing to
read (`ir.python_print` of the pre-pass entry):

```python
def add_one(self, x: pl.Tensor[[8, 8], pl.FP32]) -> pl.Tensor[[8, 8], pl.FP32]:
    out: pl.Tensor[[8, 8], pl.FP32] = pl.tensor.create([8, 8], dtype=pl.FP32, layout=pl.TensorLayout.ND)
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.tile.store(pl.tile.adds(pl.tile.load(x, [0, 0], [8, 8], [8, 8]), 1.0), [0, 0], out)
    return out
```

After the Default pipeline (entry only; surrounding program elided):

```python
def add_one(
    self,
    x__ssa_v0: pl.Tensor[[8, 8], pl.FP32, pl.MemRef(...)],
    entry_result_0: pl.Out[pl.Tensor[[8, 8], pl.FP32, pl.MemRef(...)]],
) -> pl.Tensor[[8, 8], pl.FP32]:
    pl.func_attr({"entry_output_abi": True, "entry_result_0": 1,
                  "entry_result_count": 1, "entry_synthesized_count": 1})
    ...
```

The `tensor.create` is gone; the allocation became the hidden
`entry_result_0` parameter, and `entry_result_0: 1` records that logical
result 0 is delivered by physical argument 1. A host call allocates that
buffer, dispatches, and returns it — the public call stays `add_one(x)`.

## When it runs

**Pipeline position**: after
[ConvertTensorToTileOps](12-convert_tensor_to_tile_ops.md), before
[OptimizeOrchTensors](14-optimize_orch_tensors.md). It requires the
properties the conversion produces (`SSAForm`, `SplitIncoreOrch`,
`NormalizedStmtStructure`, `IncoreTileOps`) and runs before orchestration
optimization so every downstream pass — and codegen — sees the final
physical parameter list.

**Activation**: only artifact compilation requests a host result ABI.
`ir.compile` (PROGRAM mode) first runs `prepare_program_outputs`, which
resolves the entry's return signature and stamps `entry_output_abi` on the
entry — but only when the runtime's no-metadata fallback ("the pure `Out`
parameters, in declaration order") could not already describe delivery: a
fresh top-level allocation, a return order other than declaration order, a
duplicated result, or an `InOut`-backed result. The pass itself is inert
without the attribute.

**Not applied**: structural `lower()` (no host result ABI is requested);
KERNEL compilation (every result must alias an external parameter — the
borrowed-buffer contract); distributed programs beyond a sole traced host
entry; programs with L3+ orchestration; programs without a resolvable
single entry. An entry whose `tensor.create` buffers are not yet top-level
(ordinary lowering hoists them there) is re-checked at the final compile
boundary instead of being failed here.

## The transform

1. Temporarily treat the entry's top-level `tensor.create` statements as
   `Out` parameters and run the shared whole-buffer return-lineage
   analysis — the same `returned_param_indices` / `delivered_param_indices`
   walk [NormalizeReturnOrder](30-normalize_return_order.md)
   canonicalizes — to prove which buffers the entry actually returns.
   Provenance, not the declared type, decides: a forwarded callee result is
   classified by the value it carries.
2. If every delivered value already lives in a parameter, no storage
   transformation happens; the provenance map alone is persisted when it
   disagrees with the fallback.
3. Otherwise: remove each promoted allocation, append one hidden `Out`
   parameter per distinct returned buffer (`entry_result_<i>`,
   deduplicated against the entry's parameter names), substitute the
   parameter through the body, and record `result_indices` — which logical
   result index each physical argument carries, duplicates preserving
   aliasing. Mixed results (a created buffer beside a parameter-backed one)
   keep source return order.

## Refusals (fail closed)

| Refusal | Why |
| ------- | --- |
| a created output without a static tensor shape | the host must allocate the hidden buffer before dispatch |
| `manual_dep=True` on a created output | the create statement — sole carrier of the lifetime-granularity dependency opt-out — is deleted on promotion; a hidden parameter cannot preserve the semantic |
| a result position the whole-buffer ABI cannot deliver (a scalar- or tile-valued created result) | there is no parameter slot the runtime could pack |
| incomplete provenance — a returned value backed by neither a parameter nor a fresh top-level static allocation | fail closed rather than ship an empty result |

## What survives downstream

The `entry_result_*` function attributes carry the mapping through
subsequent passes, independently of return canonicalization. Artifact
metadata (`compiled_meta.json`, schema 4) persists `synthesized_outputs`
and `result_indices`; host dispatch allocates the hidden buffers on every
call; `CompiledProgram.from_dir` reload restores the same allocation and
packing rules, and `CompiledProgram.result_indices` is the single packing
rule shared by direct dispatch and the ChipWorker path.

## API

| Python | Level |
| ------ | ----- |
| `materialize_entry_returns()` — registered in the Default pipeline | Program-level |

## See Also

- [ConvertTensorToTileOps](12-convert_tensor_to_tile_ops.md) — the pass whose `IncoreTileOps` property this adapter requires.
- [NormalizeReturnOrder](30-normalize_return_order.md) — the shared return-lineage analysis both consume.
- [Functions and Program Structure](../language/03-functions.md) — created outputs from the author's perspective.
