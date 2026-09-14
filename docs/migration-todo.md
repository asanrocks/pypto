# Migration TODO — finishing the tracing-frontend sweep

Working task file for migrating the remaining parser-era surfaces in this
repository to the tracing dialect. Scope: `examples/`, `tests/st/`, and
`docs/`. **Out of scope:** `python/pypto/`, `src/`, `include/`, and
`python/bindings/` are existing logic and stay untouched.

This file is the single source of task truth. `docs/design/04-migration-guide.md`
Appendix B summarizes it; when the two disagree, this file wins.

## How this file works

One session = pick one task → work it → verify red-then-green → update
this file → commit. Tasks are independent; prefer the lowest-numbered
open task in a group. When a group's last task closes, update the
matching Appendix B row in the same commit (see *Commit protocol*).

Task states and editing rules:

- `[ ]` open, `[x]` done, `[~]` blocked — a blocked task must name its
  entry in `docs/migration-known-gaps.md`.
- Append newly discovered tasks under the right group with the next free
  id; note where the discovery happened.
- Never delete a task. A task whose red check cannot be made to fail is
  mischaracterized — re-triage it in place (it may already be migrated,
  or belong to the exclusions below).
- Fill the *Notes* line with what the next agent needs; do not encode
  the work itself here — the guide owns the rules.

## Context (read before working)

- `docs/design/04-migration-guide.md` — the translation rules. §2's three
  principles derive every change; §4–§6 are the per-tier rules; §7 pairs
  each loud error with its fix.
- `AGENTS.md` (repo root) — commands, git discipline, commit rules.
- The `spec(...)` commits (`git log --oneline --grep='^spec('`) — real
  before/after migrations; their diffs are the oracle for any spelling
  you are unsure about.
- `tests/ut/language/conformance/` — the frontend-equivalence oracle.
- `docs/migration-known-gaps.md` — problems that cannot be fixed within
  this file's constraints.

## Constraints

1. **No existing-logic changes.** If a task cannot be completed without
   touching frontend, runtime, or compiler code, the task is blocked:
   write the gaps-file entry, mark `[~]`, and stop.
2. **Red, then green.** Every task names a red check and a green check.
   Run the red check first and observe the failure; run the same check
   after the change and observe it pass. One commit may carry both the
   test and the fix, but both observations must actually happen. If the
   documented red check does not fail, re-triage before changing
   anything.
3. **Unfixable problems go to the gaps file** — never a silent
   workaround in a test, example, or doc.

## Task inventory

### E — examples

- [x] **E1** `examples/models/03_flash_attention.py` — migrate the body
  (init_values loops, per-branch `pl.yield_` → derived carries + branch
  `pl.const`, guide §4.1–§4.3). Keep the execution skip; repoint the
  dangling `KNOWN_ISSUES.md` comment at `docs/migration-known-gaps.md`
  (gap K3).
  Red (test change first): add a `__main__` invocation with sample
  torch tensors and observe the dialect failure (today the module only
  prints the wrapper, so the body is never exercised). Green: after
  migration the call no longer raises a dialect error — reaching the
  pre-existing IR-verification failure (K3) is the expected terminal
  state, and the skip guard stays.
  Notes: done. The migrated body (derived carries, plain branch
  assignments; no branch value is a host literal, so no `pl.const` was
  needed) traces and passes IR verification — the K3 IfStmt
  yield/return_vars mismatch was an artifact of the parser-era spelling
  and is gone. The call then terminated at the Vec-space capacity limit
  and the Acc-source assemble defect (both recorded as gap K3); both are
  now fixed (task F7 in `docs/gap-fixes-todo.md` — the assemble bridge in
  `InferTileMemorySpace`, the footprint relieved by
  `optimizations=[pl.cross_core_slot(slot_num=1)]`), the skip guard is
  gone, and `__main__` runs the kernel end-to-end on a2a3sim with a fp32
  golden check; no test suite imports the example.
- [x] **E2** `examples/models/04_paged_attention.py` — migrate the
  `is_first`/`is_last` branch-yield kernels (l.352–461) to branch
  `pl.const`; re-triage the current `'ir.Function' object is not
  callable` failure — it may be an additional dialect issue.
  Red: `uv run python examples/models/04_paged_attention.py` fails.
  Green: exits 0.
  Notes: done. The `'ir.Function' object is not callable` failure was
  re-triaged as a spelling issue, not a frontend gap: the `@pl.program`
  method lane adapts *closure-captured* standalone callees but not bare
  module-global names (only the standalone `@pl.function` and jit lanes
  wrap module globals). The supported spelling is builder-local capture —
  the module-level kernels stay importable (the st harness imports them)
  and each builder binds them as locals (`_init = kernel_init_inplace` …),
  the pattern example 07 already uses. Branch yields became
  `pl.const(True/False, pl.BOOL)` (guide §4.1/§4.3; spec(const)). Green
  verified end-to-end on device 11 via `task-submit --device 11 --run`:
  compiles, executes, golden validation prints PASSED, exit 0. Env note
  for E3–E6: device execution needs the simpler runtime — `uv pip
  install --no-build-isolation -e ./runtime` (after `uv pip install
  scikit-build-core`) is now in this worktree's `.venv`; the unaligned
  builder also traces device-free (main() only exercises the first).
- [x] **E3** `examples/models/05_paged_attention_batch.py` — migrate the
  branch-yield spellings (l.464 area).
  Red/green: run the file.
  Notes: done. Two spellings migrated: the `max_bn` loop-carry seed
  became `pl.const(0, pl.INT64)` (guide §4.1 — the red failure fired
  there first, at the typed staged initialization, before any
  `pl.yield_` site; the in-file comment claiming a literal-seeded carry
  was parser-era and is corrected), and the is_first/is_last branch
  yields became `pl.const(True/False, pl.BOOL)` (§4.3, spec(const),
  matching the E2 oracle). `KernelOnlineUpdate`'s flags moved to
  `pl.Scalar[pl.BOOL]` with bare truthiness checks (`if is_first:`),
  matching 04. Kernels are class methods here, so the E2
  builder-local-capture spelling was not needed. Green: the file runs
  end-to-end — traces, compiles through the full pass pipeline, dumps
  orchestration C++, exit 0 (device-free; `ir.compile` only). No test
  suite imports the builder (only doc tables and `examples/models/
  __init__.py` reference the name); ruff check/format clean.
- [x] **E4** `examples/models/06_paged_attention_dynamic.py` — same.
  Red/green: run the file.
  Notes: done. The only stale spelling was the is_first/is_last branch
  yields `pl.yield_(1/0)` (l.359–365), which became
  `pl.const(True/False, pl.BOOL)` (guide §4.1/§4.3, matching the E2/E3
  oracle). Red fired at trace time inside the builder:
  `TypeError: call to 'dyn_kernel_online_update': argument <_YieldMarker
  object> is a host value of an unsupported type`. No other migration
  was needed: the kernels are builder-local closures (so E2's
  builder-local-capture spelling does not apply) and there is no
  host-literal loop carry. Green verified end-to-end on device 11 via
  `task-submit --device 11 --run`: compiles, executes, golden validation
  prints PASSED, exit 0. Env note: under task-submit use the worktree
  `.venv/bin/python` explicitly — the default `python` cannot import
  `pypto.pypto_core` (extension lives in the worktree venv). The st
  consumers (`tests/st/runtime/framework_and_models/
  test_dynamic_paged_attention.py`, `tests/st/codegen/torch/
  test_torch_codegen_paged_attention.py`) import only the builder, which
  now traces device-free; device runs of those suites stay on demand.
- [x] **E5** `examples/models/07_paged_attention_multi_config.py` —
  migrate the remaining parser-era kernels (l.332, l.422, l.593+);
  `make_kernel_softmax_prepare` is already migrated (spec(const)) and is
  the in-file reference.
  Red: `uv run python examples/models/07_paged_attention_multi_config.py`
  → `TypeError: range() got an unexpected keyword argument 'init_values'`
  (verified). Green: exits 0.
  Notes: done. Three spellings migrated: the `kernel_qk_matmul` and
  `kernel_pv_matmul` accumulation loops became the
  `pl.intrin.carry`/`yield_`/`leak` channels (the in-file
  `make_kernel_softmax_prepare` reference; spec(carry)/spec(const)) —
  `init_values=`/iter-tuple unpacking/`pl.yield_` gone, and pv's
  `oi_l0c_out = oi_l0c` zero-trip fallback dropped (leak binds the loop
  result, which is the carried init for zero trips); the orchestration
  `is_first`/`is_last` branch yields became `pl.const(True/False,
  pl.BOOL)` (guide §4.1/§4.3, the E2–E4 oracle) with
  `kernel_online_update`'s flags moved to `pl.Scalar[pl.BOOL]` + bare
  truthiness checks, matching example 04. Red fired first in
  `make_kernel_qk_matmul` (`range() got an unexpected keyword argument
  'init_values'`). Green: the file runs end-to-end on a2a3sim —
  compiles, executes, golden validation prints PASSED, exit 0.
  Consumers: the ut codegen suite duplicates the builders (no import) —
  ran it, 1 passed; the st harness imports the module-level kernels and
  its module import traces, but its programs call them bare — the
  pre-existing E2-class spelling issue (04's harness fails identically),
  appended as S8; the st call shape (INT64 config scalars into the
  now-BOOL flags) verified to trace under closure capture. Ruff
  check/format clean.
- [x] **E6** `examples/models/09_paged_attention_spmd.py` — same.
  Notes: done. Two spellings migrated, matching the E2–E5 oracle: the
  `max_bn_cfg` branch yields became plain assignments (guide §4.3 — both
  branch values are staged: `zero_bn_cfg` is `pl.const(0, pl.INT64)`,
  `first_bn_cfg` is derived from staged reads; red fired at the subsequent
  `pl.max(max_bn_cfg, bn_b)` because the yield left `max_bn_cfg` as a
  `_YieldMarker`), and the `is_first` branch yields `pl.yield_(1/0)` became
  `pl.const(True/False, pl.BOOL)` (§4.1/§4.3) with `SpmdOnlineUpdate`'s flag
  moved to `pl.Scalar[pl.BOOL]` + bare truthiness check (`if is_first:`),
  matching examples 04–07. Kernels are class methods (`self.*` calls), so
  E2's builder-local-capture spelling does not apply. Green verified
  end-to-end on device 11 via `task-submit --device 11 --run`: compiles,
  executes, golden validation prints PASSED, exit 0 (host run reaches
  `simpler_init failed with code 507899` without the task-submit device
  lock — device execution stays on demand). The st consumer
  (`tests/st/runtime/framework_and_models/test_paged_attention_spmd.py`)
  calls only the builder/class methods; both test-case `get_program()`s
  verified to trace device-free. Ruff check/format clean.
- [x] **E7** `examples/utils/phase_fence_dep_compression.py` — the
  second builder `build_chained_snapshot_manual_dummy_phase_fence`
  (l.102+) still uses `init_values=`/`pl.yield_`; migrate it exactly like
  its sibling `build_chained_snapshot_phase_fence` (spec(carry) diff).
  Red/green: invoke the builder (`uv run python -c "from
  examples.utils.phase_fence_dep_compression import
  build_chained_snapshot_manual_dummy_phase_fence as b; b()"` — adapt
  imports as needed). Check `tests/ut/codegen/` and `tests/st/` for
  which suites consume it; extend coverage if none does.
  Notes: done. Both loops migrated to the carry/yield_/leak channels,
  exactly mirroring the sibling (spec(carry); the `deps =
  pl.system.task_dummy(deps=[tids_a])` barrier sits between the loop
  head and the parallel fan-out and needs no change). Red fired at the
  builder call: `TypeError: range() got an unexpected keyword argument
  'init_values'` (l.137). Consumers: the st scheduling suite
  (`tests/st/runtime/scheduling/test_phase_fence_dep_compression.py`,
  `_chained_snapshot_manual_dummy_case`) calls the builder but is
  device-gated; no ut suite covered it, so coverage was extended in
  `tests/ut/codegen/test_phase_fence_dep_compression.py` with
  `test_chained_snapshot_manual_dummy_example_emits_per_loop_barriers`
  (one dummy-task submission per phase loop, branch-sized fan-in from
  the carried `tids` array, real tasks depending only on the dummy
  TaskId) — its own red (same TypeError, observed against the HEAD
  example) and green both verified. Full ut codegen suite: 22 passed;
  file `__main__` exits 0; ruff check/format clean. Device runs of the
  st suite stay on demand.

- [x] **E8** the remaining example `__main__`s still call the AST-era jit
  `lower()` API: `examples/intermediate/05_assemble.py` (l.229,
  `fn.lower(*args)` over the tile-assemble cases),
  `examples/models/qwen3_jit/qwen3_decode.py` (l.138,
  `qwen3_decode.lower(*args)`), and `examples/utils/
  cross_function_calls.py` (l.58, `main_kernel.lower(a, c)`).
  Discovered during S9 (its grep companions; 06_dyn_valid_shape's main
  was fixed in S9 as the red-check module). All three callees are lazy
  `_JitSpecializer`s, so the likely replacement is `compile_for_test` —
  but triage each: qwen3_decode's arg list is large and the printed
  function-type table may need the backend set first.
  Red: run each file (`uv run python <file>`) → AttributeError.
  Green: each file exits 0.
  Notes: done. All three red failures observed as documented
  (`AttributeError: '_JitSpecializer' object has no attribute 'lower'`);
  all three callees are the lazy lane (bare `pl.Tensor` annotations), so
  the replacement is the S9 oracle `compile_for_test(*args)` (docstring
  and inline `lower` mentions updated to match). Two of the three also
  needed the backend set first, exactly as the task anticipated —
  `run_passes` queries the backend for the Mat/Acc (L1/L0C) tile layouts
  (plain-Vec kernels like cross_function_calls pass without it):
  `backend.set_backend_type(BackendType.Ascend910B)` in `__main__` before
  the compile call (fresh-process spelling, no reset; the ut oracle
  `tests/ut/jit/test_qwen3_decode.py` gets the same from its conftest's
  autouse `_setup_backend`). Green: all three mains exit 0 — 05_assemble
  prints its 6 cases (acc_mat 4 fns, others 2), qwen3_decode prints the
  16-function table (Orchestration entry + Group/AIC/AIV kernels),
  cross_function_calls prints its 3-function table. Consumers: the ut
  oracle re-ran green (2 passed; the set call lives in `__main__`, so
  importing the module has no backend side effect);
  `tests/st/runtime/ops/test_assemble.py` imports 05_assemble's kernels
  but fails collection on the pre-existing S11 harness gap (`st.case`
  rejects tracing jit kernels) — untouched here, S11 owns it; its
  device runs stay on demand. Ruff check/format clean.

### S — tests/st

Each file's program builder traces eagerly — the red check is
device-free; device execution stays on demand (`task-submit`).

- [x] **S1** `tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py` —
  module-level `@pl.jit` with `init_values=`/yield-LHS (l.60–68).
  Red: import the module (or run the builder). Green: collection passes.
  Notes: done. The chunk loop migrated per guide §4.2 (spec(carry)):
  `init_values=(out,)` + iter-tuple unpacking + `chunk_out = pl.yield_(o)`
  became a plain `for c in pl.range(0, s_dim, CHUNK):` whose
  `out = pl.store(t, [b, c, 0], out)` rebind *is* the carry. Red fired at
  import (the kernel is eagerly traced at decoration — every param
  facade-annotated): `TypeError: range() got an unexpected keyword argument
  'init_values'`. Green exceeds the contract: the test itself now passes
  device-free. The test body's `cast_3d_dynamic.lower(x, out)` was the
  AST-era jit API (`python/pypto/jit/decorator.py`) — the tracing jit has
  no `lower`, and the eager lane's only pass-pipeline route is public
  assembly: `ir.Program([fn], ...)` + `PassManager.get_strategy(Default)
  .run_passes(...)` with `backend.set_backend_type(BackendType.Ascend910B)`
  (same reset/set/reset pattern as tests/ut/conftest.py's `ascend_backend`
  fixture). Since eager means the decoration trace *is* the specialization
  (B_DYN/S_DYN stay symbolic from the annotations), the torch sample args
  were dropped; post-pass functions are `cast_3d_dynamic` plus the outlined
  `inner` InCore function. Consumers: nothing imports the module (the ut
  transform suite mentions it in a comment only — ran it, 103 passed);
  docs references are prose. Ruff check/format clean. Discovery: `.lower`
  on jit kernels is gone from the tracing jit and is used across several
  other st suites → appended S9.
- [x] **S2** `tests/st/distributed/test_l3_host_tensor_allreduce_ring.py`
  — Red (verified): call `_build_host_ring_allreduce_program(2, 128)` →
  `TypeError: range() got an unexpected keyword argument 'init_values'`.
  Green: the call traces.
  Notes: done. Both stage-in/out loops in `_build_host_ring_allreduce_program`
  migrated to the carry/yield_/leak channels, exactly matching the
  spec(carry) oracle for the sibling InCore suite
  (`tests/st/distributed/collectives/test_l3_tensor_allreduce_intrinsic.py`,
  migrated in the same commit): `(data_iter,) = pl.intrin.carry(data)` /
  `for col in pl.range(0, size, stage_cols):` / `pl.intrin.yield_(data_iter)` /
  `(staged_data,) = pl.intrin.leak()` (and the out-mirror). Red verified at
  `publish_step` (`TypeError: range() got an unexpected keyword argument
  'init_values'`). Green verified device-free for every builder shape the
  tests parameterize: n_ranks 2/4 × size 128, sizes 1/17/4097/65537, and
  the signal-reuse program (which needed no change — its steps are
  single-shot stores). Device runs stay on demand; doc mentions of the
  file are prose tables only. Ruff check/format clean.
- [x] **S3** `tests/st/distributed/test_l3_host_tensor_allreduce.py` —
  same shape as S2.
  Notes: done. Both stage-in/out loops in `_build_host_allreduce`'s
  `publish_step`/`consume_step` migrated to the carry/yield_/leak channels,
  matching the S2 spec(carry) oracle: `(data_iter,) = pl.intrin.carry(data)` /
  `for col in pl.range(0, sz, stage_cols):` / `pl.intrin.yield_(data_iter)` /
  `(staged_data,) = pl.intrin.leak()` (and the out-mirror). Red verified at the
  builder call: `TypeError: range() got an unexpected keyword argument
  'init_values'` (fired in `publish_step`, at decoration-time trace of the
  InCore function). Green verified device-free for every builder shape the
  tests parameterize: sizes 1/17/257/8193 (FP16), 17/257 (FP32), reduce ops
  Max/Min/Prod (FP16), plus the signal-reuse and loop builders and both
  module-level programs (`HostTensorAllReduce`, `HostTensorAllReduceMax` —
  these needed no change). Device runs stay on demand; doc mentions
  (`docs/en|zh/user/distributed/01-collectives.md`, `docs/en|zh/dev/
  distributed_ops.md`) are prose tables only. Ruff check/format clean.
- [x] **S4** `tests/st/distributed/collectives/test_l3_tensor_allreduce_ring_intrinsic.py`
  Notes: done. Both stage-in/out loops in `reduce_step` migrated to the
  carry/yield_/leak channels, exactly matching the in-directory spec(carry)
  oracle (`test_l3_tensor_allreduce_intrinsic.py`, migrated in the same
  commit as S2): `(data_iter,) = pl.intrin.carry(data)` /
  `for col in pl.range(0, sz, stage_cols):` / `pl.intrin.yield_(data_iter)` /
  `(staged_data,) = pl.intrin.leak()` (and the out-mirror). Red verified at
  the builder call: `TypeError: range() got an unexpected keyword argument
  'init_values'` (fired in `reduce_step`, at the decoration-time trace of
  `@pl.program`). Green verified device-free for every builder shape the
  tests parameterize: n_ranks 2/4 × sizes 1/17/4097/65537 (FP32), all four
  reduce ops at P=2 size 17, and n_ranks 2/4 × sizes 1/17/8193/32769 (FP16).
  Device runs stay on demand; the only external references are prose/doc
  tables (`docs/en|zh/{user/distributed/01-collectives.md,
  dev/distributed_ops.md}`) and a docstring mention in the S2 host-ring file.
  Ruff check/format clean.
- [x] **S5** `tests/st/runtime/control_flow/test_dyn_orch_shape.py`
  Notes: done. Two failure classes at red (all seven builders invoked
  device-free): the paged-attention orchestration failed with
  `'ir.Function' object is not callable` — the S8-class bare module-global
  example-kernel calls inside a `@pl.program` method — and
  `DynOrchReshapeAddTestCase` failed the §4.4 declared-return check
  (`the traced return type ... does not match the declared '->'`). Fixes,
  both matching the E2 oracle (example 04): the five imported example
  kernels are bound as builder locals in `get_program` (`_init =
  kernel_init_inplace` …) with the calls rebinding to them, and the
  reshape orchestrator's `->` now states the honest static type
  `pl.Tensor[[rows, cols], pl.FP32]` (the reshape bakes the static shape;
  the old `[M, N]` return referenced M as the flat 1D length, so it was
  never truthful). The `is_first`/`is_last` branch spellings became
  `pl.const(True/False, pl.BOOL)` (guide §4.1/§4.3) with the lying
  `pl.Scalar[pl.INT64]` annotations dropped, and the PA orchestration's
  dead tensor-local annotations (staged scalars embedded in shape
  annotations) were removed to match example 04's spelling. Green: all
  seven builders trace device-free and pytest collects the module (7/28
  collected, 21 platform-deselected). No consumers outside the todo
  file; device runs stay on demand.
- [x] **S6** `tests/st/runtime/control_flow/test_ctrl_flow.py`
  Notes: re-triaged as already migrated — no change needed. The red
  check (invoke every builder device-free) does not fail: all 11
  test-case `get_program()`s trace cleanly (`grep` for
  `init_values`/`pl.while_`/`pl.cond(`/`pl.yield_(` has zero hits — the
  only match is a comment naming the parser-owned spelling), and pytest
  collects the module (11/44, 33 platform-deselected). The file was
  migrated by spec commit `236dbad47 spec(carry)` — the same oracle the
  S2–S4 migrations cite: the yield-add and tile-accum kernels use the
  `pl.intrin.carry`/`yield_`/`leak` channels, the while-loop test uses
  the bare `while` + staged-predicate spelling (guide §4.2), and branch
  values are plain assignments. No other parser-era surfaces
  (`*_parser` markers, `jit_specializer`, `static_assert`/`static_print`)
  appear; no consumers outside this todo file. Device runs stay on
  demand.
- [x] **S7** `tests/st/runtime/ops/test_memory_planner_ptoas.py`
  Notes: done (unblocked by F2/gap K5 closing). Migrated: both
  `init_values=` loops to the carry/yield_/leak channels (spec(carry);
  red fired at import — `TypeError: range() got an unexpected keyword
  argument 'init_values'` in `LoopAccumProgram`; the S6 sibling
  `kernel_tile_accum` was the in-file oracle) and
  `MultiBufferConstSlotProgram` to the constant-slot
  `pl.intrin.transmute` spelling (guide §4.4/A.2), verified device-free
  against the parser pin (1 `pto.alloc_multi_tile` + 2
  `pto.multi_tile_get` under PTOAS, `alloc_tile` under PYPTO) and on
  device via `task-submit` (6/6 passed: loop-accumulator ×3 planners,
  const-slots ×3 planners). Closed by F2: the DSL `pl.MemRef`
  `__getitem__` now accepts a staged index expression (constant folds
  through the constant path, runtime rides `slot_index_` into
  InitMemRef), so `MultiBufferCoLiveProgram`,
  `MultiBufferDoubleBufferProgram`, and `MultiBufferAccProgram` were
  rewritten to the transmute spelling with their `buf[i % 2]` rotation
  indices; their skip markers and the PTOAS refusal test's strict
  xfail are removed. Green device-free: all three form their regions
  (the refusal raises again under PTOAS for the co-live shape; the
  double-buffer and acc programs compile under all three planners),
  and the tracing-side region formation is pinned in
  `tests/ut/ir/test_multi_buffer_tracing.py` (the parser pin stays in
  `tests/ut/ir/parser/test_multi_buffer_codegen.py`). Device runs stay
  on demand; no consumers outside this todo file.
- [x] **S8** `tests/st/runtime/framework_and_models/test_paged_attention.py`
  and `test_paged_attention_multi_config.py` — the harness programs call
  the imported module-level example kernels bare inside `@pl.program`
  methods (04: `kernel_qk_matmul`, `kernel_softmax_prepare`,
  `kernel_softmax_prepare_unaligned`, `kernel_pv_matmul`,
  `kernel_online_update`; 07: `kernel_softmax_prepare`,
  `kernel_online_update`), raising `'ir.Function' object is not
  callable` at trace time. Migrate to the closure-capture spelling the
  examples use — bind the imports as locals in the enclosing builder
  scope (see 07's `_sf/_up/_qk/_pv`); a body-local `_up = kernel`
  assignment inside the traced method does not adapt. Discovered during
  E5's consumer verification; 04's harness fails identically, so the gap
  predates E5 (E2's re-triage covered only the example side).
  Red: `PYTHONPATH=tests/st uv run python -c "from
  tests.st.runtime.framework_and_models.test_paged_attention_multi_config
  import OnlineUpdateTestCase; OnlineUpdateTestCase().get_program()"` →
  TypeError. Green: the same call traces.
  Notes: done. Migrated to the builder-scope closure-capture spelling (the
  E2/07 oracle `_sf/_up/_qk/_pv` pattern): every imported module-global
  kernel called bare inside a `@pl.program` method is now bound as a local
  in the enclosing `get_program` before the `@pl.program` class and the
  traced method calls the local — 04 harness: `_qk`/`_sf`/`_sfu`/`_pv`/`_up`
  in the five single-kernel test cases; 07 harness: `_sf`/`_up` (the
  `make_*` callees were already bound as builder locals). Red verified at
  the documented call (`TypeError: 'ir.Function' object is not callable`)
  and identically in 04's `QKMatmulTestCase`; green verified device-free
  for all 12 base test cases across both modules (the PTOAS variants share
  the same `get_program`), plus pytest collection (36 collected) and
  ruff check/format. The INT64-config-scalar → BOOL-flag call shape of
  `kernel_online_update` traces fine, matching E5's note. Device runs
  stay on demand; the batch/dynamic/spmd harnesses were never affected.
  The S group is now fully closed (S7 unblocked by gap-fix F2).
- [x] **S9** the st codegen suites call the AST-era jit `lower()` API,
  which the tracing jit does not expose (it lives only on the quarantined
  `pl.jit_specializer` in `python/pypto/jit/decorator.py`). Discovered
  during S1. Sites: `tests/st/codegen/dsl/test_dyn_valid_shape_loop.py`,
  `test_dynamic_valid_shape_if_else.py`, `test_add_mul_orch_codegen.py`,
  `test_split_aiv_ragged_split_axis_codegen.py`,
  `test_split_odd_axis_codegen.py`, `test_split_aiv_gather_row_codegen.py`,
  `tests/st/codegen/torch/test_torch_codegen_cross_core.py` (l.740), and
  `tests/st/runtime/framework_and_models/test_graph_execution.py`
  (l.398, l.482). Most callees are lazy-lane `_JitSpecializer`s (bare
  `pl.Tensor` annotations), whose public replacement is
  `compile_for_test(*args)`; triage each site — the `config=RunConfig(...)`
  spellings and eager-lane callees may need the S1 route (ir.Program +
  PassManager) or a frontend gap entry. Docs teach the same API
  (`docs/en/user/language/01-functions.md` l.58, `docs/en/user/
  02-quickstart.md` l.246, `docs/en/dev/08-entry-points.md`) and need the
  same decision — folded into D6.
  Red: `PYTHONPATH=. uv run python -c "from examples.intermediate.
  dyn_valid_shape import dyn_valid_shape_loop"` imports, but
  `dyn_valid_shape_loop.lower(...)` raises AttributeError. Green: each
  site's lowering call runs device-free on the tracing-jit surface.
  Notes: done. Red observed at every site class (the documented
  AttributeError on the example, plus the same error from `example_orch`,
  `TensorSplitAivQkPvProgram`, `per_layer_accumulate`, and the eager split
  kernels). Three replacements by callee lane: lazy-lane sites
  (dyn_valid_shape_loop/if_else, example_orch, TensorSplitAivQkPvProgram,
  the graph entries) → `compile_for_test(*args)` — the spelling
  `tests/ut/codegen/test_dynamic_valid_shape_if_else.py` already uses;
  eager-lane sites (fully-shaped split kernels — decoration returns a bare
  ir.Function) → the S1 route (ir.Program([fn]) + Default PassManager
  under Ascend910B, reset/set/reset) via a `_lowered()` helper per file;
  the graph compile-side `_lower` → `compile_for_test` under the ambient
  HOST_BUILD_GRAPH PassContext (run_passes propagates the ambient runtime
  kind; backend set around it), and the default-runtime rejection test
  keeps its ValueError via `compile_for_test` outside the context. Two
  assertion updates where the printed round-trip changed under spec(tpop):
  the tpop lines now carry explicit shape/dtype producers, so
  `tpop_from_aic(split=...)` substring checks became regexes. Enabling
  example fixes: `examples/models/02_vector_dag.py`'s in-trace scalar args
  to the `pl.Scalar[pl.FP32]` params are now `pl.const(1.0/2.0, pl.FP32)`
  (host floats into runtime-declared params are loud; the add_mul green
  needs it — verified 1 Orchestration + 3 AIV); `examples/intermediate/
  06_dyn_valid_shape.py`'s `__main__` (the red-check module) uses
  `compile_for_test` (exit 0 verified); `examples/advanced/
  04_moe_graph_predicate.py` and test_graph_execution.py's six graph
  fragments left the missing `@pl.jit.graph` alias (gap K6) for supported
  spellings — the st file's fragments (no jit callees) are standalone
  `@pl.function(type=pl.FunctionType.Graph)` + shaped annotations + honest
  `->` (the arity helper slices on `) ->`; a missing declared return made
  the A/B arity check vacuous), and the moe example's block (which
  dispatches a jit kernel, so the standalone spelling would orphan the
  callee — a bare `@pl.function` trace runs with no program assembly) is
  the scope form `with pl.graph(...)` inside `moe_decode`. Green verified
  device-free: the 5+6+2+1 lower-site tests, all six
  TestGraphIsNotSilentlyDroppedAtCompile tests, every remaining graph entry
  traces through the pipeline, and the moe example compiles to 1 Graph with
  expert_ffn present; pytest collects the graph file (16 tests).
  Discoveries appended: S10 (eager-lane host-dispatch sites in the split
  suites), S11 (st.case rejects tracing jit kernels), E8 (remaining example
  `.lower` mains), D6 (docs teaching `lower()` / `@pl.jit.graph`). Device
  runs stay on demand.

- [x] **S10** the split-suite `.pto` fixtures and rejection tests call the
  eager-lane jit kernels as host callables (`ragged_rows(*args,
  config=RunConfig(...))`, `odd_rows(*_args(...), config=cfg)`,
  `runtime_tail_rows(...)`, `runtime_tail_fillpad_first/store_first(...)`,
  `sparse_kv_qk(...)` in `tests/st/codegen/dsl/test_split_odd_axis_codegen.py`,
  `test_split_aiv_ragged_split_axis_codegen.py`,
  `test_split_aiv_gather_row_codegen.py`). These kernels are fully
  annotation-shaped, so the tracing jit traces them at decoration and
  returns a bare `ir.Function` — not callable, so every such site raises
  `TypeError: 'ir.Function' object is not callable` (the eager lane has no
  host dispatch surface; only the lazy `_JitSpecializer` lane does).
  Discovered during S9. Red: `PYTHONPATH=tests/st:. uv run python -c "from
  tests.st.codegen.dsl.test_split_odd_axis_codegen import odd_rows; import
  torch; odd_rows(torch.randn(15,128,dtype=torch.bfloat16),
  torch.randn(128,128,dtype=torch.bfloat16),
  torch.empty(16,128,dtype=torch.float32), config=None)"` → TypeError.
  Green: each site compiles + (where the test asserts on it) emits its
  artifacts device-free. Likely route (triage per site): `ir.compile(
  ir.Program([fn], fn.name, fn.span), platform="a2a3",
  **run_config_compile_kwargs(cfg))` + `compiled(*args, config=cfg)` — the
  same field mapping `_compile_program` uses for the lazy lane; the two
  `runtime_tail_*` rejection tests expect their ValueError to keep firing.
  Notes: done. Red verified at the documented call (`TypeError:
  'pypto.pypto_core.ir.Function' object is not callable`). Every site now
  uses a per-file `_compiled(fn, cfg)` helper implementing exactly the
  suggested route — `ir.compile(ir.Program([fn]), platform=cfg.platform,
  **run_config_compile_kwargs(cfg))`, mirroring jit.py's `_compile_program`
  — followed by `compiled(*args, config=cfg)` where the site dispatches.
  The two `runtime_tail_*` rejection tests keep their ValueError (verified:
  it fires inside `ir.compile`, at the pass pipeline, so `pytest.raises`
  wraps the compile; the now-unused `_runtime_tail_args` helper was
  removed). One enabling spelling fix: the rank-2 `out[:]` writes became
  explicit per-axis slices (`out[0:ROWS, 0:COLS]`, all four sites in the
  odd file) — a bare `t[:]` is a one-axis subscript under tracing
  (`assemble(..., [0])`), which codegens a rank-mismatched
  `pto.partition_view` that ptoas rejects; recorded as gap K7 in
  `docs/migration-known-gaps.md` (the parser lane compiles the same kernel
  cleanly, so this was a real tracing-lane regression the old fixture's
  error capture would have silently absorbed — with the explicit slice all
  four kernels compile through ptoas and the fixtures' captured error is
  the expected device-free `simpler_init 507899`, matching the parser-era
  terminal state). Green verified device-free: all 20 tests across the
  three modules pass, and all four dispatch sites (odd_rows,
  runtime_tail_rows, ragged_rows, sparse_kv_qk) compile through ptoas and
  emit their `.pto` artifacts. Device runs stay on demand.

- [x] **S11** the st harness's jit-kernel case lane only accepts the
  quarantined `pl.jit_specializer`: `st.case(kernel, *sample_args)`
  isinstance-checks `pypto.jit.JITFunction` (`tests/st/harness/st.py`
  l.116), so a tracing `_JitSpecializer` (every modern `@pl.jit` entry with
  bare annotations) is rejected with `ValueError: ... sample arguments only
  apply to a @pl.jit kernel` at collection. Discovered during S9's
  consumer check (test_dag.py imports the migrated vector_dag example).
  Broken at collection: `tests/st/runtime/control_flow/test_dag.py`,
  `tests/st/runtime/cross_core/test_spmd_dynamic_gm_pipe.py`,
  `test_split_reduce_parity.py`, `test_subview_tmov_valid_shape.py`,
  `test_sync_set_wait.py` (re-grep `st\.case\(` for the full set).
  Red: `PYTHONPATH=tests/st:. uv run python -m pytest
  tests/st/runtime/control_flow/test_dag.py --collect-only` → ValueError.
  Green: the suites collect. The fix is harness-side (teach `st.case` /
  `JitKernel` the `_JitSpecializer` surface — compile via the kernel's own
  dispatch lane) unless that needs frontend support, in which case write
  the gaps-file entry instead.
  Notes: done, harness-side only (`tests/st/harness/`). The full broken set
  was larger than the five listed files — a whole-tree `--collect-only`
  found **25** suites dying on the same ValueError (every st file declaring
  a jit case; the eager-lane subset died on a bare `ir.Function` instead of
  a `_JitSpecializer`). Two new sources in
  `tests/st/harness/core/kernel_source.py`: `TracingJitKernel` (the lazy
  `_JitSpecializer` lane — builds the pre-pass program through the jit's
  own `_bridge_args` → `_build_program` route, the same one `compile()`
  uses; there is no public pre-pass entry point, `compile`/`compile_for_test`
  both run a pipeline first) and `EagerJitKernel` (fully-shaped annotations
  trace at decoration, so the kernel already *is* a bare `ir.Function`;
  program = the S1/S10 one-function `ir.Program`; the samples are
  cross-checked against the baked signature — shape constants and dtype,
  with dynamic extents exempt since they bind at dispatch — so a wrong
  buffer fails at declaration, not on device). `st.case` dispatches both
  (`_JitSpecializer` → TracingJitKernel, `_ir.Function` + samples →
  EagerJitKernel; an `ir.Function` without samples keeps the legacy
  ProgramKernel fall-through), `case.py` registers both in its isinstance
  gate. Regression coverage: `TestTracingJitDeclaration` in
  `tests/st/harness/test_case_declaration.py` (spec derivation incl. Out
  directions, program build, loud missing-sample / shape-mismatch errors,
  cache-id binding-style agreement, quarantined lane unchanged) — its own
  red verified against HEAD before the fix. Green: whole-tree collection
  now has **zero** `st.case` errors (1790 collected); the only 2 remaining
  collection errors are a different root cause, appended as S12. The
  harness's own device-free suites: 61 passed. Device triage: the jit's own
  dispatch lane runs green on device (`_abs_entry(a, out)` via task-submit,
  err 0.0), but the harness `st.cases` **Case device-run path fails on
  device for every kernel source** (tracing jit: `prepare_native_run code
  13`; a `@pl.program` ProgramKernel probe: AICPU 507018) while the legacy
  `PTOTestCase` path passes on the same box (3/3
  `test_loop_carried_accumulator`) — pre-existing and never exercisable
  before, since every st.cases jit case failed collection; appended as S13,
  not a regression of this change. Ruff check/format and pyright clean on
  all touched files.
- [x] **S12** `tests/st/runtime/cross_core/test_cross_core.py` and
  `test_cross_core_grouped_tpop_tfree.py` fail collection with
  `InvalidOperationError: tpop_from_aiv() missing 1 required positional
  argument: 'shape'` — their `pl.tpop_from_aiv(split=..., id=...)` call
  sites predate the spec(tpop) explicit shape/dtype-producer change (the
  same change S9 saw from the printed-text side, where it regex-patched
  the tpop assertions). Discovered during S11's whole-tree collection run
  (the only 2 remaining collection errors).
  Red: `PYTHONPATH=tests/st:. uv run python -m pytest
  tests/st/runtime/cross_core/test_cross_core.py --collect-only` →
  InvalidOperationError. Green: both suites collect (and the grouped file's
  printed-IR assertions still hold).
  Notes: done. Both suites share one root: the grouped file imports
  `V2CUDProgram` from `test_cross_core.py`, so its collection died on the
  same import-time trace. The four stale sites (two in
  `MultiPipeNoSplitProgram.cube_multi_pipe`, two in `ExplicitSlotNumProgram`)
  now pass the explicit producers exactly per spec(tpop)'s fixture diff —
  `shape=[MULTI_PIPE_DIM, MULTI_PIPE_DIM], dtype=pl.FP32, split=0, id=N`
  (module-level int constants trace fine as shape elements). Red observed
  on both suites; green: 12 + 1 tests collected. The grouped file's
  printed-IR assertions verified device-free: `ir.compile(V2CUDProgram)`
  (Default strategy, a2a3) then `_rewrite_consecutive_tpop_tfree_pto`
  against the generated `main_incore_0.pto` — its 2-tpop / 2-tfree /
  8-line-block assertions all pass (V2CUDProgram itself has no source-level
  tpop; the lowered tpop comes from the split pass, so the migration could
  not have changed that form). Full device runs stay on demand. Ruff
  check/format clean on the touched file.
- [x] **S13** the `st.cases` **Case device-run path** fails on device for
  every kernel source: `test_case_pipeline.py::test_declared_case_runs`
  (a tracing-jit case) dies at `prepare_native_run failed with code 13`,
  and a `@pl.program` ProgramKernel probe case dies with AICPU 507018
  (`aicpu execute failed`), while the legacy `PTOTestCase` path runs green
  on the same box. Discovered during S11's device verification.
  **Re-triage (red check no longer fails):** both lanes pass on device
  now — the break was runtime-side (the A2/A3 worker-return ordering
  race, fixed by runtime #2137 `da65a1ee5`), present in the worktree's
  runtime submodule since the E2-era `uv pip install -e ./runtime`;
  Case TaskArg assembly and the compiled-artifact run are innocent.
  Resolution: a permanent `@pl.program` ProgramKernel case
  (`test_declared_program_case_runs`) joins the pipeline suite so the
  Case device-run lane keeps coverage; device-verified green (4 passed).
  **Infra note for device tasks:** a root-owned auto-update holds the
  device-submission lock daily ~12:17–14:17 (stale `.task.tmp` files
  make it think the queue is busy). Do not poll through it — pick a task
  that is device-free, or reschedule; two agent sessions were spent
  sleeping on this window.

- [x] **S14** `tests/st/runtime/control_flow/test_ctrl_flow.py`
  `WhileLoopAddTestCase` — the `kernel_while_add` counter seed is the
  annotated spelling `i: pl.Scalar[pl.INDEX] = 0`, which does **not** stage
  under tracing (local annotations never flow into IR), so the `while i < 4:`
  executes as a *host* loop at trace time and the InCore function emits four
  unrolled load/add/store blocks — no WhileStmt. The test still passes on
  device (unrolled code computes the same c = a + b), but its stated intent
  ("Validates that PTO codegen emits scf.while correctly") is no longer
  exercised. Discovered during D1's doc verification (the doc's old
  while section and this kernel shared the stale spelling).
  Red: `PYTHONPATH=tests/st uv run python -c "from
  tests.st.runtime.control_flow.test_ctrl_flow import WhileLoopAddTestCase;
  import tests.st.runtime.control_flow.test_ctrl_flow as m"` … or simpler:
  trace the InCore kernel standalone and print it — no `while` in the
  printed IR. Green: the seed is `pl.const(0, pl.INDEX)` (guide §4.1/§4.2)
  and the printed IR carries a real staged while.
  Notes: done. The seed became `i = pl.const(0, pl.INDEX)` (guide §4.1) —
  that one line is the whole change: bare host literals in the staged
  predicate/arithmetic (`i < 4`, `i * 64`, `i + 1`) stage fine (verified by
  probe — only *initialization* of staged state needs `pl.const`), and
  `out`/`i` rebinding derives the carries with no declaration (§4.2).
  Red observed: `get_program()` printed four unrolled load/add/store
  blocks and no `while` (the annotated seed `i: pl.Scalar[pl.INDEX] = 0`
  never staged). Green: the InCore body is `AssignStmt, WhileStmt,
  ReturnStmt` — one representative iteration in a real WhileStmt — and
  the device-free Default-strategy `ir.compile` emits
  `scf.while (%0 = %c0_index)` in `ptoas/kernel_while_add.pto`, so the
  test's stated intent (scf.while codegen) is genuinely exercised again.
  Device-verified green via `task-submit --device auto`:
  `test_while_loop_add[a2a3]` PASSED (golden c = a + b). Module collects
  11/44 (33 platform-deselected, unchanged); ruff check/format clean;
  no consumers outside this todo file.

### D — documentation

en and zh move together (parity is lint-enforced,
`tests/lint/check_docs_en_zh_parity.py`). Green check for all D tasks:
`uv run python tests/docs/run_doc_examples.py --pages docs/en/user --check-parity -p a2a3sim`
(plus the pre-commit doc hooks).

- [x] **D1** `docs/en/user/language/02-control-flow.md` (+zh) — the worst
  offender: teaches `init_values=`, `pl.while_`, `pl.cond()` as the
  language. Rewrite around derived carries, bare `while`, and the
  assembly channels; `pl.unroll`'s no-carry rule stays.
  Notes: done, both trees. The page now teaches: carried values derived
  from rebinding (no declaration), staged state seeded by operators or
  `pl.const` (a host-literal seed is loud — verified error text
  `pl.add: expected Tensor or Tile operands, got (float, Tensor)`), bare
  `while` over a staged predicate (`pl.while_`/`pl.cond` do not exist —
  `hasattr` False), branch values by plain assignment with `pl.const` for
  host values crossing the join, and the assembly channels
  (`pl.intrin.carry`/`yield_`/`leak`, oracle: `test_ctrl_flow.py`'s
  `kernel_add_yield`) as the one explicit-carry surface. `pl.unroll`'s
  no-carry rule restated as what it now is: the explicit carry channel is
  rejected (`pl.unroll() cannot be combined with init_values` — verified;
  the error text still says init_values), while a *derived* carry in an
  unroll loop traces fine, so the old "Carried values: No" table cell is
  now "Derived only". Two facts corrected while rewriting: typed carry
  overloads go to **four** values (the old text said five), and a local
  annotation (`i: pl.Scalar[pl.INDEX] = 0`) does not stage a seed — see the
  appended S14. The page gained its first runnable blocks (quickstart
  accumulate + a while-loop chunked_square) in the passing-pages' idiom
  (lazy-lane bare `pl.Tensor` annotations, rank-2 `pl.load`/`pl.store` to
  stay clear of gap K7's bare `out[:]` on rank≥2): both PASS on a2a3sim.
  Red: the page had no doctest marks, so the runner never executed it —
  the concrete reds were observed by tracing the page's own snippets
  (`init_values=` → TypeError; `pl.while_`/`pl.cond` gone). Green: the
  page's 2 run blocks pass, `--check-parity` OK, md-refs/nav/en-zh-parity
  hooks OK; the D-group invocation is still 17/31 overall — every
  remaining failure is on other open D pages (the eager-lane `.compile()`
  AttributeError in quickstart/scopes/types and D6's `lower()` sites);
  check_docs_symbol_coverage fails at HEAD too (5 symbols, none from this
  page). Deliberate grep hits remain: the round-trip prose, the
  deprecated `pl.yield_` alias note, and quoted error-message strings —
  descriptive, not teaching the old spellings.
- [x] **D2** `docs/en/dev/language/00-python_syntax.md` (+zh) — stale
  loop examples at l.158, l.224, l.226.
  Notes: done, both trees. Red observed by tracing the page's own
  snippets (D1's precedent — the page has no doctest marks, so the
  runner never executes it): the l.158 rotation loop raises
  `TypeError: range() got an unexpected keyword argument
  'init_values'`, the l.224 pipeline block the same from
  `pipeline()`, both at decoration-time trace. Migrations: the
  rotation loop became `for i in pl.range(N):` with the annotated
  tile kept as the documented *round-trip* spelling plus a new
  paragraph stating the tracing reality — constant slot index via
  `pl.intrin.transmute` at the operation (the S7 oracle), runtime
  index has no tracing spelling yet (gap K5: transmute raises, bare
  annotation silently drops), so the runtime rotation stays
  parser-owned round-trip text. The pipeline block migrated to
  derived carries (`for stack in pl.pipeline(STACKS, stage=2):` /
  bare inner `pl.range(0, N, 2 * STEP)`, the rebind is the carry),
  `b_l1`'s placement moved to `target_memory=pl.Mem.Mat` (§4.4), and
  ping/pong's declared allocations moved to the transmute spelling
  over a `pl.tile.extract` producer. The l.10 overview bullet
  ("Uses SSA with `pl.yield_()`") was reworded to keep `pl.yield_`
  only as round-trip-descriptive (D1's deliberate-hit precedent).
  Green: the migrated snippets verified device-free — the rotation
  traces with a derived carry and the constant slot surviving into
  the printed IR (`pl.MemRef("", slots=2)[0]`), the runtime-slot
  transmute raises the documented K5 TypeError, and the pipeline
  block traces and runs the full Default pass pipeline (post-pass
  the inner range loop carries the accumulator via iter args and
  the replicated pipeline stages chain it; the declared allocations
  survive InitMemRef — unnamed `pl.MemRef()` declarations print with
  an empty name but stay distinct, auto-numbered). Doc hooks green:
  en-zh parity OK (189), md-refs OK, nav OK; the D-group invocation
  unchanged at 17/31 (this page has no runnable blocks; every
  failure is on other open D pages, per D1's baseline). Discovery:
  the page's remaining Declared-Allocations staleness appended as
  D7.
- [x] **D3** `docs/en/dev/language/01-statements.md` (+zh) — the largest
  dev page (~24 hits), including the `pl.while_`/`pl.cond` walkthrough.
  Notes: done, both trees. Red observed by tracing the page's own snippets
  (D1/D2 precedent — the page has no doctest marks, so the runner never
  executes it): `pl.while_`/`pl.cond` are gone (AttributeError),
  `pl.range(init_values=)` → TypeError, if-branch `pl.yield_` → the
  typed-staged-init error (plus the deprecation warning), bare `yield` →
  RewriteError (rewrite-unsafe construct), `static_print`/`static_assert`
  are silent no-ops under tracing (a failing assert never raises), the
  annotated while seed does not stage (S14 class — the page's "natural
  while" silently degenerated to `while n > 0: pass`), and Tile/Tensor
  aug-assign is unsupported while staged-Scalar `acc += 1` traces fine.
  Migrations: Assignment (annotations never flow into IR; aug-assign
  reality — rebind through the operand's binary op, works on Scalars,
  Python TypeError on Tile/Tensor), If → plain assignment both arms +
  `pl.const` for host values crossing the join, For → derived carries
  with a `pl.const` seed (host-literal seed error text quoted), While →
  bare `while` over a staged predicate with the annotated-seed warning,
  Yield Statement → bare `yield` forbidden; `return` and the
  `pl.intrin.yield_` channel (deprecated `pl.yield_` alias noted),
  Compile-Time Debugging → static_* framed as parser-lane-only silent
  no-ops (dump_tag *is* honored by tracing — verified it lands on the
  consuming call's `dump_vars`; failure now `TypeError`), and the SSA
  summary became "How SSA Is Derived" (ConvertToSSA owns the dataflow;
  the printer's `init_values=`/`pl.yield_`/`pl.while_`+`pl.cond` forms
  described as round-trip only, with the assembly-channel exception per
  guide §6.3/A.1). Break/continue restrictions re-verified under tracing
  (new error strings quoted); the scope-table/spmd sections were never
  flagged and are unchanged. Green: every migrated snippet traces
  device-free (incl. a real WhileStmt in the printed IR and dump_tag in
  an Orchestration method); the D-group invocation is 17/31 — unchanged
  from the D1/D2 baseline, every failure on other open D pages (getting-
  started/quickstart/execution/types/scopes/precision/tools); parity OK;
  en-zh path parity 189, nav OK, md-refs OK; symbol coverage fails at
  HEAD with the same 5 symbols, none from this page. Remaining grep hits
  on the page are deliberate round-trip prose (the D1/D2 precedent).
- [x] **D4** `docs/en/dev/language/03-functions.md` (+zh) — stale
  examples at l.282, l.410.
  Notes: done, both trees. Red observed by tracing the page's own
  snippets (D1–D3 precedent — the page has no doctest marks, so the
  runner never executes it): both fire `TypeError: range() got an
  unexpected keyword argument 'init_values'` at decoration-time trace
  (l.282 concretely in an InCore reduce_step adapted from the ut
  oracle, l.410 directly). Migrations: the OutParamWriteDropped
  loop-carry example became the derived-carry spelling (`for col in
  pl.range(0, n, chunk):` / `data = pl.store(local, [0, col], data)` /
  allreduce over the carried `data`) — guide §4.2, P2 — and its
  doc-claim was re-verified with the ut harness's own check runner
  (`passes.DiagnosticCheckRegistry` PRE_PIPELINE over a program shaped
  like `test_distributed_collective_rebind_is_silent`): zero
  OutParamWriteDropped warnings, so "data-flow, not syntactic" still
  holds under the rebind spelling. The Complete Example's loop_sum
  became a standalone `@pl.function` with `pl.Scalar[pl.INT64]`
  params, a `pl.const(0, pl.INT64)` seed, and the rebind-as-carry body
  (§4.1/§4.2; heading "Loop with iter_args" → "Loop with a derived
  carry"; the `# pypto.program:` module marker dropped — that is
  round-trip module structure, not DSL); new prose notes the printed
  round-trip renders the carry as `init_values=`/`pl.yield_()` after
  SSA (verified: Default-strategy `run_passes` then `as_python()`
  shows exactly that form — the D1/D2 deliberate-hit precedent).
  Discovery: the page's jit-constants section (source re-parsing,
  folding) and "How a `@pl.program` Class Is Located" section still
  describe AST-specializer/parser behavior — appended as D8. Green:
  both migrated snippets trace device-free, the D-group invocation is
  17/31 (unchanged from the D1–D3 baseline — this page contributes no
  runnable blocks; every failure is on other open D pages), parity/
  nav/md-refs hooks OK.
- [x] **D5** `docs/en/dev/passes/` sweep (+zh) — triage every page the
  discovery grep flags: printed IR text (round-trip dialect) is correct
  and stays; DSL input examples are stale and migrate (confirmed stale:
  `12-optimize_orch_tensors.md` l.43, l.52, l.218). Record per-page
  verdicts in the task notes as you go.
  Notes: done, both trees. Red observed by tracing every flagged DSL-input
  snippet in its doc form (D1–D4 precedent — the dev pass pages carry no
  doctest marks): all fired `TypeError: range()/pipeline()/unroll() got an
  unexpected keyword argument 'init_values'`. Migrated (en+zh): `12` — the
  Pattern 1 Before/After mini-blocks and the Example (Pattern 1) programs
  now use the assembly channel spelling, mirroring the owning tests
  (`(iter_arg,) = pl.intrin.carry(buf)` / bare `pl.range` /
  `pl.intrin.yield_(result)` / `(loop_result,) = pl.intrin.leak()`,
  `pl.create_tensor` for the redundant alloc), with a one-line note that
  the printer renders the same loop as `init_values=`/`pl.yield_`
  round-trip text; `14` — the "user-written" chunk loop is the S1 oracle
  (bare `pl.range`, `out = pl.store(...)` rebind is the carry); `20` — the
  residency example's "# Tensor source" block uses the test's channel
  spelling and the post-pass block is now labeled "(printed round-trip)";
  `50` — the Example is the test's program verbatim (program class with
  the AIV `accumulate`, channel-spelling loop). Prose fixes (en, and zh
  where the sentence exists): `31` — "`init_values=` is supported" on
  `pl.pipeline` became "carried values are derived from rebinding, exactly
  as with `pl.range`" (zh never translated that sentence); `23` — "
  `pl.range(..., init_values=(tup,))` carries one directly" became the
  derived carry (a tuple reassigned in the loop body), matching
  `test_tuple_loop_carry_propagates_the_halved_type`; `02` — the unroll
  constraint's reason cell no longer claims "No loop-carried state in
  unrolled loops" (a derived carry traces; only the explicit channel is
  rejected — verified, error text `unroll() got an unexpected keyword
  argument 'init_values'`); `18` — After blocks stay (pass-generated IR)
  but gained a clarifier that they are the printer's round-trip spelling
  users never write. Verdicts — stays, round-trip/IR-internal (deliberate
  grep hits): `04` (SSA printer output + pass prose), `05`/`06`/`30`/`42`
  (IR-mechanics prose), `09` (printed SSA IR), `49` (printer/parser
  round-trip prose), `20` l.48/73 + `23` l.824/902 (IR-construct prose;
  `pl.while_` there names the printer's WhileStmt spelling, D1's
  deliberate-hit precedent), `02` l.44 constraint text (matches the
  actual error). Out of D5's scope, pre-assigned to D6: the
  `@pl.jit.graph` mentions in `08-outline_graph_scopes.md`/
  `48-legalize_graph_boundary.md`; `15`/`99`'s `static_assert` hits are
  pto-isa C++ asserts, not `pl.static_assert`. Green: every migrated
  snippet traced verbatim device-free and its pass claim re-verified —
  `12` create eliminated + Out→InOut under OptimizeOrchTensors, `14`
  through the Default pipeline, `20` single hoisted LHS Mat load, `50`
  `iter_arg_rebind_0=False` under ClassifyIterArgCarry, `31` derived
  carry lowers onto the pipeline loop. Owning ut suites green (76 + 11 +
  54 passed); doc hooks green (parity 189, md-refs, nav, english-only);
  D-group invocation 17/31 — unchanged from the D1–D4 baseline (D5 touches
  no user pages); check_headers fails only on pre-existing python/pypto
  files, none of them mine.

- [x] **D6** docs teaching the AST-era jit `lower()` API and the
  `@pl.jit.graph` decorator, both gone from the tracing jit: `docs/en/
  user/language/01-functions.md` (l.58 `lower()`, l.75 `@pl.jit.graph`
  — the alias is gap K6, the spelling tables need the standalone /
  scope forms), `docs/en/user/02-quickstart.md` (l.246), `docs/en/dev/
  08-entry-points.md`, and `docs/en/dev/passes/
  48-legalize_graph_boundary.md` (l.222 uses the decorator in an
  example; `08-outline_graph_scopes.md`'s prose mentions may stay if
  the pass docs describe the convergence, per D5's printed-IR rule).
  Folded out of S9 (its doc decision). Green check: the D-group
  invocation above.
  Notes: done, both trees. The red check fired exactly as documented
  and wider than the four listed files: the D-group invocation failed
  on six runnable `lower()` sites — `tools/00-debugging.md:28`,
  `00-getting_started.md:61`(+92, one setup serving two run blocks),
  `execution/01-run.md:49`(+95), and `precision/00-workflow.md:143`
  beyond the listed pages' prose (D1 had already attributed "D6's
  `lower()` sites" to this class) — every one
  `AttributeError: '_JitSpecializer' object has no attribute 'lower'`
  (or `'ir.Function'`, for the eager-lane callee). All were fixed in
  scope. Replacements, probed device-free first (a2a3sim): lazy-lane
  `lower(*args)` → `compile_for_test(*args)` (tools block; asserts
  hold — post-pass outlining present); the ChipWorker setups' `ir.compile(kernel.lower(...))` → `kernel.compile(*args,
  config=RunConfig(platform=...))` (full resident-tensor + register
  flows re-verified); the precision block's `fused.lower(a, b, ...)` →
  `torch_codegen(fused)` (the fully annotated kernel *is* the
  ir.Function, probed executable). Prose migrations (en+zh):
  `01-functions.md` — the quickstart read-the-IR line became
  `entry.as_python()` (eager lane) with the compile/dispatch pointer;
  the jit-family table dropped the `@pl.jit.graph` row and the
  Graph paragraph now teaches the two supported spellings (standalone
  `@pl.function(type=pl.FunctionType.Graph)` for fragments with no
  jit callees; scope `with pl.graph("name"):` inside the entry —
  K6's tracking rationale, the standalone trace runs no program
  assembly so it would orphan dispatched callees); `.graph` dropped
  from the auto-discovered aliases; "Splitting compile from dispatch"
  now teaches `compile()` (fresh artifact per call; dispatch lane
  caches) and `compile_for_test()` (Default pipeline, no config, the
  round-trip oracle), the `specialize()` paragraph became the
  eager-lane pre-pass route (`kernel.as_python()` /
  `ir.compile(ir.Program([kernel], kernel.name, kernel.span), ...)`
  with the one-function-program caveat), `output_param_names` dropped
  from the accessor list (gone; `param_names()` + `__name__` remain,
  probed), and the pitfall/edge rows moved to `compile_for_test()`;
  a quarantine note (`pl.jit_specializer` / `pl.JITFunction`) keeps
  `check_docs_symbol_coverage` at its HEAD-baseline failure set (the
  renames would otherwise have newly uncovered `pl.JITFunction`; the
  note also covers `pl.jit_specializer`, so the check now fails on
  4 pre-existing symbols instead of HEAD's 5). `02-quickstart.md` —
  the inspect section teaches the eager `add.as_python()` and the
  lazy `compile_for_test(*sample_args)` side by side. `08-entry-points.md`
  — the layer diagram/tables name `compile_for_test` and point the jit
  entries at `language/tracer/decorators/jit.py` (was the quarantined
  `python/pypto/jit/decorator.py`). `48-legalize_graph_boundary.md`
  l.222's Before example uses the standalone Graph spelling; the
  `@pl.jit.graph` mentions in `08-outline_graph_scopes.md` (4),
  `passes/index.md`, `ir/01-hierarchy.md`, and `execution/00-compile.md`
  (the `runtime=` row) were reworded to "a standalone Graph function"
  — the convergence claims stay, the dead input spelling does not.
  Green: the D-group invocation moved 17/31 → **23/31** — all six
  lower()-class blocks PASS (getting_started ×2, execution/01-run ×2,
  precision, tools) and `--check-parity` OK; the 8 remaining failures
  are the distinct eager-lane/K7 classes, appended as D9. Doc hooks:
  en-zh parity 189, md-refs, nav, english-only all OK; symbol coverage
  4 pre-existing symbols (better than HEAD's 5); check-headers fails
  only on pre-existing python/pypto files (D5 baseline).
- [x] **D7** `docs/en/dev/language/00-python_syntax.md` (+zh) — the
  Declared Allocations section still teaches parser-era spellings
  beyond D2's loop examples (discovered during D2): the `t0/t1/t2`
  block and the constant-slot `l0c[0]`/`l0c[1]` block bind
  allocations through bare local annotations — silently dropped
  under tracing (K5's evidence class; the tracing spelling is
  `pl.intrin.transmute`, demonstrated one paragraph below the slots
  block since D2) — the "unnamed declaration takes the name of the
  variable" / one-to-one rejection prose is parser-side (under
  tracing `pl.MemRef()` prints with an empty name; explicit
  `pl.MemRef("name")` is the naming spelling, unnamed declarations
  stay distinct), and the "Under `@pl.jit`" paragraph claims the jit
  "re-parses a generated source" — AST-era specializer behavior the
  tracing jit does not have.
  Red: trace each block's spelling under `@pl.function` and print —
  the annotation-bound allocations are absent from the printed IR
  (silent drop, so assert on the printed text / post-InitMemRef
  output).
  Green: blocks state allocations at the operation; prose matches
  tracing behavior.
  Notes: done, both trees. Red observed by tracing the page's own
  blocks: the t0/t1/t2 annotation-bound ping/pong prints **zero**
  `pl.MemRef(` occurrences, and the constant-slot block prints its
  matmuls with no MemRef view — both silent drops, no error.
  Migrations: both blocks now state the allocation at the operation
  via `pl.intrin.transmute(producer, pl.Tile[..., memref, space])`
  (guide §4.4/A.2); the annotated spellings are kept as the printed
  round-trip forms (D2's precedent). Prose rewritten around verified
  tracing behavior: an unnamed declaration prints `pl.MemRef("")`
  and stays distinct (3 tile refs / 2 allocations, t0/t2 share; post-
  InitMemRef two separate `pinned=True` allocs), naming is explicit
  `pl.MemRef("name")`, sharing follows the declaration *object* —
  `alias = ping` shares (no rejection), and **each evaluation of
  `pl.MemRef(...)` is its own declaration** (two inline same-string
  occurrences are two allocations that print alike). The
  `@pl.jit` paragraph now states the tracing jit executes the body
  as Python (variable-held declarations verified in scope on both
  jit lanes) and names the AST-era re-parse as `pl.jit_specializer`
  history. Green: the migrated blocks trace with the allocations
  visible in the printed IR (`pl.MemRef("ping")` shared by t0/t2;
  `pl.MemRef("", slots=2)[0]/[1]` — slots distinct by
  `slot_index_`), and InitMemRef emits `pinned=True`. Doc hooks
  green: en-zh parity 189, md-refs, nav, headers, english-only OK;
  symbol coverage at its D6 baseline (4 pre-existing symbols, none
  from this page); D-group invocation 23/31 — unchanged (this dev
  page has no doctest marks; all failures are D9's classes).
- [x] **D8** `docs/en/dev/language/03-functions.md` (+zh) — two sections
  beyond D4's examples still describe the AST-era frontends as current
  (discovered during D4): the "JIT constants and compilation reuse"
  section teaches that a `@pl.jit` body "is parsed into `@pl.program`
  source and re-parsed in a namespace holding only `pl` and `pld`" and
  that the key "hashes the *emitted text*" — quarantined
  `pl.jit_specializer` mechanics; the tracing jit traces the callable
  and keys host knobs on the baked value (§5 of the guide), and its
  eager lane (the section's own `slice_kernel.compile()` example is
  fully facade-annotated) returns a bare `ir.Function` with no
  `.compile()` — verified `AttributeError` at the page's example. The
  "How a `@pl.program` Class Is Located" section teaches
  parser source-location resolution and the `ParserSyntaxError`
  two-branches example — under tracing both branches trace fine and
  return distinct Programs (verified); no source is located.
  Red: trace the page's own `make(case)` two-branches example (both
  Programs collect — no error) and the jit-constants example
  (AttributeError at `.compile()`).
  Green: prose describes the tracing jit's actual constant/key
  semantics and the class section states tracing behavior (or moves to
  round-trip/parser-owned framing per the D2 precedent).
  Notes: done, both trees. Red observed exactly as documented: the
  `make(case)` two-branches example collects both branches under tracing
  (distinct `ir.Program`s, no `ParserSyntaxError`, no source located) and the
  jit-constants example dies with `AttributeError: 'ir.Function' object has no
  attribute 'compile'` (fully facade-annotated = eager lane). Migrations — the
  section (retitled "Constants and compilation reuse") now teaches tracing
  semantics with every claim probe-verified device-free: the body is traced as
  Python and inherited names are read at trace time (the "must have a source
  form" rule framed as `pl.jit_specializer` history); lane timing (eager bakes
  at decoration — verified the artifact is stable across a `BLOCK` rebind;
  lazy re-reads at each compile — verified 32-row vs 64-row loads); host-knob
  values join the dispatch cache key and `pl.runtime` keys on dtype only (both
  verified via `_cache_key`); the dispatch cache keys on bridged signature +
  compile identity and does **not** track inherited names (verified: the key
  is identical across a rebind — the warm-cache staleness is stated plainly,
  with "rebind before first dispatch or use compile()" as the route);
  `compile()` is always fresh (verified distinct `CompiledProgram`s). The two
  subsections inside the section were triaged and migrated as well:
  "Compile options and diagnostic requests" (anchor kept —
  08-artifact-identity.md links it) now states that diagnostics route through
  `compile()` because the dispatch cache deliberately ignores diagnostic knobs
  (verified on a2a3sim: a warm dispatch + `dump_passes=True` reused the cached
  artifact and emitted nothing; `compile()` honors the dumps); "Prepare
  binaries without executing" teaches `compile(*args, config=...)` as the
  prepare-without-execute surface (verified device-free with
  `platform="a2a3"`) with the `warmup()`/persistent-cache protocol framed as
  quarantined `pl.jit_specializer` surface (the tracing jit has no `warmup` —
  verified AttributeError on both lanes). The class-location section
  (retitled "`@pl.program` Classes Defined in Branches") states the tracing
  reality (each branch is its own decoration; both code examples trace
  verbatim, including the closure-`scale` one) and frames the line-number /
  `ParserSyntaxError` machinery as parser-owned `@pl.program_parser` behavior
  (that entrypoint verified to still collect both branches). All the page's
  own snippets trace/compile device-free verbatim. Doc hooks green: parity
  189, nav, md-refs, english-only OK; symbol coverage and check_headers fail
  only on the pre-existing baselines (4 symbols / python/pypto files);
  D-group invocation 23/31 — unchanged (this dev page has no doctest marks;
  every failure is D9's classes). Discovery: `10-jit-cache.md` /
  `09-artifact-store.md` teach the quarantined persistent-cache/warmup
  surface as the current `@pl.jit` surface — appended D10.
- [x] **D9** the remaining user-page doc-runner failures are one
  class with two roots (discovered during D6's green check; the D1
  notes already pointed at it as "the eager-lane `.compile()`
  AttributeError in quickstart/scopes/types"): (a) runnable blocks
  define fully-annotated jit kernels — eager lane, so the decorated
  name is a bare `ir.Function` with **no host dispatch surface** —
  then call `kernel.compile()` / `kernel(*args, config=...)`:
  `02-quickstart.md:49` (`add.compile()`), `02-quickstart.md:101`,
  `language/00-types.md:226`, `language/04-scopes.md:46`/`116`,
  `precision/00-workflow.md:94` (all `TypeError: 'ir.Function' object
  is not callable`), plus the no-args-signature compile teaching
  (`00-getting_started.md`'s "Compiling from the signature"
  `prefill_fwd.compile()` and `02-quickstart.md`'s "How `compile()`
  gets its shapes" table); (b) two blocks with separate roots:
  `language/00-types.md:43` teaches a host float into a
  `pl.Scalar[fp32]` param (the loud world-check wants
  `pl.runtime(value, dtype)` or a host-knob annotation) and
  `execution/00-compile.md:31` writes a rank-2 `out[:]` — gap K7
  (needs explicit per-axis slices or K7 to close). Also the
  parser-era "parses, not executes" prose these pages still carry
  (`02-quickstart.md` concept + fatal-pitfall row, `01-functions.md`
  concept: under tracing the body *is* executed as Python at trace
  time).
  Red: the D-group invocation — the 8 failures above (23/31 after
  D6).
  Green: runnable blocks use the D1 passing-pages idiom (lazy-lane
  bare `pl.Tensor` annotations) or an honest eager route, and the
  concept prose states tracing behavior. Likely route for (a):
  bare-`pl.Tensor` annotations, which makes the taught
  `compile()`/dispatch surface real again.
  Notes: done, both trees. Root (a) — the five eager-lane dispatch/compile blocks
  migrated to the D1 passing-pages idiom (bare `pl.Tensor`, lazy lane): quickstart
  `add`/`add_then_square`, types `apply_scale`, scopes `scale`/`spmd_add`, workflow
  `to_fp16`, so the taught `compile(*samples, config=)`/dispatch surface is real
  again; their rank-2 `out[:]` writes became explicit per-axis slices
  (`out[0:128, 0:128]`) — K7's supported spelling, since a bare `t[:]` is a
  one-axis subscript under tracing. types:43's scalar: `factor: pl.Scalar[pl.FP32]`
  is a runtime parameter, so the in-trace call stages it (`pl.const(2.0,
  pl.FP32)`); `pl.runtime` is entry-call-only (in-trace it dies on the
  annotation-only unwrap — probed). The no-args-signature teaching became the
  honest eager route: getting_started's "Compiling from the signature" teaches
  `ir.compile(ir.Program([kernel, *callees]), platform=...)` (probed end-to-end:
  one extent-independent artifact serving two extents, scalar value passed at
  dispatch; the `pl.RUNTIME`/keyword-specialization text was quarantined-
  specializer surface and is gone), quickstart's shapes table states both lanes,
  and types' dynamic block dispatches two extents through one compiled artifact
  (drive+rows in one program — a missing callee is the loud
  OrchestrationReferencesResolved error, probed). Prose: quickstart concept +
  fatal pitfall now state trace-time execution; the module-level-alias claim is
  gone (aliases resolve — probed, decoration succeeds); functions.md's concept
  rewritten around tracing; scopes' `optimizations=`-must-be-inline claim dropped
  (a variable-built list traces — probed); types' "checked at parse time" →
  trace time; compile.md's residual `lower(*args)` paragraph and edge row (a
  D6-class miss on that page) now teach `compile_for_test`. Plain-fence rank-2
  `out[:]` on the touched pages (quickstart loop/discovery fences, functions
  bad/good) moved to explicit slices for consistency. Green: the D-group
  invocation is **31/31** (was 23/31), parity OK; doc hooks green (parity 189,
  nav, md-refs, english-only, headers; symbol coverage at its 4-symbol D6
  baseline). Discovery: the tutorials chapter and several other user pages teach
  rank-2 `out[:] =` as *the* output spelling on eager kernels with no doctest
  marks, so the runner never executes them — appended D11.
- [x] **D10** `docs/en/dev/10-jit-cache.md` and
  `docs/en/dev/09-artifact-store.md` (+zh) teach the persistent JIT cache /
  artifact store as the current `@pl.jit` surface — `decode.warmup(config=...)`,
  the `python -m pypto.jit warm` CLI, READY publication — but `warmup()` and
  the persistent cache live only on the quarantined `pl.jit_specializer`
  (`python/pypto/jit/`); the tracing jit has an in-process dispatch cache and
  a cache-less `compile()` (D8's verified facts). Discovered during D8
  (03-functions.md's warmup subsection linked there; its own text now frames
  the link as quarantined-owned).
  Red: call `.warmup()` on a tracing `@pl.jit` kernel (e.g. the old
  `add_three` from 03-functions.md) → `AttributeError` (no `warmup` on
  either lane).
  Green: the pages state which runtime owns the surface they document
  (quarantine framing or migration).
  Notes: done, both trees. Red observed on both tracing lanes exactly as
  documented: lazy `_JitSpecializer` → `AttributeError: '_JitSpecializer'
  object has no attribute 'warmup'`; eager (bare `ir.Function`) → the same
  on both `warmup` and `compile`; the quarantined `pl.jit_specializer`
  counterpart (`JITFunction`) does have `warmup` (probed), confirming the
  ownership split. Ownership facts verified before writing: `RunConfig.
  cache_config` is consumed only at `python/pypto/jit/decorator.py:2519`
  (quarantined lane) — the tracing jit's compile path forwards through
  `run_config_compile_kwargs`, which drops it, and `python/pypto/language/
  tracer/` has zero cache references; behaviorally probed: a tracing
  `compile(config=RunConfig(cache_config=CacheConfig(enabled=True, ...)))`
  succeeds with `cache_stats()` recording 0 requests and an empty cache
  root (silently ignored). Migrations (en+zh): `10-jit-cache.md` gained a
  top `> [!WARNING]` runtime-ownership callout (the distributed_ops.md
  precedent) naming the quarantined runtime as owner of everything on the
  page — `warmup()`, the warm CLI, and the `configure_cache()`/
  `RunConfig.cache_config` integration — and stating the tracing jit's
  actual caching (in-process dispatch cache keyed on bridged signature +
  compile identity, cache-less `compile()`), with a link to
  03-functions.md; `09-artifact-store.md`'s opening paragraph now says the
  opt-in JIT integration connecting the store to ordinary compilation/
  execution/warmup is the quarantined runtime's surface and the tracing
  `@pl.jit` never reads or publishes through the store (the page's
  internal-module documentation — `pypto.jit.artifact_cache`,
  `pypto.runtime._artifact_runtime` — stays accurate as-is; the store
  layer itself is real infrastructure, `CompiledProgram` only consults the
  adapter when a quarantined-lane `bind_artifact` attached it, default
  `None`). The in-page mechanics (request flow, config, toolchain, stats,
  CLI, adapter contract) were left untouched — they correctly document the
  quarantined surface they describe. Green: the D-group invocation stays
  **31/31** (these dev pages carry no doctest marks), `--check-parity` OK;
  doc hooks green — en-zh parity 189, md-refs, nav, english-only OK; symbol
  coverage fails only on the 4-symbol D6 baseline (`pl.expand`,
  `pl.program_parser`, `pl.static_dim`, `pl.static_shape`, none from these
  pages).
- [x] **D11** the tutorials chapter and several other user pages teach the
  rank-2 whole-tensor write `out[:] = value` as *the* output spelling on
  fully-annotated (eager-lane) kernels — under tracing a bare `t[:]` is a
  one-axis subscript (gap K7: ptoas rejects `'pto.partition_view' op offset
  count (1) must match source rank (2)`), and the eager lane has no host
  dispatch surface, so those blocks cannot run as written. None of the pages
  carries doctest marks, so the doc runner never executes them — which is why
  the D-group invocation stayed green while the teaching was stale. Sites:
  `tutorials/00-elementwise.md` (the "`out[:] =` is how you write an output"
  walkthrough, the whole-tensor table row, the edge row), `tutorials/
  02-matmul.md`, `03-mixed-kernel.md`, `04-task-graph.md`, `language/
  05-directives.md`, `language/06-syntax.md`, `03-programming-model.md`,
  `tasks/00-model.md`, `tasks/02-submit.md`, `appendix/01-faq.md`, and
  `performance/02-runtime-overhead.md` (fence). Discovered during D9 (its
  K7/eager-lan fixes needed one consistent spelling on the pages D9 touched).
  Red: run any of the blocks as written (e.g. elementwise's `add_tensor(a, b,
  out, config=...)`) → `TypeError: 'ir.Function' object is not callable`; a
  bare-annotated copy of the same body → the K7 ptoas rejection.
  Green: the pages teach a runnable spelling (explicit per-axis slices + the
  lazy lane, D9's idiom) or K7 closes and `out[:]` is correct again —
  re-triage against K7's state when worked; giving at least the elementwise
  block doctest marks would put the chapter under the runner.
  Notes: done, both trees. Red observed exactly as documented on the
  elementwise block as written (`TypeError: ... 'ir.Function' object is not
  callable`, eager lane) and on a bare-annotated copy of the same body
  (dispatch reaches ptoas and dies with K7's `'pto.partition_view' op offset
  count (1) must match source rank (2)`). Migrations: the tutorials chapter
  moved to the D9 idiom (lazy-lane bare `pl.Tensor` annotations + explicit
  per-axis slices) and is now **under the doc runner** — 6 new runnable
  blocks (elementwise steps 1/2/4 as run + step 3's `add_chunked` definition
  as a mid-page setup block the runner prepends, matmul step 1, mixed step 1,
  task-graph step 1), each probed on a2a3sim before marking; the D-group
  invocation moved 31/31 → **37/37**. Two tutorial prose lies fixed with the
  spelling: "the parser says so" on misplaced ops (it is orchestration
  codegen's Misplaced tensor op, the quickstart quote) and "the parser
  rewrites" the subscript (it executes as `pl.assemble` — the strict_ssa
  rejection claim re-verified under tracing). Same-class fixes beyond the
  listed sites (fragments, D9's precedent of keeping full annotations on
  non-dispatching fragments where shapes are the teaching): matmul step 2
  (k-blocked, lazy lane — `c[0:128, 0:128] = acc`) and step 4 (`c[0:M, 0:N]`),
  mixed's why-bother fragment, task-graph's step-2 fragment,
  03-programming-model's quickstart (lazy lane — its `levels` entry as
  written is eager and undispatchable) plus its D6-class misses found while
  there: the whole "observable stages" section taught the AST-era `lower()`
  (now `compile_for_test(*args)`, claims probed against jit.py: Default
  pipeline, no codegen, no snapshots; the eager-lane `as_python()` parenthetical
  matches 01-functions.md), "Decorators parse source into IR" (concept +
  pipeline diagram) and "the parser renames" (SSA pass) — tasks/00-model and
  tasks/02-submit quickstarts likewise lazy lane. Two run-block sites the
  runner could not catch (their blocks take pass-only routes so the K7
  rejection never fires): `tools/00-debugging.md` (`compile_for_test`) and
  `precision/00-workflow.md` (`torch_codegen`) — slices fixed, blocks re-run
  green. Also fixed: faq's "write nothing" answer (per-axis slice + the
  one-axis caveat), performance/02-runtime-overhead's mixed-kernel fence, and
  its stale "The parser catches it" predicate claim — probed: a predicate
  producer outside `deps=` traces and runs with no diagnostic under tracing,
  so the text now says nothing checks it. Doc hooks green: parity 189, md-refs,
  nav, english-only OK; symbol coverage at its 4-symbol D6 baseline, none from
  these pages; check_headers fails only on the pre-existing non-docs files.
  Discovery: `language/05-directives.md` and `language/06-syntax.md` are
  wholesale parser-era pages (their `out[:]` sites are fixed here, but both
  teach "a decorator parses your source" as the concept, parse-time
  static_* semantics, parser closure capture) — appended as D12.
- [x] **D12** `docs/en/user/language/05-directives.md` and
  `docs/en/user/language/06-syntax.md` (+zh) are wholesale parser-era pages:
  05-directives teaches `pl.static_print`/`pl.static_assert` as parse-time
  constructs of "a decorator that parses your function's source" (under
  tracing they are silent no-ops — the D3 verdict: a failing assert never
  raises, static_print prints nothing), and 06-syntax teaches "A decorator
  parses your source; it never runs it as Python", parser-time closure
  capture, and "the parser rewrites subscripts" as the concept (its
  strict_ssa and sugar tables are accurate under tracing — verified in D11 —
  but the framing prose around them is not). Discovered during D11 (only
  their `out[:]` sites were in D11's scope).
  Red: trace the pages' own snippets — `static_assert(False, ...)` never
  raises; `static_print` never prints; the "parse time" claims have no
  referent under tracing.
  Green: both pages state tracing behavior (rewrite around execution-time
  semantics, silent no-ops, and which diagnostics remain loud), or the
  constructs are re-homed explicitly as round-trip/parser-owned.
  Notes: done, both trees. Red observed exactly as documented and wider:
  `pl.static_print` printed nothing and returned `None` (host and in-trace),
  `pl.static_assert(False, ...)` never raised (the kernel collected fine,
  nothing in the IR), the directives quickstart itself died at trace time
  with `TypeError: 'NoneType' object is not subscriptable` (`x.shape`/
  `x.dtype` on a traced facade are `None` — a staged value carries no shape
  to Python), and 06's concept claim is inverted: the body *executes* as
  Python at trace time (probe print statements inside the body ran at
  decoration). One new trap pinned while probing: binding the no-op's
  result (`s = pl.static_print(...)`) reifies `None` into the TaskId
  sentinel — a `Scalar[pl.TASK_ID]` holding `pl.system.task_invalid()`
  lands in the IR (statement-position calls are safe). Migrations:
  05-directives now frames `pl.const` as the tracing-native construct and
  `static_*` as parser-owned silent no-ops (guide §8, the D3 statements.md
  oracle), with the host-`if`-that-raises replacement (verified both ways:
  the untaken branch leaves no IR; a taken one raises a real ValueError at
  trace time), a verifier-pass pointer, the None-binding trap, and the
  shape-introspection row; 06-syntax's concept/quickstart/closure-capture
  rewritten around execution semantics — subscripts/operators dispatch
  through the staged wrappers' protocol methods (the sugar table verified
  accurate: `x[0:16, :]` → `pl.tensor.slice(x, [16, 64], [0, 0])`,
  `out[...] =` → `pl.tensor.assemble`, `pl.mul(t, 2.0)` → `muls`), closure
  capture is ordinary name reads at trace time (a host int folds by value —
  `N = 8` traces as `8`; an unstageable value dies loudly naming the
  operator: `pl.mul: expected Tensor or Tile operands, got (Tensor,
  Widget)`), the parser-era "bind `available_cluster_count()` to a name
  breaks the printed IR" claim dropped (both spellings trace and print
  self-contained locals — verified), the strict_ssa rejection re-verified
  with its current error text, and the one-axis `t[:]` caveat added (K7's
  supported spelling, matching D9/D11). Both pages gained their first
  runnable blocks (05: the no-op reality + the host-if guard; 06: the head
  quickstart asserting on the printed IR) — the D-group invocation moved
  37/37 → 40/40, `--check-parity` OK; `index.md` summary rows updated
  (en+zh). Doc hooks green: parity 189, md-refs, nav, english-only OK;
  symbol coverage at its 4-symbol D6 baseline; check_headers fails only on
  pre-existing python/pypto files. Discovery: `docs/en/dev/language/
  02-manual_dependencies.md` teaches `pl.at(..., predicate=...)`, which
  does not exist under tracing — appended D13; the guide's Appendix B user
  rows flipped to done with this commit, the dev/language row now names D13.
- [x] **D13** `docs/en/dev/language/02-manual_dependencies.md` (+zh) still
  teaches parser-era surfaces as current (discovered during D12's Appendix
  B update): the `pl.at(level=..., deps=..., predicate=...)` scope form does
  not exist under tracing (`TypeError: at.__init__() got an unexpected
  keyword argument 'predicate'` — `predicate=` lives only on `pl.submit` /
  `pl.spmd_submit`, `python/pypto/language/constructs/submit.py`), and the
  deps/predicate contract checks are described as firing "at parse time"
  ("the lookup happens while parsing the predicate", "rejected at parse
  time" ×4) — D11 already found one sibling claim stale
  (performance/02-runtime-overhead.md: a predicate producer outside `deps=`
  traces and runs with no diagnostic under tracing). Triage which contract
  checks survive as pass-level diagnostics under tracing and rewrite the
  page around what remains loud; the manual-scope + `spmd_submit(predicate=)`
  form and the expressiveness rules (single-comparison `DispatchPredicate`)
  need re-verification too.
  Red: trace the page's own `pl.at(..., predicate=...)` block → TypeError
  (verified during D12).
  Green: the page states tracing behavior — which forms exist, which checks
  remain loud.
  Notes: done, both trees. Red observed exactly as documented
  (`TypeError: at.__init__() got an unexpected keyword argument
  'predicate'`). Every claim on the page was then probe-verified device-free
  (real-file `@pl.program` probes, tracing only): the canonical
  manual-scope + `spmd_submit(predicate=)` MoE form, the `pl.spmd` scope
  form with `deps=`/`predicate=`/`as tid`, the mirrored order `0 < rc[0,0]`,
  and `pl.at(deps=)` without `as tid` all trace — and the MoE form
  codegen-verifies green through the Default pipeline
  (`CoreTaskPredicate` + `set_predicate`, the ut oracle
  `tests/ut/codegen/test_predicate_codegen.py`). Triage results, now on the
  page: (1) `pl.at(predicate=)` is gone entirely — recorded as gap K8 (the
  fix is an `AtContext` member in existing frontend code); the page's
  `pl.at` table row and scope-form subsection now state that and point at
  the `pl.spmd`/`spmd_submit` alternatives. (2) The producer-in-`deps=` spot
  check does **not** survive — a predicate whose producer is outside
  `deps=` traces and runs with no diagnostic at any stage (re-verified on
  the page's canonical form; D11's sibling finding), and neither does the
  scope-producer variant (tensor assigned inside `with pl.spmd(...) as
  tid:` then predicated without that dep); the "does not see through"
  table is gone, replaced by a plain "nothing checks this under tracing"
  statement. (3) The expressiveness rejections survive but move: chained
  comparisons die at **trace time** (bool-coercion), while arithmetic,
  boolean combination, and non-literal RHS trace and are rejected by
  **orchestration codegen** (all three verified end-to-end with the
  codegen error text quoted). (4) One stale claim found beyond the task
  text: "`deps=` is accepted on all three spmd forms" was parser-era
  (ast_parser accepted it everywhere); the tracing `pl.spmd` rejects
  `deps=` without `as tid` on both the plain `with` and the `for` form
  (verified TypeError), while `pl.at` still accepts `deps=` without
  capture — the table row, the scope-forms bullets, and the cluster-nesting
  "parse time" wording all now state this.

  **Correction (2026-09-14, corpus port):** the (4) reading above aligned
  the docs to the tracer instead of the tracer to the parser, and was
  itself wrong — the parser is the round-trip owner, it accepts `deps=` on
  every form (its `_build_spmd_scope_attrs` docstring documents the
  outliner-synthesised uncaptured TaskId), and the pypto-lib corpus
  (``gate.py``'s for-form, ``expert_routed.py``'s plain with-form; 13
  files failing at decoration) depends on the permissive forms. The
  tracing rejection was removed (scopes.py), the conformance anchors
  converted from "intended divergence" pins to structural-equality
  anchors, and the dev docs (en+zh) restored the permissive rule with the
  `pl.at`-style outliner explanation. The migration guide's "existing
  dependency lists keep working" promise is now literally true. Also: Examples 1–3 were
  standalone `@pl.function` orchestrators with a `self` param (loud
  "parameter 'self' needs ... annotation" under tracing) and now use the
  `@pl.program` class spelling (all three verified to trace); the
  fire-and-forget prose now quotes the actual trace-time error; the
  predicate blockquote now says "evaluated at trace time into a staged
  expression, but never executed"; the fence example's ambiguous `1.0`
  positional (a host float into a possibly-runtime scalar param) was
  dropped; remaining `parser` mentions are deliberate parser-era framing.
  Green: every migrated snippet on the page traces device-free verbatim;
  the D-group invocation stays 40/40 with `--check-parity` OK (this dev
  page has no doctest marks); doc hooks green (en-zh parity, md-refs, nav,
  english-only); symbol coverage and check_headers at their pre-existing
  baselines. The D group is now fully closed — Appendix B's `docs/en/dev/
  language/` row flipped to done in the same commit.

### G — guide

- [x] **G1** Correct the guide's Appendix B examples/st rows (they
  claimed complete; the E/S inventories above are the truth) and point
  Appendix B at this file as the live tracker.

## Discovery protocol

Re-run this before appending or closing tasks; a clean grep with all
exclusions applied is the completion signal for the whole file.

```bash
rg -n "init_values|pl\.while_|pl\.cond\(|pl\.yield_\(" examples/ tests/ docs/
```

Exclusions — files the grep flags that are **not** stale (verdicts
pre-established; re-confirm only if the hit moves):

| Path | Verdict |
| ---- | ------- |
| `examples/utils/parse_from_text.py` | parser-era by design (guide §3) |
| `tests/ut/**/parser/`, `tests/ut/language/conformance/` | parser-owned by contract (`AGENTS.md`) |
| `tests/ut/tools/test_memory_map.py` | round-trip text inside a string literal |
| `tests/ut/codegen/test_orchestration_misc.py` | prose about IR `init_values` (iter_args), not DSL |
| `tests/ut/codegen/test_golden_writer.py` | golden-writer API naming, host-side concept |
| `tests/ut/language/test_intrin_while_assembly.py` | parser round-trip oracle |
| `tests/ut/language/test_pipeline_kwarg.py` | round-trip `pl.parse_program` test |
| `docs/en/dev/ir/01-hierarchy.md` | round-trip grammar spec; per-hit check |
| `docs/en/dev/ir/07-parser.md` | the parser's own documentation |
| pass-doc hits quoting printer output | round-trip dialect is correct (guide §9) |
| `docs/design/` | the design series and the migration guide's deliberate before/after pairs |

**Sweep status (2026-09-12):** all groups closed (S7 was unblocked by
gap-fix F2, which gave runtime MemRef slot subscripts a tracing
spelling). The
discovery grep is clean across the sweep scope; the fixable gaps are
closing through `docs/gap-fixes-todo.md` (K5 closed by F2), and the
remaining entries in `docs/migration-known-gaps.md` carry their own
evidence and blocker. New tasks may still be appended by future
discovery passes — the protocol above is the completion signal.

Beyond the grep: any new `@pl.function_parser`/`@pl.program_parser` use
outside the parser suites, any `pl.jit_specializer` use outside
`python/pypto/jit/`'s own tests, and any example or doc that imports
`pl.static_assert`/`pl.static_print` expecting a check to fire.

## Verification by task type

Targeted checks only — the full gate is the acceptance check for
`tests/ut/` work, and no task here changes `tests/ut/` (E7 may extend
it; then the new test is its own red/green check). To find a changed
file's consumers outside its group: `rg -l "<module name>" tests/`, and
run those suites directly.

| Type | Red/green check | Extra |
| ---- | --------------- | ----- |
| Example (E) | run the file / invoke its builder, exit 0 | if a `tests/ut/` suite imports the example, run that suite directly (e.g. `uv run python -m pytest tests/ut/codegen/test_phase_fence_dep_compression.py -n auto`) |
| tests/st (S) | invoke the module's builder — traces device-free | device run on demand via `task-submit` |
| Docs (D) | the `run_doc_examples.py` invocation above | pre-commit doc hooks; `check_md_refs` |

## Commit protocol

- The work and this file's state edit land in **one commit**; stage the
  task's files plus `docs/migration-todo.md` explicitly (never `git add .`).
- Message convention: `examples:`, `test(st):`, `docs(user):`, … with the
  task id in the body (e.g. `migration-todo E5`).
- When a group's last task closes, flip the guide's Appendix B row to
  done in the same commit.
- Run the applicable verification from the table before committing;
  `git status` clean of unrelated paths before `git commit`.
