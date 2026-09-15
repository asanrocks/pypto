# ci-repro Triage Tracker

Execution-ordered tasks derived from the Open items in
`scripts/ci-repro-worklog.md`. Order: cheap deterministic fixes first
(T3, T2, T5, T6), then the codegen-test port (T1), then device-tier
characterization (T4, T7, T8).

## Worker rules (binding)

- One task per session. Read your entry, then **confirm the documented red
  check actually fails before changing anything** — if it does not fail,
  change nothing and report (the entry may be stale or mischaracterized).
- Stay inside the allowed paths. Update this file (state flip, completion
  notes, newly discovered tasks appended with the next free id) **in the
  same commit as the work**.
- Exactly one commit per task, staging explicit paths only. Never stage
  `scripts/ci-repro-worklog.md` (its unstaged modification is not yours),
  the `runtime` submodule, or anything under `3rdparty/`.
- Device evidence only via bounded single-line
  `task-submit --device auto --max-time N` (add `--device-num 2` where the
  entry says so). Never poll locks or sleep-loop.
- Do not launch sub-agents. English only in code, comments, and docs.
- Size to the targeted check, not the full gate; if you cannot finish,
  leave the tracker entry `in-progress` with precise notes so a resumed
  session can pick up.

State legend: `open` | `in-progress` | `done` | `blocked` (needs an owner).

## T3 — pre-commit debt (mechanical) — state: done

- Mode: mutating. Model: openai/gpt-5.6-luna.
- Allowed paths: the files named in the red checks below;
  `tests/lint/check_docs_symbol_coverage.py` and `docs/en/user/**` (for the
  symbol-coverage item); any repo file modified by the end-of-file /
  clang-format / markdownlint auto-fix hooks (never `3rdparty/`, never
  `runtime/`, never submodule content).
- Red checks (all re-verified by the orchestrator on r10-dev-2):
  1. `uv run python tests/lint/check_headers.py` — 9 files fail:
     `scripts/review/check_api_docs.py`, `check_comment_churn.py`,
     `check_construct_parity.py`, `check_deprecated_runtime.py`,
     `check_rebind_prevalence.py`,
     `tests/ut/jit/test_torch_dtype_bridge.py`,
     `tests/ut/language/conformance/test_generic_scopes.py`,
     `test_intrin.py`, `test_statement_handlers.py`.
     Fix to the canonical template the checker prints.
  2. `uv run python tests/lint/check_no_broad_raises.py` — 8 sites:
     `tests/ut/language/conformance/test_binding_assignment.py:268,465,481,483`,
     `test_for.py:491`, `test_statement_handlers.py:454,617,619`.
     Round-1 pattern: narrow to the actual exception type (e.g.
     `pypto.Error`, `SyntaxError`); allowlist via `<path>::<qualname>` only
     if the test genuinely asserts the exception *hierarchy*.
  3. `uv run python tests/lint/check_op_name_literals.py` — 7 sites:
     `tests/ut/language/conformance/test_inline.py:405,410,436,441`,
     `test_spmd_split_aiv.py:126,295,791`. Route through
     `ir.get_op("...").name` module constants (Round-1 pattern).
  4. `uv run python tests/lint/check_docs_symbol_coverage.py` —
     `pl.static_dim` and `pl.static_shape` appear in no code span under
     `docs/en/user/`. Note: Round 1 already *documented* them in
     `00-types.md` / `05-directives.md`, yet the checker still does not
     detect them — suspect either the docs mention them outside a
     recognized code span, or a span-detection gap in the checker.
     Investigate which; fix the right side.
  5. Hook convergence: the end-of-file, clang-format, and markdownlint
     hooks auto-modified files during the last ci-repro run and those
     changes were restored. Re-apply them and **commit** the result this
     time (the markdownlint hook already excludes `3rdparty/` — verify it
     stays out).
- Green check: `./scripts/ci-repro --skip clang-tidy,unit-tests,codegen-tests,examples-tests,docs-examples`
  — every hook green **except the pyright hook** (its ~445-error debt over
  `tests/` is a separate campaign, explicitly out of scope here). The four
  checkers above also green when run directly.
- Notes: do not "fix" pyright errors; do not touch `python/pypto/` typing.
- Completion: red reproduced before edits (9 headers, 8 broad raises, 7 bare op-name literals, and 2 missing docs code-span symbols). Canonical headers, concrete exception classes, registry-backed op-name constants, and explicit `pl.static_dim`/`pl.static_shape` docs spans now pass all four direct checkers. Hook auto-fixes from the repro were retained, and the three affected review documents now satisfy `tests/lint/.markdownlint.yaml`: MD056 is 0 and MD060 is 0. The prescribed repro passes every pre-commit hook except the separate pyright debt (503 errors), which remains out of scope.

## T2 — port 3 doc examples to the tracing-era spelling — state: done

- Mode: mutating. Model: openai/gpt-5.6-luna.
- Allowed paths: `docs/en/user/language/00-types.md`,
  `docs/en/user/language/06-syntax.md`,
  `docs/en/user/precision/00-workflow.md`.
- Red check: `./scripts/ci-repro --skip pre-commit,clang-tidy,unit-tests,codegen-tests,examples-tests`
  (docs-examples stage) fails on exactly three blocks:
  - `00-types.md:233` — uses a JIT wrapper where an `ir.Function` is
    required;
  - `06-syntax.md:35` — calls `.as_python()` on a specializer wrapper;
  - `00-workflow.md:143` — passes a specializer wrapper to `torch_codegen`.
- Fix pattern: the tracing-era function/program spelling — **not** a
  compatibility shim on `_JitSpecializer`. Round 3's jit-lane prescription
  (eager entry = fully annotated params → bare `ir.Function`; lazy lane =
  bare `pl.Tensor` params) is the reference. The doc runner cannot select
  individual blocks, so the whole docs-examples stage is the check.
- Green check: docs-examples stage PASS (40/40 blocks).
- Notes: Reproduced the three documented failures before editing. Changed
  `rows` and `drive` in `00-types.md` to explicit `pl.function` tracing
  decorators with `InCore` and `Orchestration` types, changed `head` in
  `06-syntax.md` to `InCore`, and changed the failing `fused` block in
  `00-workflow.md` to `Orchestration`. Mirrored the identical tracing-era
  code in the three Chinese twin blocks while leaving surrounding Chinese
  prose unchanged. The prescribed command now passes parity and all 40/40
  runnable docs examples.

## T5 — `tpop_from_aic()` required `shape` arg (16 st failures) — state: done

- Mode: triage, then fix if the verdict is mechanical. Model:
  openai/gpt-5.6-luna.
- Allowed paths: `tests/st/runtime/cross_core/**` (and the intrinsic's
  Python surface if the verdict says the call sites were right).
- Red check: the ~16 `tests/st/runtime/cross_core/` failures —
  `tpop_from_aic()` gained a required `shape` argument the tests don't
  pass. Confirm via a bounded device run (single-line
  `task-submit --device auto --max-time 600`) on a focused subset first.
- Triage question: did the API legitimately gain the argument (→ update
  the 16 call sites), or did the argument become required by accident
  (→ fix the signature/default)? Check the intrinsic's declaration and
  recent history before deciding.
- Green check: the affected cross_core tests pass on device.
- Completion: Confirmed the red check with `task-submit --device auto --timeout 900 --max-time 600 --run "cd /data/wangmingzhe/workspace/subprojects/pypto-2 && .venv/bin/python -m pytest tests/st/runtime/cross_core/test_cross_core_dynamic_valid_shape.py::TestCrossCoreDynamicValidShape::test_c2v_tpop_dynamic_valid_shape -v --platform=a2a3"`: tracing failed with `InvalidOperationError: tpop_from_aic() missing 1 required positional argument: 'shape'`. The intrinsic declaration and commit `5a9c6d280` show that `shape` and `dtype` are intentional required value-producer inputs, so the seven stale call sites were updated to pass the physical `[ROWS, COLS]` shape and `pl.FP32`. The three stale strict xfails in `test_cross_core_split_parity.py` were removed because the explicit full-box shape makes all six parity regimes pass. Focused device evidence passed 4/4; the affected four-file family passed 18/18. A broader 57-item cross-core run passed 52, with three of those now-removed xfails passing and one unrelated grouped-tpop test failing with `ACL_ERROR_RT_AICPU_EXCEPTION (507018)`; that failure is tracked separately as T9.

## T6 — `.warmup` on eagerly-traced `ir.Function` (stale baseline) — state: done

- Mode: triage, then fix if mechanical. Model: openai/gpt-5.6-luna.
- Allowed paths: `tests/st/runtime/framework_and_models/test_jit.py` (and
  the jit decorator surface only if the verdict points there).
- Red check: the documented count was stale on the current checkout. The
  focused device run collected 10 tests and produced 7 failures, including
  `.warmup` on an eagerly-traced `_JitSpecializer`, specializer-only
  persistent-artifact assertions, a lazy traced facade read through
  `a.shape`, and dynamic eager dispatch creating two cache entries.
- Triage verdict: process-persistent artifact reuse and device-free
  `.warmup` belong to the quarantined `pl.jit_specializer` surface; the
  tracing `pl.jit` surface provides compile/dispatch and
  `compile_for_test`. The remaining tests exercise tracing lazy dispatch
  and eager dynamic-artifact reuse.
- Green check: the 8 retained tracing-jit tests pass on device via bounded
  `task-submit`.
- Completion: Removed the two specializer-only ST tests and their helper,
  changed the lazy add fixture to use `pl.tensor.dim` with a fixed
  `[128, 128]` tile and runtime `valid_shape`, and normalized declared
  dynamic dimensions to `None` in the tracing jit cache key. Before edits,
  `task-submit --device auto --max-time 600 --run "cd
  /data/wangmingzhe/workspace/subprojects/pypto-2 && .venv/bin/python -m
  pytest tests/st/runtime/framework_and_models/test_jit.py --platform=a2a3
  -v"` reported 7 failed, 3 passed; after edits the same bounded command
  reported 8 passed. Device-free focused jit checks passed 33/33 and Ruff
  passed for both changed source files. The removed user-observable behavior
  is recorded as T10 below.

## T1 — port 20+4 codegen tests off `_JitSpecializer` — state: done

- Mode: mutating, designed for session resume. Model: volce/glm-5.3.
- Allowed paths: the failing test files under `tests/ut/codegen/` (and
  `tests/st/codegen/dsl/` if the same pattern appears); no changes to
  `python/pypto/jit/` or C++.
- Red check: `./scripts/ci-repro --skip pre-commit,clang-tidy,unit-tests,examples-tests,docs-examples`
  (codegen-tests stage) — collect the failing nodeids; expect the ~24
  known ones (20 fail + 4 error) failing with `AttributeError` on `.name`
  or `.span` before the pass under test runs.
- Fix pattern (Round 2 precedent): eager tracing tests use the resulting
  `ir.Function`/`ir.Program` directly; lazy JIT tests use the
  compile/dispatch helper (`_compile(fn, cfg)` route). Keep assertions
  focused on the pass behavior after the fixture surface is corrected.
- Sizing: this is the largest task; if it overflows, leave `in-progress`
  notes listing which files are done and which remain, then a resumed
  session continues. One commit per task is the rule — if you must split,
  finish the current file family, commit, and record the remainder.
- Green check: codegen-tests stage PASS (62 passed at Round-2 baseline;
  count may grow with the ported tests).
- Completion: red reproduced before edits (exactly the documented 20 failed
  and 4 errored, all with `AttributeError` on `.name`/`.span` in
  `tests/st/codegen/dsl/`). Root cause refined: the six files had already
  been migrated to `@pl.jit` in `481ec526e`, but `d955eb1aa` ("serve the
  eager lane's host surface from the decoration record") changed the eager
  decoration to return the jit object instead of a bare `ir.Function`, so
  every site passing the decorated object to `_lowered`/`_compile`/
  `ir.Program` broke. Port: each call site now passes the decoration trace
  (`jit_obj.fn`, the annotation-determined `ir.Function`), keeping the
  established `_lowered`/`_compile(fn, cfg)` helper routes; the helpers'
  stale "the decoration returns a bare `ir.Function`" docstrings were
  corrected. No `python/pypto/jit/` or C++ changes. After edits: the 24
  nodeids pass focused (7 + 17), and the prescribed repro passes the
  codegen-tests stage 62 passed / 30 deselected (the 24 were already inside
  the 62 collected, so the count did not grow). Ruff check and format pass
  on the six changed files.

## T4 — ptoas `pto.tmul` element-type mismatch — state: done (verdict)

- Mode: **read-only** (no edits; findings + evidence only). Model:
  volce/glm-5.3.
- Scope: `tests/st/runtime/ops/test_gather_composite_scenarios.py` conv and
  grid kernels; the `tmul` op typing surface for reading.
- Red check: after Round 4's shape/cast fixes, the kernels trace cleanly
  but fail at ptoas with `pto.tmul` src0/src1 element-type mismatch.
- **Verdict: kernel-side bug, not an op-typing move.** Commit `8e7c5e9dd`
  (Round 4's port) removed the conv kernel's inner `pl.cast(values,
  pl.FP32)` on the false premise that flat-form `pl.gather` already yields
  FP32 — the op's documented contract is "dtype follows input", and an
  FP16 source yields an FP16 tile. Without the widening cast,
  `pl.mul(values, valid)` is FP16×FP32: IR typing promotes the result
  without inserting operand converts, and ptoas rejects the emitted
  `pto.tmul` ("expects src0 and src1 to have the same element type").
  Verified against ptoas 0.45–0.61 — the rule never moved. Restoring the
  original cast compiles the full conv kernel clean through ptoas
  host-side; the grid kernel (FP32 source) was never affected. The
  one-line fix is T11.
- Evidence: full session `.sessions/T4-tmul-triage.session` (host-side
  repro scripts under `/tmp`, ptoas version matrix, minimal mixed-dtype
  IR, restored-cast compile).

## T11 — restore the conv kernel's widening cast — state: in-progress

- Mode: mutating. Origin: discovered by T4. Fixed in-session by the
  orchestrator (one line + comment; below worker-launch overhead).
- Scope: `tests/st/runtime/ops/test_gather_composite_scenarios.py`.
- Red check: the conv st cases fail at ptoas with the `pto.tmul`
  element-type mismatch (see T4).
- Green check: the gather-composite st cases (conv + grid) pass on
  device.

## T12 — mixed-element-type tile-tile elementwise ops fail only at ptoas — state: open

- Mode: owner decision (op owner). Origin: discovered by T4.
- Finding: a mixed-dtype tile-tile `tile.mul` (e.g. FP16×FP32) passes IR
  verification — `DeduceTileOpElementwiseBinaryType`
  (`src/ir/op/tile_ops/elementwise.cpp`) promotes the result without
  inserting operand converts — and only fails in ptoas.
- Options: reject mixed element types at IR typing (loud at trace time,
  like same-dtype casts), or have codegen insert a `tcvt` on the narrower
  operand.

## T7 — l3 tensor collectives on two-card runs (~17 failures) — state: open

- Mode: **read-only** diagnosis. Model: volce/glm-5.3.
- Scope: `tests/st/distributed/collectives/test_l3_tensor_*_intrinsic.py`
  (reading); runtime/collective implementation for reading.
- Red check: the family fails on two-card runs; not yet diagnosed beyond
  the family signature. Use `task-submit --device auto --device-num 2
  --max-time 600` on one representative test.
- Deliverable: failure signature, first divergence point, and a
  hypothesis ranked list — not a fix.

## T8 — paged-attention numeric-compare (2 failures) — state: open

- Mode: **read-only** characterization. Model: openai/gpt-5.6-luna.
- Red check: 2 paged-attention numeric-compare failures in the device
  environment.
- Deliverable: which comparisons fail, tolerance vs real-divergence, and
  whether the numbers match a known family.

## T9 — grouped tpop/tfree reordered artifact AICPU exception — state: open

- Mode: triage. Model: openai/gpt-5.6-luna.
- Allowed paths: `tests/st/runtime/cross_core/test_cross_core_grouped_tpop_tfree.py` (and the owning runtime surface only if triage identifies it).
- Red check: the full `tests/st/runtime/cross_core/` run on `a2a3` fails `TestCrossCoreGroupedTpopTfree::test_v2c_updown_consecutive_tpop_then_tfree_reordered_artifact[a2a3]` during device execution with `ACL_ERROR_RT_AICPU_EXCEPTION (507018)` after the reordered PTO artifact is rebuilt. The other selected tests passed.
- Triage question: determine whether the failure is a grouped-artifact/runtime regression or a device/infrastructure failure; do not bundle it with the T5 API migration.
- Green check: the representative grouped-tpop/tfree test passes on device.

## T10 — port persistent READY reuse and device-free warmup to tracing jit — state: open

- Mode: mutating. Owner: tracing jit owner.
- Scope: tracing jit implementation and its focused integration/unit tests;
  the quarantined AST-era implementation is reference-only.
- Debt: restore the user-observable behavior removed from T6's stale ST
  coverage: persistent READY artifact reuse across processes, including
  readonly restoration without rerunning compiler stages, and device-free
  `.warmup` that prepares chip-level binaries without initializing a runtime
  worker or executing the kernel.
- Existing quarantined implementations: persistent artifact resolution is in
  `python/pypto/jit/_persistent.py` and its `JITFunction` integration is in
  `python/pypto/jit/decorator.py`, including `JITFunction.warmup`.
- Blocked: `pl.jit` in
  `python/pypto/language/tracer/decorators/jit.py` currently has no equivalent
  persistent-artifact or `.warmup` surface. Port the behaviors only after the
  tracing jit gains those public surfaces; do not move the tests back to
  `pl.jit_specializer` as a permanent solution.
- Green check: dedicated tracing-jit coverage for both behaviors passes,
  including a cross-process READY reuse check and worker-free warmup.

## Explicitly out of scope (no task)

- **pyright hook debt (~445 errors over `tests/`)** — campaign-sized; 315
  sit in three files around `collector.instrument` not exported to the
  import surface, ~25 are `pl.Tuple[...]` facade annotation mismatches.
  Needs the entrance/typing owners. (A read-only inventory task may be
  added later by the orchestrator.)
- **Environment items** — `git fetch origin main` unreachable (stale
  clang-tidy diff base), venv 3.12.13 vs CI 3.10, `PTO_ISA_ROOT` vs
  `runtime/pto_isa.pin` drift, pre-existing `runtime` submodule untracked
  content. Host repairs handled by the orchestrator, not workers.
