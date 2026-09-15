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
- Completion: red reproduced before edits (9 headers, 8 broad raises, 7 bare op-name literals, and 2 missing docs code-span symbols). Canonical headers, concrete exception classes, registry-backed op-name constants, and explicit `pl.static_dim`/`pl.static_shape` docs spans now pass all four direct checkers. Hook auto-fixes from the repro were retained, including EOF/clang-format/markdownlint changes; the pre-commit run passes the custom checks, EOF, clang-format, cpplint, ruff, and format hooks. The repro still reports markdownlint failure on the existing large review tables and the separate pyright debt (503 errors); neither was expanded into this task.

## T2 — port 3 doc examples to the tracing-era spelling — state: open

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

## T5 — `tpop_from_aic()` required `shape` arg (16 st failures) — state: open

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

## T6 — `.warmup` on eagerly-traced `ir.Function` (9 st failures) — state: open

- Mode: triage, then fix if mechanical. Model: openai/gpt-5.6-luna.
- Allowed paths: `tests/st/runtime/framework_and_models/test_jit.py` (and
  the jit decorator surface only if the verdict points there).
- Red check: 9 failures in that file — `.warmup` and friends called on
  eagerly-traced `ir.Function` objects; the lazy-lane surface moved.
- Triage question: which lane do these tests mean to exercise? If they
  test the lazy jit surface, port them to the lazy spelling (bare
  `pl.Tensor` params); if they test eager artifacts, drop the moved
  methods. Confirm intent from the assertions before editing.
- Green check: the 9 tests pass on device (bounded task-submit).

## T1 — port 20+4 codegen tests off `_JitSpecializer` — state: open

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

## T4 — ptoas `pto.tmul` element-type mismatch — state: open

- Mode: **read-only** (no edits; findings + evidence only). Model:
  volce/glm-5.3.
- Scope: `tests/st/runtime/ops/test_gather_composite_scenarios.py` conv and
  grid kernels; the `tmul` op typing surface for reading.
- Red check: after Round 4's shape/cast fixes, the kernels trace cleanly
  but fail at ptoas with `pto.tmul` src0/src1 element-type mismatch.
- Deliverable: a verdict — kernel-side bug (illegal dtype mix reaches
  `tmul`) vs op-typing move (the op's element-type rule changed) — with
  the minimal reproducing IR and the ptoas error text. Bounded device run
  only as needed for the repro.
- This needs the op owner's attention; the verdict unblocks them.

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
