# Changeset review — tracing-migration line (r10-base)

Review of the work over `origin/main`. The changeset is the frontend tracing
migration recorded in `AGENTS.md`: the parser-first collection frontend
replaced by a tracing frontend, with the `pl.jit_specializer` AST runtime
quarantined.

This file is the single communication channel between the orchestrator and the
worker agents, and the durable record of the review. Every worker reads it to
cold-start and writes it back in the same commit as its work.

## 0. Fact base

| | |
| --- | --- |
| Base ref | `origin/main` = `8ecc5d19f0ac1062ae6779ba8aca8cee75c22667` |
| Head at seed | `3020f8650` |
| Size | 54 commits, 530 files, +56929 / −10793 |
| Oracles | `scripts/review/check_{deprecated_runtime,comment_churn,api_docs}.py` |

Pre-existing dirty state, **not part of any review task and never to be
staged**: `M runtime` (submodule pointer drift — see
`docs/hard-won-lessons.md`), untracked `docs/plan-landing.md`, untracked
`pi-sessions/` (agent session scratch).

## 1. Contract (every task, every worker)

1. **One task per session. The session ends with a commit.** The work and the
   tracker update go in *one* commit so the record and the change cannot drift.
2. **Git rules, absolute.** Stage explicit paths only (`git add <path> ...`);
   never `git add -A`, `git add .`, `git reset --hard`, `git checkout .`,
   `git clean -fd`, `git stash`, `git rebase`, `git commit --amend`, or
   `git commit --no-verify`. Never touch the `runtime` submodule,
   `docs/plan-landing.md`, or `pi-sessions/`.
3. **Evidence rule.** A fix is not done until the task's oracle command shows
   the transition. Quote the before/after numbers in the commit message. Prefer
   a test that is red before the fix and green after; where that is impossible,
   the oracle's exit status is the test.
4. **Red-then-green is also a re-triage trigger.** If the documented red check
   does not fail, **change nothing** — correct the record in this file and say
   so. Do not fabricate work to justify the task.
5. **Verdict vocabulary.** `unnecessary` (fix it) / `necessary` (leave the code,
   record why) / `needs-decision` (record only, change nothing). Only
   `unnecessary` items are edited. Anything ambiguous is `needs-decision`.
6. **Scope discipline.** Fix the finding, not the neighbourhood. No drive-by
   reformatting, renaming, or doc rewrites. If a fix requires work beyond the
   task, append it as a discovered task (§4) with the next free id.
7. **Output contract.** The final message *is* the report: terse, no
   deliberation, no scratchpad narration. End with `CHECKS-RUN:` listing only
   commands actually run.
8. **Do not run the full `./scripts/gate`** — it can exceed the session budget.
   Run focused checks for touched files; if a gate re-run is unavoidable, use
   `PYPTO_GATE_LOG=/tmp/gate-<task>.log ./scripts/gate -- <paths>`.

## 2. Scope

- In scope: the code diff `git diff origin/main..HEAD`.
- **Out of scope:** docs prose (`docs/**/*.md`). The migration rewrote it
  deliberately (commits `87d7f25c7`, `5061fe4dc`); reviewing its wording is a
  different task. Code comments in `.py/.cpp/.h/.rs/.toml/.yaml` are in scope
  for R2.
- The oracles are **report-only**: not wired into `./scripts/gate`. They exist
  to give each task a reproducible red/green signal.

## 3. Tasks

### R1 — deprecated runtime usage

**Axis.** Unnecessary uses of the deprecated parser-era runtime:
`program_ast` (removed by the migration) and `pl.jit_specializer` (quarantined).

**Oracle.** `python3 scripts/review/check_deprecated_runtime.py` — exit 1 is red.

**Red state at seed.** `program_ast`: **0 hits**, upstream and in HEAD — there
is no work on that name; record it and move on. `pl.jit_specializer`: 119
usages, 85 allowlisted, **34 candidates** in 8 test files.

**Allowlist rationale.** Production references are inherent, not leftovers: the
runtime itself (`python/pypto/jit/`), its public alias export
(`python/pypto/language/__init__.py:48`), and error text explaining the
quarantine (`python/pypto/language/typing/tensor.py`). `tests/ut/jit/**` is
specializer-mechanics coverage, which `AGENTS.md` assigns to this runtime.

**The finding class.** Test fixtures outside `tests/ut/jit/**` that use
`@pl.jit_specializer` as a convenience kernel builder — pass/codegen/runtime
suites, not specializer-mechanics suites:

| File | Sites |
| --- | --- |
| `tests/ut/ir/transforms/test_block_nz_tensor_views.py` | 9 |
| `tests/ut/ir/transforms/test_narrow_loop_carry_valid_shape.py` | 7 |
| `tests/ut/ir/transforms/test_auto_tile_matmul_acc_mn.py` | 6 |
| `tests/ut/codegen/test_orchestration_returned_param_map.py` | 5 |
| `tests/ut/runtime/test_prebuilt_artifact.py` | 3 |
| `tests/ut/ir/transforms/test_lower_auto_vector_split.py` | 2 |
| `tests/ut/ir/transforms/test_block_mx_scale_tensor_views.py` | 1 |
| `tests/ut/codegen/test_matmul_init_cond.py` | 1 |

**Method.** Per file, decide whether the kernel is expressible in the tracing
(ergonomics) dialect. Rewrite **one** site first and run that file's tests.
If green, migrate the file's remaining sites. If it fails for a documented
tracing gap, the verdict is `necessary`: record the gap (id from
`tests/ut/_migration_xfail.txt` if listed) and leave the code untouched —
do not add comments to explain it.

**Deliverable.** Per-file verdict table (below), minimal rewrites for
`unnecessary` files, oracle before/after in the commit message. Allowlist
growth is itself a finding and must be justified here — the orchestrator
reviews every addition.

**Outcome (round 1).** 34 candidates → 0 (oracle PASS). 7 of the 8 files were
`unnecessary` and migrated; 1 (`tests/ut/runtime/test_prebuilt_artifact.py`,
3 sites) is `necessary` and its path was added to the oracle's
`TEST_ALLOWLIST` with the justification recorded in §7. `program_ast` was 0
hits upstream and in HEAD — no work existed on that name. Verdict table in §7.

**Failure mode to avoid.** These are pass/codegen suites: a rewrite that
merely moves the fixture onto the tracing lane but silently drops the
construct under test is worse than leaving it. Pin what the test asserts
before and after.

### R2 — unnecessary changes to upstream comments

**Axis.** Comment-only edits to code files that the dialect migration did not
require.

**Oracle.** `python3 scripts/review/check_comment_churn.py` — exit 1 is red.

**Red state at seed.** **167 comment-only lines in 38 code files**. Largest:
`python/pypto/language/parser/ast_parser.py` (18),
`tests/ut/jit/test_qwen3_decode.py` (16),
`src/codegen/orchestration/orchestration_codegen.cpp` (13),
`tests/ut/jit/test_decorator.py` (8), `python/pypto/ir/compile.py` (7).

**Method.** Per file, inspect the comment-only hunks
(`git diff origin/main..HEAD -- <file>`; upstream text via
`git show origin/main:<file>`). Two outcomes only:

- *Unrelated to the dialect change* → restore the upstream text exactly
  (`unnecessary`).
- *Genuinely required* (e.g. the comment described the parser API that no
  longer exists) → add the file's path prefix to
  `scripts/review/comment_churn_allowlist.txt` **and** a row to the R2
  decisions table with the reason (`necessary`).

Growing the allowlist without a recorded reason is a contract violation. The
end state is: flagged set empty, allowlist non-empty only where argued.

**Guard.** A restored comment must still describe the code beneath it. If the
code changed, the comment may be `necessary` after all — check the hunk, not
just the wording.

### R3 — non-minimal spec changes

**Axis.** `spec(...)` commits whose diff does more than the dialect migration
requires.

**Worklist.** `git log --format='%h %s' origin/main..HEAD | rg '^[0-9a-f]+ spec\('` —
**13 commits**. Use the subject-only form: `--grep='^spec'` also matches commit
bodies and reports 14, pulling in `b8f6f2327 chore(build): ...`.

```text
d3bd63e67 spec(st)        b9cb6f04b spec(examples) 57c9018da spec(fixtures)
c71009928 spec(placement) 29c56ec43 spec(spmd)     f66b3bd09 spec(const)
1e2c80e77 spec(transmute) 236dbad47 spec(carry)   bdefaec5b spec(tpop)
d54679982 spec(jit)       edef64804 spec(frontend) a8c4cc223 spec(annotations)
02d3360cc spec(frontend)
```

**No automatic oracle.** This axis is a judgement, deliberately so. Method:
per commit, read the diff and ask whether each hunk is *required* by the
dialect swap or is an incidental improvement bundled into a migration commit
(a renamed fixture, a tightened assertion, a reordered test, a new assertion
that the migration did not need). Record the minimal-unit question, not a
style opinion.

**Deliverable.** Verdict table, committed. **This task makes no code changes**;
accepted findings become separate `F*` tasks. `needs-decision` is the correct
verdict whenever intent is arguable — the orchestrator escalates those to the
human rather than guessing.

**Cross-check.** Dual-verdict by design: this axis is reviewed independently by
both models, and disagreement is escalated, never averaged.

### R4 — new API surfaces not properly documented

**Axis.** Public names added by the changeset that a user cannot find in the
documentation.

**Oracle.** `python3 scripts/review/check_api_docs.py` — exit 1 is red.

**Red state at seed.** 8 public names added to
`python/pypto/language/__init__.py::__all__`: `expand`, `function_parser`,
`intrin`, `jit_specializer`, `program_parser`, `runtime`, `static_dim`,
`static_shape`. Two are undocumented:

- `static_dim`, `static_shape` — **0 doc references.** Added by `3e145880f`
  and `e0e760996` as host-side shape queries; mentioned only in
  `docs/migration-todo.md`, a planning file, which the oracle excludes.

**Second lead — an API removal.** `cond` and `while_` were **removed** from
`__all__` (`249 → 255` with 8 added means 2 removed) while 22 and 13 doc files
still reference them. Determine: intentional removal (then the docs are stale)
or accidental (then `__all__` regressed and `from pypto.language import *`
silently loses them). An accidental removal is a finding above this task's
weight — record it as `needs-decision` if intent is not provable from the
commits.

**Deliverable.** Minimal doc additions for genuinely new public API,
matching the existing `docs/en/user/language/` structure (and the `docs/zh/`
mirror only if that mirror is maintained per-name, not wholesale). Verdicts
for the remaining six names (documented — where, and is the anchor real?).
`needs-decision` rows for the removal.

## 4. Discovered tasks

Append here with the next free id (`F*` for fixes, `R*` for new review axes),
including: what, why it was found, the red check, and the owning files. Twelve
of the migration campaign's 37 tasks were found mid-flight; discovery is
expected, not exceptional.

- **F1 — `pl.spmd` body region boundary differs between the parser lane and
tracing.** Found in R1 (test_orchestration_returned_param_map.py). For
`with pl.spmd(N, name_hint="x"):` whose body reads the per-block index, the
parser synthesises ONE `InCoreScopeStmt` (CORE_GROUP) carrier around the whole
body (`ast_parser._emit_spmd_body`, "inline body (reads the per-block index) →
wrap in `InCoreScopeStmt`"). Tracing's `auto_scope` instead wraps each
`pl.split_aiv` region separately and leaves the statements between them (a
`pl.system.syncall` + a bare `pl.matmul`) outside any CORE_GROUP region; after
`ConvertTensorToTileOps` that matmul lands in Vec and `AivSplitValid` rejects
the fixture's `pl.aiv_shard` ("operand is in Vec … requires Acc"). Repro: the
fixture in `tests/ut/codegen/test_orchestration_returned_param_map.py`, whose
migrated text states the carrier explicitly
(`pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_fused")` nested in the spmd)
and then reproduces the parser lane's orchestration byte-for-byte. Decide:
intended (the tracing lane requires an explicit carrier — then document it on
`pl.spmd`) or a frontend-equivalence divergence (then `auto_scope` should wrap
the whole index-reading body). Owning files:
`python/pypto/language/tracer/decorators/function.py` (auto-scope),
`python/pypto/language/constructs/scopes.py`, `language/parser/ast_parser.py`.
Red check: trace the fixture body without the explicit carrier and run
`tests/ut/codegen/test_orchestration_returned_param_map.py`.

## 5. Round ledger

| Round | Task | Model | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | seed tracker + oracles | — (orchestrator) | done | `1acbadcc8` | oracles frozen: R1 red 34, R2 red 167, R4 red 2 |
| 0.1 | churn-oracle scope fix | — (orchestrator) | done | *(this commit)* | R2 red restored to the seed 167/38; the tooling had been auditing itself (`scripts/review/comment_churn_allowlist.txt` reported as churn) |
| 1 | R1 | `opencode-go/deepseek-v4.1-flash:high` | done | `be0618dcf` | 34 → 0 candidates (oracle PASS); 7 files migrated, 1 allowlisted as `necessary`; F1 found. **Orchestrator-verified**: 240 touched tests pass, D10 justification confirmed verbatim, oracle edit narrows scope without weakening it |
| 1 | R2 | `opencode-go/deepseek-v4.1-flash:high` | pending | | |
| 1 | R3 | `openai/gpt-5.6-luna:max` | pending | | |
| 1 | R4 | `opencode-go/deepseek-v4.1-flash:high` | pending | | |

**Convergence.** A round is clean when it introduces no new findings and
commits only ledger updates, with every oracle green and `./scripts/gate`
green. **Two consecutive clean rounds** end the loop — one clean round can be a
reviewer omission (observed: a model dropped a file from a 21-item enumeration
on a one-line `rg`). Cap: 3 rounds, then escalate unresolved items to the human.

**Cross-review.** Round *N*'s fix diff is reviewed in round *N+1* by the
*other* model, against the commit range recorded here.

## 6. Worker invocation

```bash
pi --provider <provider> --model '<model>:<thinking>' -p \
   --session pi-sessions/r10-review/<TASK>.session --name <task> --no-skills \
   --system-prompt "<role + §1 contract restated + git rules>" \
   "Work task <ID> from docs/review-r10.md."
```

`--session` is the resume handle: on timeout, inspect the tail of that file and
resume the *same* path with a corrective, time-boxed prompt. Do not pass
`--no-extensions`. Workers never wait on locks — prefer device-free evidence.

## 7. Decisions (append per task; orchestrator-reviewed)

### R1 verdicts

Oracle: `python3 scripts/review/check_deprecated_runtime.py`.
Before: `119 usages (85 allowlisted, 34 candidates)`, FAIL, exit 1.
After: `88 usages (88 allowlisted, 0 candidates)`, PASS, exit 0.
`program_ast`: 0 hits before and after (nothing to do).

| File | Sites | Verdict | Change |
| --- | --- | --- | --- |
| `tests/ut/ir/transforms/test_block_mx_scale_tensor_views.py` | 1 | unnecessary | frozen `@pl.jit_specializer` kernel → `@pl.program` class + `_run`; the `_bind_args_from_signature` + `_compile_to_program` pair (which built the IR through `pl.parse`) is gone. 10/10 tests, same as HEAD |
| `tests/ut/codegen/test_matmul_init_cond.py` | 1 | unnecessary | `@pl.jit` + `kernel.compile(..., config=RunConfig(codegen_only=True, save_kernels=True, ...))`. The two trip-count worlds are now the annotation: a host `int` (baked) vs bare `pl.Scalar` + `pl.runtime(4, pl.INDEX)` (a runtime scalar, and bare is what keeps the kernel on the lazy lane). Added the world pin `("to %arg3 step" in mlir) == runtime_bound`, because the annotations — not the removed `pl.RUNTIME` sentinel — now select the world; without it the parametrization is unverifiable. 14/14 tests |
| `tests/ut/ir/transforms/test_lower_auto_vector_split.py` | 2 | unnecessary | `@pl.jit` (eager) + `PassManager.get_strategy(Default).run_passes(ir.Program([fn], fn.name, fn.span))` with the a2a3 backend pinned inline; `lower(config=RunConfig(platform="a2a3"))` and the now-unused `RunConfig` import are gone. Assertions unchanged, 2/2 e2e tests |
| `tests/ut/ir/transforms/test_narrow_loop_carry_valid_shape.py` | 7 | unnecessary | module-level `@pl.function(type=pl.FunctionType.Orchestration)`; `_jit_program` reduces to `ir.Program([kernel], kernel.name, kernel.span)`. `vec_carry_narrowed_by_yield` restated in the dialect's carry form (`pl.intrin.carry`/`leak`/`transmute`, `pl.intrin.yield_`). 13/13 tests |
| `tests/ut/ir/transforms/test_auto_tile_matmul_acc_mn.py` | 6 | unnecessary | same shape as the row above. 25/25 tests |
| `tests/ut/ir/transforms/test_block_nz_tensor_views.py` | 9 | unnecessary | same shape, through a new local `_program(fn)` helper (9 call sites). 25/25 tests |
| `tests/ut/codegen/test_orchestration_returned_param_map.py` | 5 | unnecessary | `@pl.jit.inline` helpers + `@pl.jit` entries; `_orch_code(entry, *callees)` runs the Default pipeline over `ir.Program([entry, *callees], entry.name, entry.span)` with the a2a3 backend pinned. Each spmd body now states its carrier explicitly — `pl.spmd(NUM_CORES, name_hint="fa_fused", sync_start=True)` nested with `pl.at(level=pl.Level.CORE_GROUP, name_hint="fa_fused")` — which restores the parser lane's implicit carrier (see F1). The `torch` import and the torch sample tuple are gone. 3/3 tests; the emitted orchestration C++ is **byte-identical** to the pre-migration lane for all three kernels (compared against the HEAD output captured before the rewrite) |
| `tests/ut/runtime/test_prebuilt_artifact.py` | 3 | necessary | see below |

**`tests/ut/runtime/test_prebuilt_artifact.py` — necessary (3 sites, 229 / 746
/ 808).** The kernels there are `def kernel(): pass` stubs on which
`compile`/`_compile` are monkeypatched; what the tests exercise is
`JITFunction.warmup` (line 229), the automatic artifact publication through
`compile(config=RunConfig(cache_config=CacheConfig(enabled=True, ...)))` (lines
746, 808) and the quarantined `host` entry (for the DISTRIBUTED build kind) — all on
`pl.jit_specializer`'s runtime. Only the
quarantined runtime has `warmup`/`host`/`_compile`; the tracing jit has none of
them and never reads or publishes through the artifact store
(`docs/migration-todo.md` D10, verified: `rg 'def warmup' python/pypto` → only
`python/pypto/jit/decorator.py`). Migrating these sites would mean porting
`warmup`/persistent publication to the tracing jit — a feature, not a test
migration — so the code is left untouched and the gap stands as recorded in D10.

**Allowlist growth (orchestrator review required).** One entry was added to
`TEST_ALLOWLIST` in `scripts/review/check_deprecated_runtime.py`:
`tests/ut/runtime/test_prebuilt_artifact.py`. Rationale: the suite's own header
calls itself specializer-mechanics coverage, and the only specializer surface it
touches is the warmup/publication path the tracing lane does not have. The
`tests/ut/jit/` scope in the original allowlist assumed specializer-mechanics
tests live in that directory; this one does not. No other allowlist entry was
touched (production sites and `tests/ut/jit/**` are unchanged).

### R2 allowlist justifications

None recorded yet — one row per entry added to
`comment_churn_allowlist.txt`.

### R3 non-minimal findings

None recorded yet.

### R4 API documentation verdicts

None recorded yet.
