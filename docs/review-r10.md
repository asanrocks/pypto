# Changeset review — tracing-migration line (r10-base)

Review of the work over `origin/main`. The changeset is the frontend tracing
migration recorded in `AGENTS.md`: the parser-first collection frontend
replaced by a tracing frontend, with the `pl.jit_specializer` AST runtime
quarantined.

This file is the single communication channel between the orchestrator and the
worker agents, and the durable record of the review. Every worker reads it to
cold-start and writes it back in the same commit as its work.

## 0. Fact base

| Field | Value |
| ----- | ----- |
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
| ---- | ----- |
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

**Outcome (round 1).** `167 lines in 38 files` → **`0 / 0` (oracle PASS)**.
32 of the 167 lines were restored as `unnecessary` (10 files); 133 lines in 28
files are `necessary`, every one carrying a row in §7. Two lines left the
report because the oracle had miscounted them: `#include` in
`python/bindings/modules/ir.cpp` and `src/ir/op/tile_ops/cross_core.cpp` is a
preprocessor directive, not a comment (oracle scope fix in §7). One hunk is
`needs-decision` (escalated, not edited). The load-bearing finding is the
`necessary` rule itself: the changeset ships its own comment/vocabulary
authority (`docs/guide-comments.md`), which is why most of the 167 lines are
justified prose work rather than incidental churn — see §7.

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

**Outcome (round 1).** The seed's red state undercounted the axis. The R4 oracle
(`check_api_docs.py`) counts *any* doc file under `docs/` — design docs included — so it saw
2 undocumented names. The repo's own user-manual gate,
`tests/lint/check_docs_symbol_coverage.py` (pre-commit), requires each name in a code span
under `docs/en/user/` and is the authoritative reading of "a user cannot find it": green at
`origin/main` (**241 documentable, 0 missing**), red at HEAD (**4 missing**: `pl.expand`,
`pl.program_parser`, `pl.static_dim`, `pl.static_shape`). All four were documented in the
user manual (en + per-name zh mirror), so both readings are green after: R4 oracle
`FAIL (2 undocumented)` → `PASS (0)`, user-manual gate `4 missing` → `0`.
`cond` / `while_` are a confirmed **intentional** removal — verdicts and the one stale-doc
row are in §7.

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
**State: fixed** (fix-campaign task A2, as the construct-parity review's F23).
The tracing lane's carrier decision is now the parser's static, body-wide fact:
the rewriter's `reads_block_idx` (an `ast.walk` over the whole with-suite,
threaded through `rt.on_begin_with` beside `single_call_body`) makes
`pl.spmd.__trace_enter__` wrap the whole body at its first statement and
reserve the split name hints, with `_maybe_open_spmd_carriers` reduced to the
backstop for a read the AST walk cannot see. The four divergent spellings are
anchors (`test_carrier_*_anchor` in `test_spmd_split_aiv.py`) with a print →
reparse regression; the explicit-carrier spelling's residual outer-`name_hint`
divergence is filed as the construct-parity record's F36.

- **F2 — the changeset introduces 52 files whose license header violates the
repo's own `check_headers` lint.** `python3 tests/lint/check_headers.py .`
fails 52 files at HEAD and **0 at `origin/main`** (measured by running the
checker against each `origin/main` blob), so every failure is new here — the
`docs/hard-won-lessons.md` claim that `check_headers` failures are "pre-existing
license-wording drift" is stale and was not reproducible. Three variants:
`You may not use` → `You should not use` (20 files), `MERCHANTABILITY, OR` →
`MERCHANTABILITY OR` (32 files), `This program is free software,` → `;` / `:`
(5 / 6 files, several files carrying two variants), all 0 at `origin/main`; plus
pre-header lines (`#!/usr/bin/env python3` in
`scripts/fix_cusparselt_metadata.py`, `scripts/migrate_intrin.py`;
`# ruff: noqa: E501` in `python/pypto/language/protocol.py`,
`python/pypto/language/tracer/semantics.py`) and one file with no header at all
(`python/pypto/language/op/_validation.py`). R2 fixed the three files where the
drift appeared as comment-only churn (`python/pypto/ir/formatter.py`,
`python/pypto/language/parser/diagnostics/__init__.py`,
`python/pypto/language/typing/tuple.py`; check_headers 55 → 52 failing); the
remaining 52 are new files and out of R2's axis. Fix: restore the canonical
header from `tests/lint/check_headers.py` (the exact text is in that file).
Red check: `python3 tests/lint/check_headers.py .` (exit 1, 52 files).
Owner: whichever commit created each file — all 52 are in the changeset.

- **F3 — 14 files of the changeset are not `ruff format` clean, so the
pre-commit `ruff-format` hook fails.** `uv run ruff format --check .` reports
`14 files would be reformatted` at HEAD (0 outside the changeset); scoped to
the changeset's `.py` files the count was 15 before R2 removed the extra blank
line in `python/pypto/language/typing/scalar.py`. Seven are modified files that
were formatted at `origin/main` (`python/pypto/jit/decorator.py`,
`python/pypto/language/typing/tensor.py`,
`tests/ut/ir/transforms/test_auto_tile_matmul_l0.py`,
`test_expand_mixed_kernel_a5.py`, `test_expand_mixed_kernel_split_aiv.py`,
`tests/ut/language/parser/test_decorator.py`, `test_scalar_dispatch.py`); seven
are new files (`constructs/scopes.py`, `tests/ut/jit/test_program_assembly_names.py`,
`test_submit_resolve.py`, `tests/ut/language/conformance/test_jit.py`,
`test_staged_host_join.py`, `test_while_logical.py`,
`tests/ut/language/test_collector_contract.py`). Note `./scripts/gate` does not
run `ruff format`, so this is invisible to the gate. Red check:
`uv run ruff format --check .`.

- **F4 — the R2 oracle cannot see docstring-only churn, and one such edit is
wrong.** `check_comment_churn.py` recognises only `#`/`//` lines, so a hunk
that changes only docstring text is invisible even though R2's axis ("comment
edits the dialect migration did not require") covers it. Found instance:
`tests/ut/jit/test_qwen3_decode.py`'s module docstring now names the specializer
for a cross-file composition (``@pl.jit_specializer``, plus the inline form),
but every decorator in
`examples/models/qwen3_jit/` is `@pl.jit`/`@pl.jit.inline`
(`rg -n '^@pl\.' examples/models/qwen3_jit/*.py`); the blanket rename
`pl.jit` → `pl.jit_specializer` misfired on the one example that moved to the
tracing lane. The rest of the class is unenumerated because the oracle cannot
enumerate it: fixing it means teaching the oracle to accept docstring-only
hunks (a scope decision for the orchestrator, not a side effect of this task).
Red check: `rg -n 'jit_specializer' tests/ut/jit/test_qwen3_decode.py` against
the example's decorators.

- **F5 — the R1 oracle audits the tracker itself, so tracker prose can turn R1
red.** `scripts/review/check_deprecated_runtime.py` scans `docs/review-r10.md`
(`.md` is in its `CODE` set) and counts a non-prose line as a usage whenever the
quarantined runtime's name is written immediately before `.` or `(`. Two rows
added to §R2 verdicts did exactly that and flipped R1 from PASS to "2
non-allowlisted usages"; the rows were reworded. Fix: exclude
`docs/review-r10.md` from that oracle the way the churn oracle excludes its own
`scripts/review/`, or treat markdown table/prose lines as prose. Red check:
`python3 scripts/review/check_deprecated_runtime.py` after any tracker edit
(PASS = `88 allowlisted, 0 candidates`).
- **F6 — `spec(examples)` bundles a pre-existing flash-attention algorithm fix and executable golden.** Found in R3: `b9cb6f04b` changes the online-softmax else branch from a fresh `mi_102` to the carried `mi_update`, removes the skip-only `__main__`, and adds simulator execution plus an fp32 golden. Those are correctness validation/fix work, not required to replace parser-era spellings. Red check: `git diff --unified=0 b9cb6f04b^ b9cb6f04b -- examples/models/03_flash_attention.py`. Owner: `examples/models/03_flash_attention.py`.
- **F7 — `spec(st)` duplicates the return-style dispatch regression.** Found in R3: `d3bd63e67` adds `return_style_add` and its dispatch assertion even though `tests/ut/jit/test_return_style_dispatch.py` already owns the F8 pipeline/codegen/metadata regression; the ST addition belongs with the return-style fix if the hardware check is retained. Red check: `rg -n 'return_style_add|return-style' tests/ut/jit/test_return_style_dispatch.py tests/st/runtime/framework_and_models/test_jit.py`. Owners: `tests/st/runtime/framework_and_models/test_jit.py`, `tests/ut/jit/test_return_style_dispatch.py`.
- **F8 — `spec(carry)` removes the gate budget from subprocess-backed auto-deps tests.** Found in R3: `236dbad47` deletes the two 60-second markers and their rationale from a file whose tests still spawn fresh interpreters, undoing `41cb2da7f`'s documented accommodation for the gate's 10-second timeout. Red check: `git diff --unified=0 41cb2da7f 236dbad47 -- tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`. Owner: `tests/ut/ir/transforms/test_auto_derive_task_dependencies.py`. **State: fixed** (fix-campaign task G1, gate hygiene) — the `spec(carry)` rewrite was incidental to the deletion; both subprocess-backed tests still call `_run_auto_deps_debug_subprocess`, so the two 60-second budgets and their rationale are restored verbatim. Focused gate on the owning file green (`65 passed`).
- **F9 — `spec(transmute)` deletes four flatten-pass span regressions.** Found in R3: `1e2c80e77` removes `test_rebuilt_op_keeps_the_rhs_call_span_not_the_statement_span`, `test_auxiliary_synthesized_statements_keep_the_assignment_span`, `test_delegated_lowering_statements_keep_the_assignment_span`, and `test_fused_batch_matmul_stores_keep_the_consumed_store_span` without a tracing replacement. Red check: `git diff --unified=0 1e2c80e77^ 1e2c80e77 -- tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`. Owner: `tests/ut/ir/transforms/test_flatten_tile_nd_to_2d.py`.
- **F10 — the R2 oracle cannot tell a modified comment from an added one.**
  Found by orchestrator verification in round 1, so it predates F6–F9; it is
  recorded last because the orchestrator's write of it was lost (see note in
  round 2) and the next-free-id rule then applied. A comment-only hunk with no
  `-` line is new prose explaining new code (25 lines / 8 files here:
  `_metadata_program`, `defs_ordered`, new span assertions), not churn on an
  upstream comment; it should be out of scope rather than allowlisted. Red
  check: split the flagged set by presence of a removed line and assert the
  addition-only part is empty. Owning file:
  `scripts/review/check_comment_churn.py`.

## 5. Round ledger

| Round | Task | Model | State | Commit | Result |
| ----- | ---- | ----- | ----- | ------ | ------ |
| 0 | seed tracker + oracles | — (orchestrator) | done | `1acbadcc8` | oracles frozen: R1 red 34, R2 red 167, R4 red 2 |
| 0.1 | churn-oracle scope fix | — (orchestrator) | done | `cce8029da` | R2 red restored to the seed 167/38; the tooling had been auditing itself (`scripts/review/comment_churn_allowlist.txt` reported as churn) |
| 0.2 | F5 fix: R1 oracle off markdown | — (orchestrator) | done | `07132dfb2` | the tracker's own prose can no longer flip R1 red; markdown removed from the oracle's code set. **Note:** that commit's message also claims it recorded the R2 verification block and filed F10 — the edit was lost to a failed multi-edit call, so only the oracle fix and ledger rows landed. Both are recorded now |
| 1 | R1 | `opencode-go/deepseek-v4.1-flash:high` | done | `be0618dcf` | 34 → 0 candidates (oracle PASS); 7 files migrated, 1 allowlisted as `necessary`; F1 found. **Orchestrator-verified**: 240 touched tests pass, D10 justification confirmed verbatim, oracle edit narrows scope without weakening it |
| 1 | R2 | `opencode-go/deepseek-v4.1-flash:high` | done | `27ab5aeff` | 167 → 0 flagged lines (oracle PASS; 133 lines in 28 files allowlisted with reasons, 32 lines restored, 2 reclassified by an oracle scope fix); F2–F4 found. **Orchestrator-verified**: 108 modified-upstream vs 25 pure-addition lines, samples confirm the reasons, F2/F3 confirmed real. **Escalated**: the `vocabulary` group rests on a doc the changeset itself added |
| 1 | R3 | `openai/gpt-5.6-luna:max` | done | `d9c8c471c` | 13 subject-matched `spec(` commits reviewed; four residual non-minimal units recorded as F6–F9. Historical `run-log` churn was removed by `f6e56bac7`. **Orchestrator-verified**: all three spot-checked red checks reproduce (4 span tests deleted by `1e2c80e77`, 2 `timeout(60)` budgets removed by `236dbad47`, the flash-attention term change in `b9cb6f04b`) |
| 1 | R4 | `opencode-go/deepseek-v4.1-flash:high` | done | `a87fc0676` | 4 undocumented names, not the oracle's 2: the worker ran the repo's own `check_docs_symbol_coverage` gate (user-manual code spans), which also caught `pl.expand` and `pl.program_parser`; 247/247 now. **Orchestrator-verified**: gate exit 0, new sections present in en+zh, `cond`/`while_` removal documented in the migration guide, and the manual's only mentions teach the removal. Stale index rows recorded as needs-decision |

**Convergence.** A round is clean when it introduces no new findings and
commits only ledger updates, with every oracle green and `./scripts/gate`
green. **Two consecutive clean rounds** end the loop — one clean round can be a
reviewer omission (observed: a model dropped a file from a 21-item enumeration
on a one-line `rg`). Cap: 3 rounds, then escalate unresolved items to the human.

**Cross-review.** Round *N*'s fix diff is reviewed in round *N+1* by the
*other* model, against the commit range recorded here.

### Round 2 plan (and round-1 lessons)

Round 2 is the cross-review pass: each fix diff (`be0618dcf` R1, `27ab5aeff` R2,
`a87fc0676` R4, plus the orchestrator commits) is reviewed by the *other* model
against `1acbadcc8..HEAD`, with the tracker excluded from its own subject. Open
items it must carry:

- **F10** — refine the churn oracle (additions vs modifications), then re-run
  R2; the addition-only allowlist entries should fall away on their own.
- **The R2 `vocabulary` group** — escalate to the human; do not silently
  accept the circular justification.
- **F1, F2, F3, F6–F9** — fixes, each sized to the gate.

Round-1 lessons worth keeping:

1. **Script the enumeration; the model judges.** An agent dropped a file from a
   21-item `rg -l` list — on the easiest task in the campaign. Every review
   task must be handed a script-produced worklist, never asked to enumerate.
2. **A worker will take the dangerous shortcut unless it is forbidden.** R2's
   axis is one `git checkout origin/main -- <file>` away from reverting a
   file's code along with its comments. The prompt now forbids wholesale
   restores explicitly.
3. **Verify your own writes, not your intent.** An orchestrator multi-edit call
   failed atomically; the commit message still described the change as landed.
   The tree, not the message, is the record: check the diff before committing.
4. **Prefer the repo's own gate to a bespoke oracle.** R4's oracle found 2
   undocumented names; `check_docs_symbol_coverage` found 4. Where a repo lint
   already encodes the question, it is the authoritative reading.

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
| ---- | ----- | ------- | ------ |
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

### R2 verdicts

Oracle: `python3 scripts/review/check_comment_churn.py`.
Before: `167 lines in 38 files`, FAIL, exit 1.
After: `0 lines in 0 files`, PASS, exit 0, with `133 lines in 28 files`
allowlisted.

Decomposition of the 167 → 0: **32 lines restored** (10 files), **2 lines
reclassified by an oracle scope fix** (below), **133 lines allowlisted** (28
files, one row each below).

**The `necessary` rule used here.** The changeset carries its own prose
authority: `docs/guide-comments.md` (added by `90172e387`, retained at HEAD by
its own instruction). Its "Scope of the Commit" explicitly covers
*implementation comments* and *test documentation*; its Preferred Vocabulary
retires named words (`mint, minted` → create/produce/emit/wrap/bind;
`seed` → a binding proxy, with established unrelated uses still valid); and its
Docstring Rules forbid citing document paths in production comments. A
prose-only hunk is `necessary` when it (a) sits on code this changeset rewrote
or renamed, or (b) applies that retained vocabulary without reducing accuracy.
A hunk that does neither — or that makes the comment *less* accurate — is
`unnecessary` and was restored byte-for-byte to `origin/main`.

**Oracle scope fix (2 lines, not churn).** `is_comment` treated any line whose
content starts with `#` as a comment, so the two `#include` additions in
`python/bindings/modules/ir.cpp` and `src/ir/op/tile_ops/cross_core.cpp` were
counted as prose. In C/C++ that character opens a preprocessor directive, so
the `#` marker is now keyed to the hash-comment languages (`.py`, `.pyi`,
`.toml`, `.yaml`, `.yml`, `.txt`, `.cfg`, `.cmake`, `.sh`); `//` still counts
everywhere. This *narrows* the report (C/C++ comment hunks are still caught by
`//`) instead of allowlisting two whole C++ files, which would have hidden
future comment churn in them. No other hunk changes classification.

#### Unnecessary — restored (10 files, 32 lines)

| File | Lines | What was restored | Why |
| ---- | ----- | ----------------- | --- |
| `python/pypto/ir/formatter.py` | 4 | `You should not use` → `You may not use`; `MERCHANTABILITY OR` → `MERCHANTABILITY, OR` | License template drift, unrelated to any dialect or vocabulary decision — and it violates the repo's own header lint (evidence below) |
| `python/pypto/language/parser/diagnostics/__init__.py` | 4 | `free software:` → `free software,`; `MERCHANTABILITY OR` → `MERCHANTABILITY, OR` | same |
| `python/pypto/language/typing/tuple.py` | 2 | `free software;` → `free software,` | same |
| `python/pypto/language/op/__init__.py` | 3 | deleted the added second copy of the "Promoted tile-only ops …" comment | The hunk left the upstream comment in place and added a near-duplicate below it — a duplicated comment, not a required update. The upstream sentence stays true (``abs`` and the bitwise family are still re-exported from ``unified_ops``); only its enumeration became incomplete when `create_tile` moved |
| `python/pypto/language/typing/dynamic.py` | 4 | `populated on first use by unwrap()` → `by TypeResolver`; `all paths` → `all parsers` | `TypeResolver` is still a first-use populator of `_ir_var` (`parser/type_resolver.py:1109,1139,1878`), so the rewrite is narrower than the code; the new by-name sharing is already documented by the class docstring and `unwrap()` docstring added in the same commit |
| `python/pypto/language/typing/scalar.py` | 3 | `Scalar wraps an IR expression` → `symbolic expression`; removed the added blank line before `__class_getitem__` | `docs/glossary.md` (the permanent vocabulary) *reserves* "symbolic" for exactly this domain ("symbolic expressions"), so the swap contradicts the authority it claims; the blank line is a `ruff format` violation (red check below) |
| `python/pypto/language/parser/ast_parser.py` | 6 | `(populated by pl.dump_tag` → `(seeded by pl.dump_tag`; `pl.dump_tag entry is merged` → `pl.dump_tag seed is merged`; dropped the added period after `Resolve dtype from second argument` | "seed" as *initialize/create* is an established unrelated use the guide keeps valid and the tree still uses at HEAD (`tile_ops/transform.cpp:791`, `tensor_ops/broadcast.cpp:421`, `ir/op/tensor_ops.py:60`, `jit/decorator.py`); the dtype line changed only by a period |
| `tests/ut/jit/test_qwen3_decode.py` | 2 | `need torch to build random input tensors` → `to construct tensor metadata samples` | The rewrite is *less* accurate: `_make_args` allocates `torch.empty` (explicitly uninitialized, see the comment below it), nothing random is built |
| `tests/ut/language/parser/test_manual_scope_parsing.py` | 2 | `Create K producer scopes` → `Seed K producer scopes` | same "seed = create/initialize" established use as the `ast_parser.py` row |
| `python/pypto/language/distributed/op/tensor_ops.py` | 2 | `Initialize from the first dim` → `Seed from the first dim` | same; the code beneath is untouched |

Evidence for the three license restores: `python3 tests/lint/check_headers.py .` —
**55 failing files before, 52 after**; the three restored files pass, and all
three passed on their `origin/main` blobs. (The checker is a pre-commit hook,
not part of `./scripts/gate`.) Evidence for `typing/scalar.py`:
`uv run ruff format --check python/pypto/language/typing/scalar.py` —
`Would reformat: …scalar.py` before, `1 file already formatted` after.

#### Necessary — allowlisted (28 files, 133 lines)

| File | Lines | Reason |
| ---- | ----- | ------ |
| `examples/models/03_flash_attention.py` | 6 | Body rewritten to derived loop carries + staged joins (`pl.yield_` → rebinding); the three comments described the retired yield/SSA-phi shape |
| `examples/models/02_vector_dag.py` | 2 | New comment for the new `pl.const(1.0, pl.FP32)` staged-scalar spelling in the same hunk |
| `examples/intermediate/06_dyn_valid_shape.py` | 3 | `lower` → `compile_for_test` (API renamed by the changeset) |
| `examples/models/qwen3_jit/qwen3_decode.py` | 5 | same rename |
| `python/pypto/ir/compile.py` | 7 | New comment for the new `_metadata_program=transformed_program` argument (code changed in the same hunk) |
| `python/pypto/language/parser/ast_parser.py` | 12 | Relaxed-reassignment comment describes the behavior that replaced the raise (fresh let on a type-changing rebind); `literal` matches the widened accepted literal set (`bool` now under dtype validation) |
| `python/pypto/language/parser/decorator.py` | 5 | `_FUNCTION_DECORATOR_NAMES = {"function", "function_parser"}` widened the accepted set in the same commit, so the three per-branch comments naming `pl.function` as the only literal were stale |
| `python/pypto/runtime/runner.py` | 2 | Guide's Docstring Rules: production docstrings and comments must not cite document paths (`docs/en/dev/08-entry-points.md` and `runtime/docs/dfx/pmu-profiling.md` are still present, so this is a policy edit, not a broken link) |
| `src/codegen/orchestration/orchestration_codegen.cpp` | 13 | `HandleArrayRebindAssign` is a new declaration path (absent at `origin/main`): "Three lowering paths" → four and the caller list are the new code's documentation |
| `src/ir/transforms/inline_functions_pass.cpp` | 4 | New comment for the new `defs_ordered` member (deterministic first-encounter order) added in the same hunk |
| `src/ir/transforms/utils/return_lineage_utils.cpp` | 6 | Extended comment for the new `fresh_create_stop` out-param and its consumer `TraceToBackingBuffer` |
| `tests/st/codegen/dsl/test_flatten_dynamic_tile_3d.py` | 4 | "DSL function bodies are parsed as AST, not executed" → traced: the dialect claim itself |
| `tests/st/runtime/control_flow/test_dyn_orch_shape.py` | 5 | same dialect claim |
| `tests/ut/ir/transforms/test_outline_graph_scopes.py` | 2 | `what this test feeds the parser` → the construct: the test now feeds a construct (raises `TypeError`, not `ParserSyntaxError`) |
| `tests/ut/ir/verifier/test_perf_hint_tile_innermost_dim.py` | 5 | Parser-attached spans → tracing spans, with new span assertions in the same hunk |
| `tests/ut/language/parser/test_flash_attention.py` | 2 | The module's decorator is `@pl.function_parser` at HEAD (`test_flash_attention.py:41`); the comment named `@pl.function` |
| `tests/ut/jit/test_decorator.py` | 8 | The quarantined runtime's decorators (the ``pl.jit_specializer`` family) are used throughout this file (42 occurrences); the comments named the tracing `@pl.jit` family |
| `tests/ut/jit/test_spmd.py` | 2 | The file's dep is `@pl.jit.incore` (line 64) after the `from pypto.jit.decorator import jit` → `pl.jit` switch in the same commit |
| `tests/ut/jit/test_qwen3_decode.py` | 14 | `lower` → `compile_for_test` (required, line 38) + the rewritten module-cache comment (rename plus a fixture-scope rationale — see the needs-decision row) |
| `tests/ut/ir/transforms/test_lower_auto_vector_split.py` | 2 | `pl.jit(...).lower()` → `.compile_for_test()` |
| `tests/ut/codegen/test_orchestration_codegen_more.py` | 5 | Test extended from two to three chained calls (new `third` assertions), so the alias/name comments and the "Same for ``second``" addition describe the new assertions |
| `tests/ut/ir/test_distributed_ops.py` | 3 | New comment for the changed test input (`_validate_chunk(True, …)` replaced `_validate_chunk(1.5, …)`) |
| `tests/ut/ir/transforms/test_simplify_pass.py` | 3 | New comment for the rewritten `Expected` fixture (canonical const-left `Le` staged explicitly) |
| `tests/ut/runtime/test_prebuilt_artifact.py` | 3 | New module comment recording why this suite decorates with `pl.jit_specializer` — the same justification R1 allowlisted its path against the deprecated-runtime oracle |
| `tests/st/runtime/ops/test_memory_planner_ptoas.py` | 2 | Cited test relocated by the changeset: `tests/ut/codegen/test_multi_buffer_codegen.py` → `tests/ut/ir/parser/test_multi_buffer_codegen.py` |
| `tests/ut/codegen/test_orchestration_codegen.py` | 4 | Guide-retired vocabulary: "alias is minted" → "is created" ×2, no code change, assertions identical |
| `tests/ut/codegen/test_orchestration_misc.py` | 2 | same "minted" → "created" |
| `tests/ut/ir/transforms/test_convert_to_ssa_pass.py` | 2 | same ("SSA mints a fresh version" → "creates") |

#### Needs-decision (escalate; recorded, not edited)

| File | Lines | Question |
| ---- | ----- | -------- |
| `tests/ut/jit/test_qwen3_decode.py` | 12 | The module-cache comment was replaced wholesale (upstream: "Module-level memoization keeps the non-trivial full-pipeline lowering to one run…") with the `compile_for_test` rename *plus* six new lines of rationale for a fixture-scoped (rather than `scope="class"`) fixture — conftest autouse ordering and the `PYPTO_PROG_BUILD_DIR` redirect. The rename is required; the added rationale is new information the migration did not need, but it documents non-obvious ordering, which `docs/guide-comments.md`'s Comment Rules ask for. Arguable either way, so the file is allowlisted and this row is left for the orchestrator. |

One entry per line was appended to `scripts/review/comment_churn_allowlist.txt`,
grouped by reason; that file carries the one-word tag only, the argument lives
in the tables above.

#### Orchestrator verification (round 1)

Independently checked: oracle green; the 32 restored lines are byte-identical to
`origin/main`; `check_headers` reports 52 failures and **all 52 are inside the
changeset** (F2 is real, not pre-existing drift); `ruff format --check` fails on
14 files (F3 is real); the churn oracle's `#` fix is correct (`#include` is a
preprocessor directive, not prose).

Two caveats, both for round 2:

- **Scope.** The oracle cannot distinguish a *modified* upstream comment from a
  comment-only *addition*. The 133 allowlisted lines split **108 modified (22
  files — the real R2 axis)** / **25 pure additions (8 files)**: new prose
  explaining new code (`_metadata_program=` in `ir/compile.py`, `defs_ordered`
  in `inline_functions_pass.cpp`). Leaving those alone is correct, but they are
  not churn and should not need an allowlist entry. Filed as F10.
- **The `vocabulary` group is escalated, not accepted.** Its five files are
  justified by `docs/guide-comments.md`, a document the changeset itself added
  and retained. That is circular: the changeset rewrote upstream comments *and*
  supplied the prose authority that mandates the rewording. The representative
  case (`tests/ut/codegen/test_orchestration_misc.py`: "alias is minted" →
  "alias is created") is a pure vocabulary edit, on a hunk otherwise unrelated
  to the reworded line. Not reverted — the convention doc is real and retained
  at HEAD — but this is `needs-decision` for the human, not a clean
  `necessary`.

### R3 non-minimal findings

No automatic oracle exists for this axis. The documented subject-only worklist
is the count oracle.

Before: **13 `spec(` commits**, from `git log --format='%h %s'
origin/main..HEAD | rg '^[0-9a-f]+ spec\('`.
After: **13 `spec(` commits**; the review changes only this tracker record.

| Commit / minimal unit | Verdict | Decision |
| --------------------- | ------- | -------- |
| `d3bd63e67` ST dialect rewrites | `necessary` | The carry/const rewrites, closure-callee captures, eager/lazy compile-for-test adapters, explicit tpop producers, and tracing-owned assertions are required for the ST suites to leave the parser-era lane. |
| `d3bd63e67` `return_style_add` and its end-to-end dispatch test | `unnecessary` | This is a regression test for the pre-existing return-style dispatch fix (`f6109f786`), not an ST dialect rewrite. Recorded as F7. |
| `b9cb6f04b` example dialect rewrites other than `03_flash_attention.py`'s algorithm hunk | `necessary` | The examples replace parser-era loop channels and `lower()`, state typed staged values, capture program callees, and adapt tracing execution. |
| `b9cb6f04b` `examples/models/03_flash_attention.py` online-softmax correction and new simulator/golden `__main__` path | `unnecessary` | Replacing the fresh `mi_102` with carried `mi_update` changes pre-existing algorithm behavior, and the executable golden is new validation for that bug fix. The tracing carry/output spelling remains necessary. Recorded as F6. |
| `57c9018da` fixture migrations | `necessary` | The parser-owned markers, tracing carries/consts, typed expected IR, and placement-specific fixture moves are required by the corresponding dialect contracts. |
| `c71009928` placement and parser-owned fixture moves | `necessary` | Explicit `target_memory` and parser ownership of annotation-only `MemRef` fixtures are required authoring changes. |
| `29c56ec43` direct SPMD assembly fixtures | `necessary` | The `pl.intrin.spmd_direct`/`spmd_direct_call` migration and its focused coverage state the assembly dialect's dispatch form. |
| `f66b3bd09` typed staged constants and fixture migrations | `necessary` | The affected fixtures must use `pl.const` and staged predicates under tracing; the documentation hunks are out of scope under §2. |
| `1e2c80e77` transmute fixture rewrites | `necessary` | Producer-side `pl.intrin.transmute` is the required replacement for annotation-only view restamping, with carry rewrites where those fixtures also use loop channels. |
| `1e2c80e77` deletion of four `TestFlattenTileNdTo2DSpans` tests | `unnecessary` | The RHS-call, synthesized-statement, delegated-lowering, and fused-store span regressions are independent pass coverage and were deleted without a migration replacement. Recorded as F9. |
| `236dbad47` carry fixture rewrites and tracing cycle/typing adaptations | `necessary` | These hunks replace `init_values`/implicit yields and make staged values type-sound or explicitly reject recursion, as required by the tracing contract. |
| `236dbad47` removal of the two 60-second subprocess timeout overrides | `unnecessary` | `41cb2da7f` added the documented budget because these tests spawn a fresh interpreter; the subprocesses remain, and the spec commit gives no migration reason to remove the budget. Recorded as F8. |
| `bdefaec5b` explicit tpop producer semantics and fixture migrations | `necessary` | The shape/dtype producer contract must be implemented in the wrappers, registry/codegen, and affected fixtures; the documentation hunks are out of scope under §2. |
| `d54679982` tracing-jit and entrance oracle suites | `necessary` | These additions are the acceptance coverage for the tracing jit, entrance, parameter-world, submit, and variant-key contracts introduced by the migration. |
| `edef64804` frontend ownership flip and contract suites | `necessary` | Unsuffixed tracing dispatch, permanent parser markers, gate pinning, strict xfail handling, layering checks, and frontend-equivalence coverage are migration infrastructure. The generated `run-log` addition was historical churn and was removed by `f6e56bac7`. |
| `a8c4cc223` typed facade alignment | `necessary` | The direct attributes, honest heterogeneous namespaces, and inline ignores make the shared suites agree with the typed facade contract and typing gates. |
| `02d3360cc` explicit parser markers and parser-corpus rewrites | `necessary` | The aliases, decorator recognition, and parser-suite marker substitutions pin parser ownership before unsuffixed markers switch to tracing. |
| `edef64804`, `bdefaec5b`, `57c9018da` tracked `run-log` artifacts | `unnecessary` | They are generated gate output, not source or migration evidence. They were removed later by `f6e56bac7`, so no residual task is opened. |

### R3 verdict notes

The four accepted findings are historical scope decisions, not code changes in
this review session. Their owning follow-ups are listed under Discovered tasks.

### R4 API documentation verdicts

Oracles (both red before, both green after):

- R4 oracle `python3 scripts/review/check_api_docs.py`.
  Before: `public names 249 -> 255`, 8 added, `2` undocumented (`static_dim`,
  `static_shape`, 0 doc files each), `RESULT: FAIL`, exit 1.
  After: same 8 added, every added name has ≥1 doc file, `RESULT: PASS`, exit 0.
- User-manual gate `python3 tests/lint/check_docs_symbol_coverage.py` (pre-commit; scopes
  "documented" to a code span under `docs/en/user/`).
  Before: `247 documentable, 243 documented, 4 not` — `pl.expand`, `pl.program_parser`,
  `pl.static_dim`, `pl.static_shape`; `FAIL`, exit 1.
  After: `247/247`, exit 0. Base cross-check against the `origin/main` blobs:
  `241 documentable, 0 missing`, so all four violations are new in this changeset.

The R4 oracle undercounts because a design-doc mention satisfies it;
`check_docs_symbol_coverage.py` is the repo's own reading of "the user manual promises that
every user-visible capability is findable" and is the criterion used here.

| Name | Doc anchor | Anchor real? | Verdict |
| ---- | ---------- | ------------ | ------- |
| `static_dim` | none (planner-only: `docs/migration-todo.md`) | — | `unnecessary` — documented |
| `static_shape` | none (planner-only) | — | `unnecessary` — documented |
| `expand` | `docs/design/03-entrance.md` §7, `docs/design/04-migration-guide.md` | design docs only → no user-manual span, gate red | `unnecessary` — documented |
| `program_parser` | `docs/design/00-tracing.md`, `docs/design/01-language.md`, `docs/en/dev/language/03-functions.md:338` | dev docs only → gate red | `unnecessary` — documented |
| `function_parser` | `docs/en/user/language/02-control-flow.md:233`, `05-directives.md:20`, `01-functions.md:329` | yes (user manual) | `necessary` — already documented |
| `intrin` | `docs/en/user/language/02-control-flow.md:196` (§ "The assembly channels") | yes (user manual) | `necessary` — already documented |
| `jit_specializer` | `docs/en/user/language/01-functions.md:160` (quarantine note) + `docs/en/dev/` | yes | `necessary` — already documented |
| `runtime` | `pl.runtime(value, dtype)` at `docs/en/dev/language/03-functions.md:48,132`, `docs/zh/dev/language/03-functions.md:41,114`, `docs/design/04-migration-guide.md:279` | canonical spelling anchored in the dev guide; the oracle's 202-file count is incidental prose (`runtime` in unrelated code spans, e.g. `` `examples/runtime/` ``) | `necessary` — documented |

**Changes (en, with the per-name-maintained zh mirror).**

- `docs/en/user/language/00-types.md` (+ zh): new `### Reading a static shape` subsection
after `### Dynamic shapes`, covering `pl.static_dim` / `pl.static_shape` — host `int`, no
IR emitted, loud on a dynamic extent, `pl.tensor.dim` for a staged extent.
- `docs/en/user/language/01-functions.md` (+ zh): a `@pl.expand` paragraph beside the
`@pl.inline` paragraph (tracing-native expansion: one preparation, executed per call site,
not host-callable), and a sentence naming the parser-owned `@pl.function_parser` /
`@pl.program_parser` entries.

**Removal (`cond`, `while_`).** Verdict `necessary` — the removal is intentional and
proven from the changeset and the design authority:

- `13546e523` deleted `dsl_api.py` (the only definition site of `cond` / `while_`) and
dropped both from `__all__`; `yield_` survived (re-exported from `intrin`, deprecated
alias).
- `docs/design/04-migration-guide.md` §3: "**`pl.while_` and `pl.cond()` are gone.**" —
with the replacement spelling (a bare `while` over a staged predicate; carry derived from
rebinding).
- Round-trip is unaffected: the printer still emits `pl.while_(...)` / `pl.cond(...)`
(`tests/ut/ir/printing/test_python_printer.py` asserts it) and the parser matches those
spellings textually (`ast_parser._VALID_ITERATORS`); neither resolves the attribute, so no
`AttributeError`. `rg 'pl\.while_\b|pl\.cond\(' --glob '!docs/**'` finds no executable
usage outside the printer/parser/migrator.

The oracle's `STALE DOC?` note is mostly correct prose: the 22 / 13 files are
`docs/design/`, the dev IR grammar, removal-aware `02-control-flow.md` / `05-directives.md`
sections, or incidental word matches (`CHECK(cond)` in C++, `if cond:` IR pseudocode,
`pl.static_assert(cond, msg)` parameter names). One reference is genuinely stale, and it is
upstream text this changeset made stale: the control-flow index rows
`docs/en/user/language/index.md:27` and `docs/zh/user/language/index.md:19` still list
`while_` and `cond` as covered topics (the row was not touched by the migration commit
`87d7f25c7`).

**Needs-decision (recorded, not edited).**

| Location | Question |
| -------- | -------- |
| `docs/en/user/language/index.md:27`, `docs/zh/user/language/index.md:19` | The two index rows list `while_`, `yield_`, `cond` as control-flow coverage; `while_`/`cond` are gone. `yield_` is still exported (deprecated alias), so only those two are stale. Docs prose is out of scope for this review (§2) — fix in a docs pass, not here. |
