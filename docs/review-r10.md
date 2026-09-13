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

None recorded yet.

## 5. Round ledger

| Round | Task | Model | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | seed tracker + oracles | — (orchestrator) | done | *(this commit)* | oracles frozen: R1 red 34, R2 red 167, R4 red 2 |
| 1 | R1 | `opencode-go/deepseek-v4.1-flash:high` | pending | | |
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

None recorded yet.

### R2 allowlist justifications

None recorded yet — one row per entry added to
`comment_churn_allowlist.txt`.

### R3 non-minimal findings

None recorded yet.

### R4 API documentation verdicts

None recorded yet.
