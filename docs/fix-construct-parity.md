# Fix campaign — construct-parity findings

Fixes the defects found by `docs/review-constructs.md`. That file stays the
frozen findings record (the `F*` entries with their red checks and pins); this
file is the fix worklist, the contract, and the ledger.

The review campaign proved the divergences exist. This campaign changes
frontend behaviour, which is a different risk class: a fix that makes one
anchor green can silently move another, and the pins left behind by the review
assert the *buggy* behaviour on purpose.

## 1. Contract (every fix task)

`docs/review-r10.md` §1 applies (one task per session, one commit, explicit
paths, the git prohibitions, terse report), with these overrides:

1. **The red check is the F entry's red check**, made to fail *first*. Where the
   review left a pin test asserting the buggy shape, the fix **rewrites the pin
   into the correct expectation and shows it red before touching the frontend**.
   A pin says in its own docstring to delete its assertions when the fix lands;
   that is the instruction, not a suggestion.
2. **Never weaken the oracle.** `tests/ut/language/conformance/_harness.py` is
   not editable; neither is the equivalence notion (or the `INT64`/`INDEX`
   relaxation in `structural_equal.cpp`, which is upstream `45fe19c24`). A fix
   that turns an anchor green by loosening the harness is a failed task.
3. **The full `./scripts/gate` is the acceptance criterion** — unlike the review
   campaign, these tasks change behaviour. Expect it to overflow a session:
   design for resumption, and use `PYPTO_GATE_LOG=/tmp/gate-<task>.log
   ./scripts/gate -- <paths>` for a focused first pass before the full run.
4. **No C++ without saying so.** The IR printer lives in
   `src/ir/transforms/python_printer.cpp`; a task that touches C++ needs a
   rebuild (`scripts/rebuild`) before its evidence counts. Flag it in the
   commit message.
5. **Update the findings record.** Flip the `F*` entry's state in
   `docs/review-constructs.md` in the same commit, and keep the review's
   `needs-decision` wording (do not convert one into a fix silently).
6. **One defect per task.** If two `F` ids are the same defect (F22/F26 are),
   they are one task; if a fix grows a second defect, record it, don't fold it.

## 2. Batch order

Ordered by blast radius, then by cost (Python-only before C++/rebuild):

**Batch A — round-trip breaks** (the tracing lane emits IR its own parser
cannot re-read; each also fails `pl.parse(str(traced))`):

| Task | Defect | Language |
| --- | --- | --- |
| A1 | **F12** — `break`/`continue` accepted in a top-level `pl.spmd` / `pl.split_aiv` region that lowers to no loop (`PtoForConstruct._validate_transfer` falls back to kind `"range"`) | Python |
| A2 | **F23** — the `pl.spmd` with-form CORE_GROUP carrier boundary diverges (dynamic on tracing, static on parser); confirms the migration campaign's F1 | Python |
| A3 | **F22 + F26** — the printer drops the `var = pl.yield_(...)` LHS inside a manual `RuntimeScopeStmt` and inside a `SplitAivScopeStmt` | **C++** |

**Batch B — parser silent loss** (the parser drops user code; tracing is already
loud, so the parser is the buggy side):

| Task | Defect |
| --- | --- |
| B1 | **F13** — a `for`/`while` `else` suite is silently dropped |
| B2 | **F15** — statements after a function-level `return` |
| B3 | **F16** — an inline callee's multi-value trailing return splices 2 values vs the parser's `MakeTuple` |
| B4 | **F17** — an inline early return under staged control flow is mis-lowered |

**Batch C — tracing validation gaps** (tracing accepts what the parser rejects;
several also fail print→parse): F18 (graph `dump_vars`), F19 (non-enum `mode=`),
F20 (manual-scope depth), F21 (name-hint validation), F24 (nested `split_aiv`),
F25 (`pl.at` name_hint).

**Batch D — shared semantic hole:** F14 (a type-changing rebind inside a staged
loop produces IR that `ConvertToSSA` rejects, on **both** lanes). Not a parity
divergence — fixing it changes shared semantics, so it goes last and alone.

**Batch E — submit / bindings surface:** F11 (`std::bad_cast` on a non-string
`attrs=` key), F27–F35 (submit constant folding, duplicate `dumps=`, extra
temporary, `dump_tag` merge, multi-output target, `core_num` validation,
annotationless callee metadata, `device=` literal dtype).

## 3. Out of scope

- The 33 `needs-decision` rows recorded by the review. Several may *redirect* a
  fix (a "bug" that turns out to be intended), so they are triaged with the
  human before Batch C, not silently resolved.
- The 5 unclaimed public tracing entries (decorator-only, no parser surface).
- The `op/` surface (27 parser helpers) — its own campaign.

## 4. Round ledger

| Task | Defect | Language | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| A1 | F12 | Python | pending | | |
| A2 | F23 | Python | pending | | |
| A3 | F22 + F26 | C++ | pending | | |
