# Hard-Won Lessons — the tracing migration

Operating record of what the parser-to-tracing migration (run as
sequential worker agents) actually taught beyond the code it changed. Each lesson cost at
least one agent session or one false lead; none of it is obvious from the
code. AGENTS.md points here. Evidence anchors are task ids
(`docs/migration-todo.md`, `docs/gap-fixes-todo.md`) and commits.

## Environment facts

- **A root-owned auto-update holds the task-submit lock daily, roughly
  12:17–14:17 PDT.** Stale `.task.tmp` files in the task queue make the
  updater see a busy queue, so it waits out its full idle window and
  blocks every submission. Do not poll through it — pick device-free
  work, or reschedule. Two full agent hours were spent sleep-polling this
  window before anyone read the updater log (S13).
- **The `runtime` submodule drift can mask bugs.** The worktree's
  `runtime` pointer can sit ahead of the recorded gitlink and be installed
  editable, so device behavior follows the *working tree*, not the
  recorded commit. A device failure that "should" reproduce may already
  be fixed by that uncommitted drift — check `git status` on submodules
  before debugging a device-only failure (S13: the Case device-run
  "failure" was fixed by an uncommitted runtime commit).
- **`build/` can be configured from a dead ephemeral uv env** (cmake/ninja
  paths into a vanished uv tmp dir). If incremental builds misbehave, do
  not debug the cache — run `./scripts/rebuild` (F2, F3 each rediscovered
  this independently).
- **Under `task-submit`, invoke the worktree `.venv/bin/python`
  explicitly** — the default `python` on PATH cannot import
  `pypto.pypto_core` (E4).

## Verification methodology

- **A green doc-runner says nothing about unmarked pages.** The doc
  example runner only executes blocks carrying `<!-- doctest: run -->`;
  most pages have none. For content-level verification, trace the page's
  own snippets and read the printed IR (the D-group's standard red
  check).
- **A passing doc block can teach a broken kernel.** A block verified
  through a pass-only route (`compile_for_test`, `torch_codegen`) passes
  the runner while the kernel it teaches would fail a real compile —
  know which route a block exercises before trusting its green (D11
  found two such blocks teaching the then-broken `out[:]` spelling).
- **A green test does not prove the construct fired.** The scariest find
  of the migration: an annotated seed (`i: pl.Scalar[pl.INDEX] = 0`)
  silently does not stage — annotations on locals never flow into IR —
  so the while-loop host-unrolled at trace time, no `WhileStmt` was
  emitted, and the device test still passed because unrolled code
  computes the same result (S14). When a test claims a construct is
  exercised, pin it on the IR shape (print, or assert on the emitted
  statement), not on the golden.
- **Pre-existing lint baselines exist; don't re-derive them.** The doc
  symbol-coverage check carries 4–5 long-standing missing symbols, and
  `check_headers` fails on a few `python/pypto` files with license-wording
  drift — all pre-existing. Compare your failure against a run on the
  untouched tree before treating it as yours.

## Debugging heuristics

- **Errors fire far from their cause.** A `pto.subview` blayout mismatch
  at codegen was actually an op-registry input-space constraint missing
  two passes upstream: with no constraint, the mixed-kernel expansion
  split the matmul from its consumer with no bridge, and DCE silently
  deleted the consumerless matmul (F7). The finding method: minimal
  repro plus a *control case* — the nearby shape that compiles (the
  Out-param variant) isolates the differing variable.
- **Sibling facades share defects.** The bare-slice rank bug existed in
  both `Tensor.__setitem__` and `Tile.__setitem__`; the gap entry had
  found only one (F1). When a facade has a defect, check its siblings
  before writing the fix — the shared-helper fix covers both.
- **A never-compiled body hides numeric bugs.** When the flash-attention
  example finally compiled and ran against a golden, it failed — its
  online-softmax rescale had been wrong since the parser era, but the
  body had never once been executed (F7). Example code that has only
  ever been printed is unproven; the first successful end-to-end run of
  any "legacy" body deserves a golden check, not just an exit-code
  check.
- **Misleading terminal errors often mean a silent deletion upstream.**
  Both `prepare_native_run code 13` / AICPU 507018 (a device-wedge race,
  S13) and the blayout mismatch (DCE'd producer, F7) presented as
  downstream failures. When the error site's inputs look absurd (an
  undefined operand, a wedged device), suspect that something upstream
  was silently removed or raced, and look for what could have deleted it.

## Agent orchestration

**Explicit opt-in.** Launching sub-agents is something the main agent
may do **only when the user requests it explicitly** for the task at
hand. It is never an optimization to apply on own initiative: it spends
the user's compute and tokens, commits land under the user's name, and
an unattended worker can touch far more than the user intended. When in
doubt, do the work in-session and ask.

How the migration was actually run: sequential `pi -p` worker agents, one
task per session, each following the tracker file's contract
(red-then-green, update-the-tracker, one commit); an orchestrator agent
launched, verified, and adjusted between runs.

### The invocation recipe

The working command line (per worker):

```bash
pi --provider volce --model glm-5.3 -p \
   --session <workdir>/<TASK>.session --name <task> --no-skills \
   --system-prompt "<worker role + contract + absolute git rules>" \
   "Work task <ID> from <tracker file>."
```

What each element earns:

- **`-p`** — non-interactive: process the prompt and exit; the
  orchestrator stays in control between runs.
- **`--session <path>`** — one file per task, in a directory kept for
  inspection. This is also the resume handle: re-invoking with the same
  path continues that session's history (see the timeout lesson below).
  Do **not** pass `--no-extensions` — custom providers (e.g. `volce`)
  ride on extension discovery, and disabling it kills the provider
  (`Unknown provider "volce"`). `--no-skills` is the safe trim.
- **One task per session** — the tracker file's contract, restated in
  the system prompt. The worker reads the tracker for scope, rules, and
  verification; the system prompt carries only what must not be
  missed: the role, the contract in one breath (red first, fix, green,
  update tracker, one commit, explicit paths), and the absolute git
  prohibitions.
- **Sequential, never parallel.** One worker at a time, and the
  orchestrator verifies each result before launching the next: read the
  commit, re-run the acceptance check itself, check the tracker state.
  Verification is where the orchestrator earns its keep — several runs
  needed correction (re-triage, scope narrowing, a forbidden-`git`
  disclosure) that a fire-and-forget launcher would have missed.
- **Timeout ~1h** — sized to the gate, not the change (see below). A
  timeout is handled by inspecting the session file's tail, then
  resuming the *same* session with a corrective, time-boxed prompt —
  not by restarting.
- **Workers never wait.** The prompt carries the standing instruction:
  do not poll locks, do not sleep-loop; prefer device-free evidence;
  use `task-submit --max-time` bounded. Without it, workers will burn a
  full session polling an infra lock.

### The tracker file as the communication medium

The orchestrator does not pass scope through prompts — the **tracker
file is the single channel**:

- Workers read it to cold-start: task entry, red/green checks,
  cross-references to the owning docs, and the previous task's Notes.
- Workers write it back in the same commit as the work: state flip,
  completion notes, and **newly discovered tasks appended with the next
  free id** — discovery during work is the norm, not the exception
  (twelve of the tracker's thirty-seven tasks were found mid-flight).
- The orchestrator seeds it once, then adjusts it between runs (a task
  that proved mischaracterized gets re-triaged *in the file*, so the
  next worker inherits the correction rather than the error).

This is the worklog style: durable state in a file under version
control, ephemeral instructions in prompts. A prompt is gone when the
session ends; the tracker accumulates. The practical test: a fresh
agent given only the tracker (and "work the next open task") must be
able to do the right thing — if it cannot, the tracker is missing what
the prompts have been silently carrying, and that gap is the next fix
to the file.

### What the runs taught

- **A timeout is recoverable, not a failure.** Resume on the *same*
  `--session` path with a corrective, time-boxed prompt; the JSONL
  carries the interrupted agent's complete investigation. S13's root
  cause was recovered verbatim from the dead session's thinking blocks;
  F8's half-built compiler pass was finished and committed on resume.
  Inspect the tail of the session file before restarting anything.
- **Hypotheses are frequently wrong — mandate investigation before
  fixing.** Three of eight gap hypotheses were wrong or inverted: the
  missing codegen case was one site over from the guess (F3), the
  feared orphan-trap did not exist (F5), and the blamed packing code was
  a symptom of a pipeline that never created the output position (F8).
  "Confirm against the code, correct the record" was a contract rule and
  it earned its keep.
- **Red-then-green doubles as a re-triage trigger.** "If the documented
  red check does not fail, change nothing" caught two false positives:
  an already-migrated file (S6) and a failure fixed by environment drift
  (S13). Without that rule, agents would have fabricated work on both.
- **Harden the worker prompt in response to incidents.** One agent ran
  the forbidden `git stash` (disclosed, undone); explicit git
  prohibitions added to every subsequent prompt; no recurrence. Rules
  that live only in a file an agent may not re-read are weaker than
  rules restated in the task prompt.
- **Size tasks to the gate, not the change.** One hour fits a mechanical
  migration with targeted checks. A logic fix that must pass the full
  gate (pyright + strict overlay + 13k tests) can overflow — design for
  resumption rather than longer timeouts, and tell the resumed agent
  what the previous run had built.
- **The tracker file is the memory.** Task states, discovered-task
  appends, and per-task Notes let every agent cold-start in one read.
  Twelve of the 37 tracker tasks were discovered mid-flight by the workers
  following the discovery protocol — the file, not the orchestrator, is
  where scope lives.
