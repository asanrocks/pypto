# ci-repro Co-evolution Worklog

Iterative verification of `scripts/ci-repro` against the real repo: run a
stage, triage every failure (script bug vs. repo finding vs. environment),
fix, commit, repeat until a full pass closes with no new open items.

Status legend: `fixed` (committed), `open` (tracked below, not fixable in
this loop), `env` (host repair, not a commit).

## Round 1 — pre-commit (uvx pre-commit run --all-files)

Environment repairs needed before the stage could run at all (`env`):

- `python3.10` on this host resolves to `/usr/local/bin/python3.10`, built
  without `_ssl`; the `check-environment-inputs` hook pins
  `language_version: python3.10`, so pre-commit's env bootstrap could not
  reach the package index. `scripts/ci-repro` now probes both `python3` and
  `python3.10` and shims a healthy interpreter (absolute path — pre-commit
  creates venvs with `cwd=/`, so a relative PATH entry silently fails).
- The first shim candidate (uv-managed 3.10.20) *also* fails virtualenv's
  discovery probe: its bundled setuptools scans a stale user-site
  (`~/.local/lib/python3.10/site-packages`) entry point and crashes, and
  discovery then falls through to the broken `/usr/local/bin/python3.10`
  anyway. `/usr/bin/python3.10` probes fine — the shim prefers it.
- npm 11.12.1 rejects pre-commit's `--allow-git root` for `-g` installs
  (EALLOWGIT) while accepting `all`; `scripts/ci-repro` shims `npm` for the
  stage with the rewrite (superset of what pre-commit asks).

Repo findings (`fixed`):

- **License headers drifted in 52 files** (wording variants, missing
  headers, one C++ file) — the migration gate never runs pre-commit, so the
  branch accumulated drift. Fixed to the canonical
  `tests/lint/check_headers.py` template; the checker now tolerates a
  leading `#!` shebang. One fixer bug caught on the way: the C++ header
  replacement left a duplicate `*/`, which broke the extension build —
  caught because ci-repro's unit-tests stage rebuilds.
- **`python/pypto/ir/utils.py:659`** — `ir.Type` typo for `_ir.Type`
  (ruff F821 + pyright).
- **`passes.pyi` missing `materialize_entry_returns`** — the C++ pass
  exists and `pass_manager.py` calls it; the nanobind stub never got the
  entry (pyright `reportAttributeAccessIssue`).
- **check-no-broad-raises**: 2 `pytest.raises(Exception)` sites narrowed to
  `pypto.Error` (the verifier's use-before-def) and `SyntaxError`
  (RewriteError's base).
- **check-environment-inputs**: `PYPTO_COLLECT` was an unclassified
  environment read; classified `semantic` in `python/pypto/_environment.json`.
- **check-op-name-literals**: 8 bare operator-name comparisons routed
  through `ir.get_op(...).name` module constants (scopes.py, submit.py,
  semantics.py ×2, three test sites).
- **check-function-attr-key-parity**: bare `"dual_aiv_dispatch"` /
  `"auto_scope"` attr keys replaced with the declared constants.
- **check-op-docstring-parity**: `pl.tile.assemble`'s DSL docstring now
  carries the memory-space rule its IR twin explains.
- **check-docs-symbol-coverage**: `pl.expand`, `pl.program_parser`,
  `pl.static_dim`, `pl.static_shape` were undocumented public symbols;
  documented in `language/01-functions.md`, `05-directives.md`,
  `00-types.md`.
- **cpplint**: one unbraced multi-clause `if` in
  `infer_tile_memory_space_pass.cpp`.
- **markdownlint**: AGENTS.md heading-level jump (five `###` promoted to
  `##`), MD060 table-style drift in nine docs files, one unfenced-language
  block, two broken `#pl.at-*` anchors (dots are stripped in GitHub slugs).
  The hook also **walked into the `3rdparty/` submodules** with `--fix` and
  whitespace-touched their READMEs/CHANGELOGs (clang-format and cpplint
  exclude `^3rdparty/`; markdownlint-cli2 did not) — submodule noise
  reverted and the exclusion added to the hook.
- **ruff**: F841/PLC0415/E501 leftovers in five test files; an invalid
  `# noqa` directive inside a prose comment in `tracer/decorators/function.py`.
- **`scripts/migrate_intrin.py`**: 10 pyright errors — predicates converted
  to `TypeIs[cst.Call]`, `BaseSuite.body` union handled with boundary
  casts, an `attr is None` guard; `libcst` added to the dev extra (the
  pyright hook resolves imports against the project venv).
- Auto-fix hook drift applied: ruff format (15 files), clang-format (3),
  end-of-file (.gitignore).

Environment repair (`env`): the **runtime submodule worktree was a broken,
stale snapshot** — no git HEAD, content older than the recorded commit
`39ce891db`, so the simpler wheel built from it lacked `ChipTensor` and 5
gate tests failed. Repaired by rsyncing the identical, clean checkout from
the sibling `pypto-1/runtime` (same recorded commit, offline-safe) and
reinstalling simpler; the managed pto-isa clone was recreated from the
host's pinned checkout (`PTO_ISA_ROOT`) instead of the unreachable GitHub.

Round-1 result: pre-commit 22/23 hooks pass; the gate (unit-tests stage) is
fully green (13751 passed).

## Round 2 — clang-tidy + codegen-tests

Script bugs (`fixed` in `scripts/ci-repro`):

- The clang-tidy stage ran the checker without the tidy venv's `bin` on
  PATH, so `--strict-version` could not find the `clang-tidy` binary (CI
  activates the venv first). The stage now prefixes PATH with the venv.
- The tidy venv bootstrap failed on a re-run (`uv venv` refuses to replace
  an existing environment) — now created with `--clear`.

Repo findings (`fixed`):

- **`include/pypto/ir/op_registry.h:702`** — clang-tidy
  `bugprone-unchecked-optional-access` (promoted to error) on
  `set_input_memory_matches_arg`; brought in line with the sibling
  `set_input_memory` (deref to a reference with the NOLINT).
- **Two stale `tests/st/codegen/dsl` files** still driving the
  specializer-era jit surface: `test_batch_matmul_pipeline.py` cleared a
  `_cache` attribute and called the kernel with `config=`, and
  `test_split_aiv_none_codegen.py` called the kernel — both now get a bare
  `ir.Function` from the eager tracing lane. Ported both to the
  established `_compile(fn, cfg)` route (the same mapping the lazy lane's
  `config=` call performs internally).

Environment repairs (`env`):

- **The runtime worktree's `pto_isa.pin` was itself stale**: the broken
  snapshot pinned pto-isa `83d01313` (matching the host checkout by
  coincidence), while the recorded runtime commit pins `5a4f74cb`. After
  syncing the runtime content, ci-repro's fidelity check caught the
  mismatch; the managed clone `runtime/build/pto-isa` was re-pinned to
  `5a4f74cb` (fetched from the sibling checkout's clone — GitHub is
  unreachable) and simpler rebuilt against it. Five gate tests that import
  `simpler.task_interface.ChipTensor` now pass.

Round-2 result: clang-tidy PASS (299s), codegen-tests PASS (62 passed).

## Round 4 — the device tier (--device)

Script bugs (`fixed` in `scripts/ci-repro`):

- The batched stages cannot run inside a `task-submit` session: the
  harness's `--execute-via-task-submit` submits per batch, and task-submit
  rejects nested submission inside its own daemon. The
  stages now run from the shell with `--task-submit-device` pinned to ONE
  device (CI's TEST MODE shape), the id taken from `CI_REPRO_DEVICE_ID` or a
  one-shot borrow probe.
- The pinning is not cosmetic: with per-batch `--device auto`, batches land
  on both dies of one physical chip concurrently and a single AICPU fault
  cascades into dozens of `507901` failures (the documented a2a3 shared-SO
  fault). One transient chip-4 AICPU exception produced a 174-failure run
  whose per-case errors were almost all cascade noise.
- A pinned device can be taken by another user between the probe and the
  batches — the batches then queue on the lock. The stage's
  `--task-queue-timeout` bounds the wait (a loud failure, not a hang);
  manual invocations must pass it too.
- Two of the author's own round-4 edits failed the hooks on the next full
  run (Chinese error text quoted in the worklog tripped
  `check-english-only`; helper insertions missed ruff format and one .cpp
  reflow) — the convergence run exists to catch exactly this, and did.

Repo findings (`fixed`):

- **`test_bitwise_binary.py` (144 failures)**: three layers of
  specializer-era drift — `.specialize()` (replaced with an eager-lane
  `_program()` helper that assembles the entry plus the decoration assembly's
  callees), `dtype=lhs.dtype` staged attributes into `pl.tile.create`
  (replaced with the closure's concrete `pl_dtype`), and full-shape `tmp`
  tiles fed to narrowed-`valid_shape` ops (narrowed via
  `pl.tile.set_validshape`). All 360 cases green on device.
- **`test_gather_composite_scenarios.py` (10 failures, partially fixed)**:
  `patches.shape` returns `None` on an expr-wrapping facade (the
  tracing-native spelling is `pl.static_shape`), and op validation is now
  loud about same-dtype casts (a `pl.cast(values, pl.FP32)` whose input is
  already FP32). Both fixed; the conv kernel then hits a **ptoas-level
  `pto.tmul` element-type mismatch** that needs the op owner — left open.

Open items found in this round (characterized, per-file porting work for
the migration owners — see Open items below).

## Open items

- **pyright hook debt (~445 errors over `tests/`)**: the repo-default
  basic-mode pyright (which the CI hook runs over all tracked files) is
  red on this branch's test tree. The migration gate only type-checks
  `python/pypto/language/`, so `tests/` drifted: 315 of the errors sit in
  three files (`tracer/collector/test_driver.py`,
  `test_func_attr_tracing.py`, `test_split_transfer_tracing.py` — one
  pattern: `collector.instrument` not exported to the import surface),
  plus ~25 `pl.Tuple[...]` facade return-annotation mismatches and
  scattered smaller families. This is a migration-campaign-sized cleanup,
  not a mechanical fix — the `Tuple`-annotation sites in particular
  interact with DSL facade semantics and should be fixed with the
  entrance/typing owners.
- **`git fetch origin main` is unreachable on this host** — the clang-tidy
  diff base may be stale; ci-repro reports it as a fidelity caveat.
- **Device-tier st-test drift (~60 failures, each characterized)**:
  - `test_gather_composite_scenarios.py` — the conv/grid kernels now trace
    past the shape/cast fixes but fail at ptoas (`pto.tmul` src0/src1
    element-type mismatch); needs the op owner to say whether the kernel
    or the op typing moved.
  - `tests/st/runtime/cross_core/` (16) — `tpop_from_aic()` gained a
    required `shape` argument the tests don't pass.
  - `test_jit.py` (9) — `.warmup` and friends on eagerly-traced
    `ir.Function` objects (the lazy-lane surface moved).
  - `tests/st/distributed/collectives/test_l3_tensor_*` (~17) — the l3
    tensor intrinsic family fails on two-card runs; not yet diagnosed
    beyond the family signature.
  - paged-attention numeric-compare (2) in the device environment.
