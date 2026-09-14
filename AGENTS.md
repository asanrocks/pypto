# Tracing Frontend Migration

This worktree is migrating PyPTO from parser-first collection to a tracing
frontend. The migration has three dialects:

- **Ergonomics dialect**: user-facing kernel code. It uses ordinary Python
  control flow and the ergonomic `pl.*` operations. Its frontend is tracing.
- **Assembly dialect**: a strict low-level extension of the ergonomics
  dialect, used by compiler developers and compiler unit tests to express
  explicit IR constructs. Its frontend is also tracing.
- **Round-trip dialect**: the lossless representation emitted from C++ IR by
  the Python printer. It is consumed by the AST parser and is not a tracing
  input.

The intended pipelines are:

```text
user code (ergonomics) --tracing--> C++ IR
compiler tests (assembly) --tracing--> C++ IR
C++ IR --IR printer--> round-trip dialect --AST parser--> C++ IR
```

The migration contract is:

- All tests except tests whose contract is AST parsing or printer round-trip
  parsing must use the ergonomics or assembly dialect.
- Every program executed by tracing must be type-sound. Do not rely on
  execution context to retype Python values, and do not preserve annotations
  that lie about the values produced by operations.
- The migration changes Python frontend code and tests only. It must not touch
  C++ code, teach the parser to recognize ergonomics or assembly semantics, or
  teach tracing to consume round-trip syntax.
- The C++ IR printer remains the source of round-trip text, and the AST parser
  remains the sole owner of parsing that text.

## Frontend Markers

The tracing frontend is now the default and only collection frontend for
ordinary DSL and assembly fixtures:

- `@pl.function` and `@pl.program` collect through tracing everywhere outside
  the parser test directories.
- `@pl.function_parser` and `@pl.program_parser` are permanent parser-owned
  entrypoints. Use them only for AST/parser tests and printer round-trip
  source fixtures under `tests/ut/ir/parser/` or `tests/ut/language/parser/`,
  with one exception: the frontend-equivalence harness under
  `tests/ut/language/conformance/` calls them as the reference oracle — it
  compares `pl.function`/`pl.program` output against the parser's output, so
  the parser side of that comparison stays parser-owned.
- The transitional `*_pending` decorators and `PYPTO_TRY_PENDING` probe have
  been removed. Do not reintroduce a third frontend marker.

Round-trip verification is always parser-owned. Do not add an IR/tracing
round-trip mode or route `pl.parse` through tracing.

## The Entrance Foundation (`pypto.language.entrance`)

`pypto.language.entrance` is the public entry tier every entrance is policy
over (design doc `docs/design/03-entrance.md`). It exposes exactly four
names — `Param`, `Decl`, `TracedCallable`, `current_trace()` — plus the
`Infer` return-contract sentinel:

- `Param`/`Decl` are immutable declaration data; invocation placeholders
  resolve by `Param` identity, never name lookup or structural equality.
- `TracedCallable.from_callable`/`from_string` do the reusable preparation
  (rewrite + compile); `invoke(args, kwargs)` performs one atomic root trace
  with fresh one-shot state and returns the `ir.Function`.
- `current_trace()` projects the one ambient channel; its opaque capability
  exposes only `expand` (borrowing expansion) and is loud when stale.
- `entrance.annotation.derive` (a framework library above the foundation)
  produces the `Decl` + flat invocation recipe from annotations.

The layering is enforced by test: shared language code imports no frontend,
the collector/Sema import no entrance policy, and the entrance foundation
never imports the parser. `@pl.expand` (`tracer/decorators/expand.py`) is the
tracing-native expansion entrance over the capability — body executed and
emitted at every occurrence, one trailing return, no recursion. Acceptance
tests: `tests/ut/language/entrance/`.

## The two jits (transitional quarantine)

`pl.jit` collects through **tracing** (`pypto/language/tracer/decorators/jit.py`);
both timings return one public jit object. Eager (every param a facade
annotation) traces at decoration — the trace (`.fn`) plus the callees its
body reached are retained in a record, and the object's host
`compile`/dispatch serve that record's program: the annotation determines
the artifact (declared dynamic extents stay dynamic; call args are
dispatch values, and a contradicting static shape re-traces). Lazy is the
full jit surface — a call with `torch.Tensor`
args bridges each to a static-shape facade, assembles the multi-function
program (entry + callees reached in-trace, emitted as detached `ir.Call`s),
compiles via `ir.compile`, and dispatches (cached on the bridged signature
plus the compile identity — platform, strategy, distributed config, planner,
dbC flag, resolved from the `config=RunConfig` and any active `PassContext`;
03-entrance.md §5.4). A host `compile`/`__call__` accepts `config=RunConfig`
(compile-side fields forward to `ir.compile`, runtime fields to
`CompiledProgram.__call__`; in-trace calls stay parser-aligned and reject it);
`compile_for_test` runs the pass pipeline only. Parameter worlds are pinned
by the declaration: a pl facade annotation (bare class included) is a runtime
param, a plain Python annotation or none is a host knob baked at trace time;
a contradicting call site is a loud TypeError in both directions, and
`pl.runtime(value, dtype)` is the entry spelling for a runtime scalar (the
dispatch cache keys on the dtype, never the value). The name-keyed compile
vocabulary (`fn.compile(config=..., scalar=pl.RUNTIME)` — the golden
runner's `compile_runtime` path) is declaration-owned: `RUNTIME` is a no-op
on an already-runtime `pl.Scalar` param, a host literal is accepted and
ignored (dispatch supplies the value; the artifact is identical), and a
name targeting a host-knob param is loud. Outputs are host-visible
only through Out parameters: `MaterializeEntryReturns` promotes a dispatch
entry's returned tensors to appended Out params (deleting the internal
`tensor.create` that held them), so a return-style kernel's host call returns
the tensors — full in-place calls get `None`, and a partial call with the
declared signature returns only the outputs it omitted. The standalone
`pl.function` lane stays parser-aligned: host literals lift to Const args.
Call and submit share one instantiation semantics — `pl.submit` /
`pl.spmd_submit` resolve their callee through `Marker.resolve_submit` (the
same per-call variant lane a call takes; only the emitted instruction
differs), and closure-captured callees specialize identically to
module-global ones (specialization is callee-owned, never a property of
where it is bound). `@pl.jit.extern` traces a
single-core stub as a header-only AIC/AIV `ir.Function` and a `mixed` stub
as an AIC + AIV member pair with the Group wrapper assembled at the call
site. Behavioral coverage: `tests/ut/jit/test_roundtrip.py` (structural
equality vs the traced `@pl.program` reference) is the oracle.

`pl.jit_specializer` is the **quarantined AST-era runtime** (`pypto/jit/`:
specialization via source rewriting, cache, device execution). Unlike the
permanent `pl.function_parser`, it has no long-term owner — it exists only
for its own unit tests (specializer-mechanics coverage), and is deleted
once their behavioral assertions are ported. It is guarded: calling it
inside a trace raises (jit-inside-jit must never silently escape the tracing
frontend). Do not build new features in `pypto/jit/`; route
specializer-mechanics tests there, but any test asserting a user-observable
jit behavior must eventually port to the tracing jit.

Two silent-degradation classes are closed on the tracing side: parameter
directions ride the runtime annotation (`Out[T]`/`InOut[T]` carry their
direction on the facade copy — no source reconstruction, so exec'd kernels
keep their directions), and `bind_dynamic` on a static dimension raises
(the override form's semantics lived in the specializer's AST scan; the
annotation-embedded `pl.dynamic` form is the supported spelling).

Known tracing gaps from this migration are listed in
`tests/ut/_migration_xfail.txt` and applied as strict xfails by
`tests/ut/conftest.py`. The manifest is a comment-friendly nodeid list with
source-ownership and root-cause notes; keep comments explanatory rather than
using them as machine-readable failure signatures. Remove an entry when
tracing support lands so a fix is reported as an XPASS. The normal gate uses
`pytest-timeout=10`.

The gate intentionally runs item-level xfail tests through normal pytest;
xfail means an expected result, not a test exclusion. Pytest emits the original
expected-failure traceback through the reporting hook in `tests/ut/conftest.py`,
so investigate `run-log` from the same gate invocation instead of rerunning
with `--runxfail`. Module-level decorator failures cannot become item xfails;
when one appears, pytest reports the collection error directly in the gate
log.

## Typing Gates and the Surface-Gap Report

Typing enforcement is tiered:

- The repo default (`[tool.pyright]` in `pyproject.toml`) is basic mode.
- The collector strict overlay (`pyrightconfig.collector.json`) holds
  `python/pypto/language/tracer/collector`, the tracer protocol/semantics/
  constructs, and `python/pypto/language/constructs` to strict typing.
- `tests/lint/check_typing_debt.py` ratchets explicit escape hatches
  (`Any`, `cast`, `# type: ignore`) per file in the overlay trees.

`scripts/report_surface_gaps.py` is the informational companion: it reports
strict-mode *signal* without the annotation-labor noise. It runs two passes —
full strict over the entrance tier (`language/tracer/decorators`,
`language/tracer/ambient.py`, `facade.py`, `intrin.py`, `typing`, `compat`,
`op`), and strict over `tests/ut` with the
mechanical rule families (`reportUnknown*`, missing-parameter, unused-*)
disabled — then classifies the residue against known root causes (collapsed
to one line each) instead of dumping raw diagnostics. Use it to find
library-surface gaps that blind kernel authors' type checkers; do not wire it
into the gate as blocking. When a root-cause fix lands, delete its classifier
block and the category disappears from the report.

`scripts/report_lib_gaps.py` is the corpus-side sibling: it strict-scans the
pypto-lib model zoo (sibling checkout, per model-variant to keep bare
sibling imports resolving) and buckets findings into gap candidates
(callee origin tagged), frontend-precision candidates (Unknown seeded at
`pl.*`/slicing — the only detector for library `Any` leaks), allowlisted
false positives (`scripts/lib_gaps_allowlist.txt`), and collapsed local
debt. Same contract: informational, never gate-blocking.

## Migration Workflow

Each commit is reconstructed from a source snapshot rather than replayed or
rebased:

1. Select source `S` from the existing line.
2. Materialize `S` with `git restore -s S .` and stage only that commit's paths.
3. Restore the worktree from the index and remove residual files.
4. Run `./scripts/gate` and fix every red result.
5. Obtain explicit confirmation before committing the next migration unit.

Every commit must pass the gate. Keep commits narrow enough that a failing
marker family or typing rule has one identifiable owner.

## Agent Orchestration (explicit opt-in)

**Never launch sub-agents unless the user requests it explicitly for the
task at hand.** Spawning `pi` worker agents is not an optimization to
apply on own initiative — it spends the user's compute, lands commits
under the user's name, and an unattended worker can touch far more than
intended. When in doubt, do the work in-session and ask. When
orchestration *is* requested, the full methodology — invocation recipe
(`pi -p` command line, model choice, one-task-per-session, sequential
runs with orchestrator verification, timeout-resume handling), and the
tracker-file-as-communication-medium pattern — is recorded in
`docs/hard-won-lessons.md` § Agent orchestration; follow it rather than
improvising.

## Hard-Won Lessons

The tracing migration left an operating record of traps and method
lessons that cost real sessions to learn: `docs/hard-won-lessons.md`. Read it before device debugging (the daily
task-submit lock window; the `runtime` submodule drift masking bugs),
before trusting a green check (a passing test does not prove the
construct fired — pin constructs on the IR shape), and before running
orchestrated agent runs (the resume-on-timeout and red-check-re-triage
contracts). It is also the canonical place to record the next lesson
worth keeping.

## Shared Plugins

Portable skills are published from `hw-native-sys/pypto-skills`; they are not
vendored into this repository. Install the plugins needed for the current
agent and scenario (e.g. `codex plugin add pypto-developer@pypto-skills`).
Repository-local workflows stay under `.claude/skills/`.

## Preferred Commands

For full testing details, follow `.claude/skills/testing/SKILL.md`.

```bash
# Install dev dependencies
uv sync --extra dev

# Configure and build
[ ! -f build/CMakeCache.txt ] && cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --parallel

# Migration gate: pyright, ruff, and pytest run once; output is also written to run-log
./scripts/gate

# Gate a focused set and write diagnostics to a separate log
PYPTO_GATE_LOG=/tmp/pypto-gate.log ./scripts/gate -- tests/ut/path/to_test.py

# Focused test
uv run python -m pytest tests/ut/core/test_error.py -n auto --maxprocesses 8 -v

# Lint and type check
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pre-commit run --all-files
```

Run system tests in `tests/st/` only when the task requires them and the
necessary hardware or environment is available.

## Repository Map

- `include/pypto/`: public C++ headers
- `src/`: C++ implementation, passes, codegen, runtime internals
- `python/bindings/`: nanobind extension bindings
- `python/pypto/`: Python API, DSL, pass manager, and type stubs
- `tests/ut/`: unit tests
- `tests/st/`: system and hardware-dependent tests
- `tests/lint/`: repo-specific lint and validation scripts
- `tests/lint/`: repo-specific lint and validation scripts
- `scripts/`: repo tooling — the migration gate (`gate`), build helper
  (`rebuild`), typing audits (`type-audit`, `report_surface_gaps.py`,
  `report_lib_gaps.py`), and
  focused migration helpers
- `docs/`: living design docs (`docs/design/`) — indexed in `docs/README.md`
- `docs/en/dev/`, `docs/zh/dev/`: developer documentation
- `examples/`: user-facing examples (beginner/, intermediate/, advanced/, models/, runtime/, utils/)
