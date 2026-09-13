# Migration Known Gaps

Standalone record for problems the migration sweep (`docs/migration-todo.md`)
cannot fix within its constraints — they require existing-logic changes,
upstream adoption, or are by-design behavior. Nothing here is a reason to
work around a check in a test, example, or doc; the gap is the answer.

Entry format: problem / evidence / why not fixable here / tracking.

Fixes for the fixable entries are tracked in `docs/gap-fixes-todo.md`
(the work list); this file is the evidence record — an entry is removed
there in the same commit that closes its fix task.

## K2 — cadence prefetch-context corpus trace

- **Problem:** the prefetch-context surface from the upstream cadence
  line does not trace.
- **Evidence:** `tests/ut/jit/test_corpus_gaps.py::TestCadenceUpstreamSurface::test_cadence_prefetch_context_traces`,
  applied as a strict xfail by `tests/ut/conftest.py`.
- **Why not fixable here:** the fix is an upstream rebase/adopt
  (upstream `2e54cd660`, `#2142`, `#2089`, `#2369`) — never a test or
  corpus edit.
- **Tracking:** `tests/ut/_migration_xfail.txt` (remove the entry when
  the surface is adopted; the fix reports as XPASS).

## K4 — `static_assert` / `static_print` are silent no-ops under tracing

- **Problem:** the two compat-tier diagnostics only the AST parser
  runs; under tracing a kernel using them stays importable but the
  check never fires.
- **Evidence:** `python/pypto/language/compat/diagnostics.py` module
  contract.
- **Why not fixable here:** by design — kernels written for the parser
  stay importable in a tracing environment. There is nothing to fix;
  this entry exists so the behavior is recorded as a decision rather
  than rediscovered as a bug.
- **Tracking:** `docs/design/04-migration-guide.md` §8 (move the
  assertion into a host `if` that raises, or into a verifier pass).
