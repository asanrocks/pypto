# Construct parity review — tracing vs parser frontends

Second review campaign, separate from `docs/review-r10.md` (the migration
changeset review). This one answers a different question: for the structural
language surface — the handlers for `if`/`else`, `for`, `while`, the scopes,
the exits — **is the tracing frontend's logic equivalent to the AST parser's?**

The two frontends are `python/pypto/language/parser/ast_parser.py` (the parser
lane) and the tracing lane (`tracer/constructs.py`, `constructs/*.py`,
`tracer/collector/`). Both are live: `pl.function` collects through tracing
everywhere outside the parser-owned test directories, while the parser stays
the round-trip owner.

## 1. What "equivalent" means here

Equivalence is **already defined** by
`tests/ut/language/conformance/_harness.py`. The work is to *populate* it, not
invent it:

```python
conformance(kernel, build, *, canonical=False)
    == assert_structural_equal(build(kernel), function_parser(kernel))
```

Both frontends parse **the same source text**, so an anchor is defined only for
spellings *both* lanes accept. Three traps follow, each of which manufactures
false findings if ignored:

1. **Regime.** Straight-line kernels compare *raw*. Staged control flow
   compares *post-`canonicalize()`*, because tracing deliberately emits **less**
   than the parser (no `iter_arg` / `return_var` / explicit phi; the SSA pass
   synthesizes them from self-referential assignment updates and convergent
   branch bindings). Every task must state its regime and why. The wrong regime
   turns an intended difference into a fake divergence.
2. **Retired spellings are out of scope.** `pl.while_(...)`, `pl.cond()`,
   `_parse_legacy_scope`: parser/round-trip-only, rejected by the tracing lane,
   so no anchor is possible.
3. **A red anchor is a finding, not a fix.** Divergence fixes are separate `F*`
   tasks. Otherwise the tempting repair is to edit the frontend — or the test.

## 2. Why the unit of work is a construct, never a file

The equivalence question cuts across both sides: a parser handler lives in a
9k-line `ast_parser.py`, and its counterpart is spread over
`tracer/constructs.py`, `constructs/loops.py`, `constructs/scopes.py`,
`tracer/collector/interpreter.py`. A file-based split produces units that cannot
answer the question they were given. **One task = one construct family,
compared end to end.**

## 3. Evidence base (at seed time)

**77 anchors across 14 of the 20 conformance suites:**

| Suite | Anchors | Suite | Anchors |
| --- | --- | --- | --- |
| `test_while_logical` | 16 | `test_spmd_with_form` | 4 |
| `test_subscript_read` | 12 | `test_spmd_submit_only_kwargs` | 4 |
| `test_at_const` | 10 | `test_at_predicate` | 4 |
| `test_if` | 7 | `test_program_harness`, `test_program`, `test_op_zoo`, `test_harness` | 2 each |
| `test_for` | 7 | `test_jit` | 1 |

**Six suites contain no `conformance(...)` call at all:** `test_auto_scope`,
`test_call`, `test_extern`, `test_inline`, `test_staged_host_join`,
`test_submit_collect`. That is not automatically a coverage hole — they may
assert through a different entry point (a `bind`/program helper, or a targeted
negative assertion). **Resolved in Round 0:** the anchor count is per *suite*,
and `test_call`/`test_extern`/`test_inline` exercise calls through a different
helper, so the per-construct table below (not the per-suite count) is the
worklist. Coverage is thin in specific places, not absent.

## 4. Round 0 — the parity inventory oracle

`scripts/review/check_construct_parity.py` (report-only, not wired into
`./scripts/gate`). It holds a hand-maintained construct map and *verifies* it
against the tree, reporting STALE for any map pattern that matches nothing, so
the map cannot silently rot into passing. Three outputs: the per-construct
coverage table; unclaimed parser handlers, bucketed into the out-of-scope
op/value surface and the structural remainder; unclaimed public tracing entries.

**Round 0 output at seed time:** 0 stale map entries; **10 of 16 constructs have
no anchor suite**; **4 unclaimed structural parser handlers** —
`_handle_at_optimizations_kw`, `_parse_and_validate_core_num`,
`_parse_pld_category_op`, `_parse_range_keyword` (the other 26 unclaimed
handlers are the op/value surface, out of scope by section 8); **17 unclaimed
public tracing entries**.

Anchors per construct: `while` 16, `binding / assignment` 12, `at family` 10,
`if / else / elif` 7, `for + iterators` 7, `spmd` 4 — and **0** for
`break / continue`, `returns / exits`, `generic scopes`, `split_aiv`,
`submit / spmd_submit`, `with-form scopes`, `intrin surface`,
`statement handlers`, `func attrs / optimizations`, `calls`.

The discipline is the same as round 0 of the migration review: **the agent
judges; it never enumerates.** An earlier round proved a model can drop a file
from a 21-item `rg -l` list on the easiest task in the campaign.

## 5. Wave 1 — structural core (answers the question directly)

| Task | Construct | Today | Gap to close |
| --- | --- | --- | --- |
| R5 | `if` / `else` / `elif` | `test_if` 7 | `elif` chains; host-vs-staged predicate dispatch; `if` nested in `while`/`spmd` |
| R6 | `for` + iterators | `test_for` 7 | `range` kwarg forms; `parallel` / `unroll` / `pipeline` + stage pinning; target forms (single, tuple, computed) |
| R7 | `break` / `continue` | 2 in `test_for`, some in `test_while_logical` | nested loops; deferred staged exit ordering; exits inside `while` |
| R8 | binding & assignment | `test_subscript_read` 12 | subscript **write**; staged rebinding; defs / store-target first-encounter order (`136a0f457`); `pl.const` staged init; tuple unpack; `pl.runtime` worlds |
| R9 | function exits | `return` mentioned in 21 suites | trailing single-return contract; early return; return inside branch/loop; `@pl.expand`; `inline` |

## 6. Wave 2 — region and remaining surface

| Task | Construct |
| --- | --- |
| R10 | generic scopes: `scope`, `manual_scope`, `graph`, `cluster` |
| R11 | `spmd` (with-form vs for-form, tid capture), `split_aiv`, CORE_GROUP carrier — the migration review's F1 lives here — plus the `at` family |
| R12 | statement handlers: `static_print`, `static_assert`, `dump_tag`, `set_cache_policy`, `func_attrs`, `optimizations` (**0 anchors today**) |
| R13 | `submit` / `spmd_submit` kwargs, `intrin.*`, kernel / external / dispatch calls |

## 7. Per-task contract

The worker contract is `docs/review-r10.md` §1, unchanged: one task per session,
the session ends with one commit containing the work and the tracker update;
explicit paths only; the absolute git prohibitions; the evidence rule; verdict
vocabulary (`unnecessary` / `necessary` / `needs-decision`); never run the full
`./scripts/gate`; the final message is the report.

Additions for this campaign:

- **Deliverable per construct** — a conformance anchor committed under
  `tests/ut/language/conformance/`, plus a verdict:
  - `equivalent` — the anchor lands green;
  - `intended divergence` — must cite the doc / ADR / manifest line that
    intends it, otherwise it is `needs-decision`;
  - `bug` — the anchor lands red; record an `F*` task and leave the frontend
    alone.
- **State the regime** (raw vs post-canonical) and why, per anchor.
- **Read both sides before verdicting.** Three of the migration campaign's eight
  gap hypotheses were wrong or inverted.

## 8. Out of scope

- **The `op/` surface** (`tile_ops.py`, `tensor_ops.py`) — a different axis with
  its own dedicated lints (`check_op_docstring_parity`,
  `check_function_attr_key_parity`, `check_op_name_literals`) and its own
  campaign.
- Retired parser-only spellings (see trap 2).
- Docs prose.

## 9. Open decisions

1. **Standalone tracker** (this file) vs extending `docs/review-r10.md` §8.
2. **The harness is the definition of equivalence** — units must not invent
   their own notion. Confirm.
3. **Divergence fixes go to separate `F*` tasks**; review tasks stay
   review-only. Confirm.
4. **Wave 1 only, or authorise wave 2 too?** Wave 1 is 5 sessions, comparable to
   the migration review's round 1.

## 10. R5 verdicts — `if` / `else` / `elif`

Both sides read before verdicting: the parser's `parse_if_statement`
(`ast_parser.py:3629`, whose `orelse[0] is ast.If` case is a recursive
`parse_if_statement` inside the else scope) and the tracer's `_if_row`
rewriter (`rewriter.py:1216`: `eval_if` / `eval_then` / `eval_else` /
`eval_else_active` rows, an elif nested under the else-active guard) +
`interpreter.eval_if`/`eval_else`/`_merge_branch_locals` (`interpreter.py:733`,
`:759`, `:797`) + `PtoIfConstruct` (`constructs.py:210`).

**Regime first.** Every anchored spelling below is **raw**-equal: plain
syntax leaks locals on both sides (no `pl.yield_`, so no `return_var`/phi is
built by either frontend; the chain nests structurally on both sides).
`canonical=True` is asserted *as well* wherever the IR carries staged
control flow (the redundant stronger form — raw equality already implies it
for deterministic passes). The one variant that cannot be an anchor is the
host predicate: same source, deliberately different IR (`docs/design/01-language.md`
§5), so the anchor rule does not apply and a shape-pinning test replaces it.

| Variant | Regime | Verdict |
| --- | --- | --- |
| `if` / `else`, per-branch locals (existing `k_merge`, `k_else_passthrough`, `k_nested`) | raw + canonical | equivalent (pre-existing anchors, unchanged) |
| `if` / `else` / `elif` chain with trailing `else` (`k_elif_chain`) | raw + canonical | equivalent |
| `if` / `elif` chain with no `else` (`k_elif_no_else`) | raw + canonical | equivalent |
| `if` / `elif` / `else` chain nested in a staged `while` (`k_elif_in_while`) | raw + canonical | equivalent |
| `if` / `else` nested in a staged `while` (`k_if_else_in_while`) | raw + canonical | equivalent |
| `if` / `else` nested in the for-form `pl.spmd` body (`k_if_in_spmd_for`) | raw + canonical | equivalent |
| `if` / `else` nested in the with-form `pl.spmd` body (`KIfInSpmdWith`, program-level) | raw + program-canonical | equivalent |
| empty then-branch (`pass`) (`k_pass_then_branch`) | raw | equivalent |
| host predicate (`gate = 4; if gate > 2:`) | n/a (no anchor possible) | **intended divergence** — `docs/design/01-language.md` §5 line 201 (*'Value-directed staging uses the kind of the predicate or iterable'*) + line 226 (*'A host predicate (plain Python int/bool) → host if: one path only'*) + §11 line 595 (*'host predicate → one path, passes through'*). Tracing emits no `IfStmt` (taken path only); the parser stages every `if` with a constant condition. Pinned by `test_host_predicate_dispatch_divergence`. |
| explicit-yield `if` (`pl.intrin.carry`/`yield_`/`leak` vs `a = pl.yield_`) | n/a | **out of scope — no shared spelling**: the assembly channel is tracing-owned (`pl.intrin.carry`/`leak`; the parser reports `Unknown operation 'pl.intrin'`), the result spelling is parser/round-trip-owned (`a = pl.yield_(...)`; tracing's `_YieldMarker` has no `unwrap`). Dialect boundary per `docs/design/01-language.md` §14 line 671. |

No red anchor, therefore no `F*` task and no frontend edit. The three
gap hypotheses in section 5's R5 row were all confirmed (elif chains,
host-vs-staged dispatch, nesting in `while`/`spmd`) — the two anchoring
surprises were that the elif chain is raw-equal (not merely
post-canonical) and that `pl.spmd` nesting needs no new regime.

## 11. Round ledger

| Round | Task | Model | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | seed this tracker | — (orchestrator) | done | `d60a4b303` | plan, contract, evidence base |
| 0 | parity inventory oracle | — (orchestrator) | done | *(this commit)* | `scripts/review/check_construct_parity.py`; 0 stale, 10/16 constructs without anchors, 4 unclaimed structural parser handlers, 17 unclaimed public tracing entries. The gap table in section 5 is now data, not hypothesis |
| 1 | R5 `if` / `else` / `elif` | worker session | done | *(this commit)* | 13 new anchors in `test_if.py` (suite 7 → 20 call sites; 70 → 83 across `test_*.py`), all green. **No `F*` task**: no red anchor. One intended divergence pinned (host predicate dispatch). |
