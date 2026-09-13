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

## 11. R6 verdicts — `for` + iterators

Both sides read before verdicting: the parser's `parse_for_loop`
(`ast_parser.py:2617`), `_parse_for_loop_target` (`:2568`), `_parse_range_call`
(`:2837`) and `_parse_range_keyword` (`:2808`) — plus its `pl.unroll` bound/step
validation (`:2677`, `:2684`) — and the tracing lane's `_for_row` rewriter
(`rewriter.py:1386`) + `_route_target` (`:669`) +
`RangeIterator`/`_LoopTargetBinding` (`constructs/loops.py:94`, `:47`) +
`SemaFrame.create_for_construct` (`semantics.py:1446`) +
`PtoForConstruct.activate_target` (`constructs.py:332`).

**Regime first.** Every anchored spelling below is a staged loop —
`pl.range`/`pl.parallel`/`pl.pipeline` are *always* staged, even with host int
bounds (`docs/design/01-language.md` §8 line 383), and `pl.unroll` is a trace
iterable too. So each anchor is asserted **raw**-equal *and*
post-`canonicalize()`-equal. Raw equality holds because neither lane emits
`iter_args` for a plain loop: on both sides the loop-carried value is the
self-referential `AssignStmt` update to a mutable local, and the parser's
iter_args path is reachable only through `init_values=` (which tracing does not
have). Canonical is the staged-flow regime: the SSA pass derives the same
loop-carried value from the same update on both sides.

| Variant | Regime | Verdict |
| --- | --- | --- |
| `pl.range` 1-arg / staged bound (`k_loop_accum`, `k_loop_staged_bound`) | raw + canonical | equivalent (pre-existing anchors, unchanged) |
| `pl.range(start, stop)` (`k_range_start_stop`) | raw + canonical | equivalent |
| `pl.range(start, stop, step)` (`k_range_start_stop_step`) | raw + canonical | equivalent |
| computed bounds over a staged Scalar, `pl.range(n - 2, n + 2, 2)` (`k_range_computed_bounds`) | raw + canonical | equivalent |
| `pl.parallel(4)` / `pl.parallel(0, 4, 2)` (`k_parallel_loop`, `k_parallel_three_arg`) | raw + canonical | equivalent |
| `pl.unroll(4)` / `pl.unroll(0, 6, 2)` (`k_unroll_loop`, `k_unroll_three_arg`) | raw + canonical | equivalent |
| `pl.pipeline(8, stage=4)` / `pl.pipeline(0, 8, 2, stage=2)` (`k_pipeline_loop`, `k_pipeline_three_arg_stage`) | raw + canonical | equivalent |
| `attrs={...}` on range, and `stage=` merged with `attrs` on pipeline (`k_range_attrs`, `k_pipeline_attrs_stage`) | raw + canonical | equivalent |
| round-trip INT64 bound `pl.range(pl.const(4, pl.INT64))` (`k_range_int64_const_bound`) | raw + canonical | equivalent **for-syntax**: the parser's round-trip inference types the loop var INT64, tracing always types it INDEX, and the oracle declares INT64/INDEX equivalent in a loop-var (and ConstInt) position (`AreForSyntaxScalarDtypesEquivalent`, `src/ir/transforms/structural_equal.cpp:50`, applied at `:1047`). The raw IR differs in that one field only. |
| staged `for` nested in staged `for` (`k_nested_for`), `for` inside a staged `while` (`k_for_in_while`) | raw + canonical | equivalent |
| iter_args tuple target `for i, (v,) in pl.range(..., init_values=(...))` (`k_iter_args_target`) | n/a | **out of scope — no shared spelling**: the tracing surface of `pl.range`/`pl.parallel`/`pl.pipeline` deliberately has no `init_values=` (`docs/design/01-language.md` §8 line 415; §14 line 674 — the loop-carried channel is `pl.intrin.carry`). Parser accepts (round-trip ownership), tracing rejects at construction. Pinned by `test_iter_args_tuple_target_is_parser_only`. |
| bare tuple target `for i, j in pl.range(4)` (`k_bare_tuple_target`) | n/a | **out of scope — both lanes reject**: parser `Iter args must be a tuple` (`_setup_iter_args`, `ast_parser.py:2591`); tracing `cannot unpack non-iterable _LoopTargetBinding` (the target binding is one proxy, `loops.py:47`). Pinned by `test_bare_tuple_target_rejected_by_both_lanes`. |
| `stage=` on a non-pipeline iterator, missing `stage=`, zero args, `unroll=` kwarg | n/a | out of scope — both lanes reject; no accepted spelling exists to anchor (tracing rejects by signature, the parser by `_parse_range_keyword`). |

All 28 new anchors land green (suite 7 → 35 call sites; conformance dir
83 → 111). No red anchor, so no `F*` task from the anchor rule and no frontend
edit. On "target forms (single, tuple, computed)" from section 5's R6 row: the
parser's `_parse_for_loop_target` (`ast_parser.py:2568`) accepts exactly a
simple name or a two-element `(loop_var, iter_args)` tuple, so "single" and
"tuple" are the whole target grammar; "computed" is the computed
bound/offset surface (the `n - 2`/`n + 2` bound anchor here and the pre-existing
`k_gemm_tile_loop`'s loop-var-into-offset `i * 64`), which both lanes share.

**Canonical-regime caveat (recorded, not a divergence).** The canonical pair
runs `convert_to_ssa` + `simplify` *without* `UnrollLoops`, which a real
pipeline runs first. For the two `pl.unroll` anchors SSA therefore derives
`iter_args` onto the surviving Unroll marker — an artifact state both lanes
reach identically; the roundtrip instrument's "IR not printable" warning is
filtered in the two tests as in `tests/ut/ir/transforms/test_unroll_loops_pass.py`.

### R6 no-anchor findings (acceptance-side divergences, needs-decision)

These have no anchor because one lane rejects the spelling. They are
divergences in the construct's logic, recorded here; nothing was edited.

1. **Non-string `attrs=` key leaks an internal C++ error.** `pl.range(4,
   attrs={1: 2})` (same for `pl.parallel`/`pl.unroll`/`pl.pipeline`) raises
   `RuntimeError: std::bad_cast` from `ConvertKwargsDict`
   (`python/bindings/modules/ir.cpp:114`, reached from `RangeIterator.begin`,
   `constructs/loops.py:181`), while the parser raises
   `ParserSyntaxError: attrs keys must be string literals` (`ast_parser.py:2893`).
   The tracing factory forwards the user's dict unvalidated. Filed as **F11**.
2. **`pl.unroll` bound/step validation is parser-only.** `pl.unroll(n)` with a
   staged bound and `pl.unroll(0, 4, 0)` are accepted by tracing and rejected
   by the parser (`ast_parser.py:2677`, `:2684`). The C++ pass still fails
   loudly later (`UnrollLoops: <what> must be a compile-time integer constant`,
   `src/ir/transforms/unroll_loops_pass.cpp:68`; `Unroll loop step cannot be
   zero`, `:72`), so this is error-*timing* parity, not a correctness gap.
   needs-decision: no doc intends tracing to skip the early check.
3. **`stage=` accepts less on the tracing side.**
   `pl.pipeline(8, stage=pl.const(2, pl.INT64))` is accepted by the parser
   (`_is_const_int` sees the ConstInt) and rejected by tracing's `_pin_stage`
   (host `int` required, `constructs/loops.py:233`). No round-trip pressure
   (the printer emits a bare int), so this is an ergonomics-spelling
   acceptance gap only. needs-decision.
4. **The parser ignores extra positional range args.** `pl.range(0, 4, 2, 9)` is
   accepted by the parser (`_parse_range_call` reads the first three,
   `ast_parser.py:2862`) and rejected by tracing (`ValueError: range() takes 1
   to 3 positional arguments`). A parser-side arity gap, not a tracing gap.
   needs-decision.

### Discovered task

- **F11 — a non-string `attrs=` key reaches nanobind and raises
  `RuntimeError: std::bad_cast` instead of a loud frontend error.** Found in
  R6 (finding 1 above); not an anchor red, so it is a discovered task rather
  than a construct divergence. Red check: trace any kernel containing
  `for i in pl.range(4, attrs={1: 2}):` (a `/tmp` module run with `uv run
  python`) → `RuntimeError: std::bad_cast`; the parser lane raises
  `ParserSyntaxError: attrs keys must be string literals`. Owner undecided
  (needs-decision on the layer): validate the key type in the Python
  frontend's attr pass-through (`constructs/loops.py`, and the sibling
  attrs-taking entries) or make `ConvertKwargsDict`
  (`python/bindings/modules/ir.cpp:114`) raise `TypeError`. Frontend-only
  fixes are the migration's rule. F-ids continue `docs/review-r10.md` §4's
  series (last F10).

## 12. R7 verdicts — `break` / `continue`

Both sides read before verdicting: the parser's `parse_break`/
`parse_continue` (`ast_parser.py:6123`, `:6133`) + `_validate_loop_control`
(`:6143`) + the `_loop_kind_stack` the loop handlers push (`:2761` for-form,
`:3551`/`:3615` while, `:835` the `pl.split_aiv` sentinel), and the tracing
lane's `interpreter.break_`/`continue_` (`interpreter.py:943`, `:956`) +
`_transfer` (`:961`, nearest loop *construct*) +
`PtoForConstruct._validate_transfer` (`constructs.py:384`)
/ `PtoWhileConstruct.on_break` (`:517`) + the rewriter's
transfer guard (`_native_transfer_row` / `_escapes_transfer` /
`_lifetime_with`, `rewriter.py:1340`, `:1283`, `:1273`).

**Regime first.** Every anchored spelling below is a *staged* loop
(`docs/design/01-language.md` §8), so each is asserted **raw**-equal *and*
post-`canonicalize()`-equal. Raw is the load-bearing half: a staged transfer
suppresses the native keyword (`if rt.break_(): break`) and both lanes record
a plain `BreakStmt`/`ContinueStmt` inline in its region. Canonical is the
staged-flow regime proper (the SSA pass derives the loop-carried values of a
transfer-containing loop). The *deferred staged exit ordering* shows up in
both regimes as structure, not as an ordering event: the branch Construct
closes at its lexical boundary (rewriter's unwind scope guard) *before* the
loop's, so the `BreakStmt` lands inside the branch body and the statements
after the branch stay in the loop body. The guard's native-unwind arm (a
`break` that actually unwinds) only fires when the target is a *host* loop,
and a host loop is not an anchorable target: the parser accepts only `pl.*`
iterators for `for` (a native `range`/list is rejected), and a host `while`
carries host-valued state — a separate, unshared regime. That arm is behaviorally covered in
`tests/ut/language/test_collector_contract.py`
(`test_native_break_cleans_stale_constructs_in_staged_if`), never anchorable.

| Variant | Regime | Verdict |
| --- | --- | --- |
| bare `break` / `continue` in a staged `for` (existing `k_loop_break`, `k_loop_continue`) | raw + canonical | equivalent (pre-existing anchors, unchanged) |
| bare `break` in a staged `while` (existing `k_while_break`) | raw | equivalent (pre-existing anchor, unchanged) |
| `break` / `continue` inside a staged `if` inside a staged `for` (`k_break_in_if_in_for`, `k_continue_in_if_in_for`) | raw + canonical | equivalent |
| `break` / `continue` inside a staged `if` inside a staged `while` (`k_break_in_if_in_while`, `k_continue_in_if_in_while`) | raw + canonical | equivalent |
| `for` > `while` > `if` > `break`, three Constructs around one transfer (`k_deep_nest_break`) | raw + canonical | equivalent |
| inner/outer target selection in nested `for`s (`k_nested_for_inner_break`, `k_nested_for_inner_continue`, `k_nested_for_outer_break_after_inner`) | raw + canonical | equivalent |
| `while`-in-`while`, `while`-in-`for`, `for`-in-`while` innermost targeting (`k_while_in_while_inner_break`, `k_while_in_for_inner_break`, `k_for_in_while_inner_break`) | raw + canonical | equivalent |
| bare `continue` in a staged `while` (`k_while_bare_continue`) | raw + canonical | equivalent |
| statements after a staged `break` / `continue` still collected (`k_stmts_after_break`, `k_stmts_after_continue`) | raw + canonical | equivalent |
| gating is the *innermost* loop's kind: sequential inner loop under `pl.parallel` / `pl.unroll` (`k_range_in_parallel_break`, `k_range_in_unroll_continue`) | raw + canonical | equivalent |
| `break` in a `pl.pipeline` loop (`k_pipeline_break`) | raw + canonical | equivalent |
| `break` in the for-form `pl.spmd` body *nested in a loop* (`k_break_in_spmd_for_nested_in_for`) | raw + canonical | equivalent (both lanes record it in the Spmd region) |
| `break`/`continue` in `pl.parallel` / `pl.unroll` | n/a (both lanes reject) | out of scope — no accepted spelling; the one gate both lanes share verbatim (pinned by `test_parallel_unroll_transfer_rejected_by_both_lanes`) |
| `break`/`continue` in a **top-level** for-form `pl.spmd` body or a `pl.split_aiv` region | n/a (parser rejects, tracing accepts) | **bug — F12**, fixed: both lanes reject (the spmd for-descriptor is crossed, never a transfer target) |
| a transfer into a staged loop's `else` suite; `for`/`while` + `else` generally | n/a (both lanes reject) | **bug — F13**, fixed: the parser rejects the suite the way tracing does (no shared spelling — the pair is pinned as a rejection) |

36 new anchors in `test_break_continue.py` (18 kernels, raw + canonical;
suite is new, so conformance-directory call sites 111 → 147), all green. No
red anchor, therefore no anchor-rule `F*` task and no frontend edit.

### R7 no-anchor divergences (one bug, one parser-side bug)

1. **F12 — tracing accepts a transfer in a non-loop region.** `_transfer`
   walks the construct stack for `kind in {"for", "while"}` and
   `_validate_transfer` reads the loop *kind* off a `_LoopDescriptor`,
   falling back to `"range"` for any trace iterable without the `kind`
   surface (`constructs.py:388`). The for-form `pl.spmd` descriptor and
   `pl.split_aiv` both lack it, so a `break`/`continue` inside either is
   accepted and recorded as a `BreakStmt` inside a region that lowers to a
   *scope*, not a `ForStmt`. The parser rejects both (spmd: `'break' outside
   loop` — its handler pushes no loop kind; split_aiv: `'break' not supported
   inside a 'for ... in pl.split_aiv(...)' body`). The recorded break also
   breaks the print→parse fixpoint: `pl.parse(str(traced))` raises the same
   parser error. Red check: trace `for i in pl.spmd(4): ...; break` or
   `for aiv in pl.split_aiv(2, mode=...)` with a `break` → tracing accepts
   and the printed text re-parses to `InvalidOperationError`; the parser
   rejects the source. Pinned by
   `test_split_aiv_and_spmd_transfer_divergence` (rewritten into the fix's
   expectation as `test_split_aiv_and_spmd_transfer_rejected_by_both_lanes`).
   Owner: the tracing lane
   (`PtoForConstruct._validate_transfer` should reject a transfer when the
   descriptor is neither a loop nor a sequential range). The nested
   `pl.spmd`-in-a-loop spelling keeps a stale outer entry on the parser's
   kind stack and is equivalent on both lanes (anchored above) — so the fix
   must not assume the two lanes' *nearest-target* rules coincide.
   **State: fixed** (fix-campaign task A1) — the pin is now
   `test_split_aiv_and_spmd_transfer_rejected_by_both_lanes`; the spmd
   for-descriptor is crossed rather than rejected (see the discovered-task
   entry below).
2. **F13 — the parser silently drops a loop `else` suite.**
   `parse_for_loop`/`parse_while_loop` never read `stmt.orelse`, so
   `for ...: ... else: ...` parses to IR without the else body (silent code
   loss), while tracing rejects it loudly with
   `InterpreterError: a staged for loop cannot have an 'else' suite`
   (the documented behavior, `docs/design/01-language.md` §8 *'Loop `else`
   belongs to host loops only'*, §12 guardrail row *'staged loop `else`'*;
   tracing side pinned by `test_staged_for_else_rejected`). Red check: parse
   the kernel in `test_loop_else_rejection_vs_parser_silent_drop` → the else
   body's `mul` is absent from the parsed IR. Pinned by that test (rewritten
   into the fix's expectation as `test_loop_else_rejected_by_both_lanes`). Owner:
   the parser lane (reject the suite the way tracing does, or spell `else`);
   the migration's rule is frontend-only, and a parser-lane fix is out of
   this review's scope. **State: fixed** (fix-campaign task B1) — the pin is
   now `test_loop_else_rejected_by_both_lanes`: the parser gained
   `_reject_loop_else` (`ast_parser.py`), called from `parse_for_loop` after
   iterator validation (so it covers all seven iterator spellings, `pl.spmd`
   and `pl.split_aiv` included) and from `parse_while_loop` before the
   condition is read. Because the parsed dialect has no host world, every
   loop it accepts is staged, so the rejection is unconditional (no `break`
   required) — exactly the tracing rule, whose `for`/`while` message the
   parser mirrors.

### Discovered tasks

- **F12 — a `break`/`continue` in a `pl.spmd` (top-level) / `pl.split_aiv`
  body is accepted by tracing and emitted inside a non-loop scope** (finding
  1 above; not an anchor red because the parser rejects the spelling). Red
  check and pin as above. Frontend-only. **State: fixed** (fix-campaign task
  A1). The pin is rewritten to the correct expectation
  (`test_split_aiv_and_spmd_transfer_rejected_by_both_lanes`): both lanes
  reject a transfer in a top-level spmd for-body (`'break' outside loop`) and
  in a `pl.split_aiv` region at any nesting (`'break' not supported inside a
  'for ... in pl.split_aiv(...)' body`). The fix is the interpreter's
  transfer-target walk crossing a Construct whose descriptor lowers to a
  scope (`Construct.transfer_target_action` → CROSS, rejected by
  `on_unbound_transfer` when no enclosing loop exists) instead of
  `_validate_transfer`'s kind fallback to `"range"` — so the spmd
  for-descriptor is *transparent* (the nested anchor stays green: the walk
  reaches the enclosing loop) and the `pl.split_aiv` region rejects itself.
- **F13 — the AST parser silently drops a `for`/`while` `else` suite**
  (finding 2 above). Red check and pin as above. **State: fixed**
  (fix-campaign task B1). The pin is rewritten to the correct expectation
  (`test_loop_else_rejected_by_both_lanes`): tracing still rejects at the
  loop's close (`Interpreter.exit_construct(else_follows=True)`), and the
  parser now rejects at the loop header (`_reject_loop_else`, a
  `ParserSyntaxError` whose message carries tracing's
  "cannot have an 'else' suite" clause) instead of parses-and-drops. The
  suite has no `ForStmt`/`WhileStmt` representation, and the parsed dialect
  has no host loop for it to belong to — so the parser's rule is the
  unconditional form of tracing's staged-only rule.

F-ids continue `docs/review-r10.md` §4's series (last F11, from R6).

## 13. R8 verdicts — binding & assignment

Both sides read before verdicting: the parser's `parse_assignment`
(`ast_parser.py:1834`), `parse_annotated_assignment` (`:1530`),
`_assign_or_let` (`:1774`), `_parse_subscript_assignment` (`:2209`),
`_parse_array_subscript_assignment` (`:2402`),
`_build_assemble_offsets_and_extents` (`:2442`), and the
`_parse_typed_constant` interceptor (`:9349`); the tracing lane's
`Sema.bind`/`_bind_impl`/`_bind_dispatch`/`_default_bind`
(`semantics.py:874`, `:892`, `:947`, `:1166`) + `_rebind_changes_type`
(`:1226`) + `PtoFrame.rt_bind`/`resolve_staged_local` (`:1697`, `:1268`),
`Tensor.__setitem__`/`Tile.__setitem__`/`Array.__setitem__`
(`typing/tensor.py:309`, `typing/tile.py:247`, `typing/array.py:140`),
`subscript_write_offsets` (`typing/_subscript.py:286`), the tuple-unpack
`_EmitState`/`_structural_projection` (`typing/_emit.py`), and
`typing/const.py`.

**Regime first.** Straight-line binds and subscript writes are asserted
**raw**: both lanes emit the identical `LetStmt`/`AssignStmt` chain with no
pass state. Every staged anchor (a rebind inside a staged loop, the
`pl.const` accumulator, a subscript write inside a loop) is asserted **raw
*and* post-`canonicalize()`**: raw holds because neither lane emits
`iter_arg` for a plain loop (the loop-carried value is the self-referential
`AssignStmt` update), and canonical is the staged-flow regime proper. The
one exception is the type-changing rebind inside a loop — raw-equal, but its
canonical half is red on **both** lanes (F14).

| Variant | Regime | Verdict |
| --- | --- | --- |
| tensor subscript write, partial / bare / full-axis (`k_tensor_write_*`) | raw | equivalent |
| tile subscript write (`k_tile_write`) | raw | equivalent |
| array write, const and staged index (`k_array_write_*`) | raw | equivalent |
| subscript write inside a staged `for` (`k_tensor_write_in_loop`) | raw + canonical | equivalent |
| scalar-index rank-reducing write (`out[0, :] = src`) | n/a | **needs-decision** — tracing rejects loudly, parser accepts (`_build_assemble_offsets_and_extents` + `_lift_subscript_write_source_rank`); the tracing docstring records the omission but no design doc/manifest intends it. No shared spelling, so no anchor. Pin: `test_scalar_index_write_has_no_shared_spelling`. |
| same-type rebind in a staged `for` / `while` (`k_rebind_in_for`, `k_rebind_in_while`) | raw + canonical | equivalent |
| type-changing rebind, straight-line (`k_typechange_rebind`) | raw | equivalent |
| type-changing rebind inside a staged loop (`k_typechange_rebind_in_loop`) | raw equivalent; canonical **red** | **bug — F14** |
| rebind of a parameter name (`k_param_name_rebind`) | raw | equivalent |
| typed staged init: bare literal / demotion inside staged flow | n/a | **intended divergence** — `docs/design/01-language.md` §4 lines 146–178 (*'a host literal cannot initialize staged state that must survive a control-flow join… the guard is symmetric: staged-ness is stable while staged control flow is open'*) + §7 lines 302–305. Tracing is loud; the parser has no host world in a body and accepts. Pinned by `test_typed_staged_init_bare_literal_rejected_by_tracing` / `test_typed_staged_init_demotion_rejected_by_tracing`. |
| defs / store-target first-encounter order (`k_defs_store_order`) | raw | equivalent (frontend emission order). The cited `136a0f457` is a C++ **pass** determinism fix (`inline_functions_pass` / `scope_outline_utils` iterated pointer-keyed containers whose hash order follows ASLR) — out of the frontend comparison's regime (the harness runs neither outlining nor a pointer-keyed walk). Its IR contract is pinned by `tests/ut/ir/transforms/test_outline_incore_scopes.py` (the store-target walk is encounter-ordered) and by the inline-regime anchors in `test_inline.py`; the frontend residue (statement emission order) is the anchor. |
| `pl.const` straight-line (`k_const_straightline`) | raw | equivalent |
| `pl.const` staged init accumulator (`k_const_loop_init`) | raw + canonical | equivalent |
| tuple unpack of a multi-result op (`k_tuple_unpack_gather`) | raw | equivalent — both name the shared node once as a `_tuple_tmp` `Let` and bind `TupleGetItemExpr` projections; the temp *name* differs (`_tuple_tmp` vs `_tuple_tmp_1`) and auto-mapping ignores it |
| parameter world: `pl.Scalar[pl.INT32]` param (`k_scalar_param`) | raw | equivalent |
| parameter world: plain `int` param | n/a | both lanes reject — no accepted spelling; pin `test_host_param_world_rejected_by_both_lanes` |
| `pl.runtime` | n/a | **out of scope — no shared spelling**: it is the jit **entry-call** marker for a runtime scalar (`typing/_runtime_scalar.py`), never kernel source. In a body tracing cannot unwrap it (annotation-only) and the parser reports `Unknown operation 'pl.runtime'`. Entry-call world pinned by `tests/ut/jit/test_param_worlds.py`. Pin: `test_runtime_marker_has_no_kernel_body_spelling`. |

22 new anchors in `test_binding_assignment.py` (conformance-directory call
sites 147 → 169), all green; the suite's 23 tests pass. The one red
canonical regime (F14) is pinned by an explicit `pytest.raises` test rather
than kept as a red anchor, recorded below without a frontend edit, alongside
three no-shared-spelling/divergence pins.

### R8 no-anchor finding

1. **F14 — a type-changing rebind inside a staged loop emits a
   loop-body-local fresh `let`, so the post-loop name resolves to a Var
   defined inside the loop.** `k_typechange_rebind_in_loop`
   (`a = load; for: a = reshape(a, [1,4096]); a = reshape(a, [64,64]);
   store(a)`) traces and parses to the byte-identical body — the first
   reshape starts a fresh `let` (type change), the second another; the
   post-loop `store(a)` then names the second body-local. `ConvertToSSA`
   builds the loop `init_values` from a free variable
   (`init_values=(a__FREE_VAR,)`) and its post verification rejects the IR
   (`SSAVerify: 'a' used outside its defining scope`; `UseAfterDefCheck:
   'a' used before definition`). Both frontends (parser `_assign_or_let`,
   tracing `Sema._default_bind`) produce it identically — a *shared* §7
   relaxed-reassignment hole, not a parity divergence; the same-type
   rebind path assigns through and canonicalizes fine, and the
   straight-line type change canonicalizes fine. Red check:
   `canonicalize(function_trace(k_typechange_rebind_in_loop))` (under the
   unit-test verification context) raises the `ConvertToSSA`
   post-verification error; `function_parser` of the same kernel raises it
   identically. Pinned by
   `test_typechange_rebind_in_loop_shared_free_var`. Owner undecided
   (needs-decision on the layer): make the frontend rebind loop-carried
   (an `AssignStmt` to the pre-loop Var with a fresh typed Var, or a loud
   rejection of a type-changing rebind inside staged control flow), or
   make the SSA pass's scope handling reject/repair it. Frontend-only fixes
   are the migration's rule. F-ids continue from F13.

## 14. R9 verdicts — returns / exits

Both sides read before verdicting: the parser's `parse_return`
(`ast_parser.py:6039` — the `ast.Tuple` arm at `:6050`, the inline-mode
`_inline_return_values` capture at `:6093`, and `parse_statement`'s dispatch at
`:1197`, which does **not** stop at a `return`), `_parse_inline_call`
(`:8488`); the tracing lane's rewriter `_return_row` (`rewriter.py:1625`:
the value temp + `rt.return_` + the `if rt.in_host_mode(): return` guard) →
`interpreter.return_` (`interpreter.py:658`) → `Sema.return_` /
`_return_impl` (`semantics.py:1348` / `:1379`, whose staged-element filter at
`:1404` is load-bearing) + the declared-`->` check
(`tracer/decorators/function.py:797`, `tracer/return_contract.py`). The B3 fix
extracted the inline expansion seam into `_expand_inline_call` +
`_inline_callee_return_values` (`:8532`–`:8600`), which `parse_return` uses.

**Regime first.** The trailing-return anchors are **raw**: a straight-line
return carries no pass state, and both lanes emit the identical one-value /
multi-value / void `ReturnStmt`. Every return inside staged control flow is
asserted **raw *and* post-`canonicalize()`**: raw is the interesting half
(both lanes record the `ReturnStmt` inline, at its source position, with no
return_var or phi), canonical is the staged-flow regime proper (the SSA pass
joins the per-path returns of a function whose branches each return).

| Variant | Regime | Verdict |
| --- | --- | --- |
| trailing single return (`k_trailing_return`) | raw | equivalent |
| multi-value `return a, b` (`k_multi_value_return`) | raw | equivalent |
| staged scalar return (`k_scalar_return`) | raw | equivalent |
| bare `return` (`k_bare_return`) | raw | equivalent |
| no return at all (`k_no_return`) | raw | equivalent |
| return in an `elif` chain (`k_ret_elif`) | raw + canonical | equivalent |
| return in an `if` nested in a staged `for` (`k_ret_nested_if_in_for`) | raw + canonical | equivalent |
| return in a staged `for` with a body tail (`k_ret_loop_tail`) | raw + canonical | equivalent |
| return in a staged `if` with a branch tail (`k_ret_branch_tail`) | raw + canonical | equivalent |
| return in the for-form `pl.spmd` body (`k_ret_spmd_for`) | raw + canonical | equivalent |
| return in a `pl.at` region (`k_ret_at_region`) | raw + canonical | equivalent |
| return in a `pl.at` region with a tail (`k_ret_at_region_tail`) | raw + canonical | equivalent |
| function-level early return + unreachable tail (`k_early_top`) | raw | equivalent — **F15**, fixed by B2 (both lanes stop at the function-level return) |
| second function-level return (`k_two_top_returns`) | raw | equivalent — **F15**, fixed by B2 (same root cause) |
| host-valued return element (`return 1.0`, `return t, 1.0`) | n/a | **needs-decision** |
| list-literal return (`return [a, b]`) | n/a | **needs-decision** |
| `return pl.Tuple(a, b)` | n/a | out of scope — no shared spelling |
| declared `->` lie, or a declared `->` with no return | n/a (tracing rejects, parser accepts) | **intended divergence** — `docs/design/04-migration-guide.md` §4.4 line 241 |
| `@pl.expand` call site | n/a (parser rejects) | out of scope — no shared spelling (`docs/design/03-entrance.md` §7) |
| `@pl.expand` body semantics | n/a | equivalent witness (expansion ≡ the written-out body) |
| inline single trailing return (the 4 pre-existing `_inline_anchor`s) | post-inline + canonical | equivalent (pre-existing, unchanged) |
| inline multi-value trailing return | raw + post-inline canonical | equivalent — **F16**, fixed by B3 (both lanes wire the callee's trailing values into a two-value `ReturnStmt`) |
| inline multi-value trailing return through a tuple temporary (`tmp = (a, b); return tmp`) | raw + post-inline canonical | equivalent — **F37**, fixed by the fix campaign (R4: the parser's one-`MakeTuple`-value reading is normative, so the tracing lane stages a tuple of staged facades as one tuple value) |
| inline early return under staged control flow | n/a (both lanes reject) | **bug — F17**, fixed by B4 (the parser rejects the nested return the way both tracing entrances do) |
| a second top-level return after the inline callee's trailing return (dead tail) | raw | equivalent — **F38**, fixed by the fix campaign (the inline body walk stops at the first top-level return, F15's rule, and the nested-return scan is scoped to the same collected prefix) |

19 new anchors in `test_return_exit.py` (12 kernels; conformance-directory
call sites 169 → 188; conformance-dir tests 288 → 310), all green, plus eight
pins in the same suite and two pins in `test_inline.py`. The B2 fix turned
F15's two pins into two more **raw** anchors in the same suite (six pins
left there); the B3 fix turned F16's pin into
`test_inline_multi_value_trailing_return_by_both_lanes` (raw shape +
post-inline canonical equality) plus the canonical anchor
`_inline_multi_value_anchor`, and filed F37 as a new pin. Each of the three
red findings was pinned, never kept red — F15's and F16's pins are now
anchors, F17's is now a rejection pair. The B4 fix turned F17's pin into
`test_inline_early_return_rejected_by_both_lanes` and filed F38 as a new pin.
The F37 fix turned that pin into
`test_inline_temp_tuple_trailing_return_by_both_lanes` (raw shape +
post-inline canonical equality). The F38 fix turned its pin into
`test_inline_dead_tail_after_trailing_return_by_both_lanes` and added the
dead-nested-return anchor `test_inline_dead_tail_nested_return_by_both_lanes`
(both raw shape on both lanes + raw equality).

The one *intended divergence* is the declared-`->` contract: tracing's
`_check_declared_return` rejects a declared return type that lies about the
traced body (or a declared `->` with no return) at entry —
`docs/design/04-migration-guide.md` §4.4 line 241: *'Function return
contracts are the exception: a declared `->` that lies about what the body
produces is a loud decoration-time error.'* The parser is the round-trip
owner and accepts both spellings, so no anchor exists; pinned by
`test_declared_return_contract_is_tracing_only`.

### R9 findings (all frontends left alone except F15's B2 fix)

1. **F15 — a statement after a function-level `return` diverges.** The
tracing lane's host-execution guard fires at the function level
(`rt.in_host_mode()` is True when no staged Construct is open), so
`_return_row`'s native `return` ends the trace and the trailing statements
are never collected; `parse_statement` (`ast_parser.py:1197`) keeps
dispatching the whole body, so the parser emits them as unreachable IR. A
return nested in staged control flow (branch/loop/region) is *not* affected —
there both lanes keep collecting (the anchors above). Red checks:
`function_trace(k_early_top)` has 3 body stmts vs the parser's 5 (and
`k_two_top_returns` 3 vs 4); `assert_structural_equal` raises in both raw and
canonical regime. Pinned by
`test_function_level_return_drops_unreachable_tail` /
`test_function_level_double_return_same_root_cause`. Owner undecided
(needs-decision on the layer): the tracing guard is intended and faithful to
Python execution, which points at the parser ceasing collection at a
function-level return; but the parser's faithful-source reading is
defensible and no doc decides. Frontend-only fix either way.
**State: fixed** (fix-campaign task B2) — the lane decision landed as the
finding's own lean: the **parser** is the wrong lane. §9's *'Trace-time
execution determines the IR'* (`docs/design/01-language.md`) makes the traced
execution the kernel's meaning, and the rewriter's native return is
load-bearing beyond the parser's reach (`Interpreter.return_`:
*'the host ``return`` carries the value in production'* — it is what hands a
value back to an `@pl.expand` caller or a host-evaluated helper), so tracing
cannot be made to execute a tail Python never reaches; a return inside a
*host* construct is the separate, documented host-world divergence
(`test_host_predicate_dispatch_divergence`). `_parse_body_siblings` gained
`stop_after_return`, passed only by `parse_function`: a `return` that is a
direct statement of the function body ends collection (no staged Construct is
open there, so the host guard fires and the tail is unreachable). Nested
blocks keep the default: a return inside a staged construct suppresses the
host exit and both lanes keep collecting (the branch/loop/region anchors
above). Pins rewritten into the fix's expectation as
`test_function_level_return_drops_unreachable_tail_by_both_lanes` and
`test_function_level_double_return_dropped_by_both_lanes`.
2. **F16 — an inline callee's multi-value trailing return diverges.** The
backend splice wires the callee's trailing values into the caller as a
*multi-value* `ReturnStmt` (the pass's
`inline_functions_pass.cpp` return wiring), while the parser's parse-time
expansion collapses the same tuple into one `MakeTuple` value
(`_inline_return_expr`'s `ast.Tuple` arm, `ast_parser.py:6052`) — so
`return helper(x)` returns two values on one lane and one tuple value on the
other. The parser's *own* non-inline `return a, b` is multi-value, so the
inline expansion changes the caller's return arity relative to the
un-expanded body; `docs/design/01-language.md` §9 line 503 (*'neither
frontend creates a fictional staged-tuple value'*) points at the parser lane.
Red check: `test_inline_multi_value_trailing_return_divergence` — spliced
`ReturnStmt` has 2 values, the parser's 1 `MakeTuple`, structural equality
raises. **State: fixed** (fix-campaign task B3) — the lane decision landed as
the finding's own lean, the **parser** lane: the AST caller is the
parse-time-spliced body, so a `return helper(x)` whose trailing return is
multi-value *is* a written-out `return a, b` and has to stay multi-value.
`parse_return`'s value arm now routes through `_parse_return_values`, which
takes an `@pl.inline` callee's expansion as the callee's value list
(`_inline_callee_return_values` → `_expand_inline_call`, the extracted
expansion `_parse_inline_call` also uses); the value-position reading is
unchanged, `_parse_inline_call` still wraps a multi-value expansion into a
`MakeTuple` for expression/assignment positions. The pin became
`test_inline_multi_value_trailing_return_by_both_lanes` (raw arity 2 on both
lanes + raw `assert_structural_equal`, then the post-inline canonical anchor
`_inline_multi_value_anchor`). The same multi-value return written through a
tuple temporary is **not** closed by this fix and was filed as F37 (fixed
later — finding 4). **Retro-R4 classification (H1):** unintended bug fix — the
collapsed one-value `MakeTuple` violated §9's explicit "neither frontend
creates a fictional staged-tuple value", so the parser was restored to the
already-documented rule; no spec change.
3. **F17 — the parser silently mis-lowers an inline early return.** An inline
callee with an early return inside staged control flow: tracing records the
per-branch `ReturnStmt` and the backend splice rejects it loudly
(`Inline function ... contains a non-trailing ReturnStmt; only a single
trailing return is supported`), while the parser's `parse_return` captures
the branch's expression into `_inline_return_expr` (renamed
`_inline_return_values` by B3's F16 fix) and then overwrites it
with the trailing return — the branch is emitted **empty** and its `add` is
silently lost (the caller's value is always the trailing return). The same
shape in `@pl.expand` is a loud `TypeError` (`docs/design/03-entrance.md`
§7: one trailing return; early returns under staged control flow rejected),
so tracing is the loud lane on both entrances. Red check:
`test_inline_early_return_loud_on_tracing_dropped_by_parser` —
`passes.inline_functions()` raises `InternalError`, while the parser's caller
has an empty `IfStmt` branch and no `tensor.adds` call at all. Owner: the
parser lane (reject the early return the way both tracing entrances do).
**State: fixed** (fix-campaign task B4). The pin is rewritten to the correct
expectation (`test_inline_early_return_rejected_by_both_lanes`, a rejection
pair: the pass's `InternalError` at splice time, the parser's
`ParserSyntaxError` at expansion time, both carrying the pass's "contains a
non-trailing ReturnStmt; only a single trailing return is supported" clause).
`_expand_inline_call` gained `_reject_inline_nested_return`, which scans the
callee's body AST for an `ast.Return` that is not a direct statement of the
body (nested in `if`/`for`/`while`/`with`), before any statement is emitted;
a return that is the body's trailing statement — including a scope-wrapped
one — stays accepted. No accepted spelling exists on either lane, so the pair
is pinned rather than anchored (the F12/F13 shape). **Retro-R4 classification
(H1):** unintended bug fix — the old expansion silently lost the branch's
`add` and both other entrances already reject the shape (`CloneInlineBody`;
`docs/design/03-entrance.md` §7), so there was no sanctioned AST-era behaviour
to preserve; no spec change.
4. **F37 — an inline callee's trailing return through a tuple temporary kept
the F16 divergence (filed by the fix-campaign task B3).** The direct spelling
is fixed (F16 above); the temp spelling was not. In the callee,
`tmp = (a, b); return tmp` traced to a flattened two-value
`ReturnStmt` (the tracing lane read `(a, b)` as a host container of facades),
so the spliced caller returned two values; the parser's expansion emits
`tmp = [a, b]` (one staged `MakeTuple`) plus `return tmp` (one value), so the
caller keeps an extra statement and a different arity. This is the
`return [a, b]` dialect collision above (needs-decision) applied to a
tuple-to-local binding — whether `tmp = (a, b)` is a host container or one
staged tuple value is exactly what the pending decision settles. Red check:
`test_inline_temp_tuple_trailing_return_divergence`. Frontend-only; was
blocked on the dialect decision. **State: fixed** (fix-campaign task F37).
The ratified rules (§4 of `docs/fix-construct-parity.md`) settle the dialect:
R4 makes the parser's reading normative wherever it accepts valid Python, so
the parser's one-tuple reading is the AST-era behaviour tracing must
preserve, while the `return [a, b]` *multi-value* spelling is a separate R1
type lie. The tracing lane's binding row now stages a tuple whose every
element is a staged facade as the staged `pl.Tuple(a, b)` value
(`_stage_staged_tuple` in `tracer/semantics.py`), so `tmp = (a, b)` binds one
`MakeTuple` and `return tmp` is one value — the parser's shape. The pin
became `test_inline_temp_tuple_trailing_return_by_both_lanes` (raw shape on
both lanes + raw `assert_structural_equal`, then the post-inline canonical
anchor). An all-host or mixed tuple, every list, and a *nested* tuple literal
(`tmp = ((a, b), c)`) still read as host containers on the tracing lane — the
same reading for spellings the task did not cover, not separately fixed.
5. **F38 — an inline callee's dead second top-level return is the F15 shape
on the inline-expansion seam (filed by the fix-campaign task B4).** In the
callee, `return a` followed by `return b`: the first return is the traced
execution's host exit (`_return_row`'s in-host-mode guard), so the traced
callee is one `ReturnStmt` carrying `a` and `CloneInlineBody` splices it; the
parser's `_expand_inline_call` keeps walking the body because the inline
walk is not `parse_function`'s `_parse_body_siblings(stop_after_return=True)`
(B2's F15 fix), and `parse_return` overwrites `_inline_return_values` on every
return, so the caller carries `b`. The two lanes disagree on which return the
caller returns. Red check:
`test_inline_dead_tail_after_trailing_return_divergence` — the spliced caller
returns `tensor.adds`, the parsed caller `tensor.muls`, and
`assert_structural_equal` raises on the op name. Frontend-only; the fix is to
give the inline body walk F15's `stop_after_return` semantics (first return
wins, dead tail never collected), so the spelling is pinned, not anchored, and
not folded into F17's rejection (tracing *accepts* this shape, so rejecting it
would be a new divergence in the other direction).
**State: fixed** (fix-campaign task F38) — the lane decision is B2's, applied
to this seam: the traced execution is the kernel's meaning, so the parser
stops the inline body walk at the body's first direct `return`
(`_expand_inline_call`'s walk and `_reject_inline_nested_return` both iterate
`_inline_body_prefix`). The pin became
`test_inline_dead_tail_after_trailing_return_by_both_lanes`. The widening the
same rule forces on F17's rejection: a nested `return` in the dead tail (after
the body's first direct return) is unreachable on both lanes and is no longer
rejected — anchored by
`test_inline_dead_tail_nested_return_by_both_lanes`; a *live* nested return
keeps its rejection pair.

### R9 no-anchor divergences (needs-decision, nothing edited)

1. **Host-valued return elements are silently dropped by tracing.**
`_return_impl` (`semantics.py:1404`) keeps only staged elements: `return 1.0`
emits *no* `ReturnStmt`, `return t, 1.0` keeps only `t`. The parser lowers
the literal to a `ConstFloat` ReturnStmt value (`return t, 1.0` → two
values). §9 says the *staged* elements are returned as separate IR values
(which leans tracing) but also that the frontends must agree on the IR
dataflow (which leans parser) — no doc decides the host element. Pinned by
`test_host_valued_return_elements_are_tracing_dropped`.
2. **`return [a, b]` is a dialect collision.** Tracing reads the host list as
a container of facades and emits the multi-value `ReturnStmt`; the parser
reads it as one `MakeTuple` — the printer's own spelling for a tuple return
(`_return_impl`'s comment at `semantics.py:1390` records the same split, and
the tracing lane only honors the round-trip reading when the declared return
type is a `TupleType`). Both readings are load-bearing for their dialect (the
tracing ergonomics container rule; the parser's round-trip ownership), so the
one accepted spelling cannot be anchored until the decision lands. Pinned by
`test_list_literal_return_is_a_dialect_collision`.
3. **`return pl.Tuple(a, b)` has no parser spelling** — the tracing-only
single-tuple facade; the parser reports `Unknown operation 'pl.Tuple'`.
Pinned by `test_pl_tuple_return_has_no_shared_spelling`.
4. **`@pl.expand` has no parser spelling.** The entrance is additive and
tracing-native (`docs/design/03-entrance.md` §7 line 436; §8 line 499: *'The
AST parser remains the sole owner of round-trip syntax and is not taught
entrance or tracing semantics'*), so the parser reports
`Unsupported function call` and no `conformance` anchor is possible. The
expansion's own contract is witnessed instead: the traced expansion is
structurally the body written out at the call site, which the parser *can*
parse (`test_expand_emits_the_written_out_body`), and the entrance's
trailing-return/early-return contract is already covered by
`tests/ut/language/entrance/test_expansion.py`. Pinned by
`test_expand_call_has_no_parser_spelling`.

### Discovered tasks

- **F15 — a statement after a function-level `return` is collected by the
  parser and dropped by tracing.** Red check and pin as above (finding 1).
  Frontend-only. **State: fixed** (fix-campaign task B2). The pins are
  rewritten to `test_function_level_return_drops_unreachable_tail_by_both_lanes`
  and `test_function_level_double_return_dropped_by_both_lanes`, both plain
  `conformance(..., build)` anchors asserted **raw** (straight-line kernels: no
  staged construct, no pass state). The lane decision: the parser, not tracing
  — the traced execution is the kernel's meaning (§9), and the rewriter's
  native return is what returns a value to a host caller (`Interpreter.return_`
  docstring), so the tracing side cannot be changed. The parser's
  `_parse_body_siblings` gained `stop_after_return`, passed only by
  `parse_function`; the statements after a function-level return are dead by
  construction, so nothing is lost.
- **F16 — an inline callee's multi-value trailing return: the pass splices a
  multi-value `ReturnStmt`, the parser's expansion returns one `MakeTuple`.**
  Red check and pin as above (finding 2). Frontend-only. **State: fixed**
  (fix-campaign task B3): `parse_return` takes an `@pl.inline` callee's
  expansion as the callee's value list instead of the value-position
  `MakeTuple`, so `return helper(x)` keeps the callee's arity; the value
  position is unchanged. The pin is rewritten to
  `test_inline_multi_value_trailing_return_by_both_lanes` (raw arity and raw
  equality) plus the post-inline canonical anchor
  `_inline_multi_value_anchor`.
- **F17 — the parser's inline expansion silently drops an early return under
  staged control flow (branch emitted empty), which both tracing entrances
  reject loudly.** Red check and pin as above (finding 3). Frontend-only.
  **State: fixed** (fix-campaign task B4). The pin is rewritten to the
  correct expectation (`test_inline_early_return_rejected_by_both_lanes`):
  the pass rejects the shape at splice time, the parser rejects it at
  expansion time (`_expand_inline_call`'s `_reject_inline_nested_return` scans
  the callee's body for a `return` nested in `if`/`for`/`while`/`with` and
  raises `ParserSyntaxError` with the pass's "non-trailing ReturnStmt"
  clause). A return that is the body's trailing statement — scope-wrapped
  included — still parses.
- **F37 — an inline callee's multi-value trailing return through a tuple
  temporary stayed divergent after F16's fix** (finding 4; filed by the
  fix-campaign task B3). Red check and pin as above. Frontend-only; was
  blocked on the `return [a, b]` dialect decision. **State: fixed**
  (fix-campaign task F37): the ratified R4 ruling makes the parser's
  one-`MakeTuple` reading normative, so the tracing lane now stages a tuple
  whose elements are all staged facades as one `pl.Tuple` value when bound
  (`_stage_staged_tuple`), and `tmp = (a, b); return tmp` keeps the temporary
  and returns one value on both lanes. The pin is rewritten to
  `test_inline_temp_tuple_trailing_return_by_both_lanes`.
- **F38 — an inline callee's dead second top-level return** (finding 5; filed
  by the fix-campaign task B4). The traced callee ends at the first return and
  the pass splices it, while the parser keeps collecting and its last-wins
  ``_inline_return_values`` overwrites the trailing return's values. Red check
  and pin as above (`test_inline_dead_tail_after_trailing_return_divergence`).
  Frontend-only; the fix is F15's `stop_after_return` semantics on the inline
  body walk, not a rejection (tracing accepts the spelling). **State: fixed**
  (fix-campaign task F38): `_inline_body_prefix` gives the inline expansion
  the same first-return-wins prefix B2 gave the direct function body, and both
  the walk and F17's nested-return scan read it. The pin is rewritten to
  `test_inline_dead_tail_after_trailing_return_by_both_lanes`, with
  `test_inline_dead_tail_nested_return_by_both_lanes` covering the dead
  nested return the narrowed scan no longer rejects.

F-ids continue from F14.

## 15. R10 verdicts — generic scopes (scope / manual_scope / graph / cluster)

Both sides read before verdicting: the parser's `_parse_scope`
(`ast_parser.py:4064`, the `mode=` kwarg), `_parse_manual_scope` (`:4090`),
`_emit_runtime_scope` (`:4101`, the manual-depth / AUTO-opt-in gates),
`_parse_legacy_scope` (`:4141`, `pl.cluster`), `_parse_scope_body` (`:5365`,
the single `builder.scope(...)` injection point), `_parse_graph_scope`
(`:5421`) and `_parse_scope_name_hint` (`:4039`); the tracing lane's rewriter
row `_with_row` (`rewriter.py:1485`) → `Sema.create_with_construct`
(`semantics.py:1548`) → `PtoWithConstruct` (`tracer/constructs.py:576`), the
four descriptors in `constructs/scopes.py` (`scope` `:222`, `graph` `:309`,
`manual_scope` `:448`, `cluster` `:536`), and the shared
`IRBuilder.begin_scope`/`end_scope` (`ir/builder.py:222`, `:254`).

**Regime first.** A scope is a transparent region on both lanes — neither
frontend emits `iter_arg`/`return_var` for it — so every straight-line scope
anchor is asserted **raw** and lands green. Anchors whose scope body contains
staged control flow, or which sit inside staged control flow, are asserted
**raw *and* post-`canonicalize()`**; raw holds because the region is
transparent (the loop-carried value is the self-referential `AssignStmt`
update on both sides) and canonical is the staged-flow regime proper. One
variant is raw-only: a scope *inside* a loop whose rebinding is loop-carried
(`k_manual_in_loop`) hits a printer gap in the canonical shape (F22, below) —
the frontends agree (verified with the roundtrip instrument bypassed) but
the post-SSA text cannot be re-parsed, so the canonical half is pinned rather
than asserted through the harness.

| Variant | Regime | Verdict |
| --- | --- | --- |
| `with pl.scope(mode=pl.ScopeMode.MANUAL):` straight-line (`k_scope_manual`) | raw | equivalent |
| statements around the scope (`k_scope_manual_fence`) | raw | equivalent |
| manual `pl.scope` wrapping a staged `for` (`k_scope_manual_around_loop`) | raw + canonical | equivalent |
| `with pl.scope():` under `auto_scope=False` (`ScopeAutoOff`, program) | raw | equivalent |
| `with pl.scope(mode=pl.ScopeMode.AUTO):` under `auto_scope=False` (`ScopeAutoOffExplicit`, program) | raw | equivalent |
| bare `pl.manual_scope()` (`k_manual_scope`) | raw | equivalent |
| scope body-local used after the scope (`k_manual_leak`) | raw | equivalent |
| rebind of an outer local inside the scope (`k_manual_shadow`) | raw | equivalent |
| `pl.manual_scope` wrapping a staged `for` (`k_manual_around_loop`) | raw + canonical | equivalent |
| `pl.manual_scope` inside a staged `for` (`k_manual_in_loop`) | raw | equivalent (canonical pinned — **F22**) |
| scope inside one branch of a staged `if`, divergent bind (`k_manual_cond_bind`) | raw + canonical | equivalent |
| staged `if` inside the scope (`k_manual_nested_if`) | raw + canonical | equivalent |
| early return inside the scope (`k_return_in_manual`) | raw + canonical | equivalent |
| scope in an Orchestration entry / in an InCore body (`ManualInOrch`, `ManualInInCore`, program) | raw | equivalent (neither lane enforces the docstring's placement rule) |
| bare `with pl.graph("decoder"):` (`k_graph`) | raw | equivalent |
| graph region inside a staged `for` (`k_graph_in_loop`) | raw + canonical | equivalent |
| cluster region inside a graph region (`GraphThenCluster`, program) | raw | equivalent |
| bare `pl.cluster()` / `pl.cluster(name_hint="grp")` (`k_cluster`, `k_cluster_named`) | raw | equivalent |
| cluster nested in a cluster (`k_cluster_nested`) | raw | equivalent |
| cluster wrapping a staged `for` (`k_cluster_around_loop`) | raw + canonical | equivalent |
| manual scope inside a cluster (`k_manual_in_cluster`) | raw | equivalent |
| cluster in an Orchestration entry (`ClusterInOrch`, program) | raw | equivalent |
| `pl.dump_tag(t)` before a `pl.graph` region (`GraphDumpTag`) | raw | equivalent (**F18**, fixed by C1) |
| scope-level `dump_vars` print -> parse (`GraphDumpTag` / `GraphDumpTagCluster`) | n/a (both lanes drop it) | **bug — F39, fixed** (filed by C1; the printer now emits the attr as a `dumps=[...]` kwarg on `pl.graph` / `pl.cluster` and both lanes' parser rebuilds it) |
| `pl.dump_tag(t)` before a `pl.cluster` region | raw | equivalent (the tracing cluster hook merges the sticky set) |
| `pl.graph("")`; graph nested in graph / in `pl.at` / in `pl.cluster` / in a device-side body; `as` target on any generic scope | n/a (both lanes reject) | out of scope — no accepted spelling (pinned: `test_graph_placement_rejected_by_both_lanes`, `test_graph_in_device_body_rejected_by_both_lanes`, `test_graph_empty_name_rejected_by_both_lanes`, `test_scope_as_target_rejected_by_both_lanes`) |
| `pl.scope(mode=1)` / `mode="MANUAL"` | n/a (parser rejects, tracing accepts) | **accepted divergence — F19** (R2: tracing may be a superset; no fix) |
| manual scope in a manual scope; AUTO scope in a manual scope under `auto_scope=False` | n/a (both lanes reject) | equivalent (**F20**, fixed by C2; rejection pins `test_nested_manual_scope_rejected_by_both_lanes`, `test_auto_scope_in_manual_rejected_by_both_lanes`) |
| non-identifier `pl.graph` name / `pl.cluster(name_hint=...)` | n/a (parser rejects, tracing accepts) | **accepted divergence — F21** (R2: tracing may be a superset; no fix) |
| `pl.graph(name="x")`; `pl.scope(pl.ScopeMode.MANUAL)` (positional) | n/a (parser rejects) | **needs-decision** — tracing-lenient spellings, same IR, no doc decides (pinned) |
| `pl.cluster(name_hint=1)` | n/a (both lanes reject) | **needs-decision** — tracing's rejection surfaces from nanobind, not the frontend (pinned) |

31 anchors in the new `test_generic_scopes.py` (25 function-level
`conformance(...)` call sites, 6 `program_conformance(...)`), plus the
no-shared-spelling pins in the same suite (37 test functions, **45 collected
tests**). Conformance-directory call sites 195 → 226;
conformance-directory tests 310 → 355; all green. No red anchor is kept — every
red finding is pinned instead.

### R10 findings (all frontends left alone)

1. **F18 — a `pl.graph` region drops the forward-sticky `dump_vars`.**
   `pl.dump_tag(t)` before a graph region: the parser's `_parse_scope_body`
   merges the tagged set for every non-Runtime kind
   (`_merge_forward_sticky_dump`, `ast_parser.py:5313`), while the tracing
   `graph.__trace_enter__` calls `begin_scope` with no `attrs` at all
   (`constructs/scopes.py:419`); the cluster hook *does* merge
   (`:575`), and that anchor is green. Red check:
   `test_graph_scope_drops_sticky_dump_vars` — the parsed graph scope carries
   `dump_vars`, the traced one has no attrs, `assert_structural_equal` raises
   on the attr dict. Caveat recorded with the pin: the parser's own attr is
   not print-stable either (`parse(as_python())` drops it), and
   `scope_outline_utils` consumes `dump_vars` only on the *dispatch* paths,
   not in `OutlineGraphScopes` — so the parser-side attachment may be the
   over-eager half. **State: fixed by C1** — the ratified decision rules
   (`docs/fix-construct-parity.md` §5) classify this as an R4 subset
   violation, so the tracing `graph.__trace_enter__` now merges the
   forward-sticky set at scope entry exactly like the cluster hook; the
   pin is rewritten into the anchor `test_graph_dump_tag_anchor`. The
   print-instability caveat is real and symmetric (both lanes drop the attr)
   and is filed as **F39**, not fixed here.
2. **F19 — a non-enum `mode=` silently selects AUTO on the tracing lane.**
   `pl.scope(mode=1)` / `pl.scope(mode="MANUAL")` pass the `scope` ctor
   unchanged (`constructs/scopes.py:255`), then
   `manual=(self.mode == ScopeMode.MANUAL)` (`:273`) is False and the AUTO
   gate (`:269`) passes because `self.mode == ScopeMode.AUTO` is False too —
   a hand-placed **AUTO** scope in a default `auto_scope=True` body, the
   exact spelling the parser rejects and the tracing frontend is supposed to
   gate on `auto_scope=False`. Red check:
   `test_scope_mode_is_not_validated_by_tracing` — the traced scope has
   `manual is False` while `function_parser` raises `Expected
   pl.ScopeMode.<name>`, and the printed text fails to re-parse
   (`requires @pl.function(auto_scope=False)`).
3. **F20 — the manual-depth rules are parser-only.** The parser tracks
   `_manual_scope_depth` (`ast_parser.py:4112`/`:4133`) and rejects
   manual-in-manual, and rejects AUTO-inside-MANUAL under
   `auto_scope=False`; the tracing `scope` / `manual_scope` hooks have no
   counter, so both spellings are accepted and emit nested
   `RuntimeScopeStmt(manual=True)` whose printed text fails to re-parse.
   Red checks: `test_nested_manual_scope_has_no_shared_spelling`,
   `test_auto_scope_in_manual_has_no_shared_spelling`. **State: fixed by
   C2** — ratified as an R4 subset violation (`docs/fix-construct-parity.md`
   §4/§5), so the tracing lane moves. The `IRBuilder` now mirrors the MANUAL
   flag of each open scope (`_scope_manuals`, the tracing counterpart of the
   parser's `_manual_scope_depth`) and the `scope` / `manual_scope` hooks
   gate on `ctx.is_inside_manual_scope()` before `begin_scope`, with the
   parser's messages and hints verbatim. The pins are rewritten into
   `test_nested_manual_scope_rejected_by_both_lanes` and
   `test_auto_scope_in_manual_rejected_by_both_lanes`; the complement the
   gate must not over-reach (`AUTO` then `MANUAL` under `auto_scope=False`,
   legal because only the enclosing scope's *manual* flag counts) is anchored
   by `test_manual_scope_inside_auto_scope_anchor`.
4. **F21 — the name-hint validation is parser-only.**
   `_parse_scope_name_hint` (`ast_parser.py:4039`) requires a non-keyword
   identifier; the tracing `graph` ctor checks only `str` + non-empty and
   `cluster` checks nothing, so `pl.graph("has space")` /
   `pl.cluster(name_hint="has space")` reach the IR and the printed form
   fails to re-parse. Red check:
   `test_scope_name_hint_validation_is_parser_only`.
5. **F22 — the printer loses the yield assignment LHS inside a manual
   runtime scope.** After `convert_to_ssa`, a loop-carried value updated
   inside a manual scope yields *inside* the scope body (graph/cluster place
   the yield outside — a pass-side difference, not a frontend one), and
   `IRPythonPrinter::VisitStmtBody` special-cases only
   `rscope && !rscope->manual_` (`src/ir/transforms/python_printer.cpp:2659`),
   so the text is a bare `pl.yield_(...)` that the parser rejects (`Loop with
   init_values requires an assignment-form pl.yield_(...)`). Red check:
   `test_scope_in_loop_canonical_is_printer_gap` — with the
   RoundtripInstrument bypassed (`passes.PassContext([],
   VerificationLevel.NONE)`) both lanes' canonical IR is structurally equal,
   which localizes the gap to the printer. Printer-side; needs-decision on
   the layer.

6. **F39 — a scope-level `dump_vars` attr is not print-stable** (filed by C1
   while fixing F18). `pl.dump_tag` is a trace-time marker that emits no IR
   statement, and neither `pl.graph` nor `pl.cluster` accepted a `dumps=`
   kwarg, so `parse(as_python())` dropped the attr off both scope kinds. Both
   lanes lost it identically, so it is a printer/parser-coverage hole, not a
   parity divergence. Pin: `test_scope_dump_vars_is_not_print_stable`.
   **State: fixed** (fix-campaign task F39). The printer's graph / cluster
   visitors now call `PrintScopeDumpAttr` (`python_printer.cpp`: the same
   emitter `pl.at` scopes use), and `pl.graph` / `pl.cluster` take `dumps=`
   on both lanes (`ast_parser.py`'s `_parse_at_dumps_kwarg` gained an `owner`
   parameter; `constructs/scopes.py`'s graph/cluster hooks merge the explicit
   list with the forward-sticky set through the shared
   `_explicit_dump_attrs`). The pin is rewritten into
   `test_scope_dump_vars_roundtrip_anchor` plus the explicit-surface anchor
   `test_explicit_scope_dumps_anchor`.

Frontend-side gaps 2–4 share one diagnostic home
(`constructs/scopes.py`'s four descriptors) but are recorded separately
because they are separate rules; F18 (the round's only anchor red) was fixed
by C1 and F20 by C2 under the ratified R4 ruling, while findings 2 and 4
(F19, F21) are R2-legal acceptance divergences per
`docs/fix-construct-parity.md` §5 and take no fix.

### Discovered tasks

- **F18 — a `pl.graph` region drops the forward-sticky `dump_vars` the
  parser attaches** (finding 1). Red check and pin as above. **Fixed by
  C1** in the fix campaign (`docs/fix-construct-parity.md` §6): the
  tracing lane now matches the parser's GraphScopeStmt attrs; the pin
  became the anchor `test_graph_dump_tag_anchor`.
- **F39 — a scope-level `dump_vars` attr has no printed surface** (finding
  6; filed by C1). Pin as above. Printer/parser-coverage, both lanes — not
  a parity item. **State: fixed** (fix-campaign task F39): the printer
  emits the attr as a `dumps=[...]` kwarg on `pl.graph` / `pl.cluster` and
  both lanes accept the spelling, so print -> reparse is a fixpoint. The
  pin became `test_scope_dump_vars_roundtrip_anchor` +
  `test_explicit_scope_dumps_anchor`.
- **F19 — `pl.scope(mode=<non-enum>)` silently selects AUTO in a default
  `auto_scope=True` body** (finding 2). Red check and pin as above.
  Frontend-only.
- **F20 — the tracing generic-scope descriptors do not track manual-scope
  nesting** (manual-in-manual; AUTO-in-manual under `auto_scope=False`)
  (finding 3). Red checks and pins as above. Frontend-only. **Fixed by
  C2** in the fix campaign (`docs/fix-construct-parity.md` §6): the
  tracing hooks now gate on the builder's open-scope manual mirror; the
  pins became the rejection pairs
  `test_nested_manual_scope_rejected_by_both_lanes` /
  `test_auto_scope_in_manual_rejected_by_both_lanes`, with the complement
  anchored as `test_manual_scope_inside_auto_scope_anchor`.
- **F21 — the tracing `graph`/`cluster` descriptors do not validate the
  region name / `name_hint` as an identifier** (finding 4). Red check and
  pin as above. Frontend-only.
- **F22 — the Python printer drops the `var = pl.yield_(...)` assignment LHS
  when the yield sits inside a manual `RuntimeScopeStmt` body** (finding 5).
  Red check and pin as above. Printer-side.

F-ids continue from F17.

## 16. R11 verdicts — `pl.spmd` (with/for form, tid capture), `pl.split_aiv`, the CORE_GROUP carrier, the `at` family

Both sides read before verdicting: the parser's `_emit_spmd_body`
(`ast_parser.py:4755`), `_spmd_body_reads_block_idx` (`:4578`, an `ast.walk`
over the whole suite), `_explicit_incore_carrier` (`:4613`),
`_parse_spmd_for_loop` (`:5036`), `_parse_spmd_scope` (`:4812`),
`_parse_spmd_scope_with_tid` (`:4893`), `_parse_spmd_kwargs` (`:4325`, the
`_parse_and_validate_core_num` gate), `_parse_split_aiv_for_loop` (`:5145`)
and the `pl.at` quartet `_parse_at_kwargs` / `_dispatch_at_keyword` /
`_handle_at_optimizations_kw` / `_parse_optimizations_list` (`:3707`,
`:3743`, `:3809`, `:3838`); the tracing lane's `spmd.__trace_enter__` /
`begin` / `__trace_exit__` (`constructs/scopes.py:961`, `:1130`, `:1183`),
`_submit_only_scope_attrs` (`:906`), `split_aiv.begin` (`:1335`),
`at.__trace_enter__` / `_collect_scope_attrs` (`:1624`, `:1582`),
`_maybe_open_spmd_carriers` (`tracer/semantics.py:1005`),
`PtoForConstruct.activate_target` (`tracer/constructs.py:332`) and the
rewriter's `single_call_body` channel (`interpreter.py:871` →
`on_begin_with`).

**Regime first.** A `pl.spmd` / `pl.split_aiv` / `pl.at` scope is a *scope*,
not a `ForStmt`: there is no iter_arg/phi territory, so raw equality is the
load-bearing half. Every anchor below is asserted **raw**, and **raw +
post-`canonicalize()`** wherever the anchor carries staged control flow (a
staged `if` inside the body, a `pl.range` around/inside the region, a
`deps=`-threaded capture, a `pl.at` inside a loop) — the SSA pass has
loop-carried values to derive there and both lanes must reach the same
canonical shape. The one canonical pair that cannot go through the harness is
a **printer** gap, not a frontend one (F26).

| Variant | Regime | Verdict |
| --- | --- | --- |
| for-form bare (no hint) (`k_spmd_for`) | raw | equivalent |
| for-form + `name_hint` + loop var used (`k_spmd_for_named`) | raw | equivalent |
| for-form + `sync_start=True` (`k_spmd_for_sync`) | raw | equivalent |
| for-form + `optimizations=[pl.split(MODE)]` (`k_spmd_for_split`) | raw | equivalent |
| for-form + `optimizations=[pl.cross_core_slot(slot_num=N)]` (`k_spmd_for_slot`) | raw | equivalent |
| for-form + both optimization entries (`k_spmd_for_split_slot`) | raw | equivalent |
| for-form + `allow_early_resolve=True` (`k_spmd_for_aer`) | raw | equivalent (the for-form now shares `_submit_only_scope_attrs` with the with-form — the DSV4 `expert_shared` divergence is closed) |
| for-form + `predicate=(...)` (`k_spmd_for_predicate`) | raw | equivalent |
| for-form + staged `core_num` (`k_spmd_for_staged_core`) | raw | equivalent |
| for-form body holding a staged `if` (`k_spmd_for_body_if`) | raw + canonical | equivalent |
| for-form single-call dispatch body (program, `KForFormDispatch`) | raw + program-canonical | equivalent |
| for-form inside a staged `for` (`k_spmd_for_in_staged_for`) | raw + canonical | equivalent |
| with-form capture + `deps=` chain + `pl.at(deps=)` (`k_spmd_capture_deps`) | raw + canonical | equivalent |
| capture stored into `pl.array.create(N, pl.TASK_ID)` (`k_spmd_capture_array`) | raw | equivalent |
| capture + split + slot on the single-call body (program, `KCaptureSplitSlot`) | raw + program-canonical | equivalent |
| with-form inside `pl.cluster()` (`k_spmd_in_cluster`) | raw + canonical | equivalent |
| the with-form four-cell body matrix (program, `test_spmd_with_form.py`) | raw / program-canonical | equivalent (pre-existing anchors, unchanged) |
| bare top-level `pl.split_aiv` (UP_DOWN / NONE / LEFT_RIGHT) | raw | equivalent |
| `pl.split_aiv` inside `pl.at(CORE_GROUP)` (`k_saiv_in_at`) | raw | equivalent (flattened in place; no second InCore) |
| statement after the bare region (`k_saiv_then_stmt`) | raw | equivalent |
| region inside a staged `pl.range` (`k_saiv_in_staged_for`) | raw + canonical | equivalent |
| region inside a staged loop inside `pl.at(CORE_GROUP)` (`k_saiv_in_at_staged_for`) | raw | equivalent; canonical **pinned — F26** |
| region at the top of an InCore-typed function (program, `KSaivInInCoreFunction`) | raw | equivalent (arm 2: emitted bare, both lanes) |
| `pl.at` `allow_early_resolve` / `windowize` / `cross_core_slot` | raw (+canonical) | equivalent |
| `pl.at` `no_dep_args=` + `dumps=` (`k_at_no_dep_args_dumps`) | raw + canonical | equivalent |
| `pl.at` `as tid` + `deps=` on a later scope (`k_at_capture_deps`) | raw + canonical | equivalent |
| `pl.at` Hierarchy + `role=` + `name_hint` / positional `level` | raw | equivalent |
| `pl.at(CORE_GROUP)` inside a Hierarchy `pl.at` (`k_at_in_hierarchy`) | raw | equivalent |
| `pl.at` inside a staged `for` (`k_at_in_staged_for`) | raw + canonical | equivalent |
| a statement before the block-index read (`k_carrier_pre_stmt`) | raw | equivalent — **F23**, fixed by A2 (the carrier boundary is the static, body-wide fact on both lanes) |
| a body whose only read is a single `i = pl.tile.get_block_idx()` statement (`k_carrier_single_read`) | raw | equivalent — **F23**, fixed by A2 (the read fact wins over the direct-dispatch shape; the split hints are on the scopes) |
| a read nested in staged control flow (`k_carrier_read_in_staged_for`) | raw + canonical | equivalent — **F23**, fixed by A2 |
| read inside a `pl.split_aiv` region, statement between regions (`k_carrier_split_aiv_between`) | raw | equivalent — **F23**, fixed by A2 (one carrier; the region is emitted in place) — the F1 fixture shape |
| read first (`k_carrier_read_first`) | raw + canonical | equivalent (the boundary's other half — both lanes wrap at the same point) |
| a body that supplies its own CORE_GROUP carrier, a lone nested `pl.at(level=pl.Level.CORE_GROUP, ...)` (`k_carrier_explicit_at` / `k_carrier_explicit_at_read`) | n/a (no shared spelling: `raw` diverges on the outer `name_hint` alone) | **bug — F36** (filed by A2; parser keeps the hint unsplit, tracing splits it) |
| `with pl.spmd(n, deps=...)` without `as tid`; same for the for-form | n/a (parser accepts, tracing rejects) | **intended divergence** — `docs/en/dev/language/02-manual_dependencies.md:36` (*'``deps=`` is accepted only on the `as tid` capture form'*) + `docs/migration-todo.md` (the docs corrected to that reading) |
| with-form body neither reading the index nor dispatching (`k_spmd_bare_body`, `k_spmd_bare_body_single_call`) | n/a (parser rejects, tracing accepts) | **needs-decision** |
| nested `pl.split_aiv` (`k_saiv_nested`) | n/a (parser rejects, tracing accepts) | **bug — F24** |
| `pl.at(name_hint="has space")` (`k_at_bad_name_hint`) | n/a (parser rejects, tracing accepts) | **bug — F25** |
| `pl.at(windowize=1)` / `allow_early_resolve=1`; a pre-built `optimizations=` list; `deps=None` | n/a (parser rejects, tracing accepts, same IR) | **needs-decision** (acceptance-only) |
| `pl.split_aiv(4)` / missing `mode=`; a region under a function-level `pl.split(...)`; a cluster-nested `pl.spmd` with `allow_early_resolve`/`as tid`; `pl.spmd(0)` | n/a (both lanes reject) | out of scope — no accepted spelling; pinned |
| core_num as `pl.const(4, pl.INDEX)`; `name_hint` ending in `_spmd` | raw | equivalent (check only) |

**Anchors.** 33 anchor call sites in the new `test_spmd_split_aiv.py` (28
function-level + 5 program-level, across 25 tests; the suite's remaining 14
tests are the pins — 39 collected tests in all) + 13
new `conformance(...)` call sites in `test_at_const.py` (10 → 23).
Conformance-directory call sites 226 → 272 (bare 210 → 251, program 16 → 21);
conformance-directory tests 355 → 407. All 272 anchors green; no red anchor is
kept — every red finding is pinned with explicit shape assertions. The fix
campaign's A2 promotes the four F23 pins to anchors (37 call sites / 29 anchor
tests) and adds the parametrized print → reparse regression plus the two F36
pins: 45 collected tests in the suite.

### R11 findings (all frontends left alone)

1. **F23 — the CORE_GROUP carrier boundary is dynamic on the tracing lane and
   static on the parser's (confirms and pins the migration review's F1).** The
   parser's `_spmd_body_reads_block_idx` (`ast_parser.py:4578`) walks the whole
   suite and, on a hit, wraps *the entire body* in one `InCoreScopeStmt`;
   `_maybe_open_spmd_carriers` (`tracer/semantics.py:1005`) instead opens the
   carrier at the *first top-level* read. The two agree only when that read is
   the body's first statement. Four spellings disagree (the table's four red
   rows): a statement before the read stays directly in the `SpmdScopeStmt`;
   a read nested in staged control flow opens no carrier at all; a read inside
   a `pl.split_aiv` region leaves each region to synthesize its own wrapper
   and the between-statements outside every carrier; and a single-statement
   read-only body reserves the *unsplit* name hint for the outer scope
   (`single_call_body`), so `SpmdScopeStmt.name_hint` becomes `"w"` where the
   parser's is `"w_spmd"`. The printed text of the last three cannot be
   re-parsed by the parser either, so the print → reparse fixpoint breaks.
   `docs/design/04-migration-guide.md` §6.2 line 333 intends the opposite
   (*'The with-form `pl.spmd` body — including multi-statement bodies reading
   `get_block_idx()` — is wrapped correctly by the tracer'*), and F1
   (`docs/review-r10.md` §4) is the prior record of the same divergence from
   the codegen side. Red checks: the four pins in `test_spmd_split_aiv.py`
   (`test_carrier_boundary_pre_read_statement`,
   `test_carrier_boundary_single_read_name_hint`,
   `test_carrier_boundary_read_inside_staged_flow`,
   `test_carrier_boundary_split_aiv_regions`). Owner: the tracing lane — the
   carrier decision must become a static, body-wide fact (or the parser's
   rule must be restated and the implicit-carrier spelling documented as
   tracing-rejected). Frontend-only either way.
   **State: fixed** (fix-campaign task A2). The tracing lane's own static,
   body-wide fact is the rewriter's `reads_block_idx` (`collector/rewriter.py`,
   the mirror of `_spmd_body_reads_block_idx` — an `ast.walk` over the whole
   suite, call-name matched), threaded to the with-construct through
   `rt.on_begin_with` → `create_with_construct` → `__trace_enter__` beside
   `single_call_body`. `spmd.__trace_enter__` wraps on it in the parser's
   order (read → inline wrap with the split hints; else single call →
   direct dispatch; else the deferred carrier), so the carrier opens at the
   body's *first* statement; `_maybe_open_spmd_carriers` stays as the backstop
   for a read the AST walk cannot see (an aliased `get_block_idx`). The four
   pins are promoted to the anchors
   `test_carrier_pre_read_statement_anchor`,
   `test_carrier_single_read_name_hint_anchor`,
   `test_carrier_read_inside_staged_flow_anchor`,
   `test_carrier_split_aiv_between_anchor`, plus the parametrized print →
   reparse regression `test_carrier_boundary_roundtrip`. The same commit
   closes an adjacent hole the new fact exposed: a body that supplies its own
   CORE_GROUP region now *cancels* the deferred carrier
   (`_DeferredSpmdCarrier.cancel_pending_carrier`, called by
   `_flush_deferred_scopes`' absorption branch) instead of leaving a spurious
   empty `InCoreScopeStmt` behind at the scope's close. The explicit-carrier
   spelling's residual outer-`name_hint` divergence is filed as **F36** below.
2. **F24 — nested `pl.split_aiv` is accepted by tracing and rejected by the
   parser.** `_parse_split_aiv_for_loop` rejects a region inside a region
   (`ast_parser.py:5167`, keyed on `_split_aiv_mode_stack`);
   `split_aiv.begin` (`constructs/scopes.py:1335`) has no nesting guard, so the
   region nests, the printed text fails to re-parse, and the print → reparse
   fixpoint breaks (F12's class). Red check:
   `test_nested_split_aiv_is_tracing_accepted`. Owner: the tracing lane.
3. **F25 — `pl.at(name_hint=...)` is not validated as an identifier.** The
   parser runs `_parse_scope_name_hint` (`ast_parser.py:4039`);
   `at.__trace_enter__` passes the string straight to `begin_scope`, so
   `name_hint="has space"` reaches the IR and the printed text fails to
   re-parse. This is F21's root cause (the R10 `graph`/`cluster` descriptors)
   reaching the `at` descriptor in the same file. Red check:
   `test_at_name_hint_validation_is_parser_only`. Frontend-only.
4. **F26 — the printer drops the `var = pl.yield_(...)` assignment LHS when the
   yield sits inside a `SplitAivScopeStmt` body.**
   `IRPythonPrinter::VisitStmtBody` threads `return_vars` only for a direct
   yield body or the last statement of a `SeqStmts`
   (`src/ir/transforms/python_printer.cpp:2605` / `:2634`); F22 recorded the
   `manual RuntimeScopeStmt` instance, this is the `SplitAivScopeStmt` one. A
   loop-carried value updated inside a region that sits inside a `pl.range`
   loop prints as a bare `pl.yield_(...)`, leaving the loop's return var
   undefined; the roundtrip instrument rejects the text (so the canonical
   half of `k_saiv_in_at_staged_for` cannot run through the harness). Both
   lanes' post-SSA IR is structurally equal — the gap is printer-side.
   Red check: `test_split_aiv_in_at_canonical_is_printer_gap`. Printer-side;
   needs-decision on the layer. **State: fixed by A3**; the canonical anchor is
   `test_split_aiv_in_at_staged_for_anchor` and the printer regression is
   `test_split_aiv_in_at_canonical_roundtrip`.
5. **F36 — a body that supplies its own CORE_GROUP carrier keeps the
   `pl.spmd(name_hint=...)` unsplit on the parser lane only (filed by the
   fix-campaign task A2).** `_emit_spmd_body`'s `_explicit_incore_carrier`
   branch (`ast_parser.py:4613`) treats a lone nested
   `pl.at(level=pl.Level.CORE_GROUP, ...)` as the carrier and passes
   `name_hint` through unsplit; the tracing lane's split-hint rule
   (`not single_call_body` in `spmd.__trace_enter__`) reserves the `_spmd`
   suffix for the outer scope whatever the body. Both lanes accept the source
   and the regions/bodies match — the raw comparison differs on
   `SpmdScopeStmt.name_hint` alone (`"w_spmd"` vs `"w"`), for an
   index-reading body and for a dispatch body alike; the printed text still
   reparses, so nothing is lost at codegen scale. Fixed only by a *third*
   static body fact from the rewriter (the parser's explicit-carrier check,
   which needs the `pl.at` level resolved) or by restating the parser's hint
   rule. Pin: `test_carrier_explicit_at_hint_divergence` (both spellings).
   Frontend-only. **State: filed, not fixed** (out of A2's scope).

### R11 no-anchor divergences (nothing edited)

1. **The tracing lane accepts a with-form `pl.spmd` body that neither reads the
   per-block index nor dispatches a `self.<kernel>(...)` call.** Both the dev
   doc (`docs/en/dev/language/01-statements.md`, the `pl.spmd` section) and the
   user doc (`docs/en/user/language/04-scopes.md:140`, *'rejected — every block
   would be doing identical work'*) state the rejection; the parser enforces it
   and the tracing lane emits the bare `Spmd` scope (printed text cannot be
   re-parsed). The construct's own suite (`test_spmd_with_form.py`) records the
   omission as deliberate — the property survives to the IR and its diagnostic
   home should be a pass/verifier check, which does not exist today — but no
   design doc or manifest sanctions it. **needs-decision**. Pins:
   `test_spmd_bare_body_is_tracing_accepted`.
2. **Four `pl.at` kwarg spellings the parser validates and the tracing
   descriptor does not**, all with the same IR: `windowize=1` /
   `allow_early_resolve=1` (parser: boolean literal; tracing: truthiness),
   a pre-built `optimizations=` list (parser: list *literal* — it inspects the
   AST; tracing: any sequence), `deps=None` (parser: list/tuple/comprehension;
   tracing: falsy no-op). **needs-decision** (acceptance-only). Pins:
   `test_at_bool_literal_kwargs_are_parser_only`,
   `test_at_optimizations_list_literal_is_parser_only`,
   `test_at_deps_none_is_parser_only`.

### Discovered tasks

- **F23 — the `pl.spmd` with-form CORE_GROUP carrier boundary diverges: the
  parser wraps the whole body statically, the tracing lane opens at the first
  top-level block-index read** (finding 1; the four red pins). Confirms F1.
  Frontend-only. **State: fixed** (fix-campaign task A2): the rewriter's
  `reads_block_idx` fact makes the boundary the parser's static, body-wide
  classification; the four pins became anchors plus a print → reparse
  regression, and the adjacent absorbed-carrier hole (a spurious empty
  `InCoreScopeStmt`) is closed in the same change. The explicit-carrier
  spelling's remaining outer-hint divergence is filed as F36.
- **F24 — a nested `pl.split_aiv` region is accepted by tracing and rejected
  by the parser** (finding 2). Red check and pin as above. Frontend-only.
- **F25 — `pl.at(name_hint=...)` identifier validation is parser-only**
  (finding 3). Red check and pin as above. Frontend-only.
- **F26 — the printer loses the yield assignment LHS inside a
  `SplitAivScopeStmt` body** (finding 4). Red check and pin as above.
  Printer-side. **State: fixed by A3**.
- **F36 — the explicit-CORE_GROUP-carrier spelling splits the outer
  `pl.spmd` name hint on the tracing lane only** (finding 5; filed by the
  fix-campaign task A2 while fixing F23). Red check and pin as above.
  Frontend-only.

F-ids continue from F22. `scripts/review/check_construct_parity.py`'s construct
map now points `split_aiv` at the new suite and claims
`_parse_and_validate_core_num` (spmd) and `_handle_at_optimizations_kw`
(`at family`); `spmd` keeps `test_spmd_with_form.py` as its named suite (the
map is one-suite-per-construct) and the new suite carries the for-form /
capture / carrier anchors as well.

## 17. Round ledger

| Round | Task | Model | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | seed this tracker | — (orchestrator) | done | `d60a4b303` | plan, contract, evidence base |
| 0 | parity inventory oracle | — (orchestrator) | done | *(this commit)* | `scripts/review/check_construct_parity.py`; 0 stale, 10/16 constructs without anchors, 4 unclaimed structural parser handlers, 17 unclaimed public tracing entries. The gap table in section 5 is now data, not hypothesis |
| 1 | R5 `if` / `else` / `elif` | worker session | done | *(this commit)* | 13 new anchors in `test_if.py` (suite 7 → 20 call sites; 70 → 83 across `test_*.py`), all green. **No `F*` task**: no red anchor. One intended divergence pinned (host predicate dispatch). |
| 1 | R6 `for` + iterators | worker session | done | *(this commit)* | 28 new anchors in `test_for.py` (suite 7 → 35 call sites; 83 → 111 across `test_*.py`), all green — all four iterator kinds, the 1/2/3-arg and computed bound forms, stage pinning, `attrs=`, the INT64/INDEX for-syntax bound, nesting. No red anchor. Target forms are parser-only or rejected by both (pinned): no shared spelling. Four no-anchor acceptance divergences recorded, one filed as **F11** (`attrs=` non-string key → `std::bad_cast`). |
| 1 | R7 `break` / `continue` | worker session | done | *(this commit)* | new suite `test_break_continue.py`: 36 anchors (18 kernels × raw + canonical; conformance-directory call sites 111 → 147), all green — nesting/target selection, the deferred staged exit ordering, exits inside `while`, post-transfer collection, innermost-kind gating, `pl.pipeline`, `pl.spmd`-nested. No red anchor, so no anchor-rule `F*`. Two no-anchor divergences recorded and filed: **F12** (tracing accepts a transfer in a top-level `pl.spmd` / `pl.split_aiv` region and emits an unparseable `break`), **F13** (the parser silently drops a loop `else` suite that tracing rejects loudly). |
| 1 | R8 binding & assignment | worker session | done | *(this commit)* | new suite `test_binding_assignment.py`: 22 anchors (conformance-directory call sites 147 → 169), all green — subscript write (tensor/tile/array, bare/partial/full-axis/in-loop), same-type staged rebind, type-changing rebind, param-name rebind, defs/store order, `pl.const` staged init, tuple unpack, the `pl.Scalar` param world. One red canonical regime filed as **F14** (a type-changing rebind inside a staged loop yields a free variable in the SSA loop init on *both* lanes — a shared §7 hole, not a parity divergence), pinned by an explicit `pytest.raises` test. Three no-shared-spelling/divergence pins: scalar-index rank-reducing write (**needs-decision**), the typed-staged-init guard (**intended divergence**, `docs/design/01-language.md` §4), `pl.runtime` (jit-entry-only, out of scope). |
| 1 | R9 returns / exits | worker session | done | *(this commit)* | new suite `test_return_exit.py`: 19 anchors (12 kernels; conformance-directory call sites 169 → 188; conformance-dir tests 288 → 310), all green — the trailing single-return contract (multi-value, scalar, bare, void), returns inside `elif`/branch/loop/`pl.spmd`/`pl.at` (raw + canonical), plus the `@pl.expand` no-spelling witness and the declared-`->` intended divergence. Three red findings filed and pinned: **F15** (a statement after a function-level `return` is dropped by tracing, collected by the parser), **F16** (an inline callee's multi-value trailing return: pass splice multi-value vs parser `MakeTuple`), **F17** (the parser's inline expansion silently drops an early return under staged control flow that both tracing entrances reject loudly). Two needs-decision no-anchor divergences: host-valued return elements dropped by tracing; `return [a, b]` MakeTuple-vs-multi-value dialect collision. `scripts/review/check_construct_parity.py`'s construct map now points `returns / exits` at the new suite. |
| 2 | R10 generic scopes (`scope` / `manual_scope` / `graph` / `cluster`) | worker session | done | *(this commit)* | new suite `test_generic_scopes.py`: 31 anchors (25 function-level + 6 program-level) + rejection/divergence pins (conformance-directory call sites 195 → 226; conformance-dir tests 310 → 355), all green — the four descriptors straight-line, inside/around staged control flow, the `auto_scope=False` AUTO spellings, scope leak/shadow/return/cond-bind semantics, cluster/graph nesting, placement rules, `as`-target rejection. Five findings filed and pinned: **F18** (a `pl.graph` region drops the forward-sticky `dump_vars` the parser attaches — the round's only red anchor), **F19** (`pl.scope(mode=<non-enum>)` silently selects AUTO), **F20** (no manual-scope nesting tracking on the tracing side), **F21** (graph/cluster name-hint identifier validation is parser-only), **F22** (the printer drops the yield LHS inside a manual runtime scope). Two needs-decision acceptance-only gaps (`pl.graph(name=...)` keyword, positional `pl.scope(MODE)`), one diagnostic-quality note (non-string `name_hint` fails in nanobind). `scripts/review/check_construct_parity.py`'s construct map now points `generic scopes` at the new suite (7 of 16 constructs now anchorless, down from 10). |
| 2 | R11 `spmd` (with/for, tid capture) + `split_aiv` + the CORE_GROUP carrier + the `at` family | worker session | done | *(this commit)* | new suite `test_spmd_split_aiv.py`: 33 anchors (28 function-level + 5 program-level, across 25 tests) + 14 divergence/rejection pin tests (39 collected tests), plus 13 new `pl.at` anchors/pins in `test_at_const.py` (10 → 23) — conformance-directory call sites 226 → 272 (bare 210 → 251, program 16 → 21); conformance-dir tests 355 → 407; all 272 green. Covers the for-form (hint/sync/split/slot/both/aer/predicate/staged core_num/body-if/nesting/dispatch program), the with-form capture + `deps=` chain + TASK_ID-array store + cluster nesting + split+slot, `pl.split_aiv` (three modes, bare/in-place/in-loop/InCore-function arms, region-shape pins), the `pl.at` kwarg surface (`allow_early_resolve`, `windowize`, `cross_core_slot`, `no_dep_args`/`dumps`, capture+deps, Hierarchy/positional/in-`pl.at`/in-loop), and the CORE_GROUP carrier boundary. Four findings filed and pinned: **F23** (the carrier boundary — confirms and pins the migration review's F1: the parser wraps the whole body statically, tracing opens at the first top-level read; four red spellings), **F24** (nested `pl.split_aiv` accepted by tracing, rejected by the parser), **F25** (`pl.at(name_hint=...)` identifier validation is parser-only — F21's root cause on the `at` descriptor), **F26** (the printer drops the yield LHS inside a `SplitAivScopeStmt` — F22's sibling). One **intended divergence** (`pl.spmd(deps=)` without `as tid`, documented in `docs/en/dev/language/02-manual_dependencies.md:36`), two **needs-decision** no-anchor gaps (the bare non-differentiating spmd body; four `pl.at` kwarg-spelling checks). `scripts/review/check_construct_parity.py` now points `split_aiv` at the new suite and claims `_parse_and_validate_core_num` / `_handle_at_optimizations_kw` (6 constructs still anchorless, down from 7; 2 unclaimed structural parser handlers, down from 4). |
