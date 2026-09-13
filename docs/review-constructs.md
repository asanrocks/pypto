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

## 14. Round ledger

| Round | Task | Model | State | Commit | Result |
| --- | --- | --- | --- | --- | --- |
| 0 | seed this tracker | — (orchestrator) | done | `d60a4b303` | plan, contract, evidence base |
| 0 | parity inventory oracle | — (orchestrator) | done | *(this commit)* | `scripts/review/check_construct_parity.py`; 0 stale, 10/16 constructs without anchors, 4 unclaimed structural parser handlers, 17 unclaimed public tracing entries. The gap table in section 5 is now data, not hypothesis |
| 1 | R5 `if` / `else` / `elif` | worker session | done | *(this commit)* | 13 new anchors in `test_if.py` (suite 7 → 20 call sites; 70 → 83 across `test_*.py`), all green. **No `F*` task**: no red anchor. One intended divergence pinned (host predicate dispatch). |
| 1 | R6 `for` + iterators | worker session | done | *(this commit)* | 28 new anchors in `test_for.py` (suite 7 → 35 call sites; 83 → 111 across `test_*.py`), all green — all four iterator kinds, the 1/2/3-arg and computed bound forms, stage pinning, `attrs=`, the INT64/INDEX for-syntax bound, nesting. No red anchor. Target forms are parser-only or rejected by both (pinned): no shared spelling. Four no-anchor acceptance divergences recorded, one filed as **F11** (`attrs=` non-string key → `std::bad_cast`). |
| 1 | R7 `break` / `continue` | worker session | done | *(this commit)* | new suite `test_break_continue.py`: 36 anchors (18 kernels × raw + canonical; conformance-directory call sites 111 → 147), all green — nesting/target selection, the deferred staged exit ordering, exits inside `while`, post-transfer collection, innermost-kind gating, `pl.pipeline`, `pl.spmd`-nested. No red anchor, so no anchor-rule `F*`. Two no-anchor divergences recorded and filed: **F12** (tracing accepts a transfer in a top-level `pl.spmd` / `pl.split_aiv` region and emits an unparseable `break`), **F13** (the parser silently drops a loop `else` suite that tracing rejects loudly). |
| 1 | R8 binding & assignment | worker session | done | *(this commit)* | new suite `test_binding_assignment.py`: 22 anchors (conformance-directory call sites 147 → 169), all green — subscript write (tensor/tile/array, bare/partial/full-axis/in-loop), same-type staged rebind, type-changing rebind, param-name rebind, defs/store order, `pl.const` staged init, tuple unpack, the `pl.Scalar` param world. One red canonical regime filed as **F14** (a type-changing rebind inside a staged loop yields a free variable in the SSA loop init on *both* lanes — a shared §7 hole, not a parity divergence), pinned by an explicit `pytest.raises` test. Three no-shared-spelling/divergence pins: scalar-index rank-reducing write (**needs-decision**), the typed-staged-init guard (**intended divergence**, `docs/design/01-language.md` §4), `pl.runtime` (jit-entry-only, out of scope). |
