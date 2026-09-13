# Gap-Fix TODO — closing the tracing-migration known gaps

Working task file for investigating and fixing the known gaps recorded in
`docs/migration-known-gaps.md`. This is the successor phase to the
migration sweep (`docs/migration-todo.md`): that phase's "no existing-logic
changes" constraint is **inverted here** — the fix IS the logic change.
Scope: `python/pypto/`, `src/`, `include/`, `python/bindings/`, plus the
tests and docs that pin and teach each surface.

Out of scope by triage (do not add tasks for them): **K2** (upstream
adoption — rebase/adopt, never a local fix), **K4** (by design — the
compat tier's contract). **K3** is investigate-only (task F6).

## How this file works

Same contract as `docs/migration-todo.md`: one session = pick one task →
investigate → write the regression test (red) → fix → green → update this
file → commit. The gap entry in `docs/migration-known-gaps.md` is the red
evidence — re-observe it yourself before writing the test. Prefer the
lowest-numbered open task.

Task states: `[ ]` open, `[x]` done, `[~]` blocked (must name why and what
would unblock it). Never delete a task; re-triage in place if the red
evidence does not reproduce.

## Context (read before working)

- `docs/migration-known-gaps.md` — the entry you are fixing: problem,
  evidence, the hypothesized owning surface, and the tracking notes.
- `docs/design/04-migration-guide.md` — the user-visible contract each fix
  restores; several gap entries name the guide section to revisit.
- `AGENTS.md` (repo root) — commands, git discipline.
- `./scripts/gate` — **mandatory** for every task here (logic changes are
  in scope); it runs pyright (incl. the strict collector overlay), ruff,
  and the full ut suite.
- The design series (`docs/design/00-3`) — the axioms the fix must honor,
  especially 01-language §12: a fix must never open a new silent-failure
  path; where the fix cannot be sound, it must be loud.

## Constraints

1. **Test first.** Every task lands a regression test that is red at the
   fix's parent commit and green after. One commit carries test + fix.
2. **Fix the root cause, not the symptom.** The gap entries carry a
   hypothesis for the owning surface — confirm it against the code before
   changing anything, and correct the hypothesis in the gap entry if it is
   wrong. Reject surface patches that leave the failure class open.
3. **Cross-layer consistency.** A change to C++ (`src/`, `include/`)
   rebuilds (`cmake --build build --parallel`) and checks the nanobind
   bindings and any stubs; a change to the tracing surface respects the
   typing gates (`pyrightconfig.collector.json`, strict overlay on the
   tracer trees).
4. **Close the loop.** When a gap closes: remove its entry from
   `docs/migration-known-gaps.md`, revert the workarounds its Tracking
   note lists (skips, xfails, doc caveats, example rewrites), and mark any
   task it blocked (migration-todo S7 for K5). All in the same commit.
5. **Git rules (absolute):** never amend, rebase, reset, checkout, clean,
   or stash; never `git add .`/`-A`/`--no-verify`; stage explicit paths.

## Task inventory

Ordered by user impact: the silent-failure classes first.

- [x] **F1 (K7)** bare `t[:]` writes mis-lower to one-axis subscripts.
  Root-cause `Tensor.__setitem__`'s non-tuple key wrapping
  (`python/pypto/language/typing/tensor.py`) and expand a bare slice to
  the target's rank — the gap notes the unranked-facade difficulty (rank
  known only from the resolved IR var at trace time); investigate whether
  the trace-time var is always resolvable, and be loud where it is not.
  Regression: a rank-2 `out[:] = ...` kernel that reaches ptoas (the gap
  names `examples/intermediate/07_task_graph.py`'s `inferred_edge` and
  `tests/st/runtime/cross_core/test_c2v_narrowed_acc_epilogue.py` as live
  consumers). Close: remove the K7 entry, revert the explicit-slice
  workarounds in the three named consumers, revisit the doc caveats the
  gap's Tracking note lists (tutorials FAQ, 06-syntax, types page).
  Notes: done. Hypothesis confirmed — the owning surface is the non-tuple
  key wrapping shared by `Tensor.__setitem__` and `Tile.__setitem__` (the
  tile path had the same defect); the rank is always resolvable at trace
  time from the staged target's `expr.type` (a `ShapedType`), and a
  non-shaped target type or an over-wide key is a loud `TypeError`. The
  fix ports the parser's numpy rule (dims past the written indices are
  implicit full slices, offsets always full-rank) as a shared
  `subscript_write_offsets` in `python/pypto/language/typing/_subscript.py`.
  Regression: `tests/ut/language/test_subscript_write.py` (red at the fix's
  parent: 6 of 7 failed, including the ptoas-reaching end-to-end); the
  three named consumers never carried workarounds (they kept the bare
  slices and now compile — `07_task_graph.py` and all three modes of
  `03_mixed_kernel.py` verified on a2a3sim, the c2v epilogue programs
  verified through ptoas), so the workarounds reverted were the S10
  split-suite kernels (`tests/st/codegen/dsl/test_split_odd_axis_codegen.py`,
  back to `out[:]`, 9/9 green) and the D11 doc caveats (tutorials chapter,
  FAQ, 06-syntax, en+zh; the tutorials runnable blocks now teach `out[:]`
  and the doc runner is 40/40 with `--check-parity` OK).
- [x] **F2 (K5)** runtime MemRef slot subscripts have no tracing spelling
  — and drop *silently* in annotations. Add the slot-expression path on
  the DSL `MemRef` (`python/pypto/language/typing/memref.py`) accepting a
  staged index, plus the `python/bindings/modules/ir.cpp` binding
  (cross-layer). Regression: the gap's S7 evidence — a multi-buffer
  program whose rotation index is `buf[i % 2]` must form its region, and
  the PTOAS refusal test must raise again. Close: unblock migration-todo
  S7 (rewrite the three runtime-slot programs to the transmute spelling,
  remove the skips and the strict xfail), remove the K5 entry, revisit
  guide §4.4's stale-annotation warning with the now-loud behavior.
  Notes: done. Hypothesis confirmed — the owning surface is exactly the
  DSL `MemRef.__getitem__` → `ir.MemRef.__getitem__` binding pair, which
  accepted only a constant `int` (the staged-index `TypeError` reproduced
  at the transmute descriptor); the bare-local-annotation drop is Python
  semantics (local annotations never evaluate) and stays documentation
  by contract — the loud path is the operation-side spelling, which now
  validates (single-slot declarations, out-of-range constants). The fix
  adds an `Expr`-taking `__getitem__` overload to the binding (constant
  expressions range-checked, runtime indices recorded as `slot_index_`
  for `InitMemRef` to scale) and unwraps a staged `Scalar` in the DSL
  layer (constants folding to the constant path). Regression:
  `tests/ut/ir/test_multi_buffer_tracing.py` (red at the fix's parent:
  module-level collection error — the `buf[i % 2]` transmute raised
  `TypeError`), now 9/9 green: the DSL surface, one PTOAS region per
  declaration with the slot operand an unscaled index, the PYPTO baked
  path, and the co-live refusal. S7 closed in the same commit: the three
  runtime-slot st programs rewritten to the transmute spelling, their
  skips and the strict xfail removed (refusal verified green device-free;
  all three programs compile under PYPTO/DSA_RP and form their regions
  under PTOAS); the K5 entry removed; guide §4.4, Appendix A.4, and the
  en+zh `00-python_syntax.md` slot sections now teach the runtime-index
  transmute spelling.
- [x] **F3 (K1)** ArrayType rebind-carry has no orchestration-codegen
  declaration case. Add the case in
  `src/codegen/orchestration/orchestration_codegen.cpp` (the file's own
  comment block at the carry-declaration site sketches the three existing
  paths; a fresh-array-per-iteration rebind needs a fourth — copy or
  alias semantics per what `ConvertToSSA` derives). C++ work: build first.
  Regression: the phase-fence loop with `tids = tids_next` rebinding
  (the gap's evidence command) must codegen; pin the emitted shape
  against the channel spelling's output. Close: guide §6.3 gains the
  derive-the-carry route; remove the K1 entry.
  Notes: done, with the hypothesis corrected. The carry-declaration site
  already had its ArrayType case (in-place and channel spellings compile
  through it); the missing case was one site over — the catch-all
  `AssignStmt` declaration path (`VisitStmt_(AssignStmtPtr)` →
  `GetCppType`), which the derive-the-carry rebind reaches as a body-level
  Var→Var assign (`tids__ssa_v3 = tids_next__rv_v2`). The fix adds an
  `ArrayType` branch there — `HandleArrayRebindAssign`, mirroring
  `HandleArrayUpdateElementAssign`: alias the LHS's emit name onto the
  RHS's backing array and forward the `array_carry_vars_` /
  `manual_task_id_map_` registrations (alias semantics, per the
  ArrayType-SSA-value-is-a-reference model); state propagates across
  iterations through the existing yield slot-by-slot copy. A non-Var RHS
  stays a loud `InternalError`. This also un-breaks plain non-loop
  `b = a` array aliases (same crash class, verified). Regression:
  `tests/ut/codegen/test_phase_fence_dep_compression.py::
  test_derived_rebind_carry_phase_fence` (red at the fix's parent with
  the gap's exact `GetCppType called for ArrayType` error); it pins the
  rebind shape's emitted code as byte-identical to the channel spelling's
  for the same program. No workarounds existed to revert (the phase-fence
  example and tests legitimately keep the channel spelling, which stays a
  supported surface); the K1 entry removed, guide §6.3 rewritten to teach
  the derive-the-carry route (with the channel spelling kept as the
  explicit alternative), the §4.2/§8 stale caveats dropped, and the en+zh
  `01-statements.md` / `01-orchestration_codegen.md` pages updated.
- [x] **F4 (K8)** `pl.at(predicate=)` has no tracing spelling. Thread a
  `predicate` kwarg through the tracing `AtContext`
  (`python/pypto/language/constructs/scopes.py`) the way
  `SpmdScope._submit_only_scope_attrs` lifts submit-only attrs onto the
  synthesized Submit, with the parser's placement gate (CORE_GROUP, not
  nested in a dispatch wrapper) as the validation model. Regression: the
  gap's doc example traces and emits `CoreTaskPredicate` like the
  `spmd_submit` sibling. Close: restore the `pl.at` scope-form subsection
  and table row in `docs/en/dev/language/02-manual_dependencies.md`
  (+zh); remove the K8 entry.
  Notes: done. Hypothesis confirmed — the owning surface is exactly the
  tracing `at` in `python/pypto/language/constructs/scopes.py` (no
  `predicate` parameter at all); all downstream machinery already existed
  (the outliner's `kAttrPredicate` → `Submit::predicate_` move in
  `scope_outline_utils.cpp`, `AssertDispatchPredicatesHaveACarrier`, and
  the printer's `PrintScopePredicateAttr` — the spmd scope form rides the
  same rails), so the fix is frontend-only. The kwarg validates stagedness
  and re-encodes the parser's full placement gate at the trace-protocol
  boundary (CORE_GROUP level only; not nested inside `pl.cluster()` /
  `pl.spmd()` / another CORE_GROUP `pl.at` — nesting inside a non-CORE_GROUP
  `pl.at` stays legal; the enclosing function must be one
  `OutlineIncoreScopes` rewrites, read off `ctx.function_type()`), then
  appends `("predicate", expr)` in the parser's canonical attr position
  (after `allow_early_resolve`, before `slot_num` / `windowize`, so print →
  reparse round-trips). Regression:
  `tests/ut/language/conformance/test_at_predicate.py` (red at the fix's
  parent: 11/11 failed on the kwarg `TypeError`) — four conformance
  anchors (plain / deps+capture / aer combo / nested in a Hierarchy `pl.at`),
  an attr-position IR pin, and six rejection anchors for the placement
  gates — plus `tests/ut/codegen/test_predicate_codegen.py::
  test_at_scope_predicate_emits_set_predicate_block`, the end-to-end MoE
  shape through the Default pipeline emitting `CoreTaskPredicate` +
  `set_predicate` exactly like the `spmd_submit` sibling (the doc example
  itself verified verbatim through the same pipeline). No test workarounds
  existed to revert (D13 only rewrote the doc page). The K8 entry removed;
  the en+zh manual-dependencies pages regained the `pl.at` scope-form
  subsection and the table-row `predicate=` clause (adapted to tracing:
  rejections fire at trace time, the producer-in-`deps=` contract stays
  unchecked); the 04-migration-guide row's gap caveat dropped.
- [x] **F5 (K6)** the tracing jit has no `@pl.jit.graph` alias. One
  `_JitAlias("graph", function_type=ir.FunctionType.Graph)` member in
  `python/pypto/language/tracer/decorators/jit.py` — but investigate the
  alias's callee-dispatch story first (the gap's Tracking note: the
  standalone spelling orphans dispatched jit callees; the alias must
  assemble the program like the other jit entries or the restoration
  would be a trap). Regression: `@pl.jit.graph` decorating both a
  standalone fragment and a dispatching entry traces and assembles.
  Close: revert the two S9 consumers if the alias is a full replacement;
  remove the K6 entry.
  Notes: done. Hypothesis confirmed and the trap is void: the alias routes
  through `jit_trace` like every other alias, so the callee-dispatch story
  comes from the existing jit lanes, not the alias — an eagerly shaped
  fragment traces at decoration under a `_ProgramAssembly` (its record keeps
  the callees reachable), and a bare-annotated fragment becomes a
  `_JitSpecializer` whose in-trace call takes the callee lane
  (`_detached_resolve`: per-call Graph variant, deduped into the *caller's*
  assembly, dispatched callees nested inside it registering there too). The
  orphaning the gap named was a property of the `@pl.function` spelling
  (no assembly), which the alias never shares. Regression:
  `tests/ut/jit/test_graph_alias.py` (red at the fix's parent: module-level
  collection error — the gap's exact `AttributeError`) — eager conformance
  vs `@pl.function(type=pl.FunctionType.Graph)` (structural equality,
  auto-mapping), the `level=` rejection, and the dispatching-fragment
  anti-orphan anchor (a `@pl.jit.graph` fragment calling an incore jit
  kernel from a `@pl.jit` entry: the program under HOST_BUILD_GRAPH carries
  entry + Graph + dispatched callee). Both S9 consumers reverted to
  `@pl.jit.graph`: the st file's six fragments (bare annotations, the
  pre-S9 spelling; the `compile_for_test`-based `_lower` stays — that was
  the S9 lower()-API migration) and the moe example (`moe_block` back to
  `@pl.jit.graph`, called from `moe_decode`; device-free verified — 1 Graph
  with `expert_ffn` joined, all six compile-side st tests green, 16 tests
  collected). The K6 entry removed; the en+zh `01-functions.md` jit-family
  table regained the `@pl.jit.graph` row and the Graph section now teaches
  the alias (the standalone-vs-scope constraint reworded: the *explicit*
  `@pl.function` spelling orphans dispatched callees, `@pl.jit.graph` never
  does); the D6-era rewordings in `08-outline_graph_scopes.md`,
  `passes/index.md`, and `ir/01-hierarchy.md` restored to name
  `@pl.jit.graph` where they described the standalone Graph function
  (`execution/00-compile.md`'s `runtime=` row never named the dead spelling
  and stays).
- [x] **F6 (K3, investigate-only)** flash-attention Vec-space overflow
  (197632 > 188416 bytes). Triage complete: the overflow itself is ring-depth
  capacity pressure, not a demonstrated planner defect — the automatic
  cross-core pipe reserves 65536 B at the default depth 2 (32768 B slot × 2)
  and the packed tiles need ~132 KB, which capacity-gated reuse cannot fit at
  any double-buffering depth; `pl.cross_core_slot(slot_num=1)` relieves the
  allocator (the packing then fits at ~165 KB). Relieving the footprint does
  NOT unblock the example: with slot_num=1, compilation instead fails at
  codegen on a second, newly-found defect — `pl.assemble` of a cube-produced
  (Acc) source into a *local* Vec target loses its producer in
  `ExpandMixedKernel` (no Acc→Vec bridge is inserted, the AIC lane's DCE
  silently deletes the consumerless matmul) and codegen reports a misleading
  `pto.subview` blayout mismatch. Blocked on F7 (below); evidence and root
  cause recorded in the K3 entry and pinned by the strict-xfail tests in
  `tests/ut/codegen/test_assemble_acc_source_codegen.py`. What would unblock:
  F7 landing, then a footprint re-triage (`cross_core_slot` override or tile
  sizing) to unskip the example body and remove the K3 entry.
  Notes: closed out by F7 — the example body was unskipped there with the
  `cross_core_slot(slot_num=1)` override this task identified (the footprint
  half of K3), so no separate work remains here.
- [x] **F7 (K3, found by F6's triage)** `tile.assemble` with a cube-produced
  (Acc) source into a local Vec target. Root cause: the op registers no
  input-space constraint for its source (the legal space depends on the
  target — a Vec target needs a Vec source, an Acc→Mat FIXPIPE downcast stays
  cube-side), so `InferTileMemorySpace` queues no Acc→Vec bridging
  `tile.move`; `ExpandMixedKernel` then splits the matmul (CUBE) from the
  assemble (VECTOR) with no boundary in between, and the AIC lane's DCE
  silently deletes the consumerless matmul, leaving the AIV assemble
  referencing an undefined operand. Codegen fails with a misleading
  `tile.assemble: blayout mismatch ... pto.subview` at
  `pto_ops_shared.cpp` `CheckSubviewTileCompat`. Fix the root cause — bridge
  the source to the target's space (the working pattern exists: the
  flash-attention else-branch's explicit `tile.move(Acc→Vec)` becomes the
  tpush/tpop boundary), or reject loudly at expansion time naming the
  pattern. The regression tests are already red:
  `tests/ut/codegen/test_assemble_acc_source_codegen.py` (strict xfails; the
  fix must flip them to XPASS and remove the markers). Afterwards, re-triage
  the flash-attention footprint per F6 and close K3 by unskipping the
  example body.
  Notes: done. Hypothesis confirmed — the root cause is exactly the missing
  input-space constraint, and the fix is the bridge: `tile.assemble` now
  registers a *dynamic* matches-arg constraint
  (`set_input_memory_matches_arg(1, 0, allow_acc_when_mat=true)` on
  `OpRegistryEntry`): the source must share the target's resolved space,
  with the converting Acc→Mat writeback (FIXPIPE — both the same-dtype
  full-window subview/tmov path and the bf16/f16 `pto.tinsert` path) as the
  one exemption, since the existing Mat-scratch tests pin that an Acc source
  against a Mat target must stay unbridged. `InferTileMemorySpace` evaluates
  it in Phase 2/3 through a shared `RequiredSpaceForArg` (static constraint
  first, matches-arg second, evaluated against `var_memory_` with the
  requirement operand's TileType annotation as fallback), so an Acc source
  against a Vec target gets the explicit Acc→Vec `tile.move` queued ahead of
  the assemble — the boundary `ExpandMixedKernel` folds into tpush/tpop.
  When the requirement resolved to a space nothing can copy into (an Acc
  target with a copyable source), the mismatch is a loud user-facing
  `CHECK_SPAN` naming the pattern instead of the misleading codegen layout
  error; the `TileMemoryInferred` verifier re-checks the constraint
  annotation-only as the net. Regression: the two former strict xfails in
  `tests/ut/codegen/test_assemble_acc_source_codegen.py` flipped and lost
  their markers, plus three new unit anchors in
  `tests/ut/ir/transforms/test_infer_tile_memory_space.py::
  TestInferTileMemorySpaceAssembleSource` (the bridge, the Mat exemption,
  the loud Acc-target rejection). K3 closed in the same commit: the
  flash-attention example got the F6 footprint fix
  (`optimizations=[pl.cross_core_slot(slot_num=1)]` on the `pl.at`), the
  skip guard is gone, and the body now runs end-to-end on a2a3sim and
  validates against a fp32 golden (the else-branch running max was also
  corrected to use the carried `mi_update` — the parser-era body never
  compiled, so its online-softmax rescale had never been numerically
  exercised). Found on the way and filed as F8: return-style jit dispatch
  returns `()`.
- [x] **F8 (found by F7's example re-triage)** return-style jit dispatch drops
  the return values. A `@pl.jit` kernel whose outputs are *return values* (no
  `pl.Out` parameters) compiles and executes, but the host call returns `()`.
  Root-cause hypothesis: `_invoke_compiled`
  (`python/pypto/ir/compiled_program.py`) packs only `output_indices` (pure
  Out parameters) and never the `return_types`, so the synthesized output
  buffer (`ret0__out`) is written on device and then discarded — return-style
  output packing was never implemented (every model example uses the in-place
  `pl.Out` spelling, so nothing red exists). Evidence: a two-tensor kernel
  `with pl.at(level=pl.Level.CORE_GROUP): c = pl.add(a, b)` / `return c` run
  with `config=RunConfig(platform="a2a3sim")` returns `()`, while the same
  body with `o: pl.Out[pl.Tensor]` + `o[:] = c` writes correctly;
  `compiled.has_return` is `True` and `output_indices` is `[]` on the
  return-style artifact. Pre-existing runtime logic, not migration-caused.
  Regression when fixed: the return-style kernel's dispatch result equals
  `a + b`.
  Notes: done, and the hypothesis was only half right — `_invoke_compiled`'s
  packing was a symptom, not the cause. The host has exactly one output
  channel: a caller-supplied tensor argument. `ConvertTensorToTileOps`
  appends an Out param to every InCore callee for its returns and allocates
  the buffer at the call site, but for the *entry* that call-site buffer is an
  internal `tensor.create` — orchestration codegen writes the task output
  into a scope-local allocation nothing exports, and the artifact's calling
  convention has no output position at all (so there was nothing for
  `_invoke_compiled` to pack). The fix is a new pass,
  `MaterializeEntryReturns` (runs after ConvertTensorToTileOps, before
  NormalizeReturnOrder): for each *dispatch entry* (an Orchestration function
  no other function calls — multi-orch programs have several, L3 chip tasks
  under a host orchestrator are callees and excluded), every tensor return
  that does not already write back an Out/InOut param is traced to its
  backing buffer (new `return_lineage::TraceToBackingBuffer`, the
  return-style counterpart of the param-writeback trace); a fresh top-level
  `tensor.create` (the synthesized call-site buffer, or a user
  `pl.create_tensor`) is deleted and appended to the entry's params as an Out
  param. Downstream everything reuses the in-place machinery: codegen binds
  the promoted buffer to `orch_args.tensor(<n>)`, and `ir.compile` now hands
  `CompiledProgram` the post-pass program (`_metadata_program`) so the
  dispatch metadata reports the effective signature (source-level names
  restored positionally) — `_invoke_compiled` needed no change of its own.
  Returns that trace to no buffer are deliberately left as-is: a reshaped
  alias of an Out param and an extern-kernel call result already deliver
  through their declared Out args (rejecting them loudly broke those
  in-place programs), and a graph-level computed value (`tensor.add` at entry
  level) cannot be code-generated at all — compute ops are only registered
  inside InCore blocks. `_coerce_args` gained a third convention for the
  mixed shape this creates — a *partial in-place* call (the declared
  signature of a kernel with Out params plus a materialized return) binds
  the supplied buffers positionally, auto-allocates the omitted trailing
  outputs, and returns exactly those. Regression: the red-first
  `tests/ut/jit/test_return_style_dispatch.py` (pipeline/codegen/meta
  layers), the partial-in-place coercion tests in
  `tests/ut/ir/test_compiled_program.py`, and the end-to-end dispatch check
  in `tests/st/runtime/framework_and_models/test_jit.py` — the return-style
  kernel's result equals `a + b` on a2a3sim (multi-return, mixed-style,
  `from_dir` reload, and jit cache-hit dispatch verified there too; the
  in-place examples and flash-attention still pass end-to-end).

## Discovery protocol

New gaps found during a fix are appended here as new F tasks (and, if
they are migration-relevant, as entries in `docs/migration-known-gaps.md`
first — the gap file is the evidence record; this file is the work list).

## Commit protocol

Same as `docs/migration-todo.md`: the work and this file's state edit land
in one commit; explicit paths; the task id in the body; run
`./scripts/gate` before committing (PYPTO_GATE_LOG to a /tmp log for
focused reading). When a task removes a gap entry, the commit message
states the root cause in one sentence.
