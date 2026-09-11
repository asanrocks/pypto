# ADR-0001: Expansion parameter write-through

- **Status:** Accepted. Implementation landed (rmsnorm corpus unit,
  2026-09-10); acceptance tests and the hardening item below are owed before
  the contract is load-bearing for more corpora.
- **Scope:** the `pl.expand` item-assignment contract; `PtoFrame` local
  resolution; the Tensor/Tile/Array `__setitem__` paths.
- **Affects:** `docs/design/03-entrance.md` §7 (amended by this record),
  the collector ownership note (§4.4 of the same document), and the
  corpus convention that expansion bodies write only body-bound views.

## Context

Converting a jit-inline kernel to `@pl.expand` (the DeepSeek refactor's
manual-inline recipe) exposed a gap. Expansion parameters are
**substitution**: the body reads the caller's facades directly. But
item-assignment (`x_normed[...] = chunk`) resolves its write target by
looking up the facade's `ir.Var` in the *current* frame's staged-local
map — and a substituted parameter is not in the expansion's fresh
lexical frame. Every such store raised
`Tensor item-assignment target must be a bound local name`.

The corpus's own precedent (`qkv_proj_rope`) never writes a parameter
directly: it binds views (`kv_view = pl.reshape(out.kv, ...)`) and writes
those. That works when a view is natural. For a plain out-parameter whose
shape is already the working shape (`x_normed`), no view exists, and the
workarounds are wrong (see Rejected alternatives).

## Decision

**Item-assignment targeting a substituted parameter rebinds the caller's
own binding.** Resolution is by value identity against the *owning*
frame — the expansion's own locals first, then the borrowed parent — and
the rebind is routed through the owning frame's own trace interpreter.

This is **value-level write-through, not lexical-scope inheritance**:

- The body's fresh lexical frame stands. Its own locals bind there.
- No caller name is visible to the body; the caller's locals enter only
  as argument values. The parent-frame lookup exists only to find the
  *owner of a value already in hand*.
- Plain rebinding of a parameter name in the body (`x = expr`) shadows
  locally, exactly like a Python function call; it never reaches the
  parent. A locally-shadowed name therefore never routes up.
- Control (`break`/`continue`) and `return` remain expansion-local.

The reference model is Python's mutable-object passing: `lst.append(...)`
inside a callee is visible to the caller without the callee inheriting
the caller's scope. The body can only write the binding it was explicitly
handed.

Owner-frame routing also keeps ownership intact in the letter of
02-collector §2: the parent Frame still owns its `ir.Var`s, and the
rebind executes in the parent interpreter's own bind path — branch-local
mappings, staged-ness flip guards, and the parent's control-flow context
all consult the parent's suspended-but-live state, which is exactly the
state textual inlining would run against.

## Why not the alternatives

- **Copy-let alias** (`v = x_normed; v[...] = ...`): the assigns target a
  fresh var; the caller's binding never sees the write. That is not
  inlining — under substitution the caller's later reads and `return`
  reference the un-updated var. Also survives the refactor verifier's
  cleanup prefix (known from the unpack-copy-let experience), so it does
  not even converge canonically.
- **Same-shape reshape view** (the qkv pattern): emits an extra reshape
  op and misroutes assigns exactly like the alias. Converges only when a
  view is needed anyway and consumers read memory rather than the
  binding.
- **Keep the loud error**: restricts the expansion conversion to
  view-written kernels, leaving plain-out-param kernels — the majority of
  leaf norms and mixes — on jit.inline. The refactor's goal is precisely
  to make those expansion blocks with explicit contracts.
- **Weaken the canonicalizer** to fold the alias forms: rejected by the
  parity ladder's own rule — mirroring is preferred over teaching the
  comparator new equivalences.

## Consequences

- `03-entrance.md` §7 gains the contract sentence: an item-assignment
  targeting a substituted parameter rebinds the caller's binding
  (resolved by value identity against the owning frame); the body's name
  resolution and plain rebinding stay in the fresh lexical frame.
- The collector ownership record (which excludes overlapping Frames)
  gains a sanctioned bounded exception: an expansion frame may *resolve
  and route* a rebind into the borrowed parent, one direction, values
  only. This is not the general overlap nested expansion would need.
- **Hardening owed:** the pending-result rejection in the setitem path
  now consults the owning interpreter; a write-through issued while the
  expansion's *own* staged construct awaits result bindings can slip that
  guard. It must be closed (explicit rejection or an ambient check) and
  covered by a test.
- Acceptance tests owed in `tests/ut/language/entrance/`: basic
  write-through (caller observes the write), write-through from inside a
  staged branch, the shadowing rule (plain rebind never routes up), and
  the unchanged loud error for expression-backed non-locals.
- Related observation from the same corpus unit (not part of this
  decision): because an expansion body binds against the caller's scope
  properties, `n = pl.tensor.dim(x, i)` folds to the caller's declared
  extent symbol where the jit-inline sub-kernel could not — the deferred
  "extent-symbol fold going uniform" item moves closer via expansions.

## Relationship to the ladder

The IR rung alone cannot witness this semantics once both corpora carry
the same source (mirroring makes it same-source). The load-bearing
verifier is entrance parity against the frozen AST corpus, where the
write-through form is compared post-pass against the inlining pass's
output — the same statement structure `InlineFunctions` produces.
