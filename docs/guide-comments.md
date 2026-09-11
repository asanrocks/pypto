# Guide: Comment and Documentation Rewrite

This is a reference guide for the comment and documentation rewrite that
follows the filesystem restructure and framework renames.

Rewrite design documents, module and API docstrings, implementation comments,
diagnostics, and test documentation to use `docs/glossary.md`. Do not change
runtime behavior, IR, public APIs, or framework identifiers in this commit.

Retain this guide as a cross-check reference after the rewrite. The permanent
vocabulary lives in `docs/glossary.md`.

## Source of Truth

`docs/glossary.md` owns repository-wide vocabulary. Other documents may state
local contracts and rationale, but must not create synonyms or redefine
shared terms.

When an exact API or identifier must be named, write it in backticks. Describe
its behavior with glossary vocabulary rather than turning its spelling into a
prose term.

Example:

```text
Preferred: The pending result construct returned by `pending_result()` remains
           at the top of the active-construct stack.

Avoid:     The completion barrier remains at the top of the Scope stack.
```

## Scope of the Commit

Rewrite prose in:

- `docs/design/` and tracing-related developer documentation;
- package, module, class, method, and function docstrings;
- implementation comments;
- diagnostic and assertion messages containing retired terminology;
- test module docstrings, test docstrings, and explanatory comments;
- documentation indexes that assign terminology ownership.

Do not rename methods, classes, variables, test functions, modules, or files in
this commit. Those changes belong to the preceding restructure guide.

Do not alter code merely to make a comment easier to write. When prose exposes
a behavioral defect or an inaccurate invariant, record it separately rather
than silently changing behavior in this commit.

## Preferred Vocabulary

### Architecture

| Avoid | Use |
| ----- | --- |
| collector runtime, unqualified runtime | trace interpreter (`Interpreter`) |
| spine | trace driver; use `driver` only for the module |
| protocol ceiling | instrumentation boundary |
| entrance bridge | decorator adapter |
| semantic gateway | `Frame` or Frame protocol |
| typing authority | state which module owns operand coercion and result typing |
| Scope stack | active-construct stack |
| collector scope | active construct (`Construct`) |
| symbolic scope | staged construct |
| concrete scope | host construct |
| Frame-lent handle | tracing context (`TraceContext`) |

Use **PTO runtime scope** only for `pl.scope`, `ScopeStmt`, and associated
resource/dependency semantics.

### Execution and values

| Avoid | Use |
| ----- | --- |
| symbolic value | staged value or IR-backed facade |
| concrete value | host value |
| symbolic control flow | staged control flow |
| symbolic-ness | participation in staging |
| concrete mode | host execution |
| symbolic mode | staged execution |
| mode decided by type | value-directed staging |
| trace-once | trace one representative loop iteration |
| late-classified | makes a lazy staging decision |
| mint, minted | create, produce, emit, wrap, or bind, according to the operation |
| minted-value domain | values produced by `Frame`, or the generic `V` result type |

Do not replace one verb mechanically. An IR operation is emitted, a facade is
wrapped, a variable is created, and a source name is bound.

### Trace protocol

| Avoid | Use |
| ----- | --- |
| carrier | trace iterable, trace context manager, trace-bindable object, binding proxy, or marker |
| structural carrier | trace-protocol object, followed by its exact role |
| for-carrier | trace iterable or loop descriptor |
| with-carrier | trace context manager |
| self-closing carrier | trace context manager and its `__trace_exit__` hook |
| trace dunder | trace-protocol hook, or the exact hook name |
| seed | binding proxy, followed by its exact role |
| one-step seed | loop-target binding proxy |
| leak seed | operation-result binding proxy |
| seed attachment | association with the owning active construct |
| deferred activation | deferred region construction |
| bare marker | expression-statement marker |
| bare item | expression statement |

`seed` remains valid for unrelated established uses such as random-number
seeds. Review it contextually rather than banning the word globally.

### Locals and SSA

| Avoid | Use |
| ----- | --- |
| routed namespace | instrumented locals |
| branch shadow | per-branch locals mapping |
| source-name-to-cell map | mapping from source names to mutable `ir.Var`s |
| cell-backed name | staged local |
| store-to-cell | describe the `LetStmt`/`AssignStmt` update and later SSA promotion |
| structural bind | registration of an already-created IR variable without `LetStmt` or `AssignStmt` |
| strict promotion | typed staged initialization requirement |
| concrete-to-constant conversion | host-literal lifting |
| leak path | convergence through mutable locals and SSA promotion |

Use `cell` only when referring to an exact remaining identifier or to an
actual CPython closure cell. Do not describe a PTO `ir.Var` as physical memory.

### Regions and results

| Avoid | Use |
| ----- | --- |
| carry prefix, pending prefix | pending loop-carried arguments or pending `iter_args` |
| carry bundle | group of loop-carried arguments |
| natural loop | loop with inferred loop-carried values |
| explicit loop | loop with explicit `iter_args` |
| void yield | zero-operand yield terminator |
| yield-capable scope | active construct accepting a yield terminator |
| completion barrier | pending result construct or result-binding phase |
| barrier window | result-binding phase |
| completion bind | result binding |
| completion progress | partial result binding |
| completion done | final result binding |
| completion-capable scope | result-producing active construct |
| leak bundle | result-binding group |
| leak witness | result-type witness |
| physical close | region finalization |
| capability token | interface tag |
| nearest capability | innermost active construct advertising the interface tag |

Keep `pl.intrin.carry`, `pl.intrin.yield_`, and `pl.intrin.leak` when referring
to those literal assembly-dialect APIs. Do not use carry, yield, or leak as
general frontend metaphors.

### Lifetime and invariants

| Avoid | Use |
| ----- | --- |
| lifetime bridge | unwind scope guard |
| observe != mutate | observation is side-effect free |
| value/structural split | staged values and trace-protocol objects |
| typing voice | state the exact typing responsibility |
| one-event obligation | state that the next semantic event must be the loop opener |

Prefer a literal invariant over a slogan.

## Docstring Rules

A module docstring should be understandable without reading another module
docstring first. It should state only:

1. the module's role;
2. the state or behavior it owns;
3. its direct dependencies and extension boundary;
4. lifecycle rules that callers must obey.

Do not put a glossary into every module. On first use of a cross-module term,
use the glossary spelling without redefining it.

Class and method docstrings state their local contract: inputs, outputs,
mutation, ownership, and exceptional behavior. They do not narrate the
implementation line by line.

Production docstrings and comments must not cite ADR numbers, document paths,
or historical migration phases. Preserve durable rationale locally and let
git own history.

## Comment Rules

Comments explain intent, invariants, non-obvious ordering, or rejected local
alternatives. Remove comments that merely restate a renamed variable or the
next statement.

Prefer:

```python
# Delay region finalization until every operation result has a source binding.
```

Avoid:

```python
# Leave the completion barrier for the leak seeds.
```

Use exact IR names where they improve precision: `LetStmt`, `AssignStmt`,
`IterArg`, `YieldStmt`, `ScopeStmt`, and `ConvertToSSA`.

## Diagnostic Rules

Diagnostics are user-facing prose. State the invalid program state and the
required correction; do not expose internal metaphors.

Prefer:

```text
pl.intrin.leak: result bindings for the staged for must complete before
another operation is traced
```

Avoid:

```text
ordinary operations reject the completion barrier
```

Preserve exception types, validation order, and tested semantic details. If a
message assertion must change, update only the expected wording.

## Rewrite Order

1. Rewrite `docs/design/` from architecture to detailed language behavior.
2. Rewrite package and module docstrings.
3. Rewrite class and method docstrings.
4. Rewrite implementation comments.
5. Rewrite diagnostics and their expected test messages.
6. Rewrite test documentation.
7. Update `docs/README.md` so `docs/glossary.md` owns cross-module vocabulary.
8. Retain this guide and the structure guide as implementation references.

This order lets lower-level prose rely on terminology already made consistent
at higher levels.

## Search Checklist

These phrases should have no remaining prose occurrences outside this guide or
quoted historical material:

```text
completion barrier
branch shadow
one-step seed
carry prefix
capability token
Frame-lent
store-to-cell
minted-value
leak path
lifetime bridge
protocol ceiling
semantic gateway
entrance bridge
typing authority
strict promotion
bare item
trace-once
late-classified
```

Run:

```bash
rg -n -i 'completion barrier|branch shadow|one-step seed|carry prefix|capability token|Frame-lent|store-to-cell|minted-value|leak path|lifetime bridge|protocol ceiling|semantic gateway|entrance bridge|typing authority|strict promotion|bare item|trace-once|late-classified' python/pypto tests/ut docs -g '!guide-comments.md'
```

Review these context-sensitive terms manually:

```bash
rg -n -i '\b(symbolic|concrete|carrier|seed|cell|scope|runtime|natural loop|explicit loop|minted?)\b' python/pypto tests/ut docs -g '!guide-comments.md'
```

Valid results include exact APIs and identifiers in backticks, genuine
symbolic-expression terminology, concrete dimensions, random seeds, CPython
closure cells, and PTO runtime scopes.

## Verification and Cleanup

Run formatting and the full migration gate after the prose rewrite:

```bash
uv run ruff format --check .
uv run ruff check .
./scripts/gate
```

Before committing, verify that the diff contains no behavioral code changes.
Retain `docs/guide-comments.md` and `docs/guide-restructure.md` as reference
checklists. `docs/glossary.md` remains the only permanent terminology authority.
