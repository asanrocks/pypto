# Documentation Index

## Where things live

| Path                                   | Content                                                                                                                                            |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/design/` (this series)           | Design docs for the tracing frontend, kernel language, and collector — living documents describing current state and the reasoning behind it.       |
| `docs/design/adr/`                     | Standalone ADRs — dated decision records for contract additions whose rationale and rejected alternatives must not be rewritten as the code evolves. |
| `docs/en/dev/`, `docs/zh-cn/dev/`      | Developer documentation (IR, language, passes, codegen, backend). English is ground truth; zh-cn mirrors the same file tree (`tests/lint/check_docs_en_zh_parity.py`). |
| `docs/en/user/`, `docs/en/reference/`  | User guide and reference.                                                                                                                          |
| `docs/en/dev/proposals/`               | Pre-decision proposals.                                                                                                                            |
| Module docstrings                      | The implementation contracts (see the altitude map below).                                                                                          |
| `docs/glossary.md`                     | Repository-wide vocabulary for the tracing frontends — the single owner of shared terminology; other documents use its spelling without redefining it. |

## The design series

| #  | Document                     | Subject                                                       |
| -- | ---------------------------- | ------------------------------------------------------------- |
| 00 | `design/00-tracing.md`       | The tracing architecture — execution as collection, axioms, components, dialects, spans |
| 01 | `design/01-language.md`      | The kernel language — values, typing, control flow, guardrails |
| 02 | `design/02-collector.md`     | The collector — ownership, boundaries, invariants, rejected alternatives |
| 03 | `design/03-entrance.md`     | The entrance tier — declarations, collection, staged calls, and the compiler/dispatcher split |
| —  | `design/adr/0001-expansion-write-through.md` | Expansion parameter write-through — value-level, not scope inheritance |

The design docs are **living**: they are rewritten when reality shifts,
not amended by dated entries. Rationale that must not track the code
(rejected alternatives, the reasons behind invariants) stays in the docs
precisely because it does not change with the code; history that does is
owned by git.

## Documentation altitude map

Each fact has exactly one canonical home. Other mentions are pointers or
altitude-adapted restatements — never transcriptions that must be edited
in lockstep.

| Fact type                            | Canonical home                                                 | Everywhere else                                                            |
| ------------------------------------ | -------------------------------------------------------------- | -------------------------------------------------------------------------- |
| API surfaces (`Frame`, `Interpreter`, `Construct`) | `python/pypto/language/tracer/collector/interface.py`, `interpreter.py` | Design docs point at the module; they never snapshot it                 |
| Decision and rationale               | The design doc                                                 | Docstrings state the contract plus at most a one-line reason               |
| Cross-module terminology              | `docs/glossary.md`                                             | Use the glossary spelling; never coin a synonym or redefine a shared term |
| Vocabularies (interface tags, the event protocol) | The defining module's docstring                  | Call sites use the tag; they add no prose                               |
| User-visible language rules          | `docs/en/dev/language/`, `docs/design/01-language.md`          | Nowhere in code beyond error messages                                      |

Enforcement:

- **Production code never cites documents** — no file paths, no section
  numbers. A docstring that defers to a document is admitting it did not
  state the contract; path citations also rot on every doc
  reorganization. Contracts are stated in code terms
  (`tests/ut/language/test_collector_contract.py`). One sanctioned
  exception: records of the *external* design-notes line (RFC/ADR numbers
  living outside this repository) are annotated, not path-cited — the
  frozen grandfathered set is listed next to the guard test and must not
  grow.
- Markdown references must resolve (`tests/lint/check_md_refs.py`).
