# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Targeted pyright gap report: signal only, no annotation-labor noise.

Three passes, all strict-typed but scoped to where findings change behavior:

* **Entrance tier** — ``python/pypto/language``'s kernel-author surface
  (``tracer/decorators``, ``tracer/ambient.py``, ``facade.py``,
  ``intrin.py``, ``typing``, ``compat``, ``op``, ``distributed``, and the
  language root modules) under full strict. This is the surface kernel
  authors touch; every ``Unknown`` leak here blinds a user's type checker.
* **IR contract tier** — ``python/pypto/ir/{builder.py,op,utils.py,type.py}``
  and the hand-written ``python/pypto/pypto_core`` binding stubs under full
  strict. Both frontends compile against this layer, so a stub lie or an
  arity drift poisons every downstream strict check; ``reportCallIssue``
  fires here even though the repo default disables it. The torch/ctypes
  runtime-interop files (``compiled_program.py``,
  ``distributed_compiled_program.py``, ``op_conversion.py``) stay out on
  purpose: their residue is annotation labor, not contract signal.
* **Tests residue** — the tracing-frontend tests (``tests/ut`` minus the
  parser-owned trees ``tests/ut/language/parser`` and ``tests/ut/ir/parser``)
  under strict with the *mechanical* rule families disabled
  (``reportUnknown*``, missing-parameter, unused-*). Parser trees are
  excluded because their fixtures are round-trip text typed against a
  tracing surface they never execute (e.g. ``init_values=`` is
  printer/parser-owned spelling). What remains is classified against the
  known root causes below instead of dumped raw; new diagnostic kinds fall
  into a residual bucket for triage.

Both library tiers run default strict rules only, with the ``Unknown``
families ON — on library files ``Unknown`` is the tier's charter, unlike
in the tests pass. The optional strict-only rule families
(implicit-override, missing-super-call, unused-call-result, implicit
string concatenation, uninitialized-instance-variable) stay off
everywhere: they collide systematically with deliberate DSL idioms
(side-effect calls that build IR, metaclass-supplied state, non-chaining
``__init__`` wrappers), measured by hand. Library-tier findings are raw,
not classified — they are actionable gaps; fix them and the tier goes
quiet.

Known root causes are collapsed to one line each so the report stays a gap
list, not an alarm list. Remove a collapsed category here when its library
fix lands — the diagnostics then simply disappear.

Known root causes (verified by hand, see the tracing-frontend analysis):

1. ``pl.<sym>`` sites that pypto.language's ``__init__`` fails to re-export.
   Parser tests never execute them (the AST parser reads source text), so
   pytest cannot see the latent AttributeError — pyright can.
2. ``@pl.jit`` kernels with bare ``pl.Tensor`` annotations read
   ``x.shape`` from source; the shape is sample-determined at compile
   time, statically unknowable (the jit gate rejects bare annotations
   without sample tensors — pyright's ``"None" is not iterable`` is the
   same signal).
3. Self-less kernels inside ``@pl.program`` classes (SubWorker
   declarations): ``is_class_method`` sanctions "self-contained functions
   defined inside a class", but pyright's method-protocol check demands a
   supertype first parameter.
4. ``pl.Tuple`` duality: the annotation names the symbolic facade while
   the ergonomic source returns a host tuple (``return a, b``) — the
   dual-purpose design (see ``typing/tuple.py``) makes the two statically
   disjoint; the ``__class_getitem__`` hook accepts the subscript, the
   assignability gap remains by design.

Retired categories, kept as record: the collector-import collateral died
when the tests pass learned to mirror the tracer conftest's ``sys.path``
insert in ``extraPaths`` — the suite's residue is live signal now; the
carrier-protocol negative fixtures (``test_collector_contract.py``) are
excluded from the scan — every diagnostic in that file is the fixture's
point, and pyright has no file-level ignore to say so in place; and the
Out-param rebinding category was retired by disabling
``reportRedeclaration`` in the tests pass — every test redeclaration was
the ``Out[T]``-is-``T`` idiom or fixture rebinding, and the repo default
already disables the rule for the dsl parser.

(The former ``RangeIterator.__next__`` tuple-variant leak classifier was
removed when its fix landed: ``__next__`` now returns the traced yield
Type ``T`` and the factories overload on ``init_values=``.)

Usage (informational; always exits 0 unless pyright itself fails):

    uv run python scripts/report_surface_gaps.py
    uv run python scripts/report_surface_gaps.py --only entrance
    uv run python scripts/report_surface_gaps.py --only ir
    uv run python scripts/report_surface_gaps.py tests/ut/language/tracer
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The library surface kernel authors call; the collector overlay
# (pyrightconfig.collector.json) already covers the tracer collector,
# protocol/semantics/constructs, and language/constructs — these are the
# tiers it does not reach.
ENTRANCE_INCLUDES = [
    "python/pypto/language/tracer/decorators",
    "python/pypto/language/tracer/ambient.py",
    "python/pypto/language/facade.py",
    "python/pypto/language/intrin.py",
    "python/pypto/language/typing",
    "python/pypto/language/compat",
    "python/pypto/language/op",
    "python/pypto/language/distributed",
    # Language root modules: kernel-authored imports resolve through them
    # (pl re-exports, dsl_api, scope), so a lie there reaches users the
    # same way as one in the facade.
    "python/pypto/language/__init__.py",
    "python/pypto/language/dsl_api.py",
    "python/pypto/language/scope.py",
    "python/pypto/language/tracer/__init__.py",
]

# The shared layer both frontends compile against: IR builders/op wrappers
# plus the hand-written C++ binding stubs. Runtime-interop files (compiled
# programs, op_conversion) are excluded — their strict residue is annotation
# labor over torch/ctypes, not contract signal (see the module docstring).
IR_TIER_INCLUDES = [
    "python/pypto/ir/builder.py",
    "python/pypto/ir/op",
    "python/pypto/ir/utils.py",
    "python/pypto/ir/type.py",
    "python/pypto/pypto_core",
]

# Mechanical annotation debt: cascades seeded by untyped test-local helpers,
# fixture parameter annotations, and unused test scaffolding. Dropping them
# by config is the whole point — see the module docstring.
MECHANICAL_RULES_OFF = [
    "reportUnknownMemberType",
    "reportUnknownVariableType",
    "reportUnknownArgumentType",
    "reportUnknownParameterType",
    "reportUnknownLambdaType",
    "reportMissingParameterType",
    "reportMissingTypeArgument",
    "reportUnusedVariable",
    "reportUnusedFunction",
    "reportUnusedClass",
    "reportUnusedCallResult",
    "reportUnusedImport",
    "reportUnnecessaryComparison",
    "reportUninitializedInstanceVariable",
]

# The parser-owned round-trip fixture trees (see the migration contract):
# their text is typed against a tracing surface it never executes, so
# strict findings there are dialect noise, not signal.
PARSER_TREES = ["tests/ut/language/parser", "tests/ut/ir/parser"]

TESTS_DEFAULT_INCLUDES = ["tests/ut"]
COMMON_SETTINGS: dict[str, object] = {
    "typeCheckingMode": "strict",
    # jit roundtrip is parser-owned; collector contract is a negative fixture
    # (deliberate protocol violations — see the module docstring). Pyright
    # has no file-level ignore, so both are excluded here instead.
    "exclude": [
        "**/__pycache__",
        "tests/ut/jit/test_roundtrip.py",
        "tests/ut/language/test_collector_contract.py",
    ],
    "extraPaths": ["python", "."],
    "pythonVersion": "3.10",
    "reportMissingTypeStubs": False,
    "reportMissingModuleSource": False,
    "reportPrivateUsage": False,
}

# --- classification patterns -------------------------------------------------

# `"spine" is not a known attribute of module "pypto.language"` — sites
# whose symbol the language facade fails to re-export (cause 1).
MODULE_ATTR = re.compile(r'"([^"]+)" is not a known attribute of module "([^"]+)"')
# The @pl.jit suite (cause 2).
JIT_SUITE = "tests/ut/jit"


@dataclass
class Diag:
    file: str
    line: int
    rule: str
    message: str

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"


# --- pyright driver ----------------------------------------------------------


def run_pyright(config: dict[str, object]) -> list[Diag]:
    """Run pyright with ``config`` against a temp config at the repo root.

    Pyright resolves ``include``/``extraPaths`` relative to the config file's
    directory, so the temp file must sit at the root next to ``python/``.
    """
    tmp = REPO_ROOT / "pyrightconfig.surface-scan.tmp.json"
    tmp.write_text(json.dumps(config), encoding="utf-8")
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, venv-scoped
            ["uv", "run", "pyright", "-p", tmp.name, "--outputjson"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=1800,
            check=False,
        )
        if proc.returncode not in (0, 1):  # 1 = diagnostics found, not a failure
            print(f"pyright failed (exit {proc.returncode}):\n{proc.stderr}", file=sys.stderr)
            raise SystemExit(2)
        payload = json.loads(proc.stdout)
    finally:
        tmp.unlink(missing_ok=True)

    diags: list[Diag] = []
    for item in payload.get("generalDiagnostics", []):
        if item.get("severity") != "error":
            continue
        diags.append(
            Diag(
                file=str(item["file"]).replace(f"{REPO_ROOT}/", ""),
                line=int(item["range"]["start"]["line"]) + 1,
                rule=str(item.get("rule", "?")),
                message=str(item["message"]).splitlines()[0],
            )
        )
    return diags


# --- classification ----------------------------------------------------------

# The library tiers scan a *subset* of the import graph, so pyright's
# unused-symbol rules fire for seam symbols whose consumers sit outside the
# include set — e.g. the typing seam shared between the parser and the
# tracer semantics (both outside the entrance tier). A conservative text
# scan over python/ decides: a symbol referenced by any other file is a
# cross-tier seam, not dead code, and is collapsed instead of reported.
UNUSED_RULES = frozenset({"reportUnusedFunction", "reportUnusedClass", "reportUnusedImport"})
UNUSED_NAME = re.compile(r'^\w+ "([^"]+)" is not accessed')


def _referenced_outside(symbol: str, own_file: str) -> bool:
    pat = re.compile(rf"\b{re.escape(symbol)}\b")
    for path in (REPO_ROOT / "python").rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        if str(path.relative_to(REPO_ROOT)) == own_file:
            continue
        if pat.search(path.read_text(encoding="utf-8")):
            return True
    return False


def split_cross_tier_seams(diags: list[Diag]) -> tuple[list[Diag], list[Diag]]:
    """Partition into (real signal, cross-tier seam unused-symbol findings)."""
    seams: list[Diag] = []
    signal: list[Diag] = []
    for d in diags:
        if d.rule in UNUSED_RULES and (m := UNUSED_NAME.match(d.message)):
            if _referenced_outside(m.group(1), d.file):
                seams.append(d)
                continue
        signal.append(d)
    return signal, seams


@dataclass
class Classified:
    pl_export_gaps: list[Diag]
    possibly_unbound: list[Diag]
    jit_shape_reads: list[Diag]
    selfless_class_kernels: list[Diag]
    tuple_duality: list[Diag]
    residual: list[Diag]


def classify(diags: list[Diag]) -> Classified:
    out = Classified([], [], [], [], [], [])
    for d in diags:
        if (m := MODULE_ATTR.search(d.message)) and m.group(2) == "pypto.language":
            out.pl_export_gaps.append(d)
        elif d.rule == "reportPossiblyUnboundVariable":
            out.possibly_unbound.append(d)
        # cause 2: sample-determined shapes read from source in @pl.jit bodies
        elif (
            d.rule == "reportGeneralTypeIssues"
            and '"None" is not iterable' in d.message
            and d.file.startswith(JIT_SUITE)
        ):
            out.jit_shape_reads.append(d)
        # cause 3: self-less kernels in @pl.program classes
        elif d.rule == "reportGeneralTypeIssues" and "must be a supertype of its class" in d.message:
            out.selfless_class_kernels.append(d)
        # cause 4: host-tuple values vs the symbolic pl.Tuple annotation
        elif d.rule in ("reportAssignmentType", "reportReturnType") and "Tuple" in d.message:
            out.tuple_duality.append(d)
        else:
            out.residual.append(d)
    return out


def locate_definitions(symbol: str) -> list[str]:
    """Where does the missing ``pl.<symbol>`` actually live? Definition-site
    candidates in the language tree, so the fix is a re-export, not a hunt."""
    hits: list[str] = []
    pat = re.compile(rf"^(?:def|class) {re.escape(symbol)}\b|^{re.escape(symbol)}\s*[=:]")
    for path in sorted((REPO_ROOT / "python/pypto/language").rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pat.match(line):
                hits.append(f"{path.relative_to(REPO_ROOT)}:{i}")
                break
    return hits[:3]


# --- report ------------------------------------------------------------------


def print_bucket(title: str, diags: list[Diag], note: str, samples: int = 5) -> None:
    print(f"\n== {title}: {len(diags)} ==")
    print(note)
    for d in diags[:samples]:
        print(f"  {d.where}  [{d.rule}] {d.message[:100]}")
    if len(diags) > samples:
        print(f"  ... and {len(diags) - samples} more")


def report_library_tier(title: str, diags: list[Diag]) -> None:
    signal, seams = split_cross_tier_seams(diags)
    print(f"### {title} (strict): {len(signal)} findings")
    if seams:
        names = sorted({m.group(1) for d in seams if (m := UNUSED_NAME.match(d.message))})
        print(
            f"  ({len(seams)} unused-symbol findings collapsed — {', '.join(names)}: "
            "referenced by files outside this tier's include set; cross-tier "
            "seams, not dead code)"
        )
    if not signal:
        print("  clean.")
        return
    by_rule = Counter(d.rule for d in signal)
    for rule, n in by_rule.most_common():
        hint = ""
        if rule == "reportUnnecessaryIsInstance":
            hint = "  <- annotation/runtime-check disagreement: trust signal"
        print(f"  {n:>4}  {rule}{hint}")
    by_file = Counter(d.file for d in signal)
    print("  top files:")
    for f, n in by_file.most_common(5):
        print(f"  {n:>4}  {f}")
    print("  samples:")
    for d in signal[:5]:
        print(f"    {d.where}  {d.message[:100]}")


def report_tests(diags: list[Diag]) -> None:
    print(f"\n### TESTS RESIDUE (strict, mechanical rules off): {len(diags)} findings")
    c = classify(diags)

    if c.pl_export_gaps:
        print_bucket(
            "pl.<sym> not re-exported by pypto.language (latent AttributeError)",
            c.pl_export_gaps,
            "  parser tests never execute these sites, so pytest cannot see the gap.\n"
            "  Fix: add to language/__init__.py re-exports.",
        )
        symbols = Counter(m.group(1) for d in c.pl_export_gaps for m in [MODULE_ATTR.search(d.message)] if m)
        print("  by symbol (definitions located in the language tree):")
        for sym, n in symbols.most_common():
            defs = locate_definitions(sym)
            where = ", ".join(defs) if defs else "NO DEFINITION FOUND"
            print(f"    {n:>4}  {sym:<20} -> {where}")
    if c.jit_shape_reads:
        print(f"\n== KNOWN: @pl.jit .shape reads on bare pl.Tensor params: {len(c.jit_shape_reads)} ==")
        print(
            "  root cause: the jit frontend reads `x.shape` from source; with a\n"
            "  bare pl.Tensor annotation the shape is sample-determined,\n"
            "  statically unknowable (the jit gate itself rejects bare\n"
            "  annotations without sample tensors).\n"
            "  Collapsed — no sites listed."
        )
    if c.selfless_class_kernels:
        print(f"\n== KNOWN: self-less kernels in @pl.program classes: {len(c.selfless_class_kernels)} ==")
        print(
            "  root cause: is_class_method sanctions self-contained functions\n"
            "  defined inside a class (SubWorker declarations), but pyright's\n"
            "  method-protocol check demands a supertype first parameter.\n"
            "  Collapsed — no sites listed."
        )
    if c.tuple_duality:
        print(
            f"\n== KNOWN: pl.Tuple duality (host-tuple values vs the facade "
            f"annotation): {len(c.tuple_duality)} =="
        )
        print(
            "  root cause: the annotation names the symbolic facade while the\n"
            "  ergonomic source returns a host tuple (return a, b) — the\n"
            "  dual-purpose design documented in typing/tuple.py.\n"
            "  Collapsed — no sites listed."
        )
    if c.possibly_unbound:
        print_bucket(
            "possibly-unbound in (mostly inline) kernel bodies",
            c.possibly_unbound,
            "  pyright is correct — many tests pin UnboundLocalError semantics on\n"
            "  purpose. Triage per site: documented DSL semantic -> targeted ignore;\n"
            "  otherwise the DSL's static sema may want the same check.",
        )
    if c.residual:
        print_bucket(
            "RESIDUAL (unclassified — triage me)",
            c.residual,
            "  no known root cause yet; new kinds land here first.",
        )
        by_rule = Counter(d.rule for d in c.residual)
        print("  by rule:")
        for rule, n in by_rule.most_common():
            top = Counter(d.file for d in c.residual if d.rule == rule).most_common(1)[0]
            print(f"    {n:>5}  {rule:<32} top: {top[0]} ({top[1]})")


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Targeted pyright gap report: signal only, no annotation-labor noise."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="override the tests-pass include roots (default: tests/ut)",
    )
    parser.add_argument("--only", choices=["entrance", "ir", "tests"], help="run a single pass")
    args = parser.parse_args()

    tests_includes = args.targets or TESTS_DEFAULT_INCLUDES

    if args.only in (None, "entrance"):
        report_library_tier("ENTRANCE TIER", run_pyright({**COMMON_SETTINGS, "include": ENTRANCE_INCLUDES}))
    if args.only in (None, "ir"):
        report_library_tier("IR CONTRACT TIER", run_pyright({**COMMON_SETTINGS, "include": IR_TIER_INCLUDES}))
    if args.only in (None, "tests"):
        tests_config = {
            **COMMON_SETTINGS,
            "include": tests_includes,
            "exclude": [*COMMON_SETTINGS["exclude"], *PARSER_TREES],
            # Mirror the tracer suite's conftest, which inserts
            # python/pypto/language/tracer on sys.path so `import collector`
            # resolves; pyright cannot execute conftest, so the pass does it
            # here instead. This retired the collector-import category and
            # turned the suite's collateral into live signal.
            "extraPaths": ["python", ".", "python/pypto/language/tracer"],
            # Every test redeclaration is the ergonomic Out-param idiom
            # (``Out[T]`` statically *is* ``T``, so the body's re-annotation
            # is redundant-but-idiomatic) or fixture rebinding; the repo
            # default already disables this rule for the dsl parser.
            "reportRedeclaration": False,
            **{r: False for r in MECHANICAL_RULES_OFF},
        }
        report_tests(run_pyright(tests_config))

    print("\n(report only — no gate decision; see module docstring for the category owners)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
