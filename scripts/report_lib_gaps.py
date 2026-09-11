# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

r"""Type-gap report for the pypto-lib model zoo: the stdlib's real-world corpus.

Sibling of ``scripts/report_surface_gaps.py``. That script audits pypto from
the inside (library surface + own tests); this one inverts the direction —
pypto-lib kernels are what real users write, so every error a kernel hits is
evidence about the pypto surface. The raw strict scan is ~23k findings on
``models/deepseek`` alone; the report's whole job is cheap triage into:

* **pl.\* call sites** — argument/call/attribute/index/operator errors on
  ``pl.<op>`` lines, each tagged with where the callee is defined (pypto
  stdlib / pypto-lib infra / builtin / unresolved). The headline gap
  candidates. Paired pyright diagnostics (a ``reportCallIssue`` plus one or
  more ``reportArgumentType`` for the same bad call) are coalesced so groups
  count *defects*, not diagnostic pairs.
* **pypto values** — operator/attribute errors whose message mentions a DSL
  type but whose call line is not a ``pl.`` call. Same origin tags; this is
  where pypto-lib's own infra problems (e.g. ``golden/runner.py`` signature
  invariance) land, distinguishable from stdlib gaps by the tag.
* **frontend precision** — ``reportUnknown*`` whose seed resolves, by a
  one-symbol backward search over assignments, to a ``pl.*`` call result or
  a subscript fed by clean identifiers. These are the only detector for
  library ``Any`` leaks (e.g. ``Tensor.__getitem__ -> Any`` poisons every
  downstream op), which is why the unknown rules stay ON here instead of
  being disabled by config like the tests pass of the sibling script.
* **allowlisted** — DSL false positives, collapsed via the manifest.
* **local debt** — ``reportUnknown*`` seeded by unannotated local code, plus
  missing-annotation rules, collapsed to a rule histogram. Bulk of the
  volume, no per-site output.
* **other local** — everything else (local lint, narrowness bugs), grouped.

Provenance heuristic (``Provenance``): for a flagged expression, collect its
free identifiers and follow each backward to its last assignment (plain,
``for``-loop, or ``with``-as target). An identifier is **dirty** when it is
a parameter pyright reported as missing-annotation, or a variable whose own
report line has no ``pl.*``/subscript; **pl** when its RHS contains a
``pl.*`` call or subscript, or (bounded depth) resolves through clean
identifiers to one. An Unknown is a precision *candidate* only when no
involved identifier is dirty and at least one is pl-seeded — the local-debt
case (``gamma = pl.cast(norm_w_2d[:, ...], ...)`` with ``norm_w_2d``
unannotated) demotes to debt even though the line calls ``pl.*``. This is
syntactic provenance, not real dataflow: comprehensions, aliasing, and
cross-function flows are not modeled. Known cost: some true library leaks
whose every local mention happens to share a name with a dirty variable are
demoted; the group list is a triage pool, not a verdict.

Scan roots are per model-variant (``models/<model>/<variant>``), one pyright
invocation each. This is load-bearing, not tidiness: variant modules import
their siblings bare (``from config import ...``) and different variants
define colliding names (``hc_head.LINEAR_K_CHUNK``), so a shared path would
cross-resolve them. It also keeps ``config`` resolving to the variant's own
``config.py`` rather than the CANN ``tbe/config`` namespace package that
shadows it on the venv's site search path.

Known pathology: ``models/qwen3/14b/qwen3_14b_prefill_tq_draft.py`` sends
pyright's inference into a blowup (>120s alone; structurally similar siblings
check in <7s; the suspect is the dependent triangular ``pl.range(b)`` loop).
It is excluded below until root-caused; extend the list if 32b shows the same
signature (timeout, not slow progress).

Usage (informational; exits 0 unless pyright itself fails):

    uv run python scripts/report_lib_gaps.py
    uv run python scripts/report_lib_gaps.py models/deepseek/v4-pro
    uv run python scripts/report_lib_gaps.py --lib-root /path/to/pypto-lib
"""

from __future__ import annotations

import argparse
import builtins
import json
import os
import re
import subprocess
import sys
import time
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIB_ROOT = REPO_ROOT.parent / "pypto-lib"
ALLOWLIST_DEFAULT = Path(__file__).resolve().parent / "lib_gaps_allowlist.txt"

COMMON_SETTINGS: dict[str, object] = {
    "typeCheckingMode": "strict",
    "pythonVersion": "3.10",
    "reportMissingTypeStubs": False,
    "reportMissingModuleSource": False,
    "reportPrivateUsage": False,
}

# pyright inference blowups — see module docstring. Excluded from every scan.
EXCLUDE_FILES = ["models/qwen3/14b/qwen3_14b_prefill_tq_draft.py"]

UNKNOWN_RULES = {
    "reportUnknownMemberType",
    "reportUnknownVariableType",
    "reportUnknownArgumentType",
    "reportUnknownParameterType",
    "reportUnknownLambdaType",
}

# Rules that can describe a pl.* call defect. Anything else measured on a
# pl.* line (unused-variable lint, missing annotations, ...) is local.
CALL_RULES = {
    "reportArgumentType",
    "reportCallIssue",
    "reportAttributeAccessIssue",
    "reportIndexIssue",
    "reportOperatorIssue",
    "reportOptionalOperand",
    "reportOptionalMemberAccess",
    "reportOptionalSubscript",
    "reportReturnType",
    "reportAssignmentType",
    "reportGeneralTypeIssues",
    "reportUndefinedVariable",
}

PL_CALL = re.compile(r"\bpl\.\w")
SUBSCRIPT = re.compile(r"\[[^\]\n]*[:,]")
PYPTO_TYPE_TOKEN = re.compile(
    r"\b(Tensor|Tile|Scalar|ScalarSpec|TensorSpec|DynVar|DataType|IntLike|RangeArg)\b"
)
INT_LITERAL = re.compile(r'"Literal\[-?\d+\]"')
PL_EXPORT_GAP = re.compile(r'"([^"]+)" is not a known attribute of module "pypto\.language"')

ASSIGN_RE = re.compile(r"^\s*(?P<name>\w+)\s*(?::[^=]*)?=(?!=)\s*(?P<rhs>.+)$")
FOR_RE = re.compile(r"^\s*for\s+(?P<names>[\w, ]+?)\s+in\s+(?P<rhs>.+?):\s*$")
WITH_RE = re.compile(r"^\s*with\s+(?P<rhs>.+?)\s+as\s+(?P<names>[\w, ]+):\s*$")
WORD = re.compile(r"\b[A-Za-z_]\w*\b")
KEYWORDS = {
    "as",
    "assert",
    "async",
    "await",
    "class",
    "def",
    "del",
    "elif",
    "else",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "not",
    "or",
    "and",
    "None",
    "nonlocal",
    "pass",
    "raise",
    "return",
    "True",
    "False",
    "try",
    "while",
    "with",
    "yield",
    "pl",
}
TYPE_OF = re.compile(r'Type of (?:parameter )?"(\w+)"')
DEF_LINE = re.compile(r"^(?P<indent>\s*)def\s+\w+")
KERNEL_DECOR = re.compile(r"^\s*@.*\bpl\.")
BUILTIN_NAMES = set(dir(builtins))

BUCKETS = ("pl", "types", "seed", "allow", "debt", "other")
BUCKET_TITLES = {
    "pl": "GAP CANDIDATES — errors at pl.* call sites (callee origin tagged)",
    "types": "GAP CANDIDATES — pypto values off pl.* lines (callee origin tagged)",
    "seed": "FRONTEND PRECISION — Unknown whose seed resolves to pl.*/subscript",
    "allow": "KNOWN FALSE POSITIVES (allowlist)",
    "debt": "LOCAL DEBT — Unknown/missing annotations seeded by local code (collapsed)",
    "other": "OTHER LOCAL — grouped",
}


@dataclass
class Diag:
    file: str  # relative to the lib root
    line: int
    col_start: int
    col_end: int  # character span on the line; -1 when unknown
    rule: str
    message: str

    @property
    def where(self) -> str:
        return f"{self.file}:{self.line}"


@dataclass
class Allow:
    rule: str
    pattern: re.Pattern[str]
    spec: str


def load_allowlist(path: Path) -> list[Allow]:
    entries: list[Allow] = []
    if not path.exists():
        return entries
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        rule, _, pattern = stripped.partition(":")
        if not pattern:
            print(f"warning: allowlist entry without pattern skipped: {raw!r}", file=sys.stderr)
            continue
        entries.append(Allow(rule.strip(), re.compile(pattern.strip()), stripped))
    return entries


def identifiers(expr: str) -> list[str]:
    return [w for w in WORD.findall(expr) if w not in KEYWORDS]


def callee_of(message: str) -> str | None:
    """Function an argument/call diagnostic is about, when pyright names it."""
    for pat in (r'No overloads for "(\w+)"', r'in function "(\w+)"'):
        if m := re.search(pat, message):
            return m.group(1)
    if m := PL_EXPORT_GAP.search(message):
        return m.group(1)
    return None


def coalesce_paired(diags: list[Diag]) -> list[Diag]:
    """Drop reportCallIssue diagnostics that duplicate a reportArgumentType
    at the same site and callee — pyright emits both for one bad call, and
    the argument-type message carries the detail. Groups then count defects,
    not diagnostic pairs."""
    arg_sites = {
        (d.file, d.line, c) for d in diags if d.rule == "reportArgumentType" and (c := callee_of(d.message))
    }
    return [
        d
        for d in diags
        if not (
            d.rule == "reportCallIssue" and (c := callee_of(d.message)) and (d.file, d.line, c) in arg_sites
        )
    ]


# --- per-file source indexing ------------------------------------------------


class FileIndex:
    """Assignment targets per name, for backward provenance lookups."""

    def __init__(self, lines: list[str]) -> None:
        by_name: dict[str, list[tuple[int, str]]] = {}
        for ln, line in enumerate(lines, 1):
            if m := ASSIGN_RE.match(line):
                by_name.setdefault(m.group("name"), []).append((ln, m.group("rhs")))
                continue
            if m := FOR_RE.match(line) or WITH_RE.match(line):
                for name in (n.strip() for n in m.group("names").split(",")):
                    if name:
                        by_name.setdefault(name, []).append((ln, m.group("rhs")))
        self._lines = {n: [e[0] for e in entries] for n, entries in by_name.items()}
        self._rhs = {n: [e[1] for e in entries] for n, entries in by_name.items()}

    def last_rhs(self, name: str, before: int) -> tuple[int, str] | None:
        lines = self._lines.get(name)
        if not lines:
            return None
        i = bisect_left(lines, before) - 1
        if i < 0:
            return None
        return lines[i], self._rhs[name][i]


class Classifier:
    """Line-level heuristics plus the provenance oracle; see module docstring."""

    def __init__(self, lib_root: Path, allowlist: list[Allow]) -> None:
        self.lib_root = lib_root
        self.allowlist = allowlist
        self._lines: dict[str, list[str]] = {}
        self._indexes: dict[str, FileIndex] = {}
        self._kernel_cache: dict[tuple[str, int], bool] = {}

    def lines(self, file: str) -> list[str]:
        if file not in self._lines:
            try:
                self._lines[file] = (self.lib_root / file).read_text(encoding="utf-8").splitlines()
            except OSError:
                self._lines[file] = []
        return self._lines[file]

    def index(self, file: str) -> FileIndex:
        if file not in self._indexes:
            self._indexes[file] = FileIndex(self.lines(file))
        return self._indexes[file]

    def source_line(self, d: Diag) -> str:
        lines = self.lines(d.file)
        return lines[d.line - 1] if 0 < d.line <= len(lines) else ""

    def span_text(self, d: Diag) -> str:
        line = self.source_line(d)
        if 0 <= d.col_start < d.col_end <= len(line):
            return line[d.col_start : d.col_end]
        return ""

    def in_kernel(self, file: str, line: int) -> bool:
        """Is the line inside a @pl.-decorated function?

        The corpus mixes traced kernels and torch host/golden code in one
        file. Unknowns in host code are torch-stub noise regardless of what
        the provenance resolver says, so only kernel bodies can host a
        pypto frontend-precision finding.
        """
        key = (file, line)
        if (hit := self._kernel_cache.get(key)) is not None:
            return hit
        lines = self.lines(file)
        src = lines[line - 1] if 0 < line <= len(lines) else ""
        indent = len(src) - len(src.lstrip())
        result = False
        if indent > 0:
            for ln in range(line - 1, 0, -1):
                m = DEF_LINE.match(lines[ln - 1])
                if not m or len(m.group("indent")) >= indent:
                    continue
                for dl in range(ln - 1, max(0, ln - 6), -1):
                    decor = lines[dl - 1]
                    if KERNEL_DECOR.search(decor):
                        result = True
                    if not decor.lstrip().startswith("@"):
                        break
                break
        self._kernel_cache[key] = result
        return result

    def allowlisted(self, d: Diag) -> Allow | None:
        for entry in self.allowlist:
            if entry.rule == d.rule and entry.pattern.search(d.message):
                return entry
        return None

    def bucket(self, d: Diag, prov: Provenance) -> str:
        if self.allowlisted(d) is not None:
            return "allow"
        if d.rule in {"reportMissingParameterType", "reportMissingTypeArgument"}:
            return "debt"
        if d.rule in UNKNOWN_RULES:
            return self._seed_or_debt(d, prov)
        line = self.source_line(d)
        if d.rule in CALL_RULES and PL_CALL.search(line):
            return "pl"
        if PYPTO_TYPE_TOKEN.search(d.message):
            return "types"
        return "other"

    def _seed_or_debt(self, d: Diag, prov: Provenance) -> str:
        if not self.in_kernel(d.file, d.line):
            return "debt"  # host/golden code: torch-stub noise, not pypto precision
        line = self.source_line(d)
        if d.rule == "reportUnknownArgumentType":
            # The flagged expression is the argument, not the assignment RHS.
            expr = self.span_text(d) or line
        elif m := ASSIGN_RE.match(line):
            expr = m.group("rhs")
        else:
            expr = self.span_text(d) or line
        ids = identifiers(expr)
        if any(i in prov.dirty.get(d.file, ()) for i in ids):
            return "debt"
        if PL_CALL.search(expr) or SUBSCRIPT.search(expr):
            return "seed"
        if any(prov.of(d.file, i, d.line) == "pl" for i in ids):
            return "seed"
        return "debt"


class Provenance:
    """Backward one-symbol provenance: 'dirty' | 'pl' | 'other'.

    'dirty' — locally unknown seed (unannotated parameter, or a variable
    whose own report line has no pl.*/subscript). 'pl' — produced by a
    pl.* call or subscript over (recursively, depth-bounded) clean values.
    """

    def __init__(self, classifier: Classifier, dirty: dict[str, set[str]]) -> None:
        self.classifier = classifier
        self.dirty = dirty
        self._cache: dict[tuple[str, str, int], str] = {}

    def of(self, file: str, name: str, before: int, depth: int = 0) -> str:
        if name in self.dirty.get(file, ()):
            return "dirty"
        key = (file, name, before)
        if (hit := self._cache.get(key)) is not None:
            return hit
        result = "other"
        found = self.classifier.index(file).last_rhs(name, before)
        if found is not None:
            ln, rhs = found
            ids = identifiers(rhs)
            if any(self.of(file, i, ln, depth + 1) == "dirty" for i in ids):
                result = "dirty"
            elif PL_CALL.search(rhs) or SUBSCRIPT.search(rhs):
                result = "pl"
            elif depth < 4 and any(self.of(file, i, ln, depth + 1) == "pl" for i in ids if i != name):
                result = "pl"
        self._cache[key] = result
        return result


def build_dirty(diags: list[Diag], classifier: Classifier) -> dict[str, set[str]]:
    """Identifiers that seed Unknown locally: unannotated parameters and
    unknown variables/members whose own line has no pl.*/subscript."""
    dirty: dict[str, set[str]] = {}
    for d in diags:
        name: str | None = None
        if d.rule == "reportMissingParameterType":
            m = re.search(r'parameter "(\w+)"', d.message)
            name = m.group(1) if m else None
        elif d.rule in UNKNOWN_RULES:
            line = classifier.source_line(d)
            if PL_CALL.search(line) or SUBSCRIPT.search(line):
                continue
            m = TYPE_OF.search(d.message)
            name = m.group(1) if m else None
        if name:
            dirty.setdefault(d.file, set()).add(name)
    return dirty


def build_origins(lib_root: Path) -> dict[str, str]:
    """Symbol -> where it is defined: pypto stdlib or a pypto-lib module."""
    origins: dict[str, str] = {}
    trees = [
        (REPO_ROOT / "python/pypto/language", "pypto stdlib"),
        (lib_root, None),  # tagged per file below
    ]
    for tree, tag in trees:
        if not tree.is_dir():
            continue
        for path in sorted(tree.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            where = tag or f"lib:{path.relative_to(lib_root).as_posix()}"
            try:
                text = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for line in text.splitlines():
                if m := re.match(r"^(?:def|class) (\w+)", line):
                    origins.setdefault(m.group(1), where)
    return origins


def origin_of(callee: str | None, origins: dict[str, str]) -> str | None:
    if callee is None:
        return None
    if callee in origins:
        return origins[callee]
    if callee in BUILTIN_NAMES:
        return "builtin"
    return "unresolved"


# --- pyright driver ----------------------------------------------------------


def discover_roots(lib_root: Path) -> list[Path]:
    """Default scan roots: per-variant model dirs plus the non-model tops.

    ``tests`` is deliberately not a default: its fixtures import torch and
    resolve paths through conftest sys.path inserts pyright cannot see, so
    the residue there is dialect noise (same rationale as the parser trees
    in the sibling script). Pass it explicitly when wanted.
    """
    roots: list[Path] = []
    models = lib_root / "models"
    if models.is_dir():
        for model in sorted(models.iterdir()):
            if not model.is_dir() or model.name == "__pycache__":
                continue
            variants = [
                d
                for d in sorted(model.iterdir())
                if d.is_dir() and d.name != "__pycache__" and any(d.glob("*.py"))
            ]
            if variants:
                roots.extend(variants)
            elif any(model.glob("*.py")):
                roots.append(model)
    for top in ("golden", "contract", "examples", "tools"):
        p = lib_root / top
        if p.is_dir():
            roots.append(p)
    return roots


def resolve_targets(targets: list[str], lib_root: Path) -> list[Path]:
    if not targets:
        return discover_roots(lib_root)
    roots: list[Path] = []
    for t in targets:
        p = Path(t)
        if not p.is_dir():
            p = lib_root / t
        if not p.is_dir():
            print(f"error: target not found: {t}", file=sys.stderr)
            raise SystemExit(2)
        roots.append(p.resolve())
    return roots


def run_pyright(root: Path, lib_root: Path, timeout: int) -> tuple[list[Diag], float, str]:
    """Strict-scan one root; returns (diags, seconds, status).

    The temp config must live at the lib root so its relative ``include`` and
    ``extraPaths`` resolve against the corpus, while the process itself runs
    from the pypto repo (that is where the pyright-bearing venv lives).
    """
    rel = root.relative_to(lib_root).as_posix()
    config = {
        **COMMON_SETTINGS,
        "include": [rel],
        "exclude": ["**/__pycache__", *EXCLUDE_FILES],
        "extraPaths": [
            os.path.relpath(REPO_ROOT / "python", lib_root),
            ".",
            rel,
        ],
    }
    tmp = lib_root / "pyrightconfig.lib-gaps.tmp.json"
    tmp.write_text(json.dumps(config), encoding="utf-8")
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, venv-scoped
            ["uv", "run", "pyright", "-p", str(tmp), "--outputjson"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return [], time.monotonic() - t0, "TIMEOUT"
    finally:
        tmp.unlink(missing_ok=True)
    elapsed = time.monotonic() - t0
    if proc.returncode not in (0, 1):  # 1 = diagnostics found, not a failure
        print(f"pyright failed on {rel} (exit {proc.returncode}):\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(2)
    payload = json.loads(proc.stdout)

    diags: list[Diag] = []
    stray = 0
    for item in payload.get("generalDiagnostics", []):
        if item.get("severity") != "error":
            continue
        try:
            rel_file = str(Path(str(item["file"])).relative_to(lib_root))
        except ValueError:  # diagnostic outside the corpus: config-resolution bug
            stray += 1
            continue
        start, end = item["range"]["start"], item["range"]["end"]
        col_end = end["character"] if end["line"] == start["line"] else -1
        diags.append(
            Diag(
                file=rel_file,
                line=int(start["line"]) + 1,
                col_start=int(start["character"]),
                col_end=int(col_end),
                rule=str(item.get("rule", "?")),
                message=str(item["message"]).splitlines()[0],
            )
        )
    if stray:
        print(f"warning: {stray} diagnostics outside {lib_root} were dropped", file=sys.stderr)
    return diags, elapsed, "ok"


# --- report ------------------------------------------------------------------


def group_key(d: Diag, normalize_names: bool = False) -> tuple[str, str]:
    message = " ".join(d.message.split())
    message = INT_LITERAL.sub('"int"', message)
    if normalize_names:
        # Variable names are corpus-specific; the defect kind is not.
        message = re.sub(r'Type of (?:parameter )?"[^"]+"', "Type of <id>", message)
    return d.rule, message


def locate_definitions(symbol: str) -> list[str]:
    """Where does a missing ``pl.<symbol>`` actually live in the pypto tree?"""
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


def print_groups(
    diags: list[Diag],
    classifier: Classifier,
    origins: dict[str, str],
    *,
    normalize_names: bool = False,
    snippets: int = 0,
    max_groups: int = 20,
    max_sites: int = 4,
) -> None:
    groups: dict[tuple[str, str], list[Diag]] = {}
    for d in diags:
        groups.setdefault(group_key(d, normalize_names), []).append(d)
    # Rank by distinct sites: clones and paired diagnostics must not inflate.
    ordered = sorted(groups.items(), key=lambda kv: -len({d.where for d in kv[1]}))
    for (rule, message), items in ordered[:max_groups]:
        sites = sorted({d.where for d in items})
        callee = callee_of(message)
        origin = origin_of(callee, origins)
        tag = f"{callee} → {origin}" if callee else rule
        more = f" ...+{len(sites) - max_sites} more sites" if len(sites) > max_sites else ""
        print(f"  {len(sites):>4}x [{tag}] {message}")
        print(f"        {', '.join(sites[:max_sites])}{more}")
        if snippets > 0:
            snippets -= 1
            line = classifier.source_line(items[0]).strip()
            if line:
                print(f"        e.g. {line[:100]}")


def print_headline(gap_diags: list[Diag], origins: dict[str, str]) -> None:
    groups: dict[tuple[str, str], list[Diag]] = {}
    for d in gap_diags:
        groups.setdefault(group_key(d), []).append(d)
    ordered = sorted(groups.items(), key=lambda kv: -len({d.where for d in kv[1]}))
    for (rule, message), items in ordered[:6]:
        sites = sorted({d.where for d in items})
        callee = callee_of(message)
        origin = origin_of(callee, origins)
        tag = f"{callee} → {origin}" if callee else rule
        print(f"  {len(sites):>4}x [{tag}] {message[:95]}")


def print_allow_section(diags: list[Diag], classifier: Classifier) -> None:
    allow_specs = [entry.spec for entry in (classifier.allowlisted(d) for d in diags) if entry]
    print(f"\n== {BUCKET_TITLES['allow']}: {len(diags)} ==")
    if allow_specs:
        for spec, n in Counter(allow_specs).most_common():
            print(f"  {n:>4}  {spec}")
    else:
        print("  none.")


def print_export_gaps(gap_diags: list[Diag]) -> None:
    export_gaps = [d for d in gap_diags if PL_EXPORT_GAP.search(d.message)]
    if not export_gaps:
        return
    symbols = Counter(m.group(1) for d in export_gaps if (m := PL_EXPORT_GAP.search(d.message)))
    print("\n  pl.<sym> not re-exported by pypto.language (definitions in the language tree):")
    for sym, n in symbols.most_common():
        defs = locate_definitions(sym)
        where = ", ".join(defs) if defs else "NO DEFINITION FOUND"
        print(f"    {n:>4}  {sym:<20} -> {where}")


def report(
    roots: list[Path],
    lib_root: Path,
    by_bucket: dict[str, list[Diag]],
    timings: list[tuple[Path, float, str, int]],
    classifier: Classifier,
    origins: dict[str, str],
    demoted_from_precision: int,
) -> None:
    n_files = sum(n for _, _, _, n in timings)
    total_t = sum(t for _, t, _, _ in timings)
    n_findings = sum(len(v) for v in by_bucket.values())
    print("### PYPTO-LIB TYPE-GAP REPORT")
    print(f"lib root: {lib_root}")
    for root, elapsed, status, n in timings:
        print(f"  {root.relative_to(lib_root).as_posix():<40} {n:>3} files  {elapsed:6.1f}s  {status}")
    print(f"total: {len(roots)} roots, {n_files} files, {total_t:.0f}s, {n_findings} findings")

    gap_diags = by_bucket["pl"] + by_bucket["types"]
    print(f"\n== HEADLINE — top gap-candidate groups ({len(gap_diags)} findings) ==")
    if gap_diags:
        print_headline(gap_diags, origins)
    else:
        print("  none.")

    print_allow_section(by_bucket["allow"], classifier)

    for b in ("pl", "types"):
        print(f"\n== {BUCKET_TITLES[b]}: {len(by_bucket[b])} ==")
        if by_bucket[b]:
            print_groups(by_bucket[b], classifier, origins, snippets=3)
        else:
            print("  none.")

    print_export_gaps(gap_diags)

    print(f"\n== {BUCKET_TITLES['seed']}: {len(by_bucket['seed'])} ==")
    if by_bucket["seed"]:
        print("  (syntactic provenance — triage pool, not verdicts)")
        print_groups(by_bucket["seed"], classifier, origins, normalize_names=True)
    else:
        print("  none.")

    print(f"\n== {BUCKET_TITLES['debt']}: {len(by_bucket['debt'])} ==")
    if by_bucket["debt"]:
        print(f"  ({demoted_from_precision} unknown-family findings demoted here by provenance)")
        for rule, n in Counter(d.rule for d in by_bucket["debt"]).most_common():
            print(f"  {n:>5}  {rule}")
    else:
        print("  none.")

    print(f"\n== {BUCKET_TITLES['other']}: {len(by_bucket['other'])} ==")
    if by_bucket["other"]:
        print_groups(by_bucket["other"], classifier, origins)
    else:
        print("  none.")


# --- main --------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Type-gap report for the pypto-lib model zoo: signal only, grouped."
    )
    parser.add_argument(
        "targets",
        nargs="*",
        help="scan roots relative to the lib root (default: discovered variant dirs + top dirs)",
    )
    parser.add_argument("--lib-root", type=Path, default=DEFAULT_LIB_ROOT, help="pypto-lib checkout")
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST_DEFAULT, help="rule : regex manifest")
    parser.add_argument("--timeout", type=int, default=600, help="per-root pyright timeout (seconds)")
    args = parser.parse_args()

    lib_root = args.lib_root.resolve()
    if not lib_root.is_dir():
        print(f"error: lib root not found: {lib_root}", file=sys.stderr)
        raise SystemExit(2)
    allowlist = load_allowlist(args.allowlist)
    roots = resolve_targets(args.targets, lib_root)

    all_diags: list[Diag] = []
    timings: list[tuple[Path, float, str, int]] = []
    for root in roots:
        diags, elapsed, status = run_pyright(root, lib_root, args.timeout)
        n_files = len([p for p in root.rglob("*.py") if "__pycache__" not in p.parts])
        all_diags.extend(diags)
        timings.append((root, elapsed, status, n_files))

    all_diags = coalesce_paired(all_diags)
    classifier = Classifier(lib_root, allowlist)
    prov = Provenance(classifier, build_dirty(all_diags, classifier))
    origins = build_origins(lib_root)

    by_bucket: dict[str, list[Diag]] = {b: [] for b in BUCKETS}
    for d in all_diags:
        by_bucket[classifier.bucket(d, prov)].append(d)
    demoted = sum(1 for d in by_bucket["debt"] if d.rule in UNKNOWN_RULES)

    report(roots, lib_root, by_bucket, timings, classifier, origins, demoted)
    print("\n(report only — no gate decision; see scripts/lib_gaps_allowlist.txt for false-positive kinds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
