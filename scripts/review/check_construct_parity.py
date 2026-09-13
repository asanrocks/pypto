#!/usr/bin/env python3
"""Construct-parity inventory: what has an equivalence anchor, and what does not.

The construct-parity campaign (docs/review-constructs.md) asks whether the
tracing frontend's logic for each structural construct matches the AST parser's.
This oracle does not answer that -- it produces the **worklist**, so that a
review agent judges constructs instead of enumerating them (an earlier round
proved a model drops items from a 21-line `rg -l` list on the easiest task).

Three outputs:

1. A coverage table: for each construct, where the parser handles it, where the
   tracing lane implements it, and how many ``conformance(...)`` anchors its
   suite holds.
2. **Unclaimed parser handlers** -- ``_parse_*`` / ``_handle_*`` / ``visit_*``
   symbols in ast_parser.py that no construct entry names. Each is either a
   missing construct entry (the map is incomplete) or a non-structural helper.
3. **Unclaimed tracing entries** -- public construct symbols in the tracing lane
   that no construct entry names. Same reading.

The construct map is maintained by hand and *verified* against the tree: a
``parser``/``tracer`` pattern that matches nothing is reported as STALE, so the
map cannot silently rot into passing.

Report-only: not wired into ./scripts/gate.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

PARSER_FILE = Path("python/pypto/language/parser/ast_parser.py")
TRACER_FILES = (
    "python/pypto/language/tracer/constructs.py",
    "python/pypto/language/tracer/return_contract.py",
    "python/pypto/language/tracer/decorators/function.py",
    "python/pypto/language/tracer/decorators/jit.py",
    "python/pypto/language/compat/diagnostics.py",
    "python/pypto/language/constructs/loops.py",
    "python/pypto/language/constructs/scopes.py",
    "python/pypto/language/constructs/submit.py",
    "python/pypto/language/constructs/optimizations.py",
    "python/pypto/language/constructs/func_attrs.py",
    "python/pypto/language/intrin.py",
)
CONFORMANCE_DIR = Path("tests/ut/language/conformance")

ANCHOR = re.compile(r"conformance\(|program_conformance\(")
# Captures the handler convention only (`_parse_*` / `visit_*` / `_handle_*`).
# The parser's `parse_*` methods are dispatch/expression entry points and are
# deliberately outside this scope; a construct whose only parser-side symbol is
# named `parse_*` still gets a construct entry, matched through the handler it
# dispatches to.
PARSER_SYMBOL = re.compile(r"^\s+def ((?:visit_|_handle_|_parse_)[A-Za-z_0-9]+)", re.MULTILINE)
TRACER_SYMBOL = re.compile(r"^(?:class|def) ([A-Za-z_][A-Za-z_0-9]*)", re.MULTILINE)


@dataclass(frozen=True)
class Construct:
    """One construct family: how each lane handles it, and its anchor suite."""

    name: str
    parser: str  # regex over parser handler symbols
    tracer: str  # regex over tracing symbols
    suite: str | None  # conformance suite, if one owns this construct


# Wave 1 first, then wave 2 (docs/review-constructs.md sections 5 and 6).
CONSTRUCTS = (
    Construct("if / else / elif", r"_parse_body_siblings|visit_If", r"PtoIfConstruct", "test_if.py"),
    Construct(
        "for + iterators",
        r"_parse_for_loop_target|_parse_range_call",
        r"PtoForConstruct|^range$|^parallel$|^unroll$|^pipeline$",
        "test_for.py",
    ),
    Construct("while", r"_parse_while_", r"PtoWhileConstruct", "test_while_logical.py"),
    Construct(
        "break / continue",
        r"_parse_body_siblings",
        r"_DeferredStagedExit",
        "test_break_continue.py",
    ),
    Construct(
        "binding / assignment",
        r"_parse_subscript_assignment|_parse_array_subscript_assignment",
        r"^bind$|_YieldingConstruct",
        "test_binding_assignment.py",
    ),
    Construct(
        "returns / exits",
        r"visit_return_stmt",
        r"return_types|signature_expressible",
        "test_return_exit.py",
    ),
    Construct(
        "generic scopes",
        r"_parse_scope|_parse_manual_scope|_parse_legacy_scope|_parse_graph_scope",
        r"^scope$|^manual_scope$|^graph$|^cluster$",
        "test_generic_scopes.py",
    ),
    Construct(
        "spmd",
        r"_parse_spmd|_parse_and_validate_core_num",
        r"^spmd$",
        "test_spmd_with_form.py",
    ),
    Construct("split_aiv", r"_parse_split_aiv_for_loop", r"^split_aiv$", "test_spmd_split_aiv.py"),
    Construct(
        "at family",
        r"_parse_at_|_handle_at_optimizations_kw",
        r"^at$",
        "test_at_const.py",
    ),
    Construct("submit / spmd_submit", r"_parse_submit", r"^submit$|^spmd_submit$", "test_submit_collect.py"),
    Construct("with-form scopes", r"_parse_scope_body", r"PtoWithConstruct", None),
    Construct("intrin surface", r"_parse_split_transfer_op", r"^intrin$|^carry$|^leak$|^transmute$", None),
    Construct(
        "statement handlers",
        r"_handle_static_print|_handle_static_assert|_handle_dump_tag|_handle_set_cache_policy",
        r"^static_print$|^static_assert$",
        None,
    ),
    Construct(
        "func attrs / optimizations",
        r"_parse_func_attr_value|_parse_optimizations_list|_parse_spmd_optimizations_list",
        r"^_fold_optimization_entries$",
        None,
    ),
    Construct(
        "calls (kernel / extern / dispatch)",
        r"_parse_kernel_call|_parse_external_function_call|_parse_inline_call|_parse_dispatch_device_kwarg",
        r"_JitExtern|_MixedExternJit",
        "test_call.py",
    ),
)


def read(paths) -> str:
    out = []
    for path in paths:
        p = Path(path)
        if p.exists():
            out.append(p.read_text())
    return "\n".join(out)


def anchors_per_suite() -> dict[str, int]:
    counts = {}
    for f in sorted(CONFORMANCE_DIR.glob("test_*.py")):
        n = len(ANCHOR.findall(f.read_text()))
        if n:
            counts[f.name] = n
    return counts


def main() -> int:
    parser_text = PARSER_FILE.read_text()
    tracer_text = read(TRACER_FILES)
    parser_symbols = sorted(set(PARSER_SYMBOL.findall(parser_text)))
    tracer_symbols = sorted(set(TRACER_SYMBOL.findall(tracer_text)))
    suites = anchors_per_suite()

    print("== construct coverage ==")
    print(f"{'construct':44s} {'parser':>7s} {'tracer':>7s} {'anchors':>8s}  suite")
    claimed_parser: set[str] = set()
    claimed_tracer: set[str] = set()
    stale: list[str] = []
    uncovered: list[str] = []

    for c in CONSTRUCTS:
        p_hits = [s for s in parser_symbols if re.search(c.parser, s)]
        t_hits = [s for s in tracer_symbols if re.search(c.tracer, s)]
        claimed_parser.update(p_hits)
        claimed_tracer.update(t_hits)
        n = suites.get(c.suite, 0) if c.suite else 0
        if not p_hits or not t_hits:
            stale.append(f"{c.name}: parser={len(p_hits)} tracer={len(t_hits)}")
        if n == 0:
            uncovered.append(c.name)
        print(f"{c.name:44s} {len(p_hits):7d} {len(t_hits):7d} {n:8d}  {c.suite or '-- none --'}")

    unclaimed_parser = [s for s in parser_symbols if s not in claimed_parser]
    unclaimed_tracer = [s for s in tracer_symbols if s not in claimed_tracer and not s.startswith("_")]

    # Collapse the residue into actionable buckets. The parser's op/value surface
    # is a different campaign (docs/review-constructs.md section 8); dumping its
    # 30-odd helpers next to the structural ones hides the signal.
    op_surface = re.compile(
        r"_parse_(?:(?:tensor|tile|array|pld|system|prefetch|unified|builtin)_op"
        r"|op_(?:attrs|kwargs|positional_arg)|.*subscript|attr(?:s_dict|_pair|_value)"
        r"|typed_constant|sequence_literal|dtype_get_byte|printed_alloc_call"
        r"|task_dummy|generic_attrs_dict|alloc_window_buffer_assignment)"
        r"|^visit_Name$"
    )
    structural = [s for s in unclaimed_parser if not op_surface.search(s)]
    op_helpers = [s for s in unclaimed_parser if op_surface.search(s)]

    print(f"\n== constructs with no anchor suite ({len(uncovered)}) ==")
    for name in uncovered:
        print(f"  {name}")

    print(f"\n== unclaimed parser handlers ({len(unclaimed_parser)}) ==")
    print(
        f"  {len(op_helpers)} are the op/value surface -- out of scope by section 8, listed for completeness:"
    )
    for s in op_helpers:
        print(f"      {s}")
    print(f"  {len(structural)} are structural -- either a missing construct entry or a new task:")
    for s in structural:
        print(f"      {s}")

    print(f"\n== unclaimed public tracing entries ({len(unclaimed_tracer)}) ==")
    for s in unclaimed_tracer:
        print(f"  {s}")
    print("  (private helpers are not listed; claim them from a construct pattern if one owns them)")

    if stale:
        print("\n== STALE map entries (pattern matches nothing in the tree) ==")
        for s in stale:
            print(f"  {s}")

    print(
        f"\nRESULT: {'REVIEW' if (uncovered or unclaimed_parser or unclaimed_tracer or stale) else 'CLEAN'} "
        f"({len(uncovered)} constructs without anchors, "
        f"{len(structural)} unclaimed structural parser handlers, "
        f"{len(unclaimed_tracer)} unclaimed public tracing entries, "
        f"{len(stale)} stale map entries)"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
