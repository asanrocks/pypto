#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""R3 prevalence scan: how often do kernels retype a variable in a staged loop?

R3 (docs/fix-construct-parity.md section 5) prefers the stricter design for a
type-changing rebind -- *unless the pattern is prevalent in existing kernels*,
in which case the decision inverts to "define the semantics". That clause is
conditional on a number nobody had, so this measures it.

Pure AST, no imports: the corpus carries bare sibling imports that only resolve
per model variant, so the scan must not execute anything.

Flag condition: a name is assigned a dtype token while lexically inside a
``for``/``while`` body, and the same name was already assigned a *different*
dtype token earlier in the same function. Dtype tokens are read off the
assignment's RHS / annotation text (FP32, INT32, INDEX, ...), which is a
heuristic: it is meant to bound the prevalence, not to replace the frontends.

Usage: check_rebind_prevalence.py <root> [<root> ...]
Report-only.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

DTYPE = re.compile(r"\b(pl\.)?(FP32|FP16|BF16|FP8|FP64|INT64|INT32|INT16|INT8|UINT8|UINT16|INDEX|BOOL)\b")
LOOPS = (ast.For, ast.While)


def dtype_token(node: ast.AST | None) -> str | None:
    """The first dtype token mentioned in an expression, if any."""
    if node is None:
        return None
    found = DTYPE.search(ast.unparse(node))
    return found.group(2) if found else None


def targets(node: ast.Assign | ast.AnnAssign) -> list[str]:
    t = node.targets[0] if isinstance(node, ast.Assign) else node.target
    return (
        [e.id for e in getattr(t, "elts", []) if isinstance(e, ast.Name)]
        if not isinstance(t, ast.Name)
        else [t.id]
    )


def scan_function(fn: ast.FunctionDef | ast.AsyncFunctionDef, path: Path) -> list[tuple[int, str, str, str]]:
    hits: list[tuple[int, str, str, str]] = []
    seen: dict[str, str] = {}

    def walk(body: list[ast.stmt], in_loop: bool) -> None:
        for stmt in body:
            if isinstance(stmt, LOOPS):
                walk(stmt.body, True)
                walk(stmt.orelse, in_loop)
            elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
                ann = dtype_token(stmt.annotation) if isinstance(stmt, ast.AnnAssign) else None
                tok = ann or dtype_token(getattr(stmt, "value", None))
                for name in targets(stmt):
                    if tok and name in seen and seen[name] != tok and in_loop:
                        hits.append((stmt.lineno, name, seen[name], tok))
                    if tok:
                        seen[name] = tok
            elif isinstance(stmt, ast.If):
                walk(stmt.body, in_loop)
                walk(stmt.orelse, in_loop)
            elif isinstance(stmt, ast.With):
                walk(stmt.body, in_loop)

    walk(fn.body, False)
    return hits


def main() -> int:
    roots = [Path(a) for a in sys.argv[1:]] or [Path("examples")]
    total_files = total_fn = 0
    hits: list[tuple[Path, int, str, str, str]] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if any(p in path.parts for p in (".venv", "build", "__pycache__")):
                continue
            try:
                tree = ast.parse(path.read_text(), filename=str(path))
            except SyntaxError:
                continue
            total_files += 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    total_fn += 1
                    for lineno, name, old, new in scan_function(node, path):
                        hits.append((path, lineno, name, old, new))

    print(f"scanned {total_files} files / {total_fn} functions")
    print(f"type-changing rebinds inside a staged loop: {len(hits)}")
    for path, lineno, name, old, new in hits[:40]:
        print(f"  {path}:{lineno}: {name}: {old} -> {new}")
    if len(hits) > 40:
        print(f"  ... {len(hits) - 40} more")
    # Not a verdict. R3 turns on judgement ("prevalent in existing kernels"),
    # and this heuristic reads the first dtype token anywhere in the RHS, so a
    # nested cast -- pl.assemble(acc, pl.cast(x, FP32)) -- is a FALSE POSITIVE.
    # Treat the count as an upper bound; a real count needs the frontends.
    print(f"\ncandidate sites (upper bound): {len(hits)}")
    # Treat the count as an upper bound; a real count needs the frontends.
    print("\nnote: upper bound only -- nested casts in the RHS over-count.")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
