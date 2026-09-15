#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""R4 oracle: report new public language names against documentation coverage.

The public surface of the frontend is ``__all__`` in
``python/pypto/language/__init__.py``. A name added since a base ref is a new
API surface; the question this oracle answers is whether any doc under
``docs/`` anchors it.

Coverage is a *count* of doc files mentioning the name, not a judgement: a
generic name (``runtime``) matches prose incidentally, and a name reachable
only through a namespaced spelling (``pl.intrin.spmd_direct``) is undercounted
by a bare-word match. The oracle therefore reports, and the reviewer decides.
Removed names are reported too: a removal is an API break worth documenting.

Report-only: not wired into ./scripts/gate.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

INIT = Path("python/pypto/language/__init__.py")
NAME_RE = re.compile(r'^\s+"([A-Za-z_][A-Za-z0-9_]*)",', re.MULTILINE)

# Planning/meta docs are not user- or developer-facing documentation: a new
# public name mentioned only in a tracker or lesson file is still undocumented.
META_DOCS = (
    "docs/migration-todo.md",
    "docs/gap-fixes-todo.md",
    "docs/hard-won-lessons.md",
    "docs/plan-landing.md",
    "docs/review-r10.md",
)


def names_at(rev: str | None) -> set[str]:
    if rev is None:
        text = INIT.read_text()
    else:
        text = subprocess.run(
            ["git", "show", f"{rev}:{INIT}"], check=False, capture_output=True, text=True
        ).stdout
    return set(NAME_RE.findall(text))


def docs_refs(name: str) -> list[str]:
    globs: list[str] = []
    for path in META_DOCS:
        globs += ["-g", f"!{path}"]
    proc = subprocess.run(
        ["rg", "-l", "--no-heading", *globs, rf"\b{re.escape(name)}\b", "docs/"],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="origin/main")
    ap.add_argument(
        "--min-refs", type=int, default=1, help="doc-file mentions required to count as documented"
    )
    args = ap.parse_args()

    old, new = names_at(args.base), names_at(None)
    added, removed = sorted(new - old), sorted(old - new)

    print(f"== public names: {len(old)} -> {len(new)} ==")
    undocumented = []
    print(f"-- added ({len(added)}) --")
    for name in added:
        refs = docs_refs(name)
        flag = "  <-- UNDOCUMENTED" if len(refs) < args.min_refs else ""
        print(f"  {name:16s} doc files: {len(refs):3d}{flag}")
        if len(refs) < args.min_refs:
            undocumented.append(name)

    print(f"-- removed ({len(removed)}) --")
    stale = []
    for name in removed:
        refs = docs_refs(name)
        # A removed public name still referenced by docs may be prose about the
        # underlying IR construct rather than the removed DSL helper, so this
        # is flagged for the reviewer, not failed.
        flag = "  <-- STALE DOC?" if len(refs) >= args.min_refs else ""
        print(f"  {name:16s} doc files still referencing it: {len(refs):3d}{flag}")
        if len(refs) >= args.min_refs:
            stale.append(name)
    if stale:
        print(f"  note: removed names still referenced by docs: {stale}")

    print(f"\nRESULT: {'FAIL' if undocumented else 'PASS'} (added names with no doc anchor: {undocumented})")
    return 1 if undocumented else 0


if __name__ == "__main__":
    sys.exit(main())
