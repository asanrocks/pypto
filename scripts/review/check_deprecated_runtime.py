#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""R1 oracle: report uses of the deprecated/quarantined frontend runtimes.

Two deprecated surfaces are checked:

* ``program_ast`` -- the AST-era program object. Removed by the tracing
  migration; expected to be absent everywhere.
* ``pl.jit_specializer`` -- the quarantined AST-era runtime. Legitimate in
  exactly three places (the runtime itself, its public alias export, and the
  text explaining the quarantine) plus the specializer-mechanics unit tests.

Also reports bare text mentions (comments/docstrings) separately, since a
stale mention is a documentation concern, not a usage.

Report-only: not wired into ./scripts/gate. Exit status is 1 when a usage
appears outside the allowlist, so a task can quote red/green output.
"""

from __future__ import annotations

import re
import subprocess
import sys

# Production sites where the name is inherent, not a leftover.
PROD_ALLOWLIST = (
    "python/pypto/jit/",  # the quarantined runtime itself
    "python/pypto/language/__init__.py",  # the public alias export
    "python/pypto/language/typing/tensor.py",  # error text explaining the quarantine
)

# Test suites whose contract *is* specializer mechanics.
TEST_ALLOWLIST = (
    "tests/ut/jit/",
    # The persistent artifact store and the warmup/publication path serve the
    # quarantined runtime only: ``warmup``/``host`` exist on its JITFunction and
    # nowhere on the tracing jit, and the tracing lane never reads the store
    # (docs/migration-todo.md D10). The suite's own header says so, and its
    # kernel stubs only exercise those quarantined hooks.
    "tests/ut/runtime/test_prebuilt_artifact.py",
)

# Code surfaces only. Markdown is excluded deliberately: the migration rewrote
# docs prose on purpose (see the tracker's scope section), and a tracker that
# discusses the deprecated names is not a usage of them. Scanning .md let the
# review's own notes turn R1 red (discovered as F5 during round 1).
CODE = (".py", ".cpp", ".h", ".hpp", ".cc", ".rs", ".txt")

USAGE = re.compile(r"(?<!\w)(?:pl\.)?jit_specializer(?=[.(]|$)|@jit_specializer")
PROGRAM_AST = re.compile(r"(?<!\w)program_ast(?!\w)")


def stripped_is_prose(line: str) -> bool:
    s = line.strip()
    return s.startswith("#") or s.startswith('"""') or s.startswith("'''") or s.startswith("//")


def rg(pattern: str) -> list[tuple[str, int, str]]:
    proc = subprocess.run(
        ["rg", "-n", "--no-heading", "-g", "!pi-sessions/**", "-g", "!scripts/review/**", pattern, "."],
        check=False,
        capture_output=True,
        text=True,
    )
    hits: list[tuple[str, int, str]] = []
    for raw in proc.stdout.splitlines():
        path, lineno, text = raw.split(":", 2)
        path = path.removeprefix("./")
        if not path.endswith(CODE):
            continue
        hits.append((path, int(lineno), text))
    return hits


def classify(path: str) -> str:
    if any(path.startswith(p) for p in PROD_ALLOWLIST):
        return "allowlisted"
    if any(path.startswith(p) for p in TEST_ALLOWLIST):
        return "allowlisted"
    return "candidate"


def main() -> int:
    rc = 0

    ast_hits = rg(PROGRAM_AST.pattern)
    print(f"== program_ast: {len(ast_hits)} hits (expected 0) ==")
    for path, lineno, text in ast_hits:
        print(f"  {path}:{lineno}: {text.strip()[:100]}")
    if ast_hits:
        rc = 1

    usages, mentions = [], []
    for path, lineno, text in rg("jit_specializer"):
        (usages if USAGE.search(text) and not stripped_is_prose(text) else mentions).append(
            (path, lineno, text)
        )

    candidates = [h for h in usages if classify(h[0]) == "candidate"]
    allowed = [h for h in usages if classify(h[0]) == "allowlisted"]
    print(
        f"\n== jit_specializer: {len(usages)} usages "
        f"({len(allowed)} allowlisted, {len(candidates)} candidates), "
        f"{len(mentions)} text mentions =="
    )
    print("-- candidates (verdict each: unnecessary / necessary / needs-decision) --")
    for path, lineno, text in sorted(candidates):
        print(f"  {path}:{lineno}: {text.strip()[:100]}")
    if candidates:
        rc = 1

    by_file: dict[str, int] = {}
    for path, _, _ in allowed:
        by_file[path] = by_file.get(path, 0) + 1
    print("-- allowlisted usage counts --")
    for path, count in sorted(by_file.items()):
        print(f"  {count:3d}  {path}")

    print(f"\nRESULT: {'FAIL' if rc else 'PASS'} (non-allowlisted usages: {len(candidates)})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
