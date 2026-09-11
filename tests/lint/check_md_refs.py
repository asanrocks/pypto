# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Fail on dangling relative references in markdown.

Scans repository markdown (vendored, submodule, and hidden trees excluded)
for ``[text](target)`` links, ``![alt](target)`` images, and
``[ref]: target`` definitions whose relative target does not resolve to an
existing file. Absolute URLs, ``mailto:``, and pure-anchor targets are
skipped; the anchor part of ``file.md#section`` is stripped before the
existence check (anchor validity is not verified).

Inline backtick paths (``docs/40-...``) are prose, not links — the checker
does not chase them; records of the external design-notes line are
annotated as such where they are cited.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

EXCLUDED_PREFIXES = ("3rdparty/", "ref/", "runtime/", ".venv/", "build/", "node_modules/")

# [text](target) — skip images (handled below); allow an optional title.
INLINE_LINK = re.compile(r"(?<!\!)\[[^\]\[]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
IMAGE_LINK = re.compile(r"!\[[^\]\[]*\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
# Leading up-to-3-space indented reference definition: [label]: target
REF_DEF = re.compile(r"^ {0,3}\[[^\]\n]+\]:\s*(\S+)", re.MULTILINE)

SKIP_PREFIXES = ("http://", "https://", "mailto:", "irc://", "#")


def md_files() -> list[Path]:
    files = []
    for path in ROOT.rglob("*.md"):
        rel = path.relative_to(ROOT).as_posix()
        if any(rel.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
            continue
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files)


def dangling_targets(src: Path) -> list[str]:
    text = src.read_text(encoding="utf-8")
    targets = []
    for pattern in (INLINE_LINK, IMAGE_LINK):
        targets.extend(match.group(1) for match in pattern.finditer(text))
    targets.extend(match.group(1) for match in REF_DEF.finditer(text))
    broken = []
    for target in targets:
        if target.startswith(SKIP_PREFIXES):
            continue
        path_part = target.split("#", 1)[0]
        if not path_part:  # a same-page anchor
            continue
        if not (src.parent / path_part).resolve().exists():
            broken.append(target)
    return sorted(set(broken))


def main() -> int:
    files = md_files()
    failures = []
    for src in files:
        for target in dangling_targets(src):
            failures.append(f"{src.relative_to(ROOT)}: dangling reference '{target}'")
    if failures:
        print("\n".join(failures))
        print(f"\n{len(failures)} dangling markdown reference(s)")
        return 1
    print(f"OK: markdown references resolve across {len(files)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
