#!/usr/bin/env python3
"""R2 oracle: report comment-only changes to code files vs a base ref.

"Comment-only hunk" means every added/removed line in the hunk is blank or a
comment (``#`` / ``//``), with no code change in the same hunk. Those are the
hunks whose only possible justification is prose, so they are the candidates
for unnecessary churn in upstream comments.

Docs prose (``docs/**/*.md``) is out of scope: the migration rewrote it
deliberately (commits 87d7f25c7, 5061fe4dc). Markdown headings would otherwise
be indistinguishable from code comments.

Files listed in --allowlist (default scripts/review/comment_churn_allowlist.txt,
one path prefix per line, ``#`` comments ignored) are treated as justified and
excluded from the reported total. Growing that file is the review decision; an
entry must carry a reason in the tracker.

Report-only: not wired into ./scripts/gate.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

CODE = (".py", ".cpp", ".h", ".hpp", ".cc", ".rs", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".cmake")
# The review apparatus is not the changeset under review: its own comment-only
# files (this oracle's allowlist, for one) would otherwise report as findings.
OUT_OF_SCOPE = ("scripts/review/",)
DEFAULT_ALLOWLIST = Path(__file__).with_name("comment_churn_allowlist.txt")


def is_comment(line: str) -> bool:
    s = line[1:].strip()
    return s == "" or s.startswith("#") or s.startswith("//")


def comment_only_hunks(base: str) -> dict[str, int]:
    raw = subprocess.run(
        ["git", "diff", "-U0", f"{base}..HEAD"], check=False, capture_output=True, text=True
    ).stdout
    # One entry per *hunk*: the path comes from the +++ header, the body is
    # that hunk's +/- lines. Keying on --- headers instead merges every hunk of
    # a file into one, which hides a comment-only hunk behind any code change
    # elsewhere in the file.
    hunks: list[list[str | None | list[str]]] = []
    cur: str | None = None
    for line in raw.splitlines():
        if line.startswith("diff --git"):
            cur = None
            continue
        if line.startswith("--- "):
            continue  # old-side header; the path arrives on +++
        if line.startswith("+++ "):
            cur = None if line.startswith("+++ /dev/null") else line[6:]
            continue
        if line.startswith("@@"):
            hunks.append([cur, []])
            continue
        if line.startswith("\\"):
            continue  # "\ No newline at end of file"
        if hunks and hunks[-1][0] is not None and (line.startswith("+") or line.startswith("-")):
            hunks[-1][1].append(line)  # type: ignore[union-attr]

    cov: dict[str, int] = {}
    for path, lines in hunks:
        if path is None or not lines or not path.endswith(CODE):
            continue
        if any(path.startswith(prefix) for prefix in OUT_OF_SCOPE):
            continue
        if all(is_comment(line) for line in lines):
            cov[path] = cov.get(path, 0) + len(lines)
    return cov


def load_allowlist(path: Path) -> tuple[str, ...]:
    if not path.exists():
        return ()
    entries = []
    for raw in path.read_text().splitlines():
        entry = raw.strip()
        if entry and not entry.startswith("#"):
            entries.append(entry)
    return tuple(entries)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default="origin/main")
    ap.add_argument("--allowlist", type=Path, default=DEFAULT_ALLOWLIST)
    args = ap.parse_args()

    allow = load_allowlist(args.allowlist)
    cov = comment_only_hunks(args.base)

    justified = {p: c for p, c in cov.items() if any(p.startswith(a) for a in allow)}
    flagged = {p: c for p, c in cov.items() if p not in justified}

    print(f"== comment-only hunks in code files vs {args.base} ==")
    for path, count in sorted(flagged.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {count:5d}  {path}")
    print(f"-- flagged: {sum(flagged.values())} lines in {len(flagged)} files --")

    if justified:
        print(f"-- allowlisted (justified): {sum(justified.values())} lines in {len(justified)} files --")
        for path, count in sorted(justified.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {count:5d}  {path}")

    print(
        f"\nRESULT: {'FAIL' if flagged else 'PASS'} (unjustified churn: "
        f"{sum(flagged.values())} lines, {len(flagged)} files)"
    )
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
