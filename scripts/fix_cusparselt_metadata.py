#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Repair the malformed ARM wheel metadata for nvidia-cusparselt-cu13.

The upstream aarch64 wheel contains a ``manylinux2014_sbsa`` WHEEL tag even
though its filename and binary target aarch64. uv rejects that tag and
reinstalls the package on every synchronized command.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path

PACKAGE = "nvidia_cusparselt_cu13-0.8.1.dist-info"
OLD_TAG = "Tag: py3-none-manylinux2014_sbsa"
NEW_TAG = "Tag: py3-none-manylinux2014_aarch64"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def environment_roots() -> list[Path]:
    roots: list[Path] = []
    for value in (os.environ.get("VIRTUAL_ENV"), os.environ.get("UV_PROJECT_ENVIRONMENT")):
        if value:
            root = Path(value).expanduser()
            roots.append(root if root.is_absolute() else project_root() / root)

    roots.append(project_root() / ".venv")
    return list(dict.fromkeys(root.resolve() for root in roots))


def site_packages(root: Path) -> Path | None:
    candidates = sorted((root / "lib").glob("python*/site-packages"))
    return candidates[-1] if candidates else None


def digest(path: Path) -> str:
    encoded = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest())
    return encoded.rstrip(b"=").decode("ascii")


def update_record(dist_info: Path, wheel: Path) -> None:
    record = dist_info / "RECORD"
    if not record.is_file():
        raise RuntimeError(f"missing RECORD: {record}")

    record_entry = f"{dist_info.name}/WHEEL"
    lines = record.read_text(encoding="utf-8").splitlines(keepends=True)
    for index, line in enumerate(lines):
        fields = line.rstrip("\r\n").split(",", 2)
        if fields[0] != record_entry:
            continue
        newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
        lines[index] = f"{record_entry},sha256={digest(wheel)},{wheel.stat().st_size}{newline}"
        record.write_text("".join(lines), encoding="utf-8")
        return

    raise RuntimeError(f"missing {record_entry} entry in {record}")


def repair() -> int:
    for root in environment_roots():
        packages = site_packages(root)
        if packages is None:
            continue
        dist_info = packages / PACKAGE
        wheel = dist_info / "WHEEL"
        if not wheel.is_file():
            continue

        contents = wheel.read_text(encoding="utf-8")
        if NEW_TAG in contents and OLD_TAG not in contents:
            print(f"already repaired: {wheel}")
            return 0
        if OLD_TAG not in contents:
            raise RuntimeError(f"unexpected wheel tag in {wheel}")

        wheel.write_text(contents.replace(OLD_TAG, NEW_TAG, 1), encoding="utf-8")
        update_record(dist_info, wheel)
        print(f"repaired: {wheel}")
        return 0

    raise RuntimeError("could not find nvidia-cusparselt-cu13 0.8.1 in the project virtual environment")


def main() -> int:
    try:
        return repair()
    except (OSError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
