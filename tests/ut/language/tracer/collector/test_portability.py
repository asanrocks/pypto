# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Portability litmus: the collector package is
PTO-agnostic — zero pypto imports, stdlib + typing only."""

import ast
import os
import sys

_PACKAGE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "..",
    "python",
    "pypto",
    "language",
    "tracer",
    "collector",
)


def _module_sources():
    for name in sorted(os.listdir(_PACKAGE_DIR)):
        if name.endswith(".py"):
            yield name, os.path.join(_PACKAGE_DIR, name)


def test_no_pypto_imports():
    """the no-``pypto``-imports grep test."""
    offenders = []
    for name, path in _module_sources():
        with open(path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                stripped = line.strip()
                if stripped.startswith("import pypto") or stripped.startswith("from pypto"):
                    offenders.append(f"{name}:{lineno}: {stripped}")
    assert not offenders, "collector must be PTO-agnostic (zero pypto imports):\n" + "\n".join(offenders)


def test_import_graph_is_closed():
    """every import resolves to the stdlib or the collector's own
    modules — nothing else."""
    allowed = set(sys.stdlib_module_names) | {"collector"}
    offenders = []
    for name, path in _module_sources():
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] not in allowed:
                        offenders.append(f"{name}: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                # relative imports (level > 0) are intra-package
                if node.level == 0 and node.module and node.module.split(".")[0] not in allowed:
                    offenders.append(f"{name}: {node.module}")
    assert not offenders, "collector imports must stay inside stdlib + collector:\n" + "\n".join(offenders)


def test_relocated_copy_runs_corpus(tmp_path):
    """the portability litmus: the **full** behavioral corpus
    passes from a copied package under a different name — zero pypto
    imports make the copy importable standalone (the litmus drives
    every corpus case, not a single kernel)."""
    import importlib  # noqa: PLC0415 - function-local: the module name is dynamic
    import shutil  # noqa: PLC0415

    from _corpus import CASES  # noqa: PLC0415 - the corpus kernels import nothing from the collector

    new_name = "collector_copy"
    target = tmp_path / new_name
    shutil.copytree(_PACKAGE_DIR, target)
    sys.path.insert(0, str(tmp_path))

    mod = importlib.import_module(new_name)
    for name, kernel, argsets in CASES:
        for args in argsets:
            assert mod.instrument(kernel)(*args) == kernel(*args), f"corpus case {name!r} failed"
