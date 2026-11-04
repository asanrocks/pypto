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

import pytest

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
    "tracing",
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


def _relocated_copy(tmp_path, name: str):
    """Import a fresh copy of the collector under ``name``: zero pypto
    imports make the copy importable standalone from its new location."""
    import importlib  # noqa: PLC0415 - function-local: the module name is dynamic
    import shutil  # noqa: PLC0415

    target = tmp_path / name
    shutil.copytree(_PACKAGE_DIR, target)
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(name), str(target)
    except BaseException:
        sys.path.remove(str(tmp_path))
        raise


def _drop_path(path: str) -> None:
    if path in sys.path:
        sys.path.remove(path)


def test_relocated_copy_runs_corpus(tmp_path):
    """the portability litmus: the **full** behavioral corpus
    passes from a copied package under a different name — zero pypto
    imports make the copy importable standalone (the litmus drives
    every corpus case, not a single kernel)."""
    from _corpus import CASES  # noqa: PLC0415 - the corpus kernels import nothing from the collector

    mod, path = _relocated_copy(tmp_path, "collector_copy")
    try:
        for name, kernel, argsets in CASES:
            for args in argsets:
                assert mod.instrument(kernel)(*args) == kernel(*args), f"corpus case {name!r} failed"
    finally:
        _drop_path(path)


def test_relocated_copy_interprets_the_same_construct_stream(tmp_path):
    """Relocation preserves the interpreted lifecycle, not just results: a
    recording Sema built from the relocated module's own extension surface
    (NullSema/NullFrame subclassing, Frame-created Constructs) drives the
    corpus control-flow kernel and must observe the exact ordered open /
    branch-switch / close event stream for its if, for, and while — the
    ordered event stream, asserted literally, not merely the return value."""
    from _corpus import k_control_flow  # noqa: PLC0415 - the corpus module is local to the suite

    mod, path = _relocated_copy(tmp_path, "collector_copy_events")
    try:
        import importlib  # noqa: PLC0415 - function-local: the module name is dynamic

        mod_interface = importlib.import_module("collector_copy_events.interface")

        class RecordingIf(mod.IfConstruct):
            def __init__(self, guard, log):
                super().__init__(guard)
                self._log = log

            def enter_else(self):
                self._log.append("if:else")
                super().enter_else()

            def close(self):
                self._log.append("if:close")
                return mod.Closed()

        class RecordingFor(mod.ForConstruct):
            def __init__(self, iterable, log):
                super().__init__(iterable)
                self._log = log

            def close(self):
                self._log.append("for:close")
                return mod.Closed()

        class RecordingWhile(mod.WhileConstruct):
            def __init__(self, log):
                super().__init__()
                self._log = log

            def close(self):
                self._log.append("while:close")
                return mod.Closed()

        class RecordingFrame(mod.NullFrame):
            def __init__(self, log):
                self._log = log

            def create_if_construct(self, interpreter, predicate, span=None):
                self._log.append("if:open")
                return RecordingIf(bool(predicate), self._log)

            def create_for_construct(self, interpreter, iterable, span=None):
                self._log.append("for:open")
                return mod_interface.ForOpen(RecordingFor(iterable, self._log), "target")

            def create_while_construct(self, interpreter, predicate_thunk, span=None):
                self._log.append("while:open")
                return RecordingWhile(self._log)

        class RecordingSema(mod.NullSema):
            def __init__(self, log):
                self._log = log

            def begin_frame(self):
                return RecordingFrame(self._log)

        log = []
        result = mod.instrument(k_control_flow, RecordingSema(log))(3)
        assert result == k_control_flow(3)
        # k_control_flow(3): the for runs once over range(3); per iteration one
        # if (the else-switch row runs on both paths — only the else-active
        # guard is branch-conditional); then one while of three iterations.
        assert log == [
            "for:open",
            "if:open",
            "if:else",
            "if:close",
            "if:open",
            "if:else",
            "if:close",
            "if:open",
            "if:else",
            "if:close",
            "for:close",
            "while:open",
            "while:close",
        ]
    finally:
        _drop_path(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
