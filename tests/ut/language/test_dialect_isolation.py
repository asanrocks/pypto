# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Import isolation between the dialect tier and the parser frontend.

The dialect tier (constructs, diagnostics, typing vocabulary) is shared by
both the tracing and the parsing frontends. Importing any dialect module must
not drag the parser package in; the parser must stay reachable lazily via
``pypto.language`` for existing ``pl.function`` / ``pl.parse`` spellings.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_PYTHON_DIR = Path(__file__).resolve().parents[3] / "python"


def _import_without_parser(module: str) -> None:
    code = (
        f"import sys; import {module}; "
        'assert "pypto.language.parser" not in sys.modules, '
        f'"importing {module} pulled in pypto.language.parser"'
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": str(_PYTHON_DIR), "PATH": "/usr/bin:/bin"},
    )


def test_constructs_optimizations_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.constructs.optimizations")


def test_diagnostics_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.diagnostics")


def test_typing_arg_direction_isolated_from_parser() -> None:
    _import_without_parser("pypto.language.typing.arg_direction")


def test_lazy_parser_spelling_still_works() -> None:
    code = (
        "import sys; import pypto.language as pl; "
        "assert pl.function is not None; "
        "assert pl.parse is not None; "
        "assert pl.parser.decorator.function is pl.function; "
        "assert 'pypto.language.parser' in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": str(_PYTHON_DIR), "PATH": "/usr/bin:/bin"},
    )


def test_diagnostics_exports_its_bug_class_family() -> None:
    """``BUG_CLASS_EXCEPTIONS`` is the shared classification callers widen on.

    It is defined and re-exported through the compatibility paths, so its
    absence from the canonical module's ``__all__`` made wildcard imports of the
    two spellings disagree.
    """
    from pypto.language import diagnostics  # noqa: PLC0415 - dialect-tier seam

    assert "BUG_CLASS_EXCEPTIONS" in diagnostics.__all__
    assert diagnostics.BUG_CLASS_EXCEPTIONS == (diagnostics.InternalError, AssertionError)


def test_lazy_parser_module_resolves_on_a_direct_first_access() -> None:
    # ``pl.parser`` as the very first parser-family touch must resolve to
    # the submodule itself — not go looking for a ``parser`` attribute
    # inside the parser package (an ordering the other lazy test masks,
    # because importing any parser submodule binds ``parser`` onto this
    # package as an import side effect).
    code = (
        "import sys; import pypto.language as pl; "
        "first = pl.parser; "
        "import pypto.language.parser as parser_module; "
        "assert first is parser_module; "
        "assert first.decorator.function is not None"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={"PYTHONPATH": str(_PYTHON_DIR), "PATH": "/usr/bin:/bin"},
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
