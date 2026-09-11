# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Code formatter for IR printer output using ruff.

The IR printer routes its output through a global format callback
(``register_format_callback``) so printed IR reads like formatted Python.
Two properties of that path matter for cost:

- ruff is a subprocess: formatting the same text twice spawns the process
  twice. The printer emits the same type/function strings over and over
  (every ``str(ir.Type)`` on the hot paths), so ``ruff_format`` memoizes.
- Some callers need the printer's *canonical text* for equality/hash keys,
  not its presentation. The printer is the lossless round-trip serializer,
  so raw text equality is exactly structural equality — ``raw_print()``
  temporarily bypasses formatting for those callers, keeping keying
  subprocess-free.
"""

import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# ruff is a pure function of (code, line_length): the printer emits the same
# type/function strings over and over (every `str(ir.Type)` on the hot paths —
# variant keys, spans), and each call is a process spawn, so memoize.
_FORMAT_CACHE: dict[tuple[str, int], str] = {}
_FORMAT_CACHE_MAX = 8192

# Raw-print mode: the format callback returns its input verbatim. Scoped per
# context, so a raw-printing caller cannot leak the mode into interleaved
# formatted printing.
_RAW_PRINT: ContextVar[bool] = ContextVar("pto_ir_raw_print", default=False)


@contextmanager
def raw_print() -> Iterator[None]:
    """Print IR without formatting for the duration of the context.

    The printer's raw output is the canonical (round-trip) serialization, so
    equality of raw text is structural equality — callers building identity
    keys from printed IR use this to avoid the formatter entirely.
    """
    token = _RAW_PRINT.set(True)
    try:
        yield
    finally:
        _RAW_PRINT.reset(token)


def formatting_callback(code: str) -> str:
    """The IR printer's format callback: ruff-formatted, or raw under
    :func:`raw_print`."""
    if _RAW_PRINT.get():
        return code
    return ruff_format(code)


def ruff_format(code: str, line_length: int = 200) -> str:
    """Format a Python code string using ruff.

    Returns the original code unchanged on any failure (ruff not found, timeout,
    parse error), so it is safe to use as a best-effort post-processor.

    Args:
        code: Python source code to format
        line_length: Maximum line length (default 200)

    Returns:
        Formatted code, or original code on failure
    """
    cache_key = (code, line_length)
    cached = _FORMAT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        result = subprocess.run(
            [
                "ruff",
                "format",
                "--line-length",
                str(line_length),
                "--stdin-filename",
                "ir_output.py",
                "-",
            ],
            check=False,
            input=code,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # ruff always appends a trailing newline; strip it to match the
            # raw printer output which may or may not end with one.
            formatted = result.stdout
            if formatted.endswith("\n") and not code.endswith("\n"):
                formatted = formatted[:-1]
            if len(_FORMAT_CACHE) < _FORMAT_CACHE_MAX:
                _FORMAT_CACHE[cache_key] = formatted
            return formatted
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return code
