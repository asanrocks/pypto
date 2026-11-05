# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Compile-time diagnostics — ``static_print`` and ``static_assert``.

These are **runtime no-ops** whose checks only the AST parser runs.
Under the tracing frontend they silently do nothing.
"""

from __future__ import annotations

from typing import Any


def static_print(*args: Any) -> None:
    """Print compile-time information about IR objects.

    At parse time, prints type/value info to stdout. At runtime, no-op.

    Args:
        *args: Values to print (variables, expressions, string labels)
    """


def static_assert(condition: Any, msg: str = "") -> None:
    """Assert a condition at compile time (parse time).

    At parse time, evaluates ``condition``. If false, raises ``ParserError``.
    At runtime, this is a no-op (all semantics are handled by the parser).

    Notes:
        * This is a **statement-only** construct. It must be used as a
          standalone statement, not as part of an expression.
        * The ``msg`` argument must be a **string literal** at the call site.
          Passing a variable or expression for ``msg`` will raise
          ``ParserSyntaxError``.
        * The check is evaluated at parse time only; it does not run at
          execution time.

    Args:
        condition: Condition to check (must be compile-time evaluable)
        msg: Optional error message as a string literal
    """
