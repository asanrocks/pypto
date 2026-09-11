# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""``pl.expand`` — the tracing-native expansion entrance.

Values and control have function semantics; IR placement has expansion
semantics. The decorated body is prepared once (rewrite and compile only)
and executed at every occurrence inside a traced kernel: arguments evaluate
once with normal Python binding, annotations play no role (they are not an
IR ABI), locals use a fresh lexical Frame, definition-site globals and
closures remain live, and ``return`` is expansion-local. No declaration,
function, ``GlobalVar``, or ``ir.Call`` is emitted for the expansion
itself. Host calls are rejected — the entrance executes only inside an
active trace.

The initial contract: one trailing return (an early return inside staged
control flow is loud) and no recursion (an expansion-stack diagnostic).
``FunctionType.Inline`` remains the distinct collect-once facility: it
collects one body, emits calls, and relies on the inlining pass, while
``pl.expand`` executes and emits its body at every occurrence.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable
from typing import Any

from pypto.language.entrance import TracedCallable, current_trace

__all__ = ["expand"]


def expand(kernel: Callable[..., object]) -> Any:
    """Prepare one expansion body over the entrance foundation.

    The annotation admits any callable (decorators can be applied to
    anything); the runtime check narrows to the real contract — a plain
    function — with a loud, ``@pl.expand``-attributed error.
    """
    if not isinstance(kernel, types.FunctionType):
        raise TypeError(f"@pl.expand expects a plain function, got {kernel!r}")
    body = TracedCallable.from_callable(kernel)

    @functools.wraps(kernel)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        trace = current_trace()
        if trace is None:
            raise TypeError(
                f"{kernel.__name__}() requires an active trace — @pl.expand executes "
                "its body at each occurrence inside a traced kernel; a host call "
                "is rejected"
            )
        return trace.expand(body, args=args, kwargs=kwargs)

    return wrapper
