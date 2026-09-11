# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The decorator adapter's class-method recovery.

Declared-signature direction recovery (``param_directions``) lives in the
shared language layer (:mod:`pypto.language.typing.arg_direction`) — the
``Marker`` in ``op/marker.py`` needs it too."""

from __future__ import annotations

import inspect
import types

__all__ = ["is_class_method"]


def is_class_method(func: types.FunctionType) -> bool:
    """Re-skinned from the parser frontend's ``_is_class_method``: is
    ``func`` a method inside a class (vs a standalone kernel)?

    Recognizes classic methods (leading ``self``) and self-contained
    functions defined inside a class, via ``__qualname__`` plus the source
    indentation — so a standalone function whose first parameter happens to
    be named ``self`` is not misclassified. Nested-function qualnames
    (``outer.<locals>.inner``) are not class methods.
    """
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return False
    parent = qualname.rsplit(".", 1)[0]
    if parent.endswith("<locals>"):
        return False
    try:
        source_lines_raw, _ = inspect.getsourcelines(func)
        indentation = len(source_lines_raw[0]) - len(source_lines_raw[0].lstrip())
        if indentation > 0:
            return True
    except (OSError, TypeError):
        # cannot get source — trust the qualname (a class-qualified name)
        return True
    return False
