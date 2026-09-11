# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Collector suite conftest: import the package standalone.

The collector is PTO-agnostic (zero pypto imports) — the tests
import it directly from ``python/pypto/language/tracer`` on sys.path, never through the
``pypto`` package's ``__init__`` (which pulls the C++ binding).
"""

import os
import sys

_COLLECTOR_ROOT = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "..", "python", "pypto", "language", "tracer"
)
if _COLLECTOR_ROOT not in sys.path:
    sys.path.insert(0, _COLLECTOR_ROOT)
