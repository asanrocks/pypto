# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The authoring tier — the shared surface between the stdlib and the tracer.

One runtime under the tier's protocols (see
``docs/en/dev/language/07-authoring-recipes.md``): the session that owns one
invocation, its function units, and the contracts a recipe pushes.

The stdlib's structural constructs (``pl.at``, ``pl.spmd``, ...) are this
tier's reference clients — they are built on the same public capabilities an
external framework author uses, never on tracer internals. Recipe code receives
capability views (``FunctionContext`` / ``ControlContext``) only: no builder,
no ``PtoFrame``, no interpreter.

The package splits by responsibility: the contracts and artifacts
(:mod:`.contracts`) and the session and function-unit lifecycle
(:mod:`.session`). This package is the tier's sole import root: nothing of the
authoring surface is re-exported at the top-level ``pl`` namespace — the root
stays the kernel author's, and framework authors import here.
"""

from pypto.language.authoring.contracts import (
    CollectedBody,
    FunctionArtifact,
    FunctionContract,
    ParamSpec,
)
from pypto.language.authoring.session import (
    ControlContext,
    FunctionContext,
)

__all__ = [
    "CollectedBody",
    "ControlContext",
    "FunctionArtifact",
    "FunctionContract",
    "FunctionContext",
    "ParamSpec",
]
