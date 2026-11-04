# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The authoring tier — the shared surface between the stdlib and the tracer.

Two protocols over one runtime (see
``docs/en/dev/language/07-authoring-recipes.md``): ``@pl.authoring.construct``
— the constrained region protocol, a generator whose single ``yield`` is a body
boundary, serving both the ``with`` and the ``for`` statement spellings of one
construct; and ``@pl.authoring.recipe`` — the unconstrained effect-composition
form, a program over an authoring context (function units, body placement,
dispatch, control), with ``@pl.authoring.body`` the prepared-body form awaiting
an explicit placement. Both are views over the session that owns one
invocation, its function units, and the dispatch effect.

The stdlib's structural constructs (``pl.at``, ``pl.spmd``, ...) are this
tier's reference clients — they are built on the same public capabilities an
external framework author uses, never on tracer internals. Recipe code receives
capability views (``ConstructContext`` / ``FunctionContext`` /
``ControlContext``) only: no builder, no ``PtoFrame``, no interpreter.

The package splits by responsibility: the contracts and artifacts
(:mod:`.contracts`), the session and function-unit lifecycle
(:mod:`.session`), the construct protocol (:mod:`.construct`), and the
dispatch effect (:mod:`.dispatch`). This package is
the tier's sole import root: nothing of the authoring surface is re-exported at
the top-level ``pl`` namespace — the root stays the kernel author's, and
framework authors import here.
"""

from pypto.language.authoring.construct import (
    CarrierRegion,
    ConstructContext,
    ConstructHandle,
    ConstructRecipe,
    LoopRegion,
    ScopeRegion,
    construct,
)
from pypto.language.authoring.contracts import (
    CollectedBody,
    FunctionArtifact,
    FunctionContract,
    ParamSpec,
)
from pypto.language.authoring.recipe import (
    RecipeCallable,
    RecipeContext,
    TraceableBody,
    body,
    recipe,
)
from pypto.language.authoring.session import (
    ControlContext,
    FunctionContext,
)

from . import dispatch as dispatch

__all__ = [
    "CarrierRegion",
    "CollectedBody",
    "ConstructContext",
    "ConstructHandle",
    "ConstructRecipe",
    "ControlContext",
    "FunctionArtifact",
    "FunctionContract",
    "FunctionContext",
    "LoopRegion",
    "ParamSpec",
    "RecipeCallable",
    "RecipeContext",
    "ScopeRegion",
    "TraceableBody",
    "body",
    "construct",
    "dispatch",
    "recipe",
]
