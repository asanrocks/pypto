# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The language-agnostic Python execution collector.

The package has three layers so far:

* :mod:`rewriter` rewrites kernel syntax into calls on the trace interpreter.
* :mod:`interpreter` executes the rewritten function and owns invocation
  state, instrumented locals, per-branch locals mappings, and one ordered
  stack of active Constructs.
* :mod:`interface` defines the boundary to a language frontend — the
  one-shot Sema, the Frame protocol, and typed lifecycle results. The
  collector does not inspect or construct IR; a Frame may do so behind this
  boundary.

``NullSema`` provides host execution without IR production. The package itself
remains independent of the language frontend and imports only the standard
library.
"""

from .interface import (
    AwaitingResultBindings,
    BindingResult,
    BranchDisposition,
    Closed,
    CloseResult,
    FinalResultBinding,
    Frame,
    HostBinding,
    NullFrame,
    NullSema,
    PartialResultBinding,
    Sema,
    SemaFactory,
    StagedBinding,
)
from .interpreter import (
    Construct,
    ForConstruct,
    IfConstruct,
    Interpreter,
    InterpreterError,
    WhileConstruct,
    WithConstruct,
)

__all__ = [
    "AwaitingResultBindings",
    "BindingResult",
    "BranchDisposition",
    "Closed",
    "CloseResult",
    "Construct",
    "FinalResultBinding",
    "ForConstruct",
    "Frame",
    "HostBinding",
    "IfConstruct",
    "Interpreter",
    "InterpreterError",
    "NullFrame",
    "NullSema",
    "PartialResultBinding",
    "Sema",
    "SemaFactory",
    "StagedBinding",
    "WhileConstruct",
    "WithConstruct",
]
