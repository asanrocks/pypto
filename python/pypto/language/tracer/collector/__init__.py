# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The language-agnostic Python execution collector.

The package has four layers:

* :mod:`rewriter` rewrites kernel syntax into calls on the trace interpreter.
* :mod:`interpreter` executes the rewritten function and owns invocation
  state, instrumented locals, per-branch locals mappings, and one ordered
  stack of active Constructs.
* :mod:`interface` defines the boundary to a language frontend — the
  one-shot Sema, the Frame protocol, and typed lifecycle results. The
  collector does not inspect or construct IR; a Frame may do so behind this
  boundary.
* :mod:`driver` compiles the rewritten function, restores its closure,
  injects a fresh interpreter (and a fresh Sema, for a factory) per call,
  and finalizes only on the success path.

The event protocol the source rewriter emits (an instrumented ``if``/``for``, in
order):

.. code-block:: text

    rt.enter_frame(**params)     # prologue: fresh frame, parameter bindings
    rt.eval_if(pred)             # install the IfConstruct (host or staged)
    if rt.eval_then(): ...       # then guard — True on both paths when staged
    rt.eval_else()               # branch switch — staged: swap the mapping
    if rt.eval_else_active(): ...
    rt.exit_construct()          # close; commit or discard branch mappings
    it = rt.begin_for(iterable)  # install the ForConstruct before iteration
    for i in it: ...             # staged: one representative iteration
    rt.exit_construct()          # from inside the else suite when present
    rt.close()                   # success path only (:mod:`driver`)

A construct whose body holds a native ``break``/``continue`` bound to an
enclosing loop replaces its trailing ``rt.exit_construct()`` with the unwind
scope guard ``with rt.unwind_guard():`` — the keyword's own unwind runs the
guard exit at the construct's exact lexical boundary, preserving Python
unwind order. ``with`` constructs are not wrapped because their generated
context manager already runs during native unwinding.

``while`` enters through ``eval_bool(thunk)`` and makes a lazy staging
decision: the Construct owns the host-to-staged transition and traces one
representative body iteration after activation. ``with`` installs its
Construct at ``on_begin_with`` and exits through the returned context manager,
not a trailing ``exit_construct``. Walrus targets bind through ``rt.bind``;
every other instrumented read and write goes through ``rt.vars``.

``NullSema`` provides host execution without IR production. The package itself
remains independent of the language frontend and imports only its
standard-library dependencies plus LibCST.
"""

from .driver import InstrumentedFunction, instrument
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
    "InstrumentedFunction",
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
    "instrument",
]
