# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The unsafe trace channel — the marked side-effect funnel.

Facade construction is supposed to be effect-free: value ops build
detached nodes, side effects flow only through the Frame's collection at
the recognition points the trace interpreter drives. Real IR building breaks that —
naming shared side-effecting calls, loop-output aliasing, region-mode
inheritance, extent-symbol folds — so facade construction may
observe/mutate the Frame/builder. This module is the **one channel** every
such "unsafe" site reaches through, making the escape hatch enumerable
(the Rust-``unsafe`` analog): the inventory is the caller list of
:func:`active_frame`, and this funnel is the only module that owns a
tracing ContextVar.

``active_frame()`` is the only ambient tracing entry. The trace driver
installs the current Frame for one function execution and resets it in an
unconditional ``finally``. The ``pl.split_aiv`` region-mode contextvar
retired: the mode is an exact PtoConstruct property obtained by querying the
trace interpreter through the Frame.
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypto.language.tracer.semantics import PtoFrame

__all__ = [
    "active_frame",
    "reset_active_frame",
    "set_active_frame",
]


# The funnel's backing store. Carries the active ``PtoFrame``, or None
# outside a trace (the no-trace guard).
_ACTIVE_FRAME: ContextVar[PtoFrame | None] = ContextVar("pto_active_frame", default=None)


def active_frame() -> PtoFrame | None:
    """The funnel: the active Frame, or ``None`` outside a trace.

    Every unsafe site calls this; the ``None`` return is the no-trace
    guard — an op called outside a trace must be inert or loud, never a
    mis-collect. Frame exposes the named stdlib operations (local
    rebinding, result-binding creation, approved active-construct
    queries); it never hands out the
    trace interpreter or the active-construct stack.
    """
    return _ACTIVE_FRAME.get()


# —— the trace-driver publisher (internal; sets/resets the funnel) ——
def set_active_frame(frame: PtoFrame) -> Token[PtoFrame | None]:
    """Install the current Frame for one function execution. Returns the
    token the driver resets in its unconditional ``finally``."""
    return _ACTIVE_FRAME.set(frame)


def reset_active_frame(token: Token[PtoFrame | None]) -> None:
    """Reset the funnel after the function execution."""
    _ACTIVE_FRAME.reset(token)
