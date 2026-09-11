# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The public projection of the ambient trace channel — ``current_trace()``.

``current_trace()`` returns ``None`` on the host or an opaque,
invocation-scoped capability during collection. It introduces no new
ContextVar: it projects the one existing ambient channel (the active Frame
funnel). The capability exposes exactly one operation — ``expand`` — and
must not be retained after its invocation: a stale capability is loud.

Expansion runs a prepared body inside the caller's trace with a fresh
interpreter and a borrowing expansion Sema: it borrows the parent's
IRBuilder and current insertion point, creates a fresh lexical Frame, and
emits no function, declaration, or call. Expansion-local ``return`` supplies
the value to the parent; control cannot target caller constructs; failure
discards the complete parent trace. Stdlib trace-protocol hooks keep
receiving their explicit language ``TraceContext`` — this capability is not
a replacement for that protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, final

from pypto.language.ambient import active_frame, reset_active_frame, set_active_frame
from pypto.language.tracer.collector.driver import InstrumentedFunction
from pypto.language.tracer.semantics import PtoSema

from .declaration import Param
from .traced import TracedCallable

if TYPE_CHECKING:
    from pypto.language.tracer.semantics import PtoFrame

__all__ = ["current_trace"]


def current_trace() -> TraceCapability | None:
    """The opaque capability for the active trace, or ``None`` on the host.

    The capability is invocation-scoped: it is valid only while the Frame it
    was created for is the active one, and it exposes only expansion.
    """
    frame = active_frame()
    if frame is None:
        return None
    return TraceCapability(frame)


@final
class TraceCapability:
    """The opaque, invocation-scoped tracing capability.

    Holds the active Frame privately; exposes nothing but
    :meth:`expand`. Using it after its invocation ended is a loud error, so
    a capability cannot silently address a later trace.
    """

    __slots__ = ("_frame",)

    def __init__(self, frame: PtoFrame) -> None:
        self._frame = frame

    def expand(
        self,
        body: TracedCallable,
        *,
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
    ) -> Any:
        """Expand one prepared body into the caller's trace.

        ``args``/``kwargs`` are the values the body is invoked with —
        facades, host values, structured values, or callables may cross; a
        ``Param`` placeholder may not (it belongs to a declaration's root
        invocation, not to an expansion call).
        """
        frame = self._frame
        if active_frame() is not frame:
            raise RuntimeError(
                "the trace capability is stale: its invocation has ended — a "
                "capability must not be retained past its trace"
            )
        if frame.sema.expansion_parent is not None:
            raise TypeError(
                f"expansion of {frame.sema.name!r} is already active — nested or "
                "recursive expansion is rejected (one expansion at a time per trace)"
            )
        if not isinstance(body, TracedCallable):
            raise TypeError(f"trace.expand expects a prepared TracedCallable body, got {body!r}")
        call_args = tuple(args)
        call_kwargs = dict(kwargs or {})
        for value in (*call_args, *call_kwargs.values()):
            if isinstance(value, Param):
                raise TypeError(
                    "an expansion argument may not be a Param placeholder — pass the "
                    "values (facades or host values) directly"
                )
        return _run_expansion(body, frame, call_args, call_kwargs)


def _run_expansion(
    body: TracedCallable,
    parent_frame: PtoFrame,
    args: tuple[object, ...],
    kwargs: dict[str, object],
) -> Any:
    """Execute one borrowing expansion of ``body`` inside the parent trace.

    The ambient channel switches to the expansion frame for the execution
    and is restored unconditionally afterwards; on success the trace
    interpreter closes the expansion frame (which finalizes nothing — the
    parent owns the function), and the body's return value crosses to the
    caller. An escaping exception skips every normal finalization and
    propagates, discarding the complete parent trace.
    """
    sema = PtoSema._for_expansion(parent_frame.sema, owner=body.__name__)
    frame = sema._prepare_frame()
    token = set_active_frame(frame)
    try:
        runner = InstrumentedFunction(body._preparation.live_function(), body._preparation.prefix, sema)
        return runner(*args, **kwargs)
    finally:
        reset_active_frame(token)
