# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""The recipe tier — the unconstrained authoring protocol over the session.

:class:`RecipeContext` is the capability object a general recipe receives:
the construct tier's region composition (inherited from
:class:`~pypto.language.authoring.construct.ConstructContext`) plus the
function effects (push / collect / finalize), the caller-side placements
(call / entry), the borrowed collection (splice), the dispatch effect, and
the semantic control capabilities. ``@pl.authoring.recipe`` wraps a plain function
into a :class:`RecipeCallable`; ``@pl.authoring.body`` prepares one reusable body
(:class:`TraceableBody`) awaiting an explicit placement — splice / inline /
incore.

See ``docs/en/dev/language/07-authoring-recipes.md`` for the contract and
the invariants.
"""

from __future__ import annotations

import functools
import types
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, cast, overload

from pypto.language.ambient import active_frame
from pypto.language.authoring.construct import ConstructContext
from pypto.language.authoring.contracts import FunctionArtifact, FunctionContract
from pypto.language.authoring.session import (
    AuthoringSession,
    ControlContext,
    FunctionContext,
    _active_assembly,
    _FunctionUnit,
    _static_extent,
)
from pypto.language.tracing.entry.traced import TracedCallable
from pypto.language.tracing.return_contract import pin_derived_return_types
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

if TYPE_CHECKING:
    # The dispatch effect owns the submit-operand vocabulary and the unpacking
    # view it returns; importing the module at runtime would close the
    # op/typing cycle the effect defers to its call site.
    from pypto.language.authoring.dispatch import (
        SubmitArg,
        SubmitCallee,
        SubmitEntries,
        SubmitOperand,
        _SubmitResult,
    )
    from pypto.language.typing import Tuple

__all__ = ["RecipeCallable", "RecipeContext", "TraceableBody", "body", "recipe"]


# -----------------------------------------------------------------------------------------------------------
# The recipe context
# -----------------------------------------------------------------------------------------------------------


class _BorrowedContext:
    """The ``ctx.current`` view of the borrowed caller frame.

    Its one effect is the borrowed collection — :meth:`collect_body` splices
    a prepared body into the caller's current insertion point: no function,
    no declaration, no call — the splice's semantics as an
    owner-addressed effect."""

    __slots__ = ("_session",)

    def __init__(self, session: AuthoringSession) -> None:
        self._session = session

    def collect_body(
        self,
        body: object,
        *,
        args: Sequence[object] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        """The borrowed collection — the splice (see :meth:`AuthoringSession.splice_body`)."""
        return self._session.splice_body(body, args=args, kwargs=kwargs)


class RecipeContext(ConstructContext):
    """The capability object handed to a general recipe as its first argument.

    The construct tier's region composition (inherited from
    :class:`~pypto.language.authoring.construct.ConstructContext`) plus the
    function effects (push / collect / finalize), the caller-side placements
    (call / entry), the borrowed collection (splice), and the semantic
    control capabilities. No builder access, no Frame, no interpreter — the
    closed authoring surface over the shared session.
    """

    __slots__ = ()

    def __init__(self, session: AuthoringSession) -> None:
        super().__init__(session, binding_name=None, single_call_body=False)

    # —— the function effects ——

    @property
    def current(self) -> FunctionContext | _BorrowedContext:
        """The current insertion owner: the newest unit (its effects refuse a
        finalized or discarded one loudly), the root unit, or (a borrowed
        session with no units) the borrowed caller."""
        session = self._session
        if session._units:
            return FunctionContext(session._units[-1])
        if session._root is not None:
            return FunctionContext(session._root)
        return _BorrowedContext(session)

    def push_function(self, contract: FunctionContract) -> FunctionContext:
        """Open one child function unit (see :meth:`AuthoringSession.push_function`)."""
        return self._session.push_function(contract)

    def emit_call(
        self, artifact: FunctionArtifact, *args: object, key: Any = None, **kwargs: object
    ) -> object:
        """Emit a finalized artifact's call site in the caller's trace."""
        return self._session.emit_call(artifact, *args, key=key, **kwargs)

    def materialize_entry(self, artifact: FunctionArtifact) -> FunctionArtifact:
        """Mark the root unit's artifact as the entry (root sessions only)."""
        return self._session.materialize_entry(artifact)

    def splice_body(
        self,
        body: object,
        *args: object,
        auto_scope: bool | None = None,
        **kwargs: object,
    ) -> object:
        """Execute one prepared body at the caller's insertion point (the
        borrowed collection — the splice's semantics)."""
        return self._session.splice_body(body, args=args, kwargs=kwargs, auto_scope=auto_scope)

    @property
    def control(self) -> ControlContext:
        """The semantic control capabilities — terminators and transfers as
        context operations (see :class:`ControlContext`)."""
        return ControlContext(self._session)

    # —— the dispatch effect ——

    def submit(
        self,
        callee: SubmitCallee,
        *kernel_args: SubmitArg,
        deps: SubmitEntries | None = None,
        dumps: SubmitEntries | None = None,
        allow_early_resolve: bool = False,
        predicate: SubmitOperand | None = None,
        device: SubmitOperand | None = None,
        timing_slot: int | None = None,
        attrs: dict[str, Any] | None = None,
        core_num: SubmitOperand | None = None,
        sync_start: bool = False,
    ) -> Tuple | _SubmitResult:
        """Dispatch one task — the submit effect as an authoring capability.

        The recipe-facing spelling of the dispatch family (the lowering
        ``pl.submit`` / ``pl.spmd_submit`` share): a marked callee, its kernel
        arguments, and the dependency / dump / predicate / launch-spec
        options. ``core_num`` (with ``sync_start``) is the SPMD launch spec;
        without one this is a single-block dispatch. Returns the structural
        Tuple facade the stdlib spellings return — bind it under one name
        with :meth:`bind` and index the outputs / the producer TaskId, or
        return it to the caller for the native unpacking.

        The effect lands in the borrowed caller frame (a borrowed session,
        same as the other in-trace effects) and is rejected on a terminated
        region (dead code after ``emit_return``).
        """
        from pypto.language.authoring.dispatch import dispatch  # noqa: PLC0415 - the op/typing cycle

        session = self._session
        session._require_live()
        frame = session._require_borrowed("submit")
        session._check_not_terminated("submit")
        return dispatch(
            callee=callee,
            kernel_args=kernel_args,
            construct="ctx.submit",
            frame=frame,
            deps=deps,
            dumps=dumps,
            allow_early_resolve=allow_early_resolve,
            predicate=predicate,
            device=device,
            timing_slot=timing_slot,
            attrs=attrs,
            core_num=core_num,
            sync_start=sync_start,
        )

    # —— the binding effect: inherited from the construct view (ctx.bind) ——

    # —— read-only trace facts: inherited from the construct view ——


# -----------------------------------------------------------------------------------------------------------
# The decorators
# -----------------------------------------------------------------------------------------------------------


class RecipeCallable:
    """The product of :func:`recipe`: a callable authoring program.

    Called inside a trace, the recipe runs over a borrowed session and its
    return value crosses back to the caller's expression position. Invoked
    through :meth:`invoke`, the recipe runs over a root session — the
    entrance composition (push the entry, collect, finalize, materialize).
    The factory itself is stateless and reusable."""

    def __init__(self, function: types.FunctionType) -> None:
        if not isinstance(function, types.FunctionType):
            raise TypeError(f"@pl.authoring.recipe expects a plain function, got {function!r}")
        self._function = function
        self.__name__ = function.__name__
        self.__qualname__ = function.__qualname__
        self.__doc__ = function.__doc__
        self.__module__ = function.__module__
        self.__wrapped__ = function

    def __repr__(self) -> str:
        return f"<authoring recipe {self.__name__}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Run the recipe over a borrowed session (inside a trace)."""
        frame = active_frame()
        if frame is None:
            raise TypeError(
                f"{self._function.__name__}() requires an active trace — @pl.authoring.recipe composes "
                "tracing effects inside a traced kernel; a host call is rejected"
            )
        session = AuthoringSession(frame)
        try:
            result = self._function(RecipeContext(session), *args, **kwargs)
            session._validate_complete()
            return result
        finally:
            session.close()

    def invoke(self, contract: FunctionContract, *args: Any, **kwargs: Any) -> FunctionArtifact:
        """Run the recipe over a root session — the entrance composition.

        The session opens the entry function unit from ``contract`` before
        the recipe runs (``ctx.current`` is the root unit), and the recipe
        collects the entry body into it, finalizes, and materializes the
        entry. The artifact is the return; the jit envelope's dispatch
        (specialization, caching) wraps this step, it does not replace it.
        """
        if active_frame() is not None:
            raise TypeError(
                f"{self._function.__name__}.invoke is a root entrance — it cannot run "
                "inside an active trace (use the borrowed call instead)"
            )
        sema = PtoSema(
            name=contract.name,
            return_type=_contract_return_type(contract),
            function_type=contract.function_type,
            level=contract.level,
            role=contract.role,
            attrs=contract.attrs,
            strict_ssa=contract.strict_ssa,
            requires_runtime_binding=contract.requires_runtime_binding,
        )
        root = _FunctionUnit(None, sema, contract)  # type: ignore[arg-type] - the root unit pre-dates its session
        session = AuthoringSession(frame=None, root=root)
        root.session = session
        try:
            result = self._function(RecipeContext(session), *args, **kwargs)
            session._validate_complete()
        finally:
            session.close()
        artifact = result if isinstance(result, FunctionArtifact) else root.artifact
        if artifact is None:
            raise TypeError(
                f"{self._function.__name__}.invoke: the root recipe produced no entry — "
                "finalize the root unit and materialize_entry its artifact"
            )
        return artifact


@overload
def recipe(function: types.FunctionType) -> RecipeCallable: ...


@overload
def recipe(function: None = None) -> Callable[[types.FunctionType], RecipeCallable]: ...


def recipe(
    function: types.FunctionType | None = None,
) -> RecipeCallable | Callable[[types.FunctionType], RecipeCallable]:
    """Declare an authoring recipe — one program over an authoring context.

    .. code-block:: python

        @pl.authoring.recipe
        def scaled_square(ctx, body, x):
            fn = ctx.push_function(
                pl.authoring.FunctionContract(name="sq", function_type=pl.FunctionType.Inline)
            )
            artifact = fn.finalize(fn.collect_body(body, args=(x,)))
            return ctx.emit_call(artifact, x)

    See :mod:`pypto.language.authoring` for the contract and the
    invariants.
    """
    if function is None:  # the parenthesized form: @pl.authoring.recipe(...)
        return functools.partial(recipe)
    if not isinstance(function, types.FunctionType):
        raise TypeError(f"@pl.authoring.recipe expects a plain function, got {function!r}")
    return RecipeCallable(function)


class TraceableBody:
    """The product of :func:`body`: a prepared body awaiting placement.

    Preparation — the source rewrite and compile — happens once and caches
    only code-shaped facts. Placement is explicit: :meth:`splice` (the
    splice placement), :meth:`inline` and :meth:`incore` (the
    callee compositions). Calling the body directly is rejected with the
    placement guidance."""

    def __init__(self, kernel: types.FunctionType) -> None:
        self._kernel = kernel
        self._traced = TracedCallable.from_callable(kernel)
        self.__name__ = kernel.__name__
        self.__qualname__ = kernel.__qualname__
        self.__doc__ = kernel.__doc__
        self.__module__ = kernel.__module__
        self.__wrapped__ = kernel

    def __repr__(self) -> str:
        return f"<traceable body {self.__name__}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError(
            f"{self.__name__} is a prepared body — place it explicitly: "
            f"{self.__name__}.splice(...) at the caller's insertion point, "
            f"{self.__name__}.inline(...) or {self.__name__}.incore(...) as a callee"
        )

    # —— the placements ——

    def splice(self, *args: object, auto_scope: bool | None = None, **kwargs: object) -> Any:
        """Execute at the caller's insertion point, per occurrence."""
        session = _borrowed_session("splice")
        try:
            return session.splice_body(self, args=args, kwargs=kwargs, auto_scope=auto_scope)
        finally:
            session.close()

    def inline(self, *args: object) -> Any:
        """The Inline callee composition: push, collect, finalize, call."""
        return _callee_composition(self, args, _inline_contract)

    def incore(self, *args: object, level: ir.Level | None = None) -> Any:
        """The InCore callee composition: push, collect, finalize, call."""
        return _callee_composition(self, args, lambda name: _incore_contract(name, level))


def _inline_contract(name: str) -> FunctionContract:
    """The Inline callee contract (the jit envelope's default policies)."""
    from pypto.language.tracing.entry.declaration import fold_auto_scope  # noqa: PLC0415

    return FunctionContract(
        name=name,
        function_type=ir.FunctionType.Inline,
        attrs=fold_auto_scope(None, True),
    )


def _incore_contract(name: str, level: ir.Level | None) -> FunctionContract:
    """The InCore callee contract (the jit envelope's default policies).

    An explicit AIC/AIV level is the specialized InCore *function kind* —
    ``FunctionTypeToLevel`` pins AIC/AIV functions to their own levels, so
    the level selects the type rather than decorating an ``InCore`` one
    (which is pinned to ``CHIP_DIE``).
    """
    from pypto.language.tracing.entry.declaration import fold_auto_scope  # noqa: PLC0415

    if level is ir.Level.AIC:
        function_type: ir.FunctionType = ir.FunctionType.AIC
    elif level is ir.Level.AIV:
        function_type = ir.FunctionType.AIV
    else:
        function_type = ir.FunctionType.InCore
    return FunctionContract(
        name=name,
        function_type=function_type,
        level=level,
        attrs=fold_auto_scope(None, True),
    )


def _borrowed_session(operation: str) -> AuthoringSession:
    """Open a borrowed session over the active trace's frame."""
    frame = active_frame()
    if frame is None:
        raise TypeError(
            f"a body placement ({operation}) requires an active trace — place the body inside a traced kernel"
        )
    return AuthoringSession(frame)


def _callee_composition(
    body: TraceableBody, args: tuple[object, ...], contract_of: Callable[[str], FunctionContract]
) -> object:
    """The shared callee composition behind :meth:`TraceableBody.inline` /
    :meth:`TraceableBody.incore`.

    Pushes the contract's function unit, derives the signature from the
    staged arguments, collects, finalizes, and emits the call. When a program
    assembly is active, the composition is variant-deduped: the same body
    with structurally identical argument types is one function, not one per
    call site.
    """
    from pypto.language.op.marker import Marker  # noqa: PLC0415 - the marker/op cycle
    from pypto.language.protocol import is_staged  # noqa: PLC0415 - the facade/typing cycle

    session = _borrowed_session("a callee composition")
    try:
        assembly = _active_assembly()
        key: Any = None
        if assembly is not None:
            key = (
                body._kernel,
                tuple(_type_key(cast("Any", a).unwrap().type) for a in args if is_staged(a)),
            )
            cached = assembly.by_key.get(key)
            if cached is not None:
                return Marker._from_ir_function(cached)(*args)
        contract = contract_of(body.__name__)
        fn = session.push_function(contract)
        collected = fn.collect_body(body, args=args)
        artifact = fn.finalize(collected)
        # The callee normalization the jit envelope applies to a traced
        # helper: pin the derived return contract so the call site and the
        # printer see the declared signature an annotated callee would have.
        pinned = pin_derived_return_types(artifact.function)
        if pinned is not artifact.function:
            artifact = FunctionArtifact(pinned, artifact.contract)
        return session.emit_call(artifact, *args, key=key)
    finally:
        session.close()


@overload
def body(kernel: types.FunctionType) -> TraceableBody: ...


@overload
def body(kernel: None = None) -> Callable[[types.FunctionType], TraceableBody]: ...


def body(
    kernel: types.FunctionType | None = None,
) -> TraceableBody | Callable[[types.FunctionType], TraceableBody]:
    """Prepare one traceable body — reusable instrumentation awaiting placement.

    .. code-block:: python

        @pl.authoring.body
        def squared(x):
            return pl.mul(x, x)

        # inside a traced kernel:
        y = squared.inline(x)        # an Inline callee
        z = squared.splice(x)        # pasted at this point, no callee

    See :mod:`pypto.language.authoring` for the placement contracts.
    """
    if kernel is None:  # the parenthesized form: @pl.authoring.body(...)
        return functools.partial(body)
    if not isinstance(kernel, types.FunctionType):
        raise TypeError(f"@pl.authoring.body expects a plain function, got {kernel!r}")
    return TraceableBody(kernel)


def _contract_return_type(contract: FunctionContract) -> ir.Type | None:
    """The Sema's single declared return type from the flat return contract.

    The entrance foundation's own rule (``_sema_return_type``): one explicit
    return is that type; a flat multi-return contract leaves the Sema without
    a structural type — the flat element list is pinned onto the completed
    function instead.
    """
    returns = contract.returns
    if not returns:
        return None
    if len(returns) == 1:
        return returns[0]
    return None


def _type_key(t: ir.Type) -> Any:
    """A hashable approximation of ``t`` for variant dedup.

    Conservative by design: an uncovered shape yields a unique sentinel, so
    an unknown type never dedupes with a different one (the worst case is an
    extra variant, never a wrong one)."""
    if isinstance(t, ir.ScalarType):
        return ("scalar", t.dtype)
    shape = cast("Sequence[object] | None", getattr(t, "shape", None))
    if isinstance(shape, (list, tuple)) and all(_static_extent(e) for e in shape):
        kind = type(t).__name__
        return (kind, getattr(t, "dtype", None), tuple(shape), repr(getattr(t, "layout", None)))
    return ("opaque", id(t))
