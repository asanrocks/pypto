# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tracing frontend's function entry.

``function_trace`` instruments the kernel through the collector's trace
driver and executes it with PtoSema: the params are bound to
annotation-derived facades, the body runs natively (the synthesized function
keeps the kernel's closure cells, so freevars resolve), the ops yield their
facades, and PtoSema builds the structural IR. Returns the built
``ir.Function``.

Sub-kernel calls resolve natively against the ``Marker`` in the kernel's
scope, emitting detached ``ir.Call`` nodes collected by the Frame's bind
machinery. Class/program tracing is not part of this frontend: ``pl.program``
and ``pl.function`` stay on the parsing lane.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
import typing
from pathlib import Path
from typing import Any

from pypto.language.diagnostics import ParserSyntaxError, ParserTypeError
from pypto.language.op.marker import EXTERNAL_SOURCE_ATTR, _fold_external_source
from pypto.language.protocol import is_staged, is_staged_annotation
from pypto.language.tracing.collector.driver import instrument
from pypto.language.tracing.entry.annotation import source_span_for_function
from pypto.language.tracing.entry.declaration import fold_auto_scope
from pypto.language.tracing.execution import (
    adapt_captured_callees,
    adapt_global_callees,
    pin_declared_returns,
    run_instrumented,
)
from pypto.language.tracing.return_contract import (
    check_return_paths,
    rebuild_with_return_types,
)
from pypto.language.tracing.semantics import PtoSema
from pypto.language.typing import (
    annotation_to_return_slots,
    annotation_to_type,
    facade_class_for_type,
)
from pypto.language.typing.arg_direction import param_directions, unmangle_hints, unmangle_param_name
from pypto.pypto_core import ir

__all__ = ["function_trace"]


def _prepare_external_source(
    kernel: types.FunctionType,
    function_type: ir.FunctionType,
    external_source: Any,
) -> str | None:
    """Validate and normalize a tracing external-kernel declaration."""
    if external_source is None:
        return None
    if function_type not in (ir.FunctionType.AIC, ir.FunctionType.AIV):
        raise ParserTypeError("external_source is only valid on FunctionType.AIC or FunctionType.AIV")

    path = Path(external_source)
    if not path.is_absolute():
        source_file = inspect.getsourcefile(kernel)
        if source_file is not None:
            path = Path(source_file).parent / path
    path = path.resolve()
    if not path.is_file():
        raise ValueError(
            f"@pl.function(external_source=...) file not found: {path}. "
            "Provide an absolute path, or one relative to the file defining the program."
        )

    try:
        source = textwrap.dedent(inspect.getsource(kernel))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError):
        # The tracing body is still skipped below for external functions. If
        # source inspection is unavailable, defer body validation to the
        # compiler rather than rejecting otherwise valid generated functions.
        return str(path)
    function_nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    body = function_nodes[0].body if function_nodes else []
    is_empty = (
        len(body) == 1
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is Ellipsis
    )
    if not is_empty:
        raise ParserSyntaxError(
            f"External kernel {kernel.__name__!r} must have an empty '...' body "
            "when backed by external_source."
        )
    return str(path)


def function_trace(  # noqa: PLR0913 - one kwarg per trace fact, mirroring _trace_with_annotations
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    auto_scope: bool = True,
    strict_ssa: bool = False,
    external_source: Any = None,
    localns: dict[str, object] | None = None,
    materialize_self: bool = False,
    specialize_global_callees: bool = False,
) -> ir.Function:
    """Build the ``ir.Function`` for ``kernel`` through the executed path.

    Args:
        kernel: The (undecorated) kernel whose body runs through the
            instrumented protocol with PtoSema.
        function_type/level/role/attrs: The function-classification facts
            — known at Marker creation, flowing
            Marker → PtoSema.__init__ →
            begin_function(type=…) at construction.
        external_source: Optional path to a hand-written C++ kernel that
            backs this (signature-only) AIC/AIV function; folded into the
            function attrs (mirroring the parser frontend's storage).
        materialize_self: Inject the jit ``self`` (the module materialized
            as an implicit program — ``self.<sibling>`` resolves the
            module's traced kernels) into the instrumented globals. The jit
            envelope's contract, never the standalone ``pl.function`` build.
        specialize_global_callees: The jit envelope's per-call variant lane
            — a module-global callee that is an eagerly traced jit function
            re-traces with the call-site argument types instead of emitting
            a verbatim call to the decoration-time function. Never the
            standalone ``pl.function`` build (the parser frontend's
            bare-name call is verbatim; the conformance oracle pins it).

    Returns:
        The built ``ir.Function`` — structurally the parser frontend's output
        for the same kernel.

    Raises:
        TypeError: On a parameter without a ``pl.*`` annotation (the AST
            frontend's discipline: every param is a typed IR param).
    """
    external_source = _prepare_external_source(kernel, function_type, external_source)
    annotations, return_type = _declared_annotations(kernel, localns=localns)
    return _trace_with_annotations(
        kernel,
        annotations,
        return_type,
        function_type=function_type,
        level=level,
        role=role,
        attrs=_fold_external_source(fold_auto_scope(attrs, auto_scope), external_source),
        strict_ssa=strict_ssa,
        localns=localns,
        materialize_self=materialize_self,
        specialize_global_callees=specialize_global_callees,
    )


def _declared_annotations(
    kernel: types.FunctionType,
    *,
    localns: dict[str, object] | None = None,
) -> tuple[dict[str, object], object]:
    """The kernel's declared signature from the runtime annotations: the
    per-param facades (validated — every param is pl.*-typed) and the
    ``->`` return facade (or None). Shared by ``function_trace`` and the
    jit envelope's eager inspection."""
    hints = unmangle_hints(kernel, typing.get_type_hints(kernel, localns=localns))
    signature = inspect.signature(kernel)
    param_names = [
        unmangle_param_name(kernel, p.name)
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    annotations: dict[str, object] = {}
    for name in param_names:
        annotation = hints.get(name)
        if annotation is None or not is_staged_annotation(annotation):
            raise TypeError(
                f"kernel {kernel.__name__!r}: parameter {name!r} needs a pl.Tensor/"
                f"pl.Tile/pl.Scalar/pl.Array/pl.Ptr annotation for the tracing frontend, "
                f"got {annotation!r}"
            )
        annotations[name] = annotation
    return annotations, hints.get("return")


def _trace_with_annotations(  # noqa: PLR0913 - one kwarg per trace fact
    kernel: types.FunctionType,
    annotations: dict[str, object],
    return_type: object,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    strict_ssa: bool = False,
    localns: dict[str, object] | None = None,
    materialize_self: bool = False,
    specialize_global_callees: bool = False,
) -> ir.Function:
    """Trace ``kernel`` with a given per-param annotation map (facade params
    create IR params; host values pass through) and a ``->`` return facade.
    The shared tail of ``function_trace`` and the jit specializer's per-call
    bake. ``localns`` is the def-site name capture: PEP 563 string
    annotations reference names that never enter the kernel's bytecode,
    and the directions read resolves against them. ``materialize_self``
    injects the jit ``self`` into the instrumented globals (see
    ``function_trace``).
    ``specialize_global_callees`` selects the jit callee lane for
    module-global AND closure-captured callees (see ``function_trace``):
    specialization is a property of the callee, never of where it is bound.

    Both callee-adaptation surfaces run here so every root-trace producer
    (standalone ``@pl.function``, the jit eager and specialize lanes) sees
    callable callees: closure-captured standalone functions become
    ``Marker``s, and module-global ones become callable in the execution
    copy of the globals. Pre-adapted kernels (the jit dispatch lane) pass
    through both adaptations unchanged."""
    kernel, _captured = adapt_captured_callees(kernel)
    kernel = adapt_global_callees(kernel)
    sema = PtoSema(
        name=kernel.__name__,
        return_type=annotation_to_type(return_type) if is_staged(return_type) else None,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        strict_ssa=strict_ssa,
        source_span=source_span_for_function(kernel),
    )
    # the decorator-adapter entry: create the param Vars (with the
    # declared Out/InOut directions) and inject the real
    # facades the interpreter's enter_frame stores unchanged. The kernel's
    # own module scope feeds the dynamic-symbol shadowing rule (a module-level
    # ``pl.dynamic`` bound by a ``pl.Scalar[pl.INDEX]`` param).
    arg_facades = sema.bind_parameters(
        annotations,
        directions=param_directions(kernel, localns=localns),
        scope=kernel.__globals__,
    )

    inst = instrument(kernel, sema)
    if materialize_self and "self" not in inst.fn.__globals__:
        # The jit contract: ``self.<sibling>`` in a jit body is the
        # module's traced-kernel registry (upstream's jit materializes
        # it; the corpus marks it with an ``F821`` noqa). The instrumented
        # globals are a private copy whose standalone callees are already
        # Marker-wrapped above, so the proxy resolves the sibling to the
        # detached-emission surface without touching the module.
        inst.fn.__globals__["self"] = _JitSelfProxy(inst.fn.__globals__, kernel.__name__)
    # The execution layer installs the current Frame for this one function
    # execution and resets it unconditionally.
    run_instrumented(inst, sema, arg_facades)
    fn = sema.result()
    return _apply_return_contract(fn, return_type)


def _apply_return_contract(fn: ir.Function, return_type: object) -> ir.Function:
    """Pin the declared ``->`` onto a completed trace, reconciling
    under-specified slots.

    A fully-concrete contract is checked against the traced values before it
    is pinned: spelling a declared signature onto a body that produced
    different values reports the declaration, not the program. A contract
    containing **bare facade classes** (``-> pl.Tensor``; ``-> tuple[pl.Tensor,
    pl.Scalar[pl.TASK_ID]]`` — the corpus's submit-style multi-return) is
    under-specified: each bare slot's shape and dtype come from the traced
    return and its kind is class-checked against the declared facade (typing
    from values — a bare class states no shape, so it cannot lie, it
    under-specifies). Concrete slots in a mixed contract are checked slot-wise.
    An extern stub has no body to derive from, so a bare slot there is loud —
    extern contracts must be concrete."""
    slots = annotation_to_return_slots(return_type)
    # The paths are verified before the slots are consulted: a body whose
    # alternative paths disagree is refused with or without a declaration,
    # because an empty signature would hide the disagreement from every
    # later caller.
    actual = check_return_paths(fn)
    if slots is None:
        return fn
    if not any(isinstance(slot, type) for slot in slots):
        return pin_declared_returns(fn, typing.cast("list[ir.Type]", slots))
    if EXTERNAL_SOURCE_ATTR in (fn.attrs or {}):
        raise TypeError(
            f"kernel {fn.name!r}: a bare facade class in the return contract "
            f"{return_type!r} cannot be derived — an extern stub has no body; "
            "annotate the return fully (pl.Tensor[[shape], dtype])"
        )
    traced_values = actual or list(fn.return_types)
    if len(traced_values) != len(slots):
        raise TypeError(
            f"kernel {fn.name!r}: the declared `->` {return_type!r} has "
            f"{len(slots)} slot(s) but the traced body returns {len(traced_values)} value(s)"
        )
    composed: list[ir.Type] = []
    for i, (slot, traced) in enumerate(zip(slots, traced_values)):
        if isinstance(slot, type):
            facade = facade_class_for_type(traced)
            if facade is None or not issubclass(facade, slot):
                raise TypeError(
                    f"kernel {fn.name!r}: return slot {i} is declared "
                    f"{slot.__name__} (a bare facade class) but the traced body "
                    f"returns {traced} — the declared kind does not match the "
                    "traced value"
                )
            composed.append(traced)
        else:
            if not _declared_return_matches(slot, traced):
                raise TypeError(
                    f"kernel {fn.name!r}: return slot {i} is declared {slot} "
                    f"but the traced body returns {traced} (a contract with "
                    "under-specified slots is checked slot-wise; a mismatch is "
                    "a loud 'the `->` was a lie')"
                )
            composed.append(slot)
    return rebuild_with_return_types(fn, composed)


def _declared_return_matches(declared: ir.Type, actual: ir.Type) -> bool:
    """Does the traced return match the declared ``->`` annotation?

    For ``Tile`` (and ``Tensor``) the annotation can express only shape +
    dtype: memory space and ``TileView`` (e.g. ``valid_shape=[M, N]`` from
    ``pl.load(..., valid_shapes=...)``) are trace-deduced details the
    annotation syntax has no word for. Comparing those extras would flag
    every Vec-tile kernel as "the declared ``->`` was a lie". Everything else
    compares structurally (auto-mapping absorbs per-trace Var identity).
    """
    if isinstance(declared, ir.ShapedType) and isinstance(actual, ir.ShapedType):
        # A distributed tensor IS-A tensor: a declared ``-> pl.Tensor`` accepts
        # a traced ``DistributedTensorType`` return (the annotation syntax has
        # no distributed flavour — ``pld.DistributedTensor`` is the only way to
        # declare it, and kernels that return ``pl.store(...)`` into a
        # distributed dest legitimately trace the subtype).
        if not isinstance(actual, type(declared)):
            return False
        if len(declared.shape) != len(actual.shape):
            return False
        if declared.dtype != actual.dtype:
            return False
        return all(
            ir.structural_equal(d, a, enable_auto_mapping=True) for d, a in zip(declared.shape, actual.shape)
        )
    return ir.structural_equal(declared, actual, enable_auto_mapping=True)


class _JitSelfProxy:
    """The jit lane's ``self``: the module materialized as an implicit
    program. Upstream's ``@pl.jit`` lets a jit body reference sibling
    kernels as ``self.<sibling>`` (the corpus contract behind the
    ``# noqa: F821``); ``__getattr__`` resolves the name through the
    execution copy of the module globals — where
    ``adapt_global_callees`` has already made standalone
    ``ir.Function`` callees callable — so both ``self.sibling(x)`` and
    ``pl.spmd_submit(self.sibling, ...)`` see the detached-emission
    ``Marker``. An unknown name is loud (the parser frontend's
    undefined-program-function discipline)."""

    __slots__ = ("_globals", "_owner")

    def __init__(self, globals_: dict[str, object], owner: str) -> None:
        self._globals = globals_
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        try:
            return self._globals[name]
        except KeyError:
            raise AttributeError(
                f"jit kernel {self._owner!r}: self.{name} does not resolve — no "
                f"module global {name!r} (the jit frontend materializes "
                f"self.<sibling> as a reference to the module's traced kernels)"
            ) from None
