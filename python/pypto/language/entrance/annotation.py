# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The standard annotation deriver — a framework library over the foundation.

``derive`` produces the declaration and flat invocation recipe for a kernel
from its annotations: ``typing.get_type_hints`` resolves the facade
annotations, ``Out``/``InOut`` markers carry the directions, the ``->``
annotation becomes the explicit return contract (or :data:`Infer`), and the
positional parameters become the declaration's ``Param`` values in
declaration order. The returned ``args``/``kwargs`` contain those ``Param``
placeholders, ready for :meth:`TracedCallable.invoke
<pypto.language.entrance.TracedCallable.invoke>`.

PEP 563 string annotations may reference host names from the enclosing
scope, which ``typing.get_type_hints`` cannot see: pass ``localns``
explicitly, or let ``derive`` capture the caller's visible names. An
entrance that derives at a different time than it captures (the decoration
boundary) captures once with :func:`capture_caller_names` and passes the
result as ``localns``.
"""

from __future__ import annotations

import inspect
import sys
import types
import typing
from dataclasses import dataclass
from typing import Any

from pypto.language.protocol import is_staged
from pypto.language.typing import annotation_to_return_types, annotation_to_type
from pypto.language.typing.arg_direction import freevar_localns, param_directions
from pypto.pypto_core import ir

from .declaration import Decl, Infer, Param

__all__ = [
    "InvocationPlan",
    "capture_caller_names",
    "derive",
    "source_span_for_function",
]


@dataclass(frozen=True)
class InvocationPlan:
    """A derived declaration plus its flat invocation recipe.

    ``args`` and ``kwargs`` contain ``Param`` placeholders and host values;
    substituting them by identity is :meth:`TracedCallable.invoke
    <pypto.language.entrance.TracedCallable.invoke>`'s job.
    """

    decl: Decl
    args: tuple[object, ...]
    kwargs: dict[str, object]


def source_span_for_function(kernel: types.FunctionType) -> ir.Span:
    """Return the function definition's coarse source span when available.

    ``co_firstlineno`` points at the ``def`` line rather than the decorator
    line returned by ``inspect.getsourcelines``. Source metadata is optional
    for generated or interactive functions, so those functions retain the
    unknown-span behavior.
    """
    source_file = inspect.getsourcefile(kernel)
    line = kernel.__code__.co_firstlineno
    if source_file is None or line <= 0:
        return ir.Span.unknown()
    # Column 1: a coarse span names the definition line, and consumers that
    # thread it onward (parameter spans, diagnostics) need a valid column.
    return ir.Span(source_file, line, 1)


def capture_caller_names() -> dict[str, object]:
    """The caller's visible names for annotation resolution.

    The decoration-boundary capture: ``{**frame.f_globals,
    **frame.f_locals}`` of the caller, walking the full frame chain
    (module → caller) so intermediate shim layers between the entrance and
    the user's code don't hide the user frame; innermost frames shadow
    outer ones, matching Python name resolution. PEP 563 string annotations
    may reference host names from the enclosing scope (e.g. ``rows`` inside
    a helper that builds a kernel in a closure), and
    ``typing.get_type_hints`` cannot see closure cells.
    """
    names: dict[str, object] = {}
    frames: list[types.FrameType] = []
    frame = sys._getframe(1)
    while frame is not None:
        frames.append(frame)
        frame = frame.f_back
    for fr in reversed(frames):
        names.update(fr.f_globals)
        names.update(fr.f_locals)
    return names


def derive(
    fn: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    auto_scope: bool = True,
    strict_ssa: bool = False,
    localns: dict[str, object] | None = None,
    span: ir.Span | None = None,
) -> InvocationPlan:
    """Derive the declaration and flat invocation recipe from a kernel's annotations.

    Every positional parameter needs a ``pl.*`` facade annotation (the AST
    frontend's discipline: every IR param is typed); non-positional
    parameters are not IR parameters. The ``->`` annotation becomes the
    explicit return contract; any other return annotation (or none) means
    :data:`Infer`.
    """
    if not isinstance(fn, types.FunctionType):
        raise TypeError(f"derive expects a Python function, got {fn!r}")
    # PEP 563 string annotations may reference names from the enclosing
    # scope: the kernel's own closure cells resolve them even when the
    # defining frame is gone, and the caller-visible names (captured here
    # when no explicit ``localns`` was supplied) complete the resolution.
    merged = freevar_localns(fn) or {}
    merged.update(localns if localns is not None else capture_caller_names())
    hints = typing.get_type_hints(fn, localns=merged or None)
    directions = param_directions(fn, localns=merged)
    signature = inspect.signature(fn)
    params: list[Param] = []
    for parameter in signature.parameters.values():
        if parameter.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue
        annotation = hints.get(parameter.name)
        if annotation is None or not is_staged(annotation):
            raise TypeError(
                f"kernel {fn.__name__!r}: parameter {parameter.name!r} needs a "
                f"pl.Tensor/pl.Tile/pl.Scalar/pl.Array/pl.Ptr annotation for the "
                f"tracing frontend, got {annotation!r}"
            )
        params.append(
            Param(
                parameter.name,
                annotation_to_type(annotation),
                directions.get(parameter.name, ir.ParamDirection.In),
            )
        )
    declared_returns = annotation_to_return_types(hints.get("return"))
    decl = Decl(
        name=fn.__name__,
        params=tuple(params),
        returns=tuple(declared_returns) if declared_returns else Infer,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        auto_scope=auto_scope,
        strict_ssa=strict_ssa,
        span=span if span is not None else source_span_for_function(fn),
    )
    return InvocationPlan(decl, tuple(params), {})
