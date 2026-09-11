# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Per-call-site direction markers for PyPTO Language DSL.

These markers attach an explicit ``pypto.ir.ArgDirection`` to a call
argument and are stored on ``ir.Call.attrs['arg_directions']`` (also
accessible via the ``ir.Call.arg_directions`` shortcut property).

The printer surfaces the direction vector via a trailing ``attrs=`` keyword
on cross-function calls so that ``python_print`` → ``parse`` round-trips
preserve the metadata::

    import pypto.language as pl

    result = self.kernel_process(
        a, is_first, result,
        attrs={"arg_directions": [pl.adir.input, pl.adir.scalar, pl.adir.inout]},
    )

Each ``pl.adir.<name>`` symbol is a direct alias of the matching
``ir.ArgDirection`` enum value — they are not callable. Per-argument
wrapper forms such as ``pl.adir.input(x)`` are intentionally not supported.

The markers correspond 1:1 with the runtime task-submission methods on
``PTOParam`` (``add_input`` / ``add_output`` / ``add_inout`` /
``add_no_dep`` / ``add_scalar``) and with ``ir.ArgDirection`` enum
values:

==================== ================================
Marker               ``ArgDirection``
==================== ================================
``pl.adir.input``            ``Input``
``pl.adir.output``           ``Output``
``pl.adir.output_existing``  ``OutputExisting``
``pl.adir.inout``            ``InOut``
``pl.adir.no_dep``           ``NoDep``
``pl.adir.scalar``           ``Scalar``
==================== ================================

User-facing parameter direction is still expressed via ``pl.Out[T]`` /
``pl.InOut[T]`` on the *callee*'s parameter list — those map to
``ir.ParamDirection`` and remain the recommended way to declare a
function's contract. :func:`param_directions` recovers exactly those
declared parameter directions from a kernel's source when the runtime
annotations cannot carry them.
"""

from __future__ import annotations

import types
import typing

from pypto.pypto_core import ir as _ir

ArgDirection = _ir.ArgDirection

# Direct aliases of the enum values. Bare references like ``pl.adir.input`` are
# the only supported surface form; ``pl.adir.input(x)`` would be a TypeError
# at runtime since enum values are not callable.
input = ArgDirection.Input  # noqa: A001 -- shadows builtin within this module only
output = ArgDirection.Output
output_existing = ArgDirection.OutputExisting
inout = ArgDirection.InOut
no_dep = ArgDirection.NoDep
scalar = ArgDirection.Scalar

# Mapping from the marker's leaf attribute name to the IR enum value.
# Used by both the printer (enum -> name) and parser (name -> enum).
NAME_TO_DIRECTION: dict[str, ArgDirection] = {
    "input": ArgDirection.Input,
    "output": ArgDirection.Output,
    "output_existing": ArgDirection.OutputExisting,
    "inout": ArgDirection.InOut,
    "no_dep": ArgDirection.NoDep,
    "scalar": ArgDirection.Scalar,
}

DIRECTION_TO_NAME: dict[ArgDirection, str] = {v: k for k, v in NAME_TO_DIRECTION.items()}


def _mangle_prefix(func: types.FunctionType) -> str | None:
    """The class-body name-mangling prefix of ``func``'s enclosing class, or None.

    CPython rewrites a ``__name`` identifier (two leading underscores, at most
    one trailing) inside a class body to ``_<classname>__name`` — classname
    stripped of its own leading underscores — in every compiled name
    (``co_varnames``, ``__annotations__``, hence ``inspect.signature``), while
    the source text and its AST keep the original spelling. The parser reads
    the source; tracing reads the compiled object, so the two frontends
    disagree on a dunder-leading *parameter name* unless the signature
    inspection un-mangles it back.
    """
    qualname = getattr(func, "__qualname__", "")
    if "." not in qualname:
        return None
    enclosing = qualname.rsplit(".", 1)[0]
    cls = enclosing.rsplit(".", 1)[-1].lstrip("_")
    return f"_{cls}__" if cls else None


def unmangle_param_name(func: types.FunctionType, name: str) -> str:
    """Undo class-body name mangling on one parameter name.

    No-op outside a class body or when the name carries no matching prefix
    (a user-written ``_Cls__x`` in class ``Cls`` is indistinguishable from a
    mangled ``__x`` — CPython's own ambiguity, resolved the same way: the
    mangled spelling wins).
    """
    prefix = _mangle_prefix(func)
    if prefix is not None and name.startswith(prefix):
        return "__" + name[len(prefix) :]
    return name


def unmangle_hints(func: types.FunctionType, hints: dict[str, object]) -> dict[str, object]:
    """Rebuild an annotation map keyed by un-mangled parameter names."""
    prefix = _mangle_prefix(func)
    if prefix is None:
        return hints
    return {unmangle_param_name(func, k) if k.startswith(prefix) else k: v for k, v in hints.items()}


def freevar_localns(func: types.FunctionType) -> dict[str, object] | None:
    """Resolve annotation names captured by the kernel's defining scope.

    The kernel's own closure cells: the names a PEP 563 string annotation may
    reference from its enclosing scope, visible even when the defining frame
    is gone (derivation after the fact)."""
    if not func.__closure__:
        return None
    localns: dict[str, object] = {}
    for name, cell in zip(func.__code__.co_freevars, func.__closure__):
        try:
            localns[name] = cell.cell_contents
        except ValueError:
            continue
    return localns


def param_directions(
    func: types.FunctionType, localns: dict[str, object] | None = None
) -> dict[str, _ir.ParamDirection]:
    """Recover ``Out``/``InOut`` directions from resolved annotations.

    Direction wrappers attach their marker to the returned facade or dynamic
    facade subclass. ``typing.get_type_hints`` is therefore the source of
    truth, with closure values and caller-provided names available for PEP 563
    annotations. An unresolvable annotation raises consistently with other
    annotation consumers.
    """
    merged = freevar_localns(func) or {}
    if localns:
        merged.update(localns)
    hints = unmangle_hints(func, typing.get_type_hints(func, localns=merged or None))
    result: dict[str, _ir.ParamDirection] = {}
    for name, hint in hints.items():
        direction = getattr(hint, "_param_direction", None)
        if direction == "Out":
            result[name] = _ir.ParamDirection.Out
        elif direction == "InOut":
            result[name] = _ir.ParamDirection.InOut
    return result


__all__ = [
    "ArgDirection",
    "DIRECTION_TO_NAME",
    "NAME_TO_DIRECTION",
    "freevar_localns",
    "inout",
    "input",
    "no_dep",
    "output",
    "output_existing",
    "param_directions",
    "scalar",
]
