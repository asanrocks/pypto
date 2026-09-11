# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Validation and error classification shared by user-facing DSL operation wrappers."""

import functools
import inspect
import sys
import types
from collections.abc import Callable
from typing import Any, TypeVar, cast

from pypto.ir.op._pad_value import normalize_pad_value
from pypto.language.ambient import active_frame
from pypto.language.diagnostics import (
    BUG_CLASS_EXCEPTIONS,
    InvalidOperationError,
    ParserError,
    concise_error_message,
)
from pypto.pypto_core.ir import Call, Expr, PadValue, set_call_attrs


def normalize_fillpad_pad_value(pad_value: object) -> PadValue:
    """Expose pad-value normalization through the DSL diagnostic contract."""
    try:
        return normalize_pad_value(pad_value)
    except (TypeError, ValueError) as exc:
        raise InvalidOperationError(str(exc)) from exc


# ─── the op-call error boundary ──────────────────────────────────────────
#
# Both frontends call the same operation wrappers, so both must classify the
# same failures the same way: an op call that raises ``TypeError`` / ``ValueError``
# — whether from the Python wrapper's own validation or from a C++ ``CHECK``
# surfacing as ``pypto::ValueError`` — is a *bad kernel*, not a PyPTO bug, and
# reaches the user as ``InvalidOperationError``. The AST parser does this at its
# ``_dispatch_op`` choke point; the tracing frontend has no dispatcher (ops are
# ordinary Python calls), so the boundary is the op modules themselves: every
# public, module-defined callable is wrapped at import time.
#
# Classification applies only to calls made *inside a trace* (``active_frame()``
# is set) — the parser likewise classifies only the calls it dispatches in
# kernel bodies, and a direct host call (test fixtures, user metaprogramming)
# keeps the host's native exception. Bug-class exceptions (``InternalError``,
# ``AssertionError``) and already-classified ``ParserError``s pass through
# untouched, and the wrapper is idempotent and alias-preserving
# (``pl.tile.arange is pl.tile.ci`` survives).

_OP_BOUNDARY_ATTR = "__pypto_op_boundary__"

_F = TypeVar("_F", bound=Callable[..., object])


def normalize_call_attr_value(value: Any) -> Any:
    """Normalize one ``attrs={...}`` value for the IR node.

    The open-world rule the printer's ``PrintAttrValue`` / the parser's
    ``_parse_attr_value`` share: a staged facade becomes its IR expression, a
    list/tuple is normalized element-wise, and every other value (host scalar,
    enum, string) passes through — the C++ kwarg converter infers the storage
    type and rejects what it cannot hold, loudly.
    """
    from pypto.language.protocol import is_staged  # noqa: PLC0415 - the protocol/op-module cycle

    if is_staged(value):
        return value.unwrap()
    if isinstance(value, (list, tuple)):
        return [normalize_call_attr_value(v) for v in value]
    return value


def normalize_call_attrs(attrs: Any, construct: str) -> dict[str, Any]:
    """Validate and normalize a call-site ``attrs={...}`` dict.

    String-literal keys only (the round-trip invariant); values are normalized
    by :func:`normalize_call_attr_value`.
    """
    if not isinstance(attrs, dict):
        raise TypeError(f"{construct}: attrs= must be a dict, got {attrs!r}")
    for key in attrs:
        if not isinstance(key, str):
            raise TypeError(f"{construct}: attrs= keys must be string literals, got {key!r}")
    return {key: normalize_call_attr_value(value) for key, value in attrs.items()}


def attach_op_attrs(result: Any, attrs: dict[str, Any]) -> Any:
    """Attach call-site attrs to a freshly built op result.

    The result is a facade (or a raw ``Call``): the underlying Call is rebuilt
    with the attrs merged over any it already carries and re-wrapped in the
    facade matching its type. A host result has nothing to attach to and passes
    through unchanged — the parser's ``_attach_op_attrs`` no-ops the same way.
    """
    from pypto.language.protocol import is_staged  # noqa: PLC0415 - the protocol/op-module cycle
    from pypto.language.typing.type_utils import wrap_expr  # noqa: PLC0415 - lazy: import-light boundary

    expr = result if isinstance(result, Expr) else (result.unwrap() if is_staged(result) else None)
    if not isinstance(expr, Call):
        return result
    merged = {**dict(expr.attrs), **attrs} if expr.attrs else attrs
    call = set_call_attrs(expr, merged)
    if isinstance(result, Expr):
        return call
    return wrap_expr(call, call.type)


def classify_op_errors(fn: _F) -> _F:
    """Re-raise one operation wrapper's user-facing errors as ``InvalidOperationError``.

    Also the ``attrs=`` interception point — the tracing dual of the parser's
    ``_dispatch_op``: a call-site ``attrs={...}`` dict is legal on every op call
    (call-site metadata, never an op parameter), so it is stripped here, the
    wrapped op runs without it, and the dict is attached to the resulting Call.
    Wrappers that genuinely declare an ``attrs`` parameter (none today) keep
    ownership of the kwarg.
    """
    takes_attrs = "attrs" in inspect.signature(fn).parameters

    @functools.wraps(fn)
    def wrapper(*args: object, **kwargs: object) -> object:
        attrs = kwargs.pop("attrs", None) if not takes_attrs else None
        if attrs is not None:
            # Validate before the op runs so a malformed dict is a loud,
            # op-named error rather than a downstream kwarg-converter failure.
            attrs = normalize_call_attrs(attrs, f"{getattr(fn, '__name__', 'op')}")
        if active_frame() is None:
            result = fn(*args, **kwargs)  # a direct host call: native exception
        else:
            try:
                result = fn(*args, **kwargs)
            except (ParserError, *BUG_CLASS_EXCEPTIONS):
                raise
            except (TypeError, ValueError) as exc:
                raise InvalidOperationError(_op_call_message(fn, exc)) from exc
        if attrs is not None:
            result = attach_op_attrs(result, cast(dict[str, Any], attrs))
        return result

    setattr(wrapper, _OP_BOUNDARY_ATTR, True)
    return cast(_F, wrapper)


def _op_call_message(fn: Callable[..., object], exc: Exception) -> str:
    """The classified message: the concise cause, prefixed with the op's name
    when the cause does not already name it (the parser's ``_dispatch_op``
    convention — a bare ``tuple.index(x)`` must still tell the user which op)."""
    msg = concise_error_message(exc, strip_trailing_span=True)
    if fn.__name__ not in msg:
        msg = f"pl.{fn.__name__}: {msg}"
    return msg


def wrap_module_ops(module_name: str) -> None:
    """Wrap a DSL op module's public, module-defined functions with
    :func:`classify_op_errors`.

    Called once at the bottom of each ``language.op`` / ``language.distributed.op``
    module, after every public function is defined. Functions imported from
    elsewhere (typing helpers, sibling modules already wrapped at their own
    definition site) and private names are left alone.
    """
    module = sys.modules[module_name]
    wrappers: dict[int, types.FunctionType] = {}
    for name, value in list(vars(module).items()):
        if name.startswith("_"):
            continue
        if not isinstance(value, types.FunctionType) or value.__module__ != module_name:
            continue
        if getattr(value, _OP_BOUNDARY_ATTR, False):
            continue
        wrapped = wrappers.get(id(value))
        if wrapped is None:
            wrapped = classify_op_errors(value)
            wrappers[id(value)] = wrapped
        setattr(module, name, wrapped)
