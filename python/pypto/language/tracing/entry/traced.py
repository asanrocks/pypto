# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The entrance foundation's prepared body — :class:`TracedCallable`.

A ``TracedCallable`` is reusable prepared instrumentation paired optionally
with a :class:`~pypto.language.tracing.entry.declaration.Decl`. Preparation — the
source rewrite and compile — happens once and caches only code-shaped facts:
the compiled code object, the aligned closure, the generated event prefix.
It never caches execution, host observations, arguments, interpreter state,
a Sema, or emitted IR.

``invoke(args, kwargs)`` performs one complete root trace atomically: a fresh
one-shot Sema from the declaration, fresh parameter Vars and facades, the
flat invocation recipe resolved by ``Param`` identity, execution with a
fresh interpreter, finalization only on success, and the escaping exception
propagated unchanged. It is not dispatch: its arguments are a tracing
invocation recipe made of owned ``Param`` placeholders and host values, not
runtime tensors for an already compiled program.
"""

from __future__ import annotations

import ast
import textwrap
import types
from collections.abc import Sequence
from typing import Protocol

from pypto.language.tracing.collector.driver import InstrumentedFunction, instrument, instrument_source
from pypto.language.tracing.execution import (
    adapt_captured_callees,
    adapt_global_callees,
    pin_declared_returns,
    run_instrumented,
)
from pypto.language.tracing.semantics import PtoSema
from pypto.pypto_core import ir

from .declaration import Decl, Param, fold_auto_scope

__all__ = [
    "TracedCallable",
]


class _Preparation(Protocol):
    """Reusable preparation: compiled code plus the live definition environment."""

    @property
    def prefix(self) -> str: ...

    def live_function(self) -> types.FunctionType: ...


class _CallablePreparation:
    """Preparation for a callable-backed body.

    Holds the kernel itself (the live definition environment: globals,
    closure cells, defaults, and keyword defaults are read from it at every
    materialization), plus the compiled instrumented code and the closure
    aligned to that code's free variables.
    """

    __slots__ = ("_code", "_closure", "_kernel", "_name", "_prefix")

    def __init__(self, kernel: types.FunctionType) -> None:
        prepared = instrument(kernel)
        self._kernel = kernel
        self._code = prepared.fn.__code__
        self._closure = prepared.fn.__closure__
        self._name = prepared.fn.__name__
        self._prefix = prepared.prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def live_function(self) -> types.FunctionType:
        """Bind the compiled code to the kernel's current definition environment."""
        fn = types.FunctionType(
            self._code,
            self._kernel.__globals__,
            name=self._name,
            argdefs=self._kernel.__defaults__,
            closure=self._closure,
        )
        fn.__kwdefaults__ = self._kernel.__kwdefaults__
        fn.__qualname__ = self._kernel.__qualname__
        fn.__module__ = self._kernel.__module__
        if self._kernel.__annotations__:
            fn.__annotations__ = dict(self._kernel.__annotations__)
        return fn


class _SourcePreparation:
    """Preparation for a source-backed body.

    Text has no globals, closures, or identity, so the environment is
    explicit and kept by reference: mutating it between invocations is
    observed by the next one, mirroring the callable-backed live globals. A
    definition that declares a parameter default is rejected at preparation
    — defaults are definition-time values, and the compiled code alone
    cannot carry them.
    """

    __slots__ = ("_code", "_environment", "_name", "_prefix")

    def __init__(self, source: str, name: str, filename: str, environment: dict[str, object]) -> None:
        _reject_definition_defaults(source, name)
        self._code, self._prefix = instrument_source(source, name, filename)
        self._environment: dict[str, object] = environment
        self._name = name

    @property
    def prefix(self) -> str:
        return self._prefix

    def live_function(self) -> types.FunctionType:
        return types.FunctionType(self._code, self._environment, name=self._name)


class TracedCallable:
    """Reusable prepared instrumentation paired optionally with a declaration.

    ``from_callable`` and ``from_string`` perform the expensive reusable
    preparation; ``invoke`` always creates fresh execution and construction
    state. ``decl`` may be omitted for a body prepared only for expansion —
    calling root ``invoke`` without a declaration is a loud error.
    """

    def __init__(self, preparation: _Preparation, decl: Decl | None, name: str) -> None:
        self._preparation = preparation
        self._decl = decl
        self.__name__ = name

    # —— the preparation entries ——

    @classmethod
    def from_callable(cls, fn: types.FunctionType, *, decl: Decl | None = None) -> TracedCallable:
        """Prepare a Python function (a ``def`` body).

        A lambda cannot be prepared: the source rewriter locates the body by
        its function name, and a lambda has none — prepare a named ``def``
        instead. The instance retains the kernel's live definition
        environment: current globals, closure cells, defaults, and keyword
        defaults are read at every invocation, never cached by preparation.
        """
        if not isinstance(fn, types.FunctionType):
            raise TypeError(f"TracedCallable.from_callable expects a Python function, got {fn!r}")
        if decl is not None and not isinstance(decl, Decl):
            raise TypeError(f"decl must be a Decl or None, got {decl!r}")
        return cls(_CallablePreparation(fn), decl, fn.__name__)

    @classmethod
    def from_string(
        cls,
        source: str,
        *,
        name: str,
        filename: str = "generated.py",
        environment: dict[str, object],
        decl: Decl | None = None,
    ) -> TracedCallable:
        """Prepare a named function definition from source text.

        ``name`` selects the definition inside ``source``; ``environment`` is
        the body's module globals — required, because text carries no
        closures, defaults, or identity of its own.
        """
        if not isinstance(source, str):
            raise TypeError(f"source must be a str, got {source!r}")
        if not isinstance(name, str) or not name.isidentifier():
            raise TypeError(f"name must be an identifier naming a function in the source, got {name!r}")
        if not isinstance(environment, dict):
            raise TypeError(f"environment must be a dict (the body's module globals), got {environment!r}")
        if decl is not None and not isinstance(decl, Decl):
            raise TypeError(f"decl must be a Decl or None, got {decl!r}")
        return cls(_SourcePreparation(source, name, filename, environment), decl, name)

    # —— the surfaces ——

    @property
    def decl(self) -> Decl | None:
        """The paired declaration, or ``None`` for a splice-only body."""
        return self._decl

    def invoke(
        self,
        *,
        args: Sequence[object] | None = None,
        kwargs: dict[str, object] | None = None,
    ) -> ir.Function:
        """Perform one complete root trace and return the built function.

        ``args``/``kwargs`` form the flat invocation recipe: direct
        ``Param`` values owned by the declaration are replaced by the fresh
        facades of this invocation (by identity); ordinary host values pass
        through unchanged; containers are not inspected. ``args=None`` means
        all declared parameters in declaration order — a kwargs-only recipe
        passes ``args=()`` explicitly, because ``args=None`` supplies every
        parameter positionally and normal Python binding then rejects the
        duplicated names loudly.

        Callees the body reaches through closure cells or module globals are
        made callable for this one execution (the entrance's verbatim
        adaptation) — a standalone traced body can call its siblings without
        mutating the module it lives in.
        """
        decl = self._decl
        if decl is None:
            raise TypeError(
                f"cannot invoke {self.__name__!r} without a declaration — a body "
                "prepared without one is splice-only; root invocation needs "
                "the declaration's parameter and return contract"
            )
        sema = PtoSema(
            name=decl.symbol_name,
            return_type=_sema_return_type(decl),
            function_type=decl.function_type,
            level=decl.level,
            role=decl.role,
            attrs=fold_auto_scope(decl.attrs, decl.auto_scope),
            strict_ssa=decl.strict_ssa,
            requires_runtime_binding=decl.requires_runtime_binding,
            source_span=decl.span,
        )
        facades = sema.bind_typed_parameters([(p.name, p.type, p.direction) for p in decl.params])
        call_args, call_kwargs = _resolve_recipe(decl, facades, args, kwargs)
        runner = InstrumentedFunction(
            _with_callable_callees(self._preparation.live_function()), self._preparation.prefix, sema
        )
        run_instrumented(runner, sema, call_args, call_kwargs)
        return pin_declared_returns(sema.result(), decl.explicit_returns)


def _with_callable_callees(live: types.FunctionType) -> types.FunctionType:
    """Make the live body's closure/global callees callable for this execution.

    The entrance's callee contract: a body that reaches a sibling through a
    closure cell or a module global sees a callable callee — a plain
    ``ir.Function`` becomes a verbatim ``Marker`` over an execution copy of
    the environment, never a mutation of the module it lives in. The shared
    adaptation lives in the execution layer beneath this foundation, so the
    import is module-scope: the foundation no longer reaches up into the
    decorator tier for its own machinery.
    """
    live, _captured = adapt_captured_callees(live)
    return adapt_global_callees(live)


def _reject_definition_defaults(source: str, name: str) -> None:
    """Reject a source definition that declares parameter defaults.

    Defaults are definition-time values: preparation keeps only the compiled
    code, so ``def f(x, scale=2.0)`` from text would silently lose ``scale``'s
    default and fail at invocation with a missing-argument error far from
    the cause. The preparation-time spelling of "text carries no defaults":
    loud, with the remedy (pass the value through the environment) named.
    """
    tree = ast.parse(textwrap.dedent(source))
    node = next(
        (
            candidate
            for candidate in ast.walk(tree)
            if isinstance(candidate, (ast.FunctionDef, ast.AsyncFunctionDef)) and candidate.name == name
        ),
        None,
    )
    if node is None:
        return  # the missing name is instrument_source's loud SyntaxError
    if node.args.defaults or any(default is not None for default in node.args.kw_defaults):
        raise TypeError(
            f"from_string body {name!r} declares a parameter default — text carries no "
            "definition-time defaults; pass the value through the environment instead"
        )


def _sema_return_type(decl: Decl) -> ir.Type | None:
    """The Sema's single declared return type from the flat return contract.

    One explicit return is that type (a structural ``TupleType`` stays one
    structural return); a flat multi-return contract leaves the Sema without
    a structural type — the flat element list is pinned onto the completed
    function instead, matching the flat multi-return signature convention.
    """
    returns = decl.explicit_returns
    if not returns:
        return None
    if len(returns) == 1:
        return returns[0]
    return None


def _resolve_recipe(
    decl: Decl,
    facades: Sequence[object],
    args: Sequence[object] | None,
    kwargs: dict[str, object] | None,
) -> tuple[list[object], dict[str, object]]:
    """Resolve the flat invocation recipe against the fresh facades."""
    facade_by_identity = {id(param): facade for param, facade in zip(decl.params, facades)}

    def resolve(value: object, where: str) -> object:
        if not isinstance(value, Param):
            return value
        facade = facade_by_identity.get(id(value))
        if facade is None:
            raise TypeError(
                f"invocation of {decl.name!r}: the Param placeholder {value.name!r} is not "
                f"owned by this declaration (owned: {[p.name for p in decl.params]}) — {where}"
            )
        return facade

    if args is None:
        # All declared parameters, in declaration order.
        call_args: list[object] = list(facades)
    else:
        call_args = [resolve(value, f"args position {index}") for index, value in enumerate(args)]
    call_kwargs = {key: resolve(value, f"kwargs key {key!r}") for key, value in (kwargs or {}).items()}
    return call_args, call_kwargs
