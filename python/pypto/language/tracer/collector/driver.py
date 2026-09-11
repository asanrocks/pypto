# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Preparation and invocation of instrumented kernels — the trace driver.

The trace driver rewrites a kernel into a synthesized function with the original
signature. Keeping a real function preserves native ``return``, ``break``,
``continue``, nested definitions, lambdas, comprehensions, and closure lookup.

At call time, a fresh interpreter is injected into a copy of the function globals.
Each invocation acquires a fresh Sema from a :data:`SemaFactory` when one is
supplied (a directly supplied mutable Sema is a one-shot low-level input).
The frame-entry prologue runs after the source docstring, and the interpreter is
closed **only on the success path**: an escaping exception skips every
normal finalization (interpreter close, Frame finish, staged construct close,
builder ``end_*``) — the caller drops that one-shot Sema and propagates the
original exception unchanged.
"""

from __future__ import annotations
import __future__

import ast
import functools
import inspect
import textwrap
import types
from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from .interface import NullSema, Sema, SemaFactory
from .interpreter import Interpreter
from .rewriter import RewriteError, emit_kernel

V = TypeVar("V")

# The default Sema source, typed at the Any domain so it can default a
# generic parameter (``NullSema`` is stateless and safely shared).
_NULL_SEMA: Sema[Any] = NullSema()


def _acquire_sema(source: Sema[V] | SemaFactory[V]) -> Sema[V]:
    """Resolve a Sema source: a factory yields a fresh Sema per call.

    A factory is a plain callable (or class) without an *instance* Sema
    surface; a Sema instance (e.g. the stateless ``NullSema`` or a one-shot
    language Sema driven directly) is used as-is.
    """
    if callable(source) and (isinstance(source, type) or not hasattr(source, "begin_frame")):
        return cast(SemaFactory[V], source)()
    return cast(Sema[V], source)


class InstrumentedFunction(Generic[V]):
    """Callable wrapper around a synthesized instrumented kernel.

    Each call acquires a Sema (fresh per call for a factory), creates a fresh
    :class:`Interpreter` over it, and injects the interpreter into per-call
    globals under the generated event prefix. The wrapper preserves the
    kernel's callable metadata and signature.
    """

    def __init__(
        self,
        fn: types.FunctionType,
        prefix: str,
        sema: Sema[V] | SemaFactory[V] = _NULL_SEMA,
    ):
        self.fn = fn
        self.prefix = prefix
        self._sema_source = sema
        # surface the kernel's metadata (class attributes would shadow
        # __getattr__ delegation)
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        if fn.__annotations__:
            self.__annotations__ = fn.__annotations__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        sema = _acquire_sema(self._sema_source)
        rt = Interpreter(sema, source_file=self.fn.__code__.co_filename)
        # Copy globals per call so injected Interpreter instances cannot race.
        globals_ = dict(self.fn.__globals__)
        globals_[self.prefix] = rt
        fn = types.FunctionType(
            self.fn.__code__,
            globals_,
            name=self.fn.__name__,
            argdefs=self.fn.__defaults__,
            closure=self.fn.__closure__,
        )
        fn.__kwdefaults__ = self.fn.__kwdefaults__
        try:
            value = fn(*args, **kwargs)
        except BaseException:
            # Finalization is success-path-only (the module docstring owns
            # the contract). Python has already unwound every entered
            # `with` through its own cm protocol; the one-shot Sema dies
            # with this call — but its span pin must not outlive it.
            sema.abort()
            raise
        else:
            rt.close()
            return value

    def __getattr__(self, name: str) -> Any:
        # surface the kernel's metadata (__doc__, __name__, __annotations__)
        return getattr(self.fn, name)


def _function_code(module_code: types.CodeType, name: str) -> types.CodeType:
    """Find a named function code object inside compiled module constants.

    Accepted input: a compiled module containing ``def kernel(...): ...``;
    output: the nested ``kernel`` code object.
    """
    stack = [module_code]
    while stack:
        code = stack.pop()
        for const in code.co_consts:
            if isinstance(const, types.CodeType):
                if const.co_name == name:
                    return const
                stack.append(const)
    raise RewriteError(f"compiled module has no function {name!r}")


# The rewritten-code cache: instrumentation is pure preparation over
# (source, name, file, line, freevars, flags), so a re-trace of the same
# kernel — the jit variant lane re-traces callees per call — reuses the
# compiled code object instead of re-running the parse/rewrite/compile
# pipeline (the entrance design's "preparation may cache rewritten code").
_INSTRUMENT_CACHE: dict[tuple[str, str, str, int, tuple[str, ...], bool], tuple[types.CodeType, str]] = {}


@functools.lru_cache(maxsize=1024)
def _prepared_body(code: types.CodeType, name: str) -> tuple[types.CodeType, str]:
    """The rewritten code and event prefix for one kernel body, memoized.

    Everything here is a pure function of the kernel's source: the line
    fetch (``inspect.getsourcelines`` re-tokenizes the source block on
    every call — the dominant per-call cost once ``_compile_instrumented``
    hits its cache), the dedent, and the (itself source-keyed) rewrite
    compilation. The per-call parts of ``instrument`` — the globals
    snapshot, the closure alignment, and the ``FunctionType`` build — stay
    out so each re-trace observes the module state at its own call time.

    Keyed by the code object: a module re-exec mints fresh code objects,
    so entries never cross an exec boundary, and the cache holds no
    reference to the kernel, its globals, or its module.
    """
    source_lines, source_start = inspect.getsourcelines(code)
    source = textwrap.dedent("".join(source_lines))
    source_file = inspect.getsourcefile(code) or f"<collector:{getattr(code, 'co_qualname', code.co_name)}>"
    return _compile_instrumented(
        source,
        name,
        source_file,
        source_start,
        freevars=code.co_freevars,
        future_annotations=bool(code.co_flags & __future__.annotations.compiler_flag),
    )


def _compile_instrumented(
    source: str,
    name: str,
    source_file: str,
    source_start: int,
    *,
    freevars: tuple[str, ...] = (),
    future_annotations: bool = False,
) -> tuple[types.CodeType, str]:
    """Rewrite and compile one function definition from dedented source text.

    ``source_start`` anchors the emitted spans: the rewritten definition is
    shifted back to its original file position. Returns the function's code
    object and the generated event prefix.
    """
    cache_key = (source, name, source_file, source_start, freevars, future_annotations)
    cached = _INSTRUMENT_CACHE.get(cache_key)
    if cached is not None:
        return cached
    emitted, prefix = emit_kernel(source, name, freevars, future_annotations, source_start - 1)
    emitted_tree = ast.parse(emitted)
    original_tree = ast.parse(source)
    original_fn = next(
        node for node in ast.walk(original_tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    emitted_fn = next(
        node for node in ast.walk(emitted_tree) if isinstance(node, ast.FunctionDef) and node.name == name
    )
    ast.increment_lineno(emitted_tree, source_start + original_fn.lineno - emitted_fn.lineno - 1)
    # Compile with the kernel's annotation flag only. Do not inherit this
    # module's future flags, which would change annotation evaluation.
    module_code = compile(
        emitted_tree,
        source_file,
        "exec",
        flags=__future__.annotations.compiler_flag if future_annotations else 0,
        dont_inherit=True,
    )
    result = (_function_code(module_code, name), prefix)
    _INSTRUMENT_CACHE[cache_key] = result
    return result


def instrument(
    kernel: Callable[..., Any], sema: Sema[V] | SemaFactory[V] = _NULL_SEMA
) -> InstrumentedFunction[V]:
    """Compile a kernel through the source rewriter and bind its Sema source.

    Instrumentation is performed when this function is called. ``sema`` is
    either a Sema instance (reused across invocations — one-shot for a
    mutable language Sema) or a zero-argument factory yielding a fresh
    Sema per invocation; the default ``NullSema`` provides host
    execution.
    """
    fn_code, prefix = _prepared_body(kernel.__code__, kernel.__name__)

    # The rewritten body can drop annotation-only free variables. Align the
    # remaining closure cells by name rather than tuple position, and reject
    # any newly referenced name that was not present in the original.
    orig_freevars = kernel.__code__.co_freevars
    if not set(fn_code.co_freevars) <= set(orig_freevars):
        raise RewriteError(
            f"instrumented {kernel.__name__!r} references free variables "
            f"({fn_code.co_freevars}) not present in the original ({orig_freevars})"
        )

    globals_ = dict(kernel.__globals__)
    closure = None
    if fn_code.co_freevars:
        # ``co_freevars`` implies a closure, but the type checker cannot infer
        # that relationship from FunctionType metadata.
        cell_by_name = dict(zip(orig_freevars, kernel.__closure__ or ()))
        closure = tuple(cell_by_name[name] for name in fn_code.co_freevars)
    fn = types.FunctionType(
        fn_code,
        globals_,
        name=kernel.__name__,
        argdefs=kernel.__defaults__,
        closure=closure,
    )
    fn.__qualname__ = kernel.__qualname__
    fn.__module__ = kernel.__module__
    fn.__kwdefaults__ = kernel.__kwdefaults__
    fn.__annotations__ = kernel.__annotations__  # FunctionType does not copy annotations
    return InstrumentedFunction(fn, prefix, sema)


def instrument_source(
    source: str,
    name: str,
    filename: str = "generated.py",
) -> tuple[types.CodeType, str]:
    """Rewrite and compile a named function definition from source text.

    The string-backed preparation entry: text has no closure, defaults, or
    identity, so the caller owns the execution environment when it
    materializes the compiled code. The definition is compiled without the
    ``annotations`` future flag — generated bodies carry value annotations
    or none, never PEP 563 strings.
    """
    return _compile_instrumented(textwrap.dedent(source), name, filename, 1)


__all__ = ["InstrumentedFunction", "instrument", "instrument_source"]
