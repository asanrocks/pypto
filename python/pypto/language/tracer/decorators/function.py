# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The tracing frontend's entries — the trace-driver PTO adapter
and the mark/trace entries.

``function_trace`` instruments the kernel through the collector's trace driver and
executes it with PtoSema: the params are bound to annotation-derived
facades, the body runs natively (the synthesized function
keeps the kernel's closure cells, so freevars resolve), the ops yield their
facades, and PtoSema builds the structural IR. Returns the built
``ir.Function``.

The mark/trace decoupling: ``trace`` is the executed
build — sub-kernel calls resolve natively (pass-through) against
the ``Marker`` in the kernel's scope, emitting detached ``ir.Call``
nodes collected by the Frame's bind machinery — and ``function_mark``
splits standalone (eager-trace →
``ir.Function``) from in-program (mark → ``Marker``).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import types
import typing
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pypto.language.ambient import reset_active_frame, set_active_frame
from pypto.language.diagnostics import ParserSyntaxError, ParserTypeError
from pypto.language.entrance.annotation import capture_caller_names, source_span_for_function
from pypto.language.entrance.declaration import fold_auto_scope
from pypto.language.op.marker import EXTERNAL_SOURCE_ATTR, Marker, _fold_external_source
from pypto.language.protocol import is_staged, is_staged_annotation
from pypto.language.tracer.collector.driver import instrument
from pypto.language.tracer.decorators.signature import is_class_method
from pypto.language.tracer.return_contract import (
    derived_return_types,
    rebuild_with_return_types,
    return_value_types,
    same_return_types,
)
from pypto.language.tracer.semantics import PtoSema
from pypto.language.typing import annotation_to_return_types, annotation_to_type
from pypto.language.typing.arg_direction import param_directions, unmangle_hints, unmangle_param_name
from pypto.pypto_core import ir

__all__ = ["function_mark", "function_trace", "trace", "trace_program"]


def _source_span_for_class(cls: type) -> ir.Span:
    """Return the class definition's coarse source span when available."""
    try:
        source_file = inspect.getsourcefile(cls)
    except TypeError:
        # An exec-defined class (a printed program re-executed from a string —
        # the verbose-printer round-trip) has no source file to name.
        return ir.Span.unknown()
    if source_file is None:
        return ir.Span.unknown()
    try:
        source_lines, source_start = inspect.getsourcelines(cls)
        tree = ast.parse(textwrap.dedent("".join(source_lines)))
    except (OSError, TypeError, SyntaxError):
        return ir.Span.unknown()

    class_def = next(
        (node for node in ast.walk(tree) if isinstance(node, ast.ClassDef) and node.name == cls.__name__),
        None,
    )
    if class_def is None:
        return ir.Span.unknown()
    return ir.Span(source_file, source_start + class_def.lineno - 1, -1)


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


def _marker_scope(marker: Marker, kernel: types.FunctionType) -> Mapping[str, object]:
    """The name scope a program method's parameters may shadow.

    The decoration-boundary ``_localns`` (the caller's visible names — the
    AST's ``closure_vars``) when captured, else the kernel's own module
    globals; ``_localns`` is the superset (it merges the caller chain), so it
    wins when present."""
    if marker._localns:
        return marker._localns
    return kernel.__globals__


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
    ``function_trace``; the program build passes its registry proxy as a
    leading argument instead — ``_trace_program_method``).
    ``specialize_global_callees`` selects the jit callee lane for
    module-global AND closure-captured callees (see ``function_trace``):
    specialization is a property of the callee, never of where it is bound.

    Both callee-adaptation surfaces run here so every root-trace producer
    (standalone ``@pl.function``, the jit eager and specialize lanes) sees
    callable callees: closure-captured standalone functions become
    ``Marker``s, and module-global ones become callable in the execution
    copy of the globals. Pre-adapted kernels (the jit dispatch lane, program
    methods) pass through both adaptations unchanged."""
    kernel, _captured = _adapt_captured_functions(kernel, specialize=specialize_global_callees)
    kernel = _wrap_global_callee_functions(kernel, specialize=specialize_global_callees)
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

    # The trace driver installs the current Frame for this one function
    # execution and resets it unconditionally.
    frame = sema.frame
    assert frame is not None
    _token = set_active_frame(frame)
    try:
        inst = instrument(kernel, sema)
        if materialize_self and "self" not in inst.fn.__globals__:
            # The jit contract: ``self.<sibling>`` in a jit body is the
            # module's traced-kernel registry (upstream's jit materializes
            # it; the corpus marks it with an ``F821`` noqa). The instrumented
            # globals are a private copy whose standalone callees are already
            # Marker-wrapped above, so the proxy resolves the sibling to the
            # detached-emission surface without touching the module.
            inst.fn.__globals__["self"] = _JitSelfProxy(inst.fn.__globals__, kernel.__name__)
        inst(*arg_facades)
    finally:
        reset_active_frame(_token)
    fn = sema.result()
    declared_return_types = annotation_to_return_types(return_type)
    if declared_return_types and not same_return_types(fn.return_types, declared_return_types):
        fn = rebuild_with_return_types(fn, declared_return_types)
    return fn


def trace(
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    strict_ssa: bool = False,
) -> ir.Function:
    """The tracing frontend's build: the executed trace.

    The kernel's body runs with the callee ``Marker``s in scope (resolved
    natively from the kernel's closure cells, pass-through) —
    a call emits the detached ``ir.Call`` from the Marker's *declared*
    signature. Deduction never reads the callee body (the Marker is
    self-sufficient); inline callees are spliced by the backend
    ``inline_functions`` pass via the conformance harness's
    ``canonicalize(inline=True, callees=[...])`` regime.
    """
    return function_trace(
        kernel,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        strict_ssa=strict_ssa,
    )


def function_mark(
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
    auto_scope: bool = True,
    strict_ssa: bool = False,
    external_source: Any = None,
) -> ir.Function | Marker:
    """``pl.function``: the mark-vs-eager split.

    Standalone kernels eager-trace (returns the ``ir.Function``); inside a
    ``@pl.program`` class the
    kernel *marks* (returns the ``Marker``) and defers to the class
    decorator (the parser frontend's ``_is_class_method`` check, re-skinned).
    """
    # Capture the caller's visible names once, at the decoration boundary
    # (the AST's ``closure_vars``): PEP 563 annotations on the kernel may
    # reference enclosing-scope host names.
    localns = capture_caller_names()
    external_source = _prepare_external_source(kernel, function_type, external_source)
    if is_class_method(kernel):
        return Marker(
            kernel,
            function_type=function_type,
            level=level,
            role=role,
            attrs=fold_auto_scope(attrs, auto_scope),
            strict_ssa=strict_ssa,
            external_source=external_source,
            localns=localns,
        )
    return function_trace(
        kernel,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        auto_scope=auto_scope,
        strict_ssa=strict_ssa,
        external_source=external_source,
        localns=localns,
    )


def trace_program(cls: type, *, strict_ssa: bool = False) -> ir.Program:
    """The tracing frontend's program build: the two-pass
    ``pl.program`` envelope (the placeholder is retired).

    ``strict_ssa`` is the program-level round-trip toggle (text-parser syntax): the
    parser frontend's ``@pl.program(strict_ssa=True)`` governs every method (the
    per-function ``strict_ssa`` in a program class body does not execute —
    the decorator returns the function unchanged), so the traced build
    applies it to every Marker (strict if either the program or the Marker
    is on).

    **Pass 1** collects the marked methods in class-body (definition) order
    — ``Marker`` class attributes (``@function_mark``) or raw methods with
    the class-method shape (``@pl.function`` returns them unchanged) — and
    builds the ``gvar`` + ``Marker`` registry. **Pass 2** traces each
    method with the registry in scope: the method's ``self`` is a
    ``_SelfProxy`` (a host param — passed through, never an IR
    param) whose ``__getattr__`` yields the sibling ``Marker``, so
    ``self.method(x)`` executes the detached emission natively.

    Class-body order: no topo-order — the static return types make
    every ``Marker`` self-sufficient, so a forward-ref call (entry before
    helper) emits with the helper's *declared* signature. Post-trace: the
    declared-return-type check (a wrong ``->`` is loud) and the
    recursion guard (a cycle is loud, a *check* not an ordering).
    """
    registry: dict[str, Marker] = {}
    localns = capture_caller_names()
    for name in cls.__dict__:
        member = cls.__dict__[name]
        if isinstance(member, Marker):
            registry[name] = member
        elif inspect.isfunction(member) and is_class_method(member):
            registry[name] = _mark_from_raw_method(member, localns)
    if not registry:
        raise TypeError(
            f"program class {cls.__name__!r}: contains no marked methods — mark "
            f"the kernels with @pl.function (or @function_mark)"
        )

    functions: list[ir.Function] = []
    external_functions: dict[str, ir.Function] = {}
    for name, marker in registry.items():
        kernel, captured_functions = _adapt_captured_functions(marker.kernel)
        for external in captured_functions:
            existing = external_functions.get(external.name)
            if existing is not None and existing is not external:
                raise TypeError(
                    f"program {cls.__name__!r}: captured functions named {external.name!r} conflict"
                )
            if external.name in registry:
                raise TypeError(
                    f"program {cls.__name__!r}: captured function {external.name!r} "
                    "conflicts with a program method"
                )
            external_functions[external.name] = external
        fn = _trace_program_method(
            marker,
            registry,
            kernel=kernel,
            strict_ssa=strict_ssa or marker.strict_ssa,
        )
        # The declared signature is the ``->`` annotation only; a marker whose
        # derived effective contract was pinned by an earlier trace of this
        # class (``marker.return_types``) must not make the re-trace materialize
        # that inference onto the Function.
        if marker.declared_return_types and not same_return_types(
            fn.return_types, marker.declared_return_types
        ):
            # A builtin ``tuple[T1, T2]`` is a flat function signature, but
            # the tracing Sema intentionally sees no structural TupleType so
            # ``return a, b`` remains two ReturnStmt values. Pin the declared
            # flat signature onto the completed Function before callers or the
            # printer inspect it.
            fn = rebuild_with_return_types(fn, marker.declared_return_types)
        if not marker.declared_return_types:
            # An annotationless callee — derive the *effective* call-site return
            # contract from the actual trace (the ReturnStmt values, the same
            # source the declared-return check compares against) so later
            # callers' ``deduce_call_return_type`` see the real type instead of
            # UnknownType, and pin the marker for callers traced after this one
            # (class-body definition order). The Function's own signature stays
            # the declaration the source spells (empty) — the parser's
            # declared/effective split (``_parse_kernel_call`` +
            # ``_effective_return_types``), so the printed ``->`` is never a
            # materialized inference. The derivation is the shared
            # all-or-nothing guard (see ``derived_return_types``): a body-local
            # extent Var (e.g. an Inline callee's ``n = pl.tensor.dim(...)``
            # result — the fold is Orchestration-only) has no name in scope at
            # the signature, so a printed ``->`` would fail reparse (the AST
            # keeps ``return_types=[]`` for exactly this case).
            derived = derived_return_types(fn)
            # Pin only a sound observation: a void callee is *known* void
            # (empty raw return set), and a successfully derived callee is
            # pinned — but a callee whose paths disagree (an UnknownType in
            # the raw set) or whose types cannot be spelled must NOT claim a
            # known return set: a submit referencing it stays loud instead
            # of guessing TASK_ID-only.
            marker.implicit_returns_pinned = bool(derived) or not return_value_types(fn)
            if derived:
                marker.return_types = list(derived)
        _check_declared_return(fn, marker)
        functions.append(fn)
    all_functions = functions + list(external_functions.values())
    _detect_cycles(all_functions, cls.__name__)
    return ir.Program(all_functions, cls.__name__, _source_span_for_class(cls))


def _adapt_captured_functions(
    kernel: types.FunctionType, *, specialize: bool = False
) -> tuple[types.FunctionType, list[ir.Function]]:
    """Make captured standalone functions callable for one traced invocation.

    ``specialize`` — the jit envelope's lanes: a captured callee routes
    through the per-call variant lane (the same wrapper the module-global
    lane uses), so a call-site host knob bakes identically whether the
    callee was reached through a closure cell or a global name. The
    standalone ``pl.function`` build keeps the plain ``Marker`` — the
    parser-aligned verbatim lane the conformance oracle pins."""
    closure = kernel.__closure__
    if not closure:
        return kernel, []

    captured: list[ir.Function] = []
    cells: list[types.CellType] = []
    changed = False
    for cell in closure:
        value = cell.cell_contents
        from pypto.language.tracer.decorators import jit as _jit  # noqa: PLC0415

        if specialize and isinstance(value, ir.Function):
            captured.append(value)
            cells.append(types.CellType(_jit._LazyGlobalFn(value, _jit._eager_trace_record(value))))
            changed = True
        elif not specialize and (frozen := _jit._parser_aligned_callee(value)) is not None:
            # An eagerly traced jit object captured in a closure contributes
            # its decoration trace to the parser-aligned lane — the callee it
            # was before the jit object became the decoration result.
            captured.append(frozen)
            cells.append(types.CellType(Marker._from_ir_function(frozen)))
            changed = True
        else:
            cells.append(cell)
    if not changed:
        return kernel, []

    adapted = types.FunctionType(
        kernel.__code__,
        kernel.__globals__,
        name=kernel.__name__,
        argdefs=kernel.__defaults__,
        closure=tuple(cells),
    )
    adapted.__kwdefaults__ = kernel.__kwdefaults__
    adapted.__annotations__ = kernel.__annotations__
    adapted.__qualname__ = kernel.__qualname__
    adapted.__module__ = kernel.__module__
    return adapted, captured


def _wrap_global_callee_functions(
    kernel: types.FunctionType, *, specialize: bool = False
) -> types.FunctionType:
    """Make module-global standalone ``ir.Function`` callees callable for one
    traced execution.

    The parser frontend resolved a bare-name call through its closure-var
    registry; tracing has no ambient registry, so a module-global standalone
    function (an eager-traced ``ir.Function``) reaches the traced body
    uncallable. This rebinds the kernel over an execution copy of its globals
    with every ``ir.Function`` value wrapped callable; the module's real
    globals are never touched, and a global the body never calls keeps the
    cost at one dict copy.

    ``specialize`` — the jit envelope's lanes: every ``ir.Function`` global
    becomes the jit lazy wrapper, which routes a callee registered at eager
    decoration through the per-call variant lane (re-trace with the
    call-site argument types) and every other function through the
    verbatim+registering lane. ``specialize=False`` (the standalone
    ``pl.function`` build and the program methods) keeps the plain
    ``Marker`` — the parser frontend's bare-name call is verbatim, and the
    conformance oracle pins that. The jit dispatch lane pre-wraps globals
    with the same lazy wrapper, so those values are no longer
    ``ir.Function`` and pass through.
    """
    if specialize:
        from pypto.language.tracer.decorators import jit as _jit  # noqa: PLC0415 — jit imports this module

        def _wrap(value: ir.Function) -> Any:
            return _jit._LazyGlobalFn(value, _jit._eager_trace_record(value))

        def _callee(value: Any) -> ir.Function | None:
            return value if isinstance(value, ir.Function) else None

    else:
        from pypto.language.tracer.decorators import jit as _jit  # noqa: PLC0415

        def _wrap(value: ir.Function) -> Any:
            # The parser-aligned lane's callee is the frozen function — an
            # eagerly traced jit object unwraps to its decoration trace
            # (no envelope policy: no world checks, no re-trace).
            return Marker._from_ir_function(value)

        _callee = _jit._parser_aligned_callee

    patched = False
    g = dict(kernel.__globals__)
    for name, value in g.items():
        frozen = _callee(value)
        if frozen is not None:
            g[name] = _wrap(frozen)
            patched = True
    if not patched:
        return kernel
    adapted = types.FunctionType(
        kernel.__code__,
        g,
        name=kernel.__name__,
        argdefs=kernel.__defaults__,
        closure=kernel.__closure__,
    )
    adapted.__kwdefaults__ = kernel.__kwdefaults__
    adapted.__annotations__ = kernel.__annotations__
    adapted.__qualname__ = kernel.__qualname__
    adapted.__module__ = kernel.__module__
    return adapted


def _mark_from_raw_method(member: types.FunctionType, localns: dict[str, object]) -> Marker:
    """Mark a raw program method, recovering its decorator classification.

    The parser-frontend ``@pl.function`` returns class methods unchanged (the ``@pl.program``
    walker parses them), so ``@pl.function(type=Orchestration, auto_scope=False)``
    would otherwise re-mark with the ``Opaque`` defaults. The parser decorator
    stashes the classification values
    on the function (``_pl_function_class`` + ``_pl_external_source``); the
    tracing path's ``function_mark`` carries them on the ``Marker`` instead.
    """
    classification: dict[str, Any] = getattr(member, "_pl_function_class", None) or {}
    return Marker(
        member,
        function_type=classification.get("function_type", ir.FunctionType.Opaque),
        level=classification.get("level"),
        role=classification.get("role"),
        attrs=fold_auto_scope(classification.get("attrs"), classification.get("auto_scope", True)),
        strict_ssa=classification.get("strict_ssa", False),
        external_source=getattr(member, "_pl_external_source", None),
        localns=localns,
    )


def _capture_host_subworker_body(kernel: types.FunctionType) -> tuple[str, bool]:
    """Capture a HOST SubWorker body without executing it as DSL.

    HOST SubWorkers are runtime Python callbacks. Their IR keeps the function
    signature and stores the body as an ``InlineStmt``; an ellipsis-only body
    is the abstract callback form and therefore carries no source text.
    """
    try:
        source = textwrap.dedent(inspect.getsource(kernel))
        tree = ast.parse(source)
    except (OSError, TypeError, SyntaxError) as exc:
        raise ParserSyntaxError(f"could not capture HOST SubWorker {kernel.__name__!r} source") from exc

    function_nodes = [
        node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not function_nodes:
        raise ParserSyntaxError(f"could not find HOST SubWorker {kernel.__name__!r} definition in its source")
    func_def = function_nodes[0]
    if func_def.args.args and func_def.args.args[0].arg == "self":
        raise ParserSyntaxError(
            f"SubWorker function {func_def.name!r} must not declare 'self'; "
            "declare it as a self-contained function inside @pl.program."
        )

    non_doc = [
        stmt
        for stmt in func_def.body
        if not (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )
    ]
    is_abstract = (
        len(non_doc) == 1
        and isinstance(non_doc[0], ast.Expr)
        and isinstance(non_doc[0].value, ast.Constant)
        and non_doc[0].value.value is ...
    )
    if is_abstract:
        return "", True
    return "\n".join(ast.unparse(stmt) for stmt in func_def.body), False


def _is_host_subworker(marker: Marker) -> bool:
    return (
        marker.level is not None
        and ir.level_to_linqu_level(marker.level) >= 3
        and marker.role == ir.Role.SubWorker
    )


def _trace_program_method(
    marker: Marker,
    registry: dict[str, Marker],
    *,
    kernel: types.FunctionType | None = None,
    strict_ssa: bool | None = None,
) -> ir.Function:
    """Trace one program method with the registry in scope (the envelope's
    Pass-2 body). The leading ``self`` is the host-proxy; the rest are
    the declared staged params (annotations + directions).

    A self-less kernel is the round-trip dialect's self-contained form: the
    C++ printer emits SubWorker functions without ``self`` ("SubWorker
    functions are declared self-contained", python_printer.cpp) — and the
    tracing frontend's own class methods always declare it. The proxy is a host
    param, so it must be passed only when the kernel declares it;
    unconditionally prepending it made every self-less re-parse die with a
    spurious ``takes N positional arguments but N+1 were given`` arity
    error."""
    kernel = marker.kernel if kernel is None else kernel
    inline_body: str | None = None
    requires_runtime_binding = False
    if _is_host_subworker(marker):
        inline_body, requires_runtime_binding = _capture_host_subworker_body(kernel)

    hints = unmangle_hints(kernel, typing.get_type_hints(kernel, localns=marker._localns))
    signature = inspect.signature(kernel)
    names = [
        unmangle_param_name(kernel, p.name)
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    has_self = bool(names) and names[0] == "self"
    if has_self:
        names = names[1:]

    annotations: dict[str, object] = {}
    for name in names:
        annotation = hints.get(name)
        if annotation is None or not is_staged_annotation(annotation):
            raise TypeError(
                f"program method {marker.name!r}: parameter {name!r} needs a "
                f"pl.Tensor/pl.Tile/pl.Scalar/pl.Array/pl.Ptr annotation, got "
                f"{annotation!r}"
            )
        annotations[name] = annotation

    return_type = hints.get("return")
    sema = PtoSema(
        name=marker.name,
        return_type=annotation_to_type(return_type) if is_staged(return_type) else None,
        function_type=marker.function_type,
        level=marker.level,
        role=marker.role,
        attrs=marker.attrs,
        strict_ssa=marker.strict_ssa if strict_ssa is None else strict_ssa,
        requires_runtime_binding=requires_runtime_binding,
        source_span=source_span_for_function(kernel),
    )
    facades = sema.bind_parameters(
        annotations,
        directions=param_directions(kernel, localns=marker._localns),
        scope=_marker_scope(marker, kernel),
    )
    proxy = _SelfProxy(registry, marker.name)
    frame = sema.frame
    assert frame is not None
    _token = set_active_frame(frame)
    try:
        if inline_body is not None:
            sema.finish_inline_body(inline_body)
        elif has_self:
            instrument(kernel, sema)(proxy, *facades)
        else:
            # A self-less kernel — the round-trip dialect's self-contained
            # SubWorker form (the python printer drops the leading ``self``)
            # may still reference sibling kernels via ``self.method(...)`` in
            # the printed IR. The instrumented fn's globals are a private
            # copy (``instrument`` copies the kernel's ``__globals__``), so
            # injecting the proxy there resolves the reference without touching
            # the kernel's module namespace.
            inst = instrument(kernel, sema)
            inst.fn.__globals__["self"] = proxy
            inst(*facades)
    finally:
        reset_active_frame(_token)
    return sema.result()


def _declared_return_matches(declared: ir.Type, actual: ir.Type) -> bool:
    """Does the traced return match the declared ``->`` annotation?

    For ``Tile`` (and ``Tensor``) the annotation can express only shape +
    dtype: memory space and ``TileView`` (e.g. ``valid_shape=[M, N]`` from
    ``pl.load(..., valid_shapes=...)``) are trace-deduced details the
    annotation syntax has no word for (historical
    ``test_load_via_pl_load_with_valid_shapes`` declared-return over-fire).
    Comparing those extras would flag every Vec-tile kernel as "the declared
    ``->`` was a lie". Everything else compares structurally (auto-mapping
    absorbs per-trace Var identity).
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


def _check_declared_return(fn: ir.Function, marker: Marker) -> None:
    """The declared-return-type check: the ``->`` annotation is
    *declared*, then checked against the trace — a mismatch is a loud "the
    declared `->` was a lie". An annotation-less method
    has nothing to compare; a host-only return has no staged value.
    Staged-control-flow returns are already gated by ``Frame``, so the
    traced return is straight-line and its type is stable under the
    canonical regime. A declared `->` whose trace produced no return value
    (or only Unknown-typed ones) is the same lie — loud.

    A function backed by ``external_source=`` is skipped: its body is the
    ``...`` signature-only placeholder (the implementation is the hand-written
    C++ source), so the trace legitimately produces no return — mirroring the
    C++ verifier's ``HasAttr("external_source")`` skip
    (verify_return_params_explicit.cpp)."""
    if marker.attrs is not None and EXTERNAL_SOURCE_ATTR in marker.attrs:
        return
    # HOST SubWorkers carry opaque Python bodies in InlineStmt. Their return
    # annotations remain the runtime/codegen contract, but cannot be checked
    # against a DSL trace that is deliberately not executed.
    if _is_host_subworker(marker):
        return
    declared = list(marker.return_types)
    if not declared:
        return
    actual = return_value_types(fn)
    if not actual:
        raise TypeError(
            f"program method {fn.name!r}: the declared `->` {declared} produced no "
            f"traced return value (a declared return type with no matching "
            f"return is a loud 'the `->` was a lie')"
        )
    if len(declared) == len(actual) and all(_declared_return_matches(d, a) for d, a in zip(declared, actual)):
        return
    raise TypeError(
        f"program method {fn.name!r}: the traced return type {actual} does not "
        f"match the declared `->` {declared} (the declared return "
        f"type is checked against the trace; a mismatch is a loud 'the `->` "
        f"was a lie')"
    )


def _detect_cycles(functions: list[ir.Function], program_name: str) -> None:
    """The recursion guard: walk each traced function's
    IR for sub-kernel ``Call`` gvars, build the call graph, and DFS for a
    cycle — a self- or mutual-call is loud. A *check* after the trace, not
    an ordering (class-body order already handled forward refs). Calls to
    names outside the program (external functions) are not graph nodes."""
    graph: dict[str, set[str]] = {}
    for fn in functions:
        called: set[str] = set()

        class _V(ir.IRVisitor):
            def visit_call(self, op: ir.Call) -> None:
                if isinstance(op.op, ir.GlobalVar):
                    called.add(op.op.name)
                super().visit_call(op)

        _V().visit_function(fn)
        graph[fn.name] = called

    white, gray, black = 0, 1, 2
    color = {name: white for name in graph}
    stack: list[str] = []

    def _dfs(node: str) -> None:
        color[node] = gray
        stack.append(node)
        for callee in sorted(graph[node]):
            if callee not in graph:
                continue  # an external callee, not a program node
            if color[callee] == gray:
                cycle = stack[stack.index(callee) :] + [callee]
                raise TypeError(
                    f"program {program_name!r}: recursive call cycle detected: "
                    f"{' -> '.join(cycle)} (recursion is forbidden; "
                    f"the call graph must be acyclic)"
                )
            if color[callee] == white:
                _dfs(callee)
        stack.pop()
        color[node] = black

    for name in graph:
        if color[name] == white:
            _dfs(name)


class _SelfProxy:
    """The ``self``-proxy: the method's ``self`` at trace
    time. A host value in the method's namespace (passed through
    ``bind_parameters`` unchanged, never an IR param); ``__getattr__`` returns
    the sibling ``Marker`` from the registry, so ``self.method(x)``
    executes the Marker's detached emission. An unknown sibling is loud
    (the parser frontend's undefined-program-function discipline)."""

    __slots__ = ("_registry", "_owner")

    def __init__(self, registry: dict[str, Marker], owner: str) -> None:
        self._registry = registry
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        marker = self._registry.get(name)
        if marker is None:
            raise AttributeError(
                f"program method {self._owner!r}: unknown sibling {name!r} — not a "
                f"marked @pl.function method of the program class "
                f"(available: {sorted(self._registry)})"
            )
        return marker


class _JitSelfProxy:
    """The jit lane's ``self``: the module materialized as an implicit
    program. Upstream's ``@pl.jit`` lets a jit body reference sibling
    kernels as ``self.<sibling>`` (the corpus contract behind the
    ``# noqa: F821``); ``__getattr__`` resolves the name through the
    execution copy of the module globals — where
    ``_wrap_global_callee_functions`` has already made standalone
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
