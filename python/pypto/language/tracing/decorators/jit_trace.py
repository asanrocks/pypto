# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The ``pl.jit`` envelope over the tracing frontend.

Eager ⊂ lazy: the same trace driver, eager is a pre-execution with
synthetic args. Both timings return the same public jit object (the
``_JitSpecializer`` surface); the split is internal. **Eager** — every
param is a facade annotation (concrete or DynVar-shaped — the symbolic-
extent shape ``pl.Tensor[[N, 64], pl.FP32]`` IS annotation-determined) —
traces at first use; the eager trace (``.fn``) and the callees its
body reached are retained in a record, and the jit's host ``compile`` /
dispatch reuse them — the annotation determines the artifact, so a call
argument supplies a dispatch value, never a re-specialization. **Lazy** —
an unannotated/bare/Callable param forces deferral: at call, the real
args specialize the kernel (tensor facades → static-shape params; host
scalars → passed through, lifted to ``Const`` by the ops at the operand
boundary), re-traced per call (no cache).

The jit object is the full jit surface:

- **eager lane** — the eager record's program (the eager trace
  plus its collected callees) serves every ``compile``/dispatch whose
  arguments are compatible with the declaration (a declared dynamic
  extent accepts any concrete shape; a static dim must match — a
  contradicting static call is refused like any other argument outside its
  declaration);
- **specialize lane** — a call with facades/host values re-traces and
  returns the ``ir.Function`` (O9: an ``ir.Function`` is not executable);
- **dispatch lane** — a call with ``torch.Tensor`` args bridges each to an
  annotation-mode facade (shape + dtype), assembles the multi-function
  ``ir.Program`` (entry + the callees its body reached), compiles via
  ``ir.compile``, and dispatches; cache keyed on the bridged signature;
- **callee lane** — a call while another trace is active traces the callee
  with the call-site facade types and emits the detached ``ir.Call``, and
  registers the callee for the caller's program assembly;
- ``compile_for_test(*args)`` — the same build + the Default pass pipeline,
  no codegen, post-pass ``ir.Program`` for structural-equality tests.

The family spellings this envelope verifies: ``@pl.jit_trace`` (an
Orchestration entry) and its ``.incore`` / ``.inline`` callees. The
remaining parsing-frontend spellings — ``host``, ``opaque``, ``graph``,
``extern`` — reject at decoration; declare those kernels on ``@pl.jit``.
"""

from __future__ import annotations

import functools
import inspect
import types
import typing
import warnings
from collections.abc import Iterable, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, NoReturn

from pypto.ir.formatter import raw_print
from pypto.language.ambient import active_frame
from pypto.language.op.marker import Marker
from pypto.language.protocol import is_staged
from pypto.language.tracing.argument_contract import (
    CALL_SITE_KEYWORDS,
    bind_declared_keywords,
    coerce_declared_arguments,
    reject_bare_data_type_annotations,
)
from pypto.language.tracing.decorators.function import _trace_with_annotations
from pypto.language.tracing.entry.annotation import capture_caller_names, source_span_for_function
from pypto.language.tracing.entry.declaration import fold_auto_scope
from pypto.language.tracing.execution import adapt_captured_callees
from pypto.language.tracing.return_contract import check_specializing_return_values, pin_derived_return_types
from pypto.language.typing._ir_facade import facade_from_ir_type
from pypto.language.typing._runtime_scalar import RuntimeScalar
from pypto.language.typing.arg_direction import (
    freevar_localns,
    param_directions,
    unmangle_hints,
    unmangle_param_name,
)
from pypto.language.typing.constexpr import constexpr_value_key
from pypto.language.typing.dynamic import DynVar
from pypto.language.typing.scalar import RUNTIME, Scalar
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import ir

if TYPE_CHECKING:
    from pypto.runtime.runner import RunConfig

__all__ = ["jit", "jit_trace"]


def _drop_inline_directions(fn: ir.Function, function_type: ir.FunctionType) -> ir.Function:
    """Normalize Out/InOut param directions away on an Inline jit kernel.

    The specializer's generated-source normalization, ported: an Inline
    body is spliced at the call site (``inline_functions_pass``), so the
    parameter is already an in-place alias of the caller's variable — a
    direction left on it would make the call pass the arg as Out, and every
    later read of the caller's name (the corpus's ``return x_next`` after
    ``hc_post(..., x_next)``) an InOutUseDiscipline violation at
    pipeline_input verification. Upstream dropped the wrapper from the
    generated source with this same deprecation warning.
    """
    if function_type != ir.FunctionType.Inline:
        return fn
    dropped = [i for i, d in enumerate(fn.param_directions) if d != ir.ParamDirection.In]
    if not dropped:
        return fn
    warnings.warn(
        f"@pl.jit.inline helper {fn.name!r} uses pl.Out[...]/pl.InOut[...] on "
        f"parameter(s) {[fn.params[i].name_hint for i in dropped]!r}. Direction "
        "annotations are deprecated for inline helpers because the body is "
        "spliced at the call site — the parameter is already an in-place "
        "alias of the caller's variable. Drop the wrapper; bare pl.Tensor[...] "
        "works the same.",
        DeprecationWarning,
        stacklevel=3,
    )
    params = [(p, ir.ParamDirection.In) for p in fn.params]
    return ir.Function(
        fn.name,
        params,
        fn.return_types,
        fn.body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
    )


def _eager_trace_record_for(
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType,
    level: ir.Level | None,
    role: ir.Role | None,
    attrs: dict[str, Any] | None,
    localns: dict[str, object] | None = None,
) -> _EagerTraceRecord:
    """Run the one-time eager trace and register its record.

    The ``_eager_record`` property materializes it on first use for every
    annotation-determined lane — the same body, one trace, run against the
    declaration rather than call arguments."""
    from pypto.language.tracing.decorators.function import function_trace  # noqa: PLC0415 - avoid a cycle

    # The eager trace owns an assembly: in-trace calls to eagerly
    # traced callees specialize into it (per-call variants), and the
    # record keeps it reachable so a program build can join the entry
    # with exactly the callees its body reached — the closure text-scan
    # cannot see variant functions (they are minted at call sites, never
    # bound to module globals).
    assembly = _ProgramAssembly()
    token = _ACTIVE_ASSEMBLY.set(assembly)
    try:
        # An annotationless ``->`` pins the derived return contract (the
        # specializer's inferred return, materialized as the Function
        # signature — its synthesized ``->``). The pin is what lets a later
        # ``pl.spmd_submit`` callee see the recovered types as declared
        # facts (the Submit's Tuple[<ret>, TASK_ID] derives from them). An
        # Inline alias additionally drops Out/InOut param directions (the
        # specializer's generated-source normalization — see
        # ``_drop_inline_directions``).
        fn = _drop_inline_directions(
            pin_derived_return_types(
                function_trace(
                    kernel,
                    function_type=function_type,
                    level=level,
                    role=role,
                    attrs=attrs,
                    localns=localns,
                    materialize_self=True,
                    specialize_global_callees=True,
                )
            ),
            function_type,
        )
    finally:
        _ACTIVE_ASSEMBLY.reset(token)
    record = _EagerTraceRecord(fn, kernel, localns, assembly)
    _EAGER_TRACES[fn] = record
    return record


def _reject_undelivered_returns(entry: ir.Function, program: ir.Program) -> None:
    """Check fresh result allocations without changing the logical signature.

    Parameter-backed values keep their existing delivery path. Fresh static
    buffers can be exported by the shared ABI adapter. Unlowered InCore results
    are checked after their backing allocations become explicit in compilation.
    """
    if not entry.return_types:
        return
    delivered = ir.delivered_param_indices(entry, program)
    undelivered = [position for position, param in enumerate(delivered) if param is None]
    if undelivered:
        from pypto.ir._entry_outputs import materialize_entry_outputs  # noqa: PLC0415

        # The shared ABI adapter proves whether these results can be delivered.
        # Keep the logical program here; compilation owns the physical signature.
        materialize_entry_outputs(program, entry, defer_unlowered=True)
        return


def jit_trace(
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
) -> _JitSpecializer:
    """The ``pl.jit`` envelope: eager ⊂ lazy, one public object.

    Eager (every param a facade annotation) → deferred to first use:
    decorating must not execute the body. The eager record is the
    annotation-determined trace (the traced ``ir.Function`` — ``.fn`` — plus
    the callees the body reached), and the jit object's host
    ``compile``/dispatch reuse it, so the artifact keeps its declared
    dynamic extents (the conformance gate compares ``.fn`` to
    ``function_ast``). Lazy (an unannotated/bare/Callable param — the IR
    depends on call-time info) → a ``_JitSpecializer`` callable that
    re-traces per call with the baked args and returns the IR.
    Both lanes materialize ``self`` in the traced body as the module's
    traced-kernel registry (``self.<sibling>`` — upstream's jit contract).
    """
    # PEP 563 string annotations may name anything visible where the kernel
    # was written — a factory-local extent, a module alias, a closure cell.
    # Capture that namespace once, before classification: the classification
    # resolves hints, and a lookup without it raises NameError before any
    # later capture could help.
    localns = _defining_namespace(kernel)
    if _eagerly_traceable(kernel, localns):
        # Every annotation-determined lane defers its eager trace to first
        # use: decorating must not execute the body, so host side effects
        # stay out of import, and a body that references a sibling defined
        # later in the module resolves instead of raising ``NameError`` at
        # decoration. The record is the identical annotation-determined trace
        # — the ``_eager_record`` property materializes it at the first
        # ``.fn`` / host compile / dispatch / ``specialize`` / ``lower``;
        # in-trace calls take the per-call variant lane and never need it.
        return _JitSpecializer(
            kernel,
            function_type=function_type,
            level=level,
            role=role,
            attrs=attrs,
            localns=localns,
            defer_eager=True,
        )
    return _JitSpecializer(
        kernel,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        localns=localns,
    )


def _defining_namespace(kernel: types.FunctionType) -> dict[str, object]:
    """The names a PEP 563 annotation may reference, captured at decoration.

    The kernel's own closure cells first (they survive its defining frame),
    then the caller's visible names (which resolve a factory-local extent
    while the factory is still running), matching the order ``derive`` uses.
    """
    localns: dict[str, object] = dict(freevar_localns(kernel) or {})
    localns.update(capture_caller_names())
    return localns


def _eagerly_traceable(kernel: types.FunctionType, localns: dict[str, object]) -> bool:
    """Signature inspection: eager iff every positional param carries a
    facade annotation (concrete or DynVar-shaped — both annotation-
    determined). An unannotated / bare / Callable param forces lazy (the IR
    depends on the call-time argument)."""
    hints = unmangle_hints(kernel, typing.get_type_hints(kernel, localns=localns))
    signature = inspect.signature(kernel)
    for p in signature.parameters.values():
        if p.kind not in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            continue
        annotation = hints.get(unmangle_param_name(kernel, p.name))
        if annotation is None or not is_staged(annotation):
            return False
    return True


class _JitSpecializer:
    """The lazy jit: the full jit surface over per-call re-tracing.

    **Specialize lane** (facades / host values): re-trace per call (no
    cache), baking the real args — tensor facades become static-shape
    params, host scalars pass through (the ops reify them as ``Const`` at
    the op-arg boundary). Returns the freshly traced
    ``ir.Function`` (O9).

    **Dispatch lane** (torch tensors): bridge → build the program →
    ``ir.compile`` → dispatch, cached on the bridged signature.

    **Callee lane** (a call inside another trace): trace the callee with
    the call-site types, emit the detached ``ir.Call``, register the
    callee for the caller's program assembly."""

    def __init__(
        self,
        kernel: types.FunctionType,
        *,
        function_type: ir.FunctionType = ir.FunctionType.Opaque,
        level: ir.Level | None = None,
        role: ir.Role | None = None,
        attrs: dict[str, Any] | None = None,
        localns: dict[str, object] | None = None,
        eager_record: _EagerTraceRecord | None = None,
        defer_eager: bool = False,
    ) -> None:
        self.kernel = kernel
        self._function_type = function_type
        self._level = level
        self._role = role
        self._attrs = attrs
        self._localns = localns
        # The eager lane's record: when present, the annotation determined
        # the trace, so host compile/dispatch reuse the record's program
        # instead of re-tracing with the bridged call shapes (which would
        # demote declared dynamic extents to static — the exact silent
        # specialization the declaration exists to prevent).
        # ``defer_eager`` — every annotation-determined lane: the record
        # materializes at first use (the ``_eager_record`` property), running
        # the identical eager trace then. A genuinely lazy jit (an
        # unannotated/bare/Callable param) keeps ``None`` forever — ``.fn``
        # stays loud there.
        self.__eager_record = eager_record
        self.__defer_eager = defer_eager
        # The parameter declarations pin each parameter's world (types flow
        # from values): a pl facade annotation is a runtime parameter, a
        # plain Python annotation (or none) is a host knob baked at trace
        # time. The world checks consult this map.
        self._param_hints: dict[str, Any] = typing.get_type_hints(kernel, localns=localns)
        # A jit kernel declares its parameters: each one crosses the trace
        # boundary as an IR parameter or a baked host knob, so a variadic
        # parameter has no lane — it would be silently omitted from
        # ``_param_names`` (the caller's keyword would be an unknown name),
        # which is a refusal at decoration, not a silent drop. A keyword-only
        # parameter has no positional slot either — arguments bind in
        # declaration order — so it is the same silent omission and the same
        # refusal at decoration.
        for p in inspect.signature(kernel).parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                spelling = "*" if p.kind is inspect.Parameter.VAR_POSITIONAL else "**"
                raise TypeError(
                    f"@pl.jit kernel {kernel.__name__!r}: parameter "
                    f"{spelling + p.name!r} is variadic — a traced kernel declares "
                    "its parameters (each one becomes an IR parameter or a baked "
                    "host knob), so *args and **kwargs cannot cross the trace "
                    "boundary. Name the parameters the body uses."
                )
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                raise TypeError(
                    f"@pl.jit kernel {kernel.__name__!r}: parameter {p.name!r} is "
                    "keyword-only — a traced kernel binds its arguments in "
                    "declaration order (a keyword names a declared parameter, it "
                    "does not declare one), so a keyword-only parameter has no "
                    "slot on the traced signature. Spell it as a regular parameter."
                )
        self._param_names = [
            unmangle_param_name(kernel, p.name)
            for p in inspect.signature(kernel).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        self._return_type = self._param_hints.get("return")
        reject_bare_data_type_annotations(self._param_hints, self._param_names, kernel.__name__)
        self.__name__ = kernel.__name__
        self.__doc__ = kernel.__doc__
        self.__module__ = kernel.__module__
        self.__qualname__ = kernel.__qualname__
        self._cache: dict[Any, Any] = {}

    # —— the ``JITFunction`` external surface (the mirror) ——

    @property
    def param_names(self) -> tuple[str, ...]:
        """Declared parameter names, in signature order (``self`` excluded).

        Mirrors ``JITFunction.param_names``: a property over an immutable
        tuple, so a consumer reads both jit objects the same way."""
        return tuple(self._param_names)

    @property
    def output_param_names(self) -> tuple[str, ...]:
        """Parameters the kernel writes — ``pl.Out[...]`` and ``pl.InOut[...]``.

        In declaration order, so the tuple stays aligned with the callee's
        return order. Recovered from the runtime annotations (the same
        source ``param_directions`` serves elsewhere), so a sourceless
        kernel keeps its directions."""
        directions = param_directions(self.kernel, localns=self._localns)
        return tuple(
            name
            for name in self._param_names
            if directions.get(name) in (ir.ParamDirection.Out, ir.ParamDirection.InOut)
        )

    @property
    def _func(self) -> types.FunctionType:
        """Alias for :attr:`kernel` — the ``JITFunction`` private spelling."""
        return self.kernel

    def __repr__(self) -> str:
        return f"<jit {self.__name__} at {hex(id(self))}>"

    @property
    def _eager_record(self) -> _EagerTraceRecord | None:
        """The eager record — materialized on first access for every
        annotation-determined lane, ``None`` forever for a genuinely lazy
        jit."""
        if self.__eager_record is None and self.__defer_eager:
            self.__eager_record = _eager_trace_record_for(
                self.kernel,
                function_type=self._function_type,
                level=self._level,
                role=self._role,
                attrs=self._attrs,
                localns=self._localns,
            )
        return self.__eager_record

    @property
    def fn(self) -> ir.Function:
        """The eager trace — the annotation-determined ``ir.Function``
        (directions already normalized, derived return already pinned).
        Structural consumers (the conformance gate, corpus pins, the
        printer) read the IR here. A lazily traced jit has no single
        pre-executed trace — the property is eager-only, loud otherwise."""
        if self._eager_record is None:
            raise AttributeError(
                f"{self.kernel.__name__!r} is lazily traced — there is no eager "
                "trace; a call re-traces with the call-site arguments"
            )
        return self._eager_record.fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # A keyword names a declared parameter, so it is bound into the
        # declaration order here and every lane below reads one positional
        # list — the binding the parser lane performs when it resolves a call
        # against the callee's signature. Two keywords stay call-site metadata
        # instead: ``config=RunConfig(...)`` at a top-level call selects the
        # compile knobs and dispatch options, and ``device=`` / ``attrs=`` in a
        # traced body ride the detached emission.
        in_trace = active_frame() is not None
        config = None
        if not in_trace and "config" in kwargs and "config" not in self._param_names:
            config = _checked_run_config(kwargs.pop("config"))
        args, overrides = bind_declared_keywords(
            self._param_names,
            args,
            kwargs,
            self.kernel.__name__,
            reserved=CALL_SITE_KEYWORDS if in_trace else (),
        )

        if in_trace:
            return self._emit_detached_call(args, overrides)
        bridged, has_torch = self._bridge_args(args)
        for name, arg in zip(self._param_names, bridged):
            if is_staged(arg) and not _annotation_shaped(arg):
                raise TypeError(
                    f"call to {self.kernel.__name__!r}: argument {name!r} is a "
                    f"expr-wrapping facade (a value from inside another trace) — "
                    f"a jit call specializes with annotation-mode facades (shape + dtype)"
                )
        if has_torch:
            return self._dispatch(args, bridged, config)
        if config is not None:
            raise TypeError(
                f"call to {self.kernel.__name__!r}: config= applies to compiled "
                "dispatch (torch-tensor arguments); a facade/scalar call returns "
                "the traced function and compiles nothing"
            )
        return self._trace_with(dict(zip(self._param_names, bridged)))

    # —— the shared per-call trace ———

    def _trace_with(
        self,
        annotations: dict[str, object],
        kernel: types.FunctionType | None = None,
        *,
        derive_return: bool = False,
    ) -> ir.Function:
        """Re-trace with the given annotations. ``derive_return`` — the callee
        lane — pins the return types on the actual trace instead of the
        declaration: a per-call variant's declared return would carry the
        declaration's dynamic extents while its specialized params are
        concrete, and the two meeting at a type-sensitive rebind (the relaxed
        reassignment rule) would fork the caller's variable instead of
        assigning through — the declared form would lie about the produced
        values. The declaration is still a contract on the static parts
        (kind, dtype, static extents, arity): a callee whose ``->`` the
        specialized values contradict is refused here, not silently
        retyped — only a declared dynamic extent may specialize.
        """
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        if assembly is not None:
            # Every lane that traces under an assembly funnels through here —
            # the callee lanes' own lookup misses for the in-flight variant,
            # so the cycle guard lives at this single choke point.
            key = _variant_key(
                self.kernel, annotations, self._function_type, self._level, self._role, self._attrs
            )
            assembly.enter_variant_trace(key, self.kernel.__name__)
        try:
            fn = _trace_with_annotations(
                kernel if kernel is not None else self.kernel,
                annotations,
                None if derive_return else self._return_type,
                function_type=self._function_type,
                level=self._level,
                role=self._role,
                attrs=self._attrs,
                localns=self._localns,
                materialize_self=True,
                specialize_global_callees=True,
            )
            if self._return_type is None or derive_return:
                fn = pin_derived_return_types(fn)
                if derive_return:
                    check_specializing_return_values(fn, self._return_type)
            return _drop_inline_directions(fn, self._function_type)
        finally:
            if assembly is not None:
                assembly.exit_variant_trace(key)

    # —— the callee lane: a call inside another trace ———

    def _emit_detached_call(self, args: tuple[Any, ...], kwargs: dict[str, Any] | None = None) -> Any:
        """Trace this callee with the call-site argument types and emit the
        detached ``ir.Call`` through its ``Marker``. The freshly traced
        callee joins the caller's program assembly (deduped on the kernel
        identity plus the derived annotations — host scalar values included,
        the specializer's ``ScalarCacheInfo`` semantics, so two calls baking
        different constants never share a variant). Call-site arguments whose
        value was baked into the variant's signature (their params vanished
        from the trace) are dropped from the emitted ``Call``."""
        callee, runtime_args = self._detached_resolve(args)
        return Marker._from_ir_function(callee)(*runtime_args, **(kwargs or {}))

    def _detached_resolve(self, args: tuple[Any, ...]) -> tuple[ir.Function, tuple[Any, ...]]:
        """The lazy callee's shared instantiation, for a call or a Submit
        reference: trace with the call-site argument types, dedupe in the
        active program assembly under the variant key, and return the variant
        with the surviving runtime args (positions whose values were baked
        into the variant's signature are dropped). Call and submit consume
        this identically; only the emitted instruction differs. A bare host
        scalar into a Scalar-declared slot lifts to a Const argument —
        the param stays runtime, so the variant is value-free
        and the literal rides the call node."""
        args = _lift_runtime_scalar_args(
            self._param_hints, self._param_names, args, self.kernel.__name__, in_trace=True
        )
        _check_param_worlds(self._param_hints, self._param_names, args, self.kernel.__name__)
        args = coerce_declared_arguments(self._param_hints, self._param_names, args, self.kernel.__name__)
        annotations: dict[str, object] = {}
        for name, arg in zip(self._param_names, args):
            annotations[name] = _in_trace_annotation(
                name, arg, self.kernel.__name__, hint=self._param_hints.get(name)
            )
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        callee: ir.Function | None = None
        if assembly is not None:
            key = _variant_key(
                self.kernel, annotations, self._function_type, self._level, self._role, self._attrs
            )
            callee = assembly.by_key.get(key)
        if callee is None:
            callee = self._trace_with(annotations, derive_return=True)
            if assembly is not None and key is not None:
                variant_name = assembly.unique_name(callee.name)
                if variant_name != callee.name:
                    callee = _renamed_function(callee, variant_name)
                assembly.register(callee, key=key)
        surviving = {p.name_hint for p in callee.params}
        runtime_args = tuple(a for a, n in zip(args, self._param_names) if n in surviving)
        return callee, runtime_args

    def resolve_submit(self, args: tuple[Any, ...]) -> tuple[Marker, tuple[Any, ...]] | None:
        """The Submit-reference resolution (``Marker.resolve_submit``): the
        lazy jit callee instantiates through the same lane a call takes. A
        Submit only happens inside a traced body, so this is always the
        specialized path; on the host there is nothing to instantiate."""
        if active_frame() is None:
            return None
        callee, runtime_args = self._detached_resolve(args)
        return Marker._from_ir_function(callee), runtime_args

    # —— the dispatch lane: torch args → compile → execute ———

    def _bridge_args(self, args: tuple[Any, ...]) -> tuple[list[Any], bool]:
        """Specialize-time argument baking: facades and host values pass
        through; a ``torch.Tensor`` becomes an annotation-mode facade
        (shape + dtype — the specializer's TensorMeta, re-skinned)."""
        bridged: list[Any] = []
        has_torch = False
        for name, arg in zip(self._param_names, args):
            if is_staged(arg):
                # A caller-supplied facade is the author's own type statement;
                # the aggregate check below holds it to the declaration.
                bridged.append(arg)
                continue
            if isinstance(arg, (bool, int, float, ir.Expr)):
                bridged.append(arg)
                continue
            facade = _torch_facade(arg, self._param_hints.get(name))
            if facade is not None:
                bridged.append(facade)
                has_torch = True
                continue
            hint = self._param_hints.get(name)
            if hint is not None and not _staged_declared(hint) and constexpr_value_key(arg) is not None:
                # The host-call side of the ``pl.constexpr`` contract (B9): a
                # foldable host value — a ``str`` name hint, a ``DataType`` a
                # dtype-parameterised kernel selects on — bakes at trace time
                # when the declaration says the parameter is a knob. A bare
                # facade/none annotation stays loud: the declaration owns the
                # parameter's world.
                bridged.append(arg)
                continue
            raise TypeError(
                f"call to {self.kernel.__name__!r}: argument {name!r} "
                f"({type(arg).__name__}) is not a tensor, facade, or host scalar — "
                "a jit call specializes with torch tensors, annotation-mode "
                "facades, or Python scalars"
            )
        # Dispatch follows declarations: a bare host scalar into a
        # Scalar-declared slot bridges by declaration — the runtime-scalar
        # marker, dtype-keyed, value at dispatch (the marker object stays
        # internal; the public spelling is the bare value).
        bridged = list(
            _lift_runtime_scalar_args(
                self._param_hints, self._param_names, tuple(bridged), self.kernel.__name__, in_trace=False
            )
        )
        _check_param_worlds(self._param_hints, self._param_names, bridged, self.kernel.__name__)
        # A torch tensor becomes a facade before this check, so a host call
        # and an explicit-facade call answer to the same declaration. Keeping
        # the bridged path exempt let a static parameter silently specialize
        # to a new shape while the identical facade call was refused. A dtype
        # inside the declared slot's integer lattice is not a refusal here: the
        # host lane has no value to convert, so the declaration is the ABI.
        bridged = list(
            coerce_declared_arguments(self._param_hints, self._param_names, bridged, self.kernel.__name__)
        )
        return bridged, has_torch

    def _dispatch(self, args: tuple[Any, ...], bridged: list[Any], config: RunConfig | None = None) -> Any:
        # A diagnostic request (a dump, a profile, an explicit output dir, an
        # outer context with instruments) is a request to *run* the compiler,
        # decided before any cache lookup — the shared resolver both jit lanes
        # use. It also validates pass-context conflicts here, so an invalid
        # combination raises even when a compatible artifact is already warm.
        # The request compiles fresh and leaves the ordinary cache untouched.
        from pypto.runtime.jit_cache import resolve_compile_request  # noqa: PLC0415 — torch is a runner dep

        _, bypass = resolve_compile_request(config)
        key = None if bypass else self._cache_key(bridged, config)
        compiled = self._cache.get(key) if key is not None else None
        if compiled is None:
            compiled = _compile_program(self._build_program(bridged), config)
            if key is not None:
                self._cache[key] = compiled
        # Host knobs were consumed at trace time — dispatch speaks
        # the runtime ABI only (the staged-declared positions), never the
        # baked values; a runtime-scalar marker unwraps to its raw value.
        dispatch_args = tuple(
            a.value if isinstance(a, RuntimeScalar) else a
            for a, name in zip(args, self._param_names)
            if _staged_declared(self._param_hints.get(name))
        )
        if config is None:
            return compiled(*dispatch_args)
        return compiled(*dispatch_args, config=config)

    def _cache_key(self, bridged: list[Any], config: RunConfig | None = None) -> tuple[Any, ...]:
        # The compile-identity suffix (platform, strategy, distributed config,
        # planner, dbC flag, runtime ABI, Buffer IR, source locations —
        # resolved from the RunConfig, any active PassContext, and the
        # environment) splits the cache so two compile configurations never
        # share one artifact (the dispatch-cache key in the *Collection timing
        # and reuse* section of ``docs/en/dev/language/03-functions.md``).
        # Shared with the parsing-frontend jit through pypto.runtime.jit_cache.
        from pypto.runtime.jit_cache import (  # noqa: PLC0415 — torch is a runner dep
            TensorKeyInfo,
            run_config_compile_identity,
        )

        parts: list[Any] = []
        for name, arg in zip(self._param_names, bridged):
            if isinstance(arg, RuntimeScalar):
                # The entry runtime-scalar marker: type-only — one artifact
                # serves every value (the whole point of the spelling).
                parts.append(("runtime_scalar", str(arg.dtype)))
            elif is_staged(arg) and isinstance(arg, Tensor):
                declared = self._param_hints.get(name)
                declared_shape = getattr(declared, "shape", None)
                shape = tuple(arg.shape or ())
                if declared_shape is not None and len(declared_shape) == len(shape):
                    # A declared dynamic extent is part of the artifact's
                    # runtime contract, not its specialization key. The
                    # eager record already preserves that extent in the IR;
                    # keying it by each concrete dispatch shape would rebuild
                    # an equivalent binary for every batch size.
                    shape = tuple(
                        None if isinstance(declared_dim, ir.Var) or hasattr(declared_dim, "_ir_var") else dim
                        for declared_dim, dim in zip(declared_shape, shape)
                    )
                # A bridged staged facade is annotation-mode, so its dtype is
                # always materialized; a None here would mean a runtime-mode
                # value escaped into the host dispatch path.
                assert arg.dtype is not None
                parts.append(TensorKeyInfo(name, shape, arg.dtype, arg.layout))
            else:
                # A baked host knob joins the key by value; the constexpr
                # identity keeps list-valued knobs hashable.
                folded = constexpr_value_key(arg)
                parts.append((type(arg).__name__, arg if folded is None else folded))
        parts.append(run_config_compile_identity(config))
        return tuple(parts)

    # —— the program build shared by compile / compile_for_test ———

    def _build_program(self, bridged: list[Any]) -> ir.Program:
        """Trace the entry with the bridged annotations and assemble the
        multi-function program: the entry plus every callee its body
        reached (lazy jit callees traced at their call sites, captured
        standalone functions adapted from the closure and the module
        globals).

        The eager lane short-circuits to the eager record when the bridged
        call is compatible with the declaration: the eager trace plus its
        collected callees IS the artifact, and re-tracing
        would bridge the concrete arg shapes, demoting declared dynamic
        extents to static."""
        if self._eager_record is not None and self._declaration_serves(bridged):
            rec = self._eager_record
            program = ir.Program(
                [rec.fn, *rec.assembly.functions],
                rec.fn.name,
                source_span_for_function(self.kernel),
            )
            _reject_undelivered_returns(rec.fn, program)
            return program
        assembly = _ProgramAssembly()
        token = _ACTIVE_ASSEMBLY.set(assembly)
        try:
            kernel = self._adapted_kernel(assembly)
            entry = self._trace_with(dict(zip(self._param_names, bridged)), kernel=kernel)
        finally:
            _ACTIVE_ASSEMBLY.reset(token)
        program = ir.Program(
            [entry, *assembly.functions], self.kernel.__name__, source_span_for_function(self.kernel)
        )
        _reject_undelivered_returns(entry, program)
        return program

    def _declaration_serves(self, bridged: list[Any]) -> bool:
        """Can the eager trace's program serve this bridged call? Each
        declared dynamic extent accepts any concrete shape; a declared
        static dim (or dtype) must match. The bridge checks that contract before
        this predicate runs, so a contradiction raises instead of re-tracing at
        a type the declaration does not admit; this remains as the cheap
        compatibility predicate for direct internal callers."""
        for name, arg in zip(self._param_names, bridged):
            declared = self._param_hints.get(name)
            d_shape = getattr(declared, "shape", None)
            c_shape = getattr(arg, "shape", None)
            if d_shape is None or c_shape is None:
                continue  # shape-less (scalar, Ptr, expr-wrapping): annotation-determined
            if len(d_shape) != len(c_shape):
                return False
            for d, c in zip(d_shape, c_shape):
                if isinstance(d, ir.Var) or hasattr(d, "_ir_var"):
                    continue  # a declared dynamic extent accepts any shape
                if d != c:
                    return False
            d_dtype = getattr(declared, "dtype", None)
            c_dtype = getattr(arg, "dtype", None)
            if d_dtype is not None and c_dtype is not None and d_dtype != c_dtype:
                return False
        return True

    def _adapted_kernel(self, assembly: _ProgramAssembly) -> types.FunctionType:
        """Make the jit values visible to one traced execution callable:
        closure-captured ``ir.Function``s become ``Marker``s (the shared
        adaptation, registered eagerly — the kernel explicitly closes over
        them); module-global ones become lazy callables that register on
        first call (a global the body never touches stays out of the
        program); lazy jit callees stay themselves (the callee lane)."""
        kernel, captured = adapt_captured_callees(self.kernel, specialize=True)
        for fn in captured:
            # Only the verbatim world registers eagerly: a jit-eager callee
            # (eager record present) joins the program as its per-call
            # variant at each call or submit-reference site — registering
            # the frozen eager trace would ship it as dead code alongside
            # the variants.
            if _eager_trace_record(fn) is None:
                assembly.register(fn)
        globals_patched = False
        g = dict(kernel.__globals__)
        for name, value in g.items():
            if isinstance(value, ir.Function):
                g[name] = _LazyGlobalFn(value, _eager_trace_record(value))
                globals_patched = True
        if not globals_patched:
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

    # —— the public compile surface ———

    def compile_for_test(self, *args: Any) -> ir.Program:
        """Build the program with the call-time specialization implied by
        *args* and run the Default pass pipeline — no codegen, no device.
        Returns the post-pass ``ir.Program`` for structural-equality tests."""
        from pypto.ir.pass_manager import OptimizationStrategy, PassManager  # noqa: PLC0415

        bridged, _ = self._bridge_args(args)
        program = self._build_program(bridged)
        return PassManager.get_strategy(OptimizationStrategy.Default).run_passes(program)

    def compile(self, *args: Any, config: RunConfig | None = None, **kwargs: Any) -> Any:
        """Build + compile for the signature implied by *args*; returns the
        ``CompiledProgram`` (device dispatch is its ``__call__``).

        For an eagerly traced kernel (annotation-determined), *args* carry
        no specialization power: the eager trace's program — with its
        declared dynamic extents — is the artifact, and the args only need
        to be compatible with the declaration (the golden runner's dummy
        tensors satisfy this by construction). A static extent is the
        declaration, not a specialization hint: a call at a different shape is
        refused unless the parameter declares a dynamic extent.

        Name-keyed *kwargs* are the annotation-driven scalar vocabulary (the
        golden runner's ``compile_runtime`` form: ``compile(config=...,
        layer_id=5, moe_epoch=pl.RUNTIME)``). The declaration owns every
        parameter's world, so no compile-time input re-specializes:
        ``pl.RUNTIME`` on a runtime-declared (``pl.Scalar``) parameter is a
        benign no-op — it is one already — and a host literal is accepted
        and ignored (the value arrives at dispatch; the artifact is
        identical either way). A ``pl.runtime(v, dtype)`` marker is likewise
        a no-op. A keyword naming a host-knob parameter, an unknown name, or
        a non-scalar value is loud.

        ``config=RunConfig(...)`` is consumed here — never forwarded to the
        kernel — and its compile-side fields (``strategy``, ``dump_passes``,
        diagnostics, ...) are forwarded to ``ir.compile()`` so the returned
        artifact honours the same knobs as ``kernel(*args, config=...)``.
        Runtime-side fields (``device_id``, DFX flags, ...) do not apply: they
        affect dispatch, not the compiled artifact.

        Each call compiles a fresh artifact (no cache), so an explicit
        compile-then-call freezes exactly one materialization. The returned
        handle's calling signature is the **runtime ABI** — host knobs were
        consumed at compile and are no longer parameters, so passing a value
        at their position is an arity error naming the consumed knob, never a
        silent freeze.
        """
        config = _checked_run_config(config)
        self._check_compile_kwargs(kwargs)
        bridged, _ = self._bridge_args(args)
        compiled = _compile_program(self._build_program(bridged), config)
        runtime_names = tuple(
            name for name in self._param_names if _staged_declared(self._param_hints.get(name))
        )
        knob_names = tuple(name for name in self._param_names if name not in runtime_names)
        return _CompiledImpl(compiled, runtime_names, knob_names, self.kernel.__name__)

    def specialize(self, *args: Any, **kwargs: Any) -> ir.Program:
        """Specialize and return the **pre-pass** ``ir.Program``.

        The mirror of ``JITFunction.specialize``: bridge the sample
        arguments, build the entry and every callee its body reached, and
        return the program before any pass has run. ``config=`` is rejected
        — no pass runs here to read it."""
        if "config" in kwargs:
            raise TypeError(
                f"@pl.jit function '{self.__name__}': specialize() does not accept config=. "
                "No pass runs here, so a RunConfig would have nothing to configure. Use "
                "lower(config=...) for post-pass IR, or compile(config=...) to build an artifact."
            )
        bridged, _ = self._bridge_args(self._bind_sample_args(args, kwargs))
        return self._build_program(bridged)

    def lower(self, *args: Any, config: RunConfig | None = None, **kwargs: Any) -> ir.Program:
        """Specialize and return the **post-pass** ``ir.Program``.

        The mirror of ``JITFunction.lower``: build the pre-pass program and
        run the configured pass pipeline — no codegen, no device. ``config=``
        is a ``RunConfig`` whose pass-side fields (strategy, diagnostics,
        planner, platform) select the pipeline; runtime/artifact fields are
        ignored. Without ``config`` the Default strategy runs."""
        from pypto.ir.compile import _run_pass_pipeline  # noqa: PLC0415 — cycle-free laziness

        checked = _checked_run_config(config)
        bridged, _ = self._bridge_args(self._bind_sample_args(args, kwargs))
        program = self._build_program(bridged)
        lower_kwargs: dict[str, Any] = {}
        platform = None
        if checked is not None:
            platform = checked.platform
            lower_kwargs = {
                "strategy": checked.strategy,
                "diagnostic_phase": checked.diagnostic_phase,
                "disabled_diagnostics": checked.disabled_diagnostics,
                "analyze_auto_scopes_for_deps": checked.analyze_auto_scopes_for_deps,
            }
            if checked.memory_planner is not None:
                lower_kwargs["memory_planner"] = checked.memory_planner
        return _run_pass_pipeline(
            program,
            operation="lower",
            platform=platform,
            inherit_outer_report_instruments=False,
            **lower_kwargs,
        ).transformed_program

    def warmup(self, *args: Any, config: RunConfig | None = None, **kwargs: Any) -> Any:
        """Compile and prepare the device binaries without dispatching.

        The mirror of ``JITFunction.warmup``: compile under the same
        arguments/config (no cache read or write), then load every
        chip-level binary the compiled handle exposes. Returns the compiled
        handle; execution stays a separate operation on it."""
        from pypto.ir.compiled_program import CompiledProgram  # noqa: PLC0415
        from pypto.ir.distributed_compiled_program import DistributedCompiledProgram  # noqa: PLC0415

        compiled = self.compile(*args, config=config, **kwargs)
        if isinstance(compiled, DistributedCompiledProgram):
            from pypto.runtime.distributed_runner import _assemble_chip_callables  # noqa: PLC0415

            _assemble_chip_callables(compiled)
        elif isinstance(compiled, CompiledProgram):
            if compiled.orchestration_names:
                for name in compiled.orchestration_names:
                    compiled[name].load()
            else:
                compiled.load()
        else:
            raise TypeError(
                f"@pl.jit function '{self.__name__}': device-free warmup is not supported "
                f"for compiled result {type(compiled).__name__}"
            )
        return compiled

    def _bind_sample_args(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
        """Bind positional + by-name sample arguments to signature order.

        The ``JITFunction`` sample-binding contract for the mirror lanes
        (``specialize``/``lower``): unknown names, duplicates, and missing
        parameters are loud, and extra positional arguments are rejected
        rather than silently truncated (``_bridge_args`` zips). The
        no-argument form is the annotation-driven signature mode — valid
        only when the declaration determined the trace (the eager record),
        which is exactly the ``JITFunction`` rule."""
        if not args and not kwargs:
            if self._eager_record is None:
                raise TypeError(
                    f"call to {self.__name__!r}: sample arguments are required — the kernel "
                    "is lazily traced (an unannotated or bare parameter), so the trace "
                    "depends on the call-time argument types"
                )
            return ()
        if len(args) > len(self._param_names):
            raise TypeError(
                f"call to {self.__name__!r} expects at most {len(self._param_names)} "
                f"argument(s), got {len(args)}"
            )
        bound: dict[str, Any] = dict(zip(self._param_names, args))
        for name, value in kwargs.items():
            if name not in self._param_names:
                raise TypeError(f"call to {self.__name__!r}: no parameter {name!r}")
            if name in bound:
                raise TypeError(
                    f"call to {self.__name__!r}: parameter {name!r} got both a "
                    "positional and a keyword sample argument"
                )
            bound[name] = value
        missing = [name for name in self._param_names if name not in bound]
        if missing:
            raise TypeError(f"call to {self.__name__!r}: no sample argument for parameter(s) {missing}")
        return tuple(bound[name] for name in self._param_names)

    def _check_compile_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate the name-keyed scalar vocabulary against the declaration.

        Every parameter's world was pinned at decoration, so the check can
        only confirm that each keyword speaks the declaration's language —
        it can never retype or re-specialize (the specializer's literal-
        specialization is deliberately not adopted: a ``pl.Scalar``
        parameter is runtime by declaration and dispatch supplies the
        value, so the artifact never depends on the keyword's value).
        """
        for name, value in kwargs.items():
            if name not in self._param_names:
                raise TypeError(f"compile(): {self.kernel.__name__!r} has no parameter {name!r}")
            if not _staged_declared(self._param_hints.get(name)):
                raise TypeError(
                    f"compile(): parameter {name!r} of {self.kernel.__name__!r} is a "
                    "host knob by declaration — it is baked at trace time from the "
                    "call arguments, not from compile keywords; declare it "
                    "pl.Scalar[dtype] for a runtime parameter"
                )
            if value is RUNTIME:
                warnings.warn(
                    "pl.RUNTIME is deprecated (dispatch follows declarations) — "
                    "a pl.Scalar parameter is runtime by declaration; the keyword is a no-op",
                    DeprecationWarning,
                    stacklevel=3,
                )
            if value is RUNTIME or isinstance(value, (RuntimeScalar, bool, int, float)):
                # pl.RUNTIME: already runtime by declaration — a no-op. A
                # host literal or a pl.runtime marker: accepted and ignored —
                # the declaration pinned the world and dispatch supplies the
                # value, so the artifact is identical either way.
                continue
            raise TypeError(
                f"compile(): keyword {name!r} of {self.kernel.__name__!r} must be "
                "pl.RUNTIME, pl.runtime(value, dtype), or a host scalar — the "
                "name-keyed vocabulary carries scalar values only (tensors go "
                "positionally)"
            )


def _annotation_shaped(arg: Any) -> bool:
    """Is ``arg`` an annotation-mode facade (shape/extent + dtype present,
    the author's own type statement), as opposed to an expr-wrapping facade
    carrying a value from inside another trace? The mode check leads: an
    expr-wrapping ``Tensor`` can derive a ``.shape`` (the lazily lowered
    read), so its attributes alone no longer answer the question — its
    ``_expr``/``expr`` does. A Ptr has no shape — always annotation-ok."""
    if getattr(arg, "_expr", None) is not None or getattr(arg, "expr", None) is not None:
        return False
    return (
        getattr(arg, "shape", None) is not None
        or getattr(arg, "extent", None) is not None
        or getattr(arg, "dtype", None) is not None
    )


def _is_host_scalar(value: Any) -> bool:
    """A bake-able host value (the ops reify it as ``Const`` at the
    op-arg boundary). Facades are excluded — they take the minted-param
    lane, checked separately."""
    return isinstance(value, (bool, int, float, ir.Expr))


# —— the torch bridge (optional dependency; loaded lazily) ——

_TORCH_CACHE: list[Any] = []
# [] = not probed, [None] = torch unavailable, [torch] = loaded


def _torch() -> Any:
    if not _TORCH_CACHE:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            _TORCH_CACHE.append(None)
        else:
            _TORCH_CACHE.append(torch)
    return _TORCH_CACHE[0]


def _torch_facade(arg: Any, declared: Any = None) -> Tensor | None:
    """A ``torch.Tensor`` → an annotation-mode facade (shape + dtype), or
    ``None`` when ``arg`` is not one. The specializer's TensorMeta
    extraction, re-skinned as the tracing annotation.

    Torch has no INDEX dtype; the host ABI carries an INDEX tensor as
    ``torch.int64`` (``pypto.ir.param_info`` maps both to int64). The
    declaration therefore decides the shape's world: an INT64 sample into a
    declared INDEX slot bridges to INDEX, while an explicit ``pl.INT64``
    facade stays INT64 and is refused. Without this, every host call of an
    index-typed tensor parameter failed on a storage detail rather than a
    type statement.

    The declared **layout** bridges the same way, and for the same reason: a
    torch tensor carries no layout, so a declaration that says ``pl.NZ`` is the
    only statement about how the buffer is packed. Deriving the layout from the
    sample instead made every host call of a layout-declared parameter fail with
    "a different tensor layout" -- a storage fact the sample cannot express, not
    a type the caller contradicted."""
    torch = _torch()
    if torch is None or not isinstance(arg, torch.Tensor):
        return None
    if not _TORCH_DTYPES:
        _TORCH_DTYPES.update(_load_torch_dtypes())
    dtype = _TORCH_DTYPES.get(arg.dtype)
    if dtype is None:
        raise TypeError(f"jit argument of torch dtype {arg.dtype!r} has no PyPTO DataType mapping")
    from pypto.pypto_core import DataType  # noqa: PLC0415 — only this bridge needs the enum

    # A bare annotation (``pl.Array``, a dtype-less slot) states no dtype, and
    # comparing its ``None`` against the enum raises inside nanobind instead of
    # answering: guard the lookup before the INDEX promotion.
    declared_dtype = getattr(declared, "dtype", None)
    if dtype == DataType.INT64 and declared_dtype is not None and declared_dtype == DataType.INDEX:
        dtype = DataType.INDEX
    return Tensor(
        list(arg.shape),
        dtype,
        layout=getattr(declared, "layout", None),
        _annotation_only=True,
    )


def _load_torch_dtypes() -> dict[Any, Any]:
    torch = _torch()
    if torch is None:
        return {}
    from pypto.pypto_core import DataType  # noqa: PLC0415

    table = {
        torch.float16: DataType.FP16,
        torch.float32: DataType.FP32,
        torch.bfloat16: DataType.BF16,
        torch.int8: DataType.INT8,
        torch.int16: DataType.INT16,
        torch.int32: DataType.INT32,
        torch.int64: DataType.INT64,
        torch.uint8: DataType.UINT8,
        torch.bool: DataType.BOOL,
    }
    # Optional low-precision MX dtypes (PyTorch 2.1+/2.3+/2.7+) — the same
    # guarded table the parsing-frontend jit maps (``pypto/jit/decorator.py``);
    # uint8 maps to UINT8 there too, and the bridge must preserve that behavior.
    for _torch_name, _pto_dt in (
        ("float8_e4m3fn", DataType.FP8E4M3FN),
        ("float8_e5m2", DataType.FP8E5M2),
        ("float8_e8m0fnu", DataType.FP8E8M0),
        ("float4_e2m1fn_x2", DataType.FP4E2M1X2),
    ):
        _td = getattr(torch, _torch_name, None)
        if _td is not None:
            table[_td] = _pto_dt
    return table


_TORCH_DTYPES: dict[Any, Any] = {}


def _in_trace_annotation(name: str, arg: Any, kernel_name: str, hint: Any = None) -> Any:
    """Bridge one call-site argument (inside a trace) to the callee's
    annotation: a runtime facade's wrapped expr carries the full type
    (shape + dtype + layout), which becomes the static-shape annotation the
    callee trace specializes with — no retype at the call boundary. Host
    scalars pass through.

    A host value beyond the scalar trio (a ``str``, a ``DataType``, an enum
    member, a list of those) crosses only into a **host-knob** parameter —
    one whose declaration is not a staged facade (``pl.constexpr``, a plain
    annotation, none). The declaration owns the parameter's world, so the
    value folds by the ``pl.constexpr`` contract
    (:func:`pypto.language.typing.constexpr.constexpr_value_key`) and never
    reaches a runtime slot, where it stays loud instead."""
    if is_staged(arg):
        ty = arg.unwrap().type
        try:
            # The facade is built from the value's own IR type, so every field
            # it carries — kind, layout/view, storage, tile memory space —
            # crosses; the kind order and field plumbing live in one place,
            # shared with the parser-lane import bridge.
            return facade_from_ir_type(ty)
        except TypeError:
            raise TypeError(
                f"call to {kernel_name!r} inside a trace: argument {name!r} has "
                f"type {ty}, which has no annotation form (Tensor/DistributedTensor/"
                "Tile/Scalar/Array only)"
            ) from None
    if _is_host_scalar(arg):
        return arg
    if hint is not None and not _staged_declared(hint):
        if constexpr_value_key(arg) is not None:
            return arg
        raise TypeError(
            f"call to {kernel_name!r} inside a trace: constexpr parameter {name!r} "
            f"got {arg!r}, which has no form the tracer can fold. A 'pl.constexpr' "
            "value must be an int, float, bool, str, a pl dtype, an enum member "
            "such as pl.Mem.Vec, or a list of those."
        )
    raise TypeError(
        f"call to {kernel_name!r} inside a trace: argument {name!r} "
        f"({type(arg).__name__}) is not a facade or host scalar"
    )


# —— the program assembly context (one entry build) ——


class _ProgramAssembly:
    """The callees collected during one entry build: appended in first-use
    order, deduped on object identity (adapted functions) and on the
    variant key (kernel identity + the wrapper's entry kind + derived
    annotations + baked host values
    — a helper called twice with the same shapes and constants stays one
    function; the specializer's ScalarCacheInfo semantics, one tier below
    the compiled-artifact cache)."""

    def __init__(self) -> None:
        self.functions: list[ir.Function] = []
        self.by_key: dict[Any, ir.Function] = {}
        # Variant keys whose trace is currently running, in start order — a
        # variant registers only after its trace finishes, so a kernel that
        # (transitively) calls itself misses the ``by_key`` lookup on every
        # level; the running key is held here so the re-entry is refused as a
        # cycle instead of re-tracing to the interpreter's recursion limit.
        self._in_flight: dict[Any, str] = {}
        self._ids: set[int] = set()
        self._name_counts: dict[str, int] = {}
        # Every name already spoken for: a registered function's, or one a
        # composition frontend reserved. Suffix candidates are checked against
        # this set, so a minted variant can never land on a taken name.
        self._taken: set[str] = set()

    def enter_variant_trace(self, key: Any, name: str) -> None:
        """Mark one variant trace as running, refusing a re-entrant one.

        A hit means the very variant being traced is reachable from its own
        body (directly or through other traced kernels): it has no finite
        traced form, so the refusal names the cycle — outermost first, closed
        back onto the re-entered kernel — rather than letting the tracer
        re-descend until the recursion limit.
        """
        if key in self._in_flight:
            keys = list(self._in_flight)
            cycle = [self._in_flight[k] for k in keys[keys.index(key) :]]
            cycle.append(name)
            raise TypeError(
                f"traced kernel cycle: {' -> '.join(repr(n) for n in cycle)}. A traced "
                "kernel cannot call itself, directly or through other traced "
                "kernels — the variant is still being traced, so the call cannot "
                "reuse it, and re-tracing never terminates. Unroll the recursion "
                "or restructure the cycle into one body."
            )
        self._in_flight[key] = name

    def exit_variant_trace(self, key: Any) -> None:
        self._in_flight.pop(key, None)

    def register(self, fn: ir.Function, key: Any | None = None) -> None:
        if id(fn) in self._ids:
            return
        self._ids.add(id(fn))
        # Reserve the base name: a frozen function (a Submit reference, an
        # eager capture) and a later per-call variant of the same kernel are
        # distinct functions and must not collide on ir.Program's
        # program-unique name check. unique_name suffixes on this count.
        self._name_counts[fn.name] = max(self._name_counts.get(fn.name, 0), 1)
        self._taken.add(fn.name)
        self.functions.append(fn)
        if key is not None:
            self.by_key[key] = fn

    def unique_name(self, base: str) -> str:
        """Mint a program-unique name for one variant of ``base``: the first
        variant keeps the kernel's name (the caller's Call and the
        eager-trace name agree), each further distinct variant gets a
        numeric suffix — one helper instantiated at multiple shapes never
        produces duplicate function names in one program.

        A candidate that is already taken is skipped, not returned: the suffix
        sequence (``base_2``, ``base_3``, …) can itself collide with a name the
        program already holds — an entry method called ``leaf_2`` beside a
        traced helper ``leaf`` — and handing that name out made the parser
        reject the whole program as a duplicate function.
        """
        count = self._name_counts.get(base, 0)
        candidate = base if count == 0 else f"{base}_{count + 1}"
        while candidate in self._taken:
            count += 1
            candidate = f"{base}_{count + 1}"
        self._name_counts[base] = count + 1
        self._taken.add(candidate)
        return candidate

    def reserve(self, names: Iterable[str]) -> None:
        """Take ``names`` out of circulation before any function is registered.

        A composition frontend may add functions to a program that already has
        names of its own (a parser program's generated methods). Reserving them
        makes the first variant of a same-named helper take the suffixed form
        instead of colliding with the name that is already spoken for.
        """
        for name in names:
            self._name_counts[name] = max(self._name_counts.get(name, 0), 1)
            self._taken.add(name)


_ACTIVE_ASSEMBLY: ContextVar[_ProgramAssembly | None] = ContextVar("pto_jit_assembly", default=None)


# —— the eager-trace record: what per-call specialization needs ———


class _EagerTraceRecord:
    """The eager trace's facts for one annotation-determined jit kernel.

    The eager jit object replaces the module global at decoration; this
    record (keyed by the eager trace's ``ir.Function``) retains what that
    object serves its lanes from: the trace (``fn``), the kernel (the
    re-trace source), the assembly (the callees the eager trace reached),
    and the lazily built specializer."""

    __slots__ = ("fn", "kernel", "localns", "assembly", "_specializer")

    def __init__(
        self,
        fn: ir.Function,
        kernel: types.FunctionType,
        localns: dict[str, object] | None,
        assembly: _ProgramAssembly,
    ) -> None:
        self.fn = fn
        self.kernel = kernel
        self.localns = localns
        self.assembly = assembly
        self._specializer: _JitSpecializer | None = None

    def specializer(self) -> _JitSpecializer:
        """The jit object for the recorded kernel, rebuilt from the
        declaration's classification facts. Carries the record so the host
        compile/dispatch lanes serve the eager trace's program; the
        per-call variant lane re-traces through the kernel as before."""
        if self._specializer is None:
            self._specializer = _JitSpecializer(
                self.kernel,
                function_type=self.fn.func_type,
                level=self.fn.level,
                role=self.fn.role,
                attrs=dict(self.fn.attrs) or None,
                localns=self.localns,
                eager_record=self,
            )
        return self._specializer


_EAGER_TRACES: dict[ir.Function, _EagerTraceRecord] = {}


def parser_aligned_callee(value: Any) -> ir.Function | None:
    """The frozen function a parser-aligned lane (the standalone
    ``pl.function`` build, the program methods) should see for *value*: an
    ``ir.Function`` passes through; an eagerly traced jit object contributes
    its eager trace, so the lane sees the frozen ``ir.Function`` rather than
    the jit envelope. Without this, the jit object's own ``__call__`` would
    leak the jit envelope's world checks into a lane the conformance oracle
    pins as parser-aligned. Anything else (a lazy jit, a plain callable) is
    None — not this lane's callee.

    Internal adapter contract, not public API: the collection adapter in
    ``tracing.decorators.function`` calls it across the module boundary, so
    it is this module's cross-file surface rather than a file-local helper."""
    if isinstance(value, ir.Function):
        return value
    if isinstance(value, _JitSpecializer) and value._eager_record is not None:
        return value._eager_record.fn
    return None


def _eager_trace_record(fn: ir.Function) -> _EagerTraceRecord | None:
    """The eager record for an annotation-determined jit kernel's
    ``ir.Function`` — None for every other function object (parser-produced,
    ``pl.function``-traced, extern members, or a plain ``ir.Function`` a
    module defines by hand)."""
    return _EAGER_TRACES.get(fn)


def _variant_key_part(annotation: Any) -> Any:
    """One annotation's variant-key component: a bake-able host scalar keys
    on its VALUE (a baked constant must not dedupe with a different
    constant), every staged facade keys on its primitive identity — kind,
    dtype, extent strings — the same fields the dispatch cache keys on
    (``_cache_key``), fetched straight off the facade without building or
    printing an ``ir.Type`` (the format callback otherwise runs a ruff
    subprocess per printed type, and the C++ structural hash does not cover
    every Type kind — e.g. ``DistributedTensorType``).

    A runtime scalar keys on its dtype only, never its value — one variant
    serves every value, the whole point of the spelling. A raw staged
    expression (``ir.Expr``) keys on its structural hash: free variables
    keep their identity there, so two distinct same-named Vars (two
    callers' locals — the printer prints both as their shared ``name_hint``)
    never collapse into one key, while same-value constants still dedupe.
    The raw printed text is NOT that key: printing an ``ir.Expr`` root goes
    through ``GetVarName`` without the disambiguating rename map a function
    or bare-stmt root builds, so raw text equality overstates structural
    equality whenever two distinct Vars share a name.

    A shape extent is NOT keyable by its printed text: two distinct
    caller-trace locals spelled the same (every attention helper's
    ``kv_dim = pl.tensor.dim(x, 0)`` feeding ``pl.create_tensor``) print
    identically, and a text-keyed dedupe handed the later callers the first
    caller's variant — whose signature embeds the first caller's ``kv_dim``
    Var. The inline splice then renamed that Var into every caller's branch
    and ConvertToSSA rejected the body with ``used outside its defining
    scope``. So extents key on identity, not text: constants by value, a
    ``pl.dynamic`` symbol by its underlying Var (same name → same object,
    the shared-by-name registry — deliberate sharing survives), every other
    Var / arithmetic expression by object identity (``unique_id`` / ``id``):
    a variant baked against one caller's extent Var is only reusable at
    call sites passing that same object."""
    if isinstance(annotation, (bool, int, float)):
        return ("host", type(annotation).__name__, annotation)
    if isinstance(annotation, ir.Expr):
        # Binding-aware, subprocess-free: ``structural_hash`` alpha-renames
        # bound variables and keys free variables on identity, so two
        # distinct Vars sharing a ``name_hint`` hash differently (the
        # variant they bake embeds their caller's Var — the same hazard the
        # extent lane keys on identity for), while structurally equal
        # constants still share one key. Attr payloads with no structural
        # encoding fall back to object identity: that can under-dedupe
        # (duplicate variants, still correct) but never hands one caller
        # another caller's variant.
        try:
            return ("expr", ir.structural_hash(annotation))
        except ir.StructuralComparisonUnsupported:
            return ("expr-id", id(annotation))
    folded = constexpr_value_key(annotation)
    if folded is not None:
        # A baked ``pl.constexpr`` value beyond the scalar trio — a ``str``
        # name hint, a ``DataType``, an enum member, a list of those — keys on
        # the same canonical identity the foldability contract defines, so
        # two calls baking different constants never share a variant.
        return ("host", *folded)
    dtype = getattr(annotation, "dtype", None)
    if dtype is not None:
        shape = getattr(annotation, "shape", None)
        extents = tuple(_extent_key_part(e) for e in shape) if shape is not None else ()
        return (
            "facade",
            type(annotation).__name__,
            dtype,
            extents,
            _layout_key_part(annotation),
            _memref_key_part(annotation),
        )
    from pypto.language.typing import annotation_to_type  # noqa: PLC0415 - cycle-free laziness

    with raw_print():
        return ("raw", str(annotation_to_type(annotation)))


_LAYOUT_KEY_FIELDS = (
    "layout",
    "blayout",
    "slayout",
    "fractal",
    "compact",
    "stride",
    "valid_shape",
    "start_offset",
    "pad",
)


def _variant_key(
    kernel: types.FunctionType,
    annotations: dict[str, Any],
    function_type: ir.FunctionType,
    level: ir.Level | None,
    role: ir.Role | None,
    attrs: Any,
) -> tuple[Any, ...]:
    """One callee wrapper's assembly dedupe key: the kernel, the wrapper's
    entry kind, and the per-call annotation parts — each part paired with
    its parameter name, sorted by name.

    The name pairing is load-bearing: the lanes build the annotation map
    in their own order — the callee lanes insert declared order, the
    parse-time import bridge inserts the runtime parameters then the
    sorted baked ones — so an interspersed baked/staged helper would enter
    under different keys from different lanes, defeating both the dedupe
    (duplicate variants) and the in-flight cycle guard (a cross-lane cycle
    misses and re-traces). Sorted by name, the key is the same whichever
    lane computed it.

    Two wrappers around one Python function — ``@pl.jit_trace`` and
    ``@pl.jit_trace.host`` on the same kernel — are different callees: their
    variants carry different function kinds and levels. The kind components
    join the key, or the second wrapper's call reuses the first wrapper's
    variant and the traced program hands a HOST-orchestration spelling an
    Orchestration body.
    """
    return (
        kernel,
        function_type,
        level,
        role,
        _attrs_key_part(attrs),
        *tuple(sorted((name, _variant_key_part(a)) for name, a in annotations.items())),
    )


def _attrs_key_part(attrs: Any) -> Any:
    """The wrapper's attrs identity for the variant key (``auto_scope`` is
    the tracing envelope's own knob). Hashable values fold in by content;
    a payload with no hashable form keys on object identity — under-dedupe
    (duplicate variants, still correct) rather than a silent collision.
    The candidate key must prove itself hashable before it is returned:
    a value that sorts but does not hash (``{"payload": []}`` — the items
    sort by their unique string keys, the list raises at hash time) would
    otherwise blow up at the first dict lookup, past this fallback.
    """
    if not attrs:
        return None
    try:
        key = tuple(sorted(attrs.items()))
        hash(key)
        return key
    except TypeError:
        return ("attrs-id", id(attrs))


def _memref_key_part(annotation: Any) -> Any:
    """The annotation's explicit storage, for the variant key.

    Kind, dtype, extents and layout do not decide a callee's parameter type
    on their own: ``annotation_to_type`` carries the facade's ``memref`` into
    the IR type, so a call site passing bound storage bakes that storage
    into the variant's signature. Keying without it handed a call site with
    different declared storage the first site's variant.

    Storage identity, not wrapper identity: two distinct MemRef objects
    describe the same storage when they share the allocation base and
    select the same window of it — byte offset, size and slot — (a
    re-subscripted ``slots=N`` declaration mints a fresh object per
    selection), so they key equal; keying on the wrapper's own identity
    over-split those into duplicate variants. The base is the allocation
    identity token even before ``InitMemRef`` resolves the address; where
    no base exists the storage is unresolved and object identity is the
    only fact left — a cache key may over-split (duplicate variants, still
    correct) but never collide.
    """
    memref = getattr(annotation, "memref", None)
    if memref is None:
        return None
    base = getattr(memref, "base_", None)
    if base is None:
        return ("memref", memref.unique_id)
    return (
        "memref",
        base.unique_id,
        _extent_key_part(memref.byte_offset_),
        memref.size_,
        None if memref.slot_index_ is None else _extent_key_part(memref.slot_index_),
    )


def _layout_key_part(annotation: Any) -> Any:
    """The annotation's storage-layout identity, for the variant key.

    Shape and dtype do not decide a callee's parameter type on their own: an
    ND buffer and an NZ one of the same extent produce different
    ``TensorType``/``TileType`` parameters, and the callee is traced from the
    first call site. Keying without this handed the second call the first
    call's variant, which the TypeChecked verifier then rejected
    ("parameter 'x' is declared ND but the argument is NZ").

    Reads only the fields each view exposes — a tensor carries ``layout``
    (sometimes the whole ``TensorView``), a tile its ``memory_space`` and
    ``tile_view`` — so an argument with no view keys as ``(None,)`` exactly.
    """
    view = getattr(annotation, "layout", None)
    if view is None:
        view = getattr(annotation, "tile_view", None)
    memory_space = getattr(annotation, "memory_space", None)
    if view is None:
        return (memory_space,)
    fields = tuple(
        (name, _view_field_key(getattr(view, name))) for name in _LAYOUT_KEY_FIELDS if hasattr(view, name)
    )
    # A third-slot annotation carries the bare layout enum; a bridged argument
    # carries the full view. Both are structural facts about the parameter.
    return (memory_space, view) if not fields else (memory_space, fields)


def _view_field_key(value: Any) -> Any:
    """One view field for the variant key: extent sequences by identity/value."""
    if isinstance(value, (tuple, list)):
        elements = typing.cast("Sequence[object]", value)
        return tuple(_extent_key_part(item) for item in elements)
    return value


def _extent_key_part(extent: Any) -> Any:
    """One shape extent's variant-key component (see ``_variant_key_part``).

    Annotation-mode extents arrive as plain ints, ``ir.ConstInt`` (value),
    ``DynVar`` wrappers / ``ir.Var`` (extent symbols and caller-trace
    locals), or arithmetic ``ir.Expr`` over them; runtime facades carry no
    shape at all (``Tensor.shape`` is ``None`` in expr mode) and never reach
    this helper.
    """
    if isinstance(extent, bool):
        return ("const", "bool", extent)
    if isinstance(extent, int):
        return ("const", "int", extent)
    if isinstance(extent, ir.ConstInt):
        return ("const", "ConstInt", extent.value)
    var = extent.unwrap() if isinstance(extent, DynVar) else extent
    if isinstance(var, ir.Var):
        # ``unique_id`` is the sanctioned same-object key (Var's binding):
        # same-name pl.dynamic symbols unwrap to one Var (shared by design),
        # while two callers' same-named locals keep distinct ids.
        return ("var", var.unique_id)
    if isinstance(var, ir.Expr):
        # A compound extent (e.g. ``(t + 15) // 16 * 16``) embeds its
        # caller's Vars; only the same expression object is the same extent.
        return ("expr", id(var))
    with raw_print():
        return ("raw", str(extent))


def _staged_declared(hint: Any) -> bool:
    """Whether a parameter annotation declares the staged (runtime) world.

    A subscripted pl annotation (``pl.Tensor[[8], pl.FP32]``, ``pl.Out[T]``,
    ``pld.DistributedTensor[[...], dt]``) is already an annotation-mode facade
    instance; a bare facade class (``pl.Tensor``, ``pl.Scalar`` — the lazy
    entries' deferred-shape spelling) is staged with the concrete type left
    to the call site. Everything else — plain Python types, no annotation —
    is the host world.
    """
    if hint is None:
        return False
    if is_staged(hint):
        return True
    if isinstance(hint, type):
        from pypto.language.facade import FACADE_TYPES  # noqa: PLC0415 — import cycle
        from pypto.language.typing.tensor import Tensor  # noqa: PLC0415 — import cycle

        if issubclass(hint, Tensor):  # covers the pybind DistributedTensor subclass
            return True
        # FACADE_TYPES is derived from the Facade union (all classes), so
        # every member is a type — no isinstance guard needed.
        return any(issubclass(hint, t) for t in FACADE_TYPES)
    return False


def _check_param_worlds(
    param_hints: dict[str, Any],
    param_names: Sequence[str],
    args: tuple[Any, ...] | list[Any],
    fn_name: str,
) -> None:
    """The post-lift residue guard: dispatch follows declarations.

    A parameter declared with a pl facade annotation is a **runtime
    parameter**; a plain Python annotation (or none) is a **host knob**
    (baked at trace time, never IR). Bare host scalars lift into
    ``Scalar``-declared slots before this guard runs
    (``_lift_runtime_scalar_args``), so what remains loud here is the
    residue: a host scalar into a *tensor/array-shaped* runtime slot
    (nothing can lift it), and a staged value into a host-knob parameter
    (a staged value cannot become host — the no-context-retyping rule
    applied to call arguments).
    """
    for name, arg in zip(param_names, args):
        hint = param_hints.get(name)
        staged_declared = _staged_declared(hint)
        if staged_declared and isinstance(arg, (bool, int, float)):
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared {hint!r} "
                "(a runtime parameter) but received a host "
                f"{type(arg).__name__} — a host scalar lifts only into a "
                "Scalar-declared slot; pass a facade (or a torch tensor at a "
                "host call)"
            )
        if staged_declared and constexpr_value_key(arg) is not None:
            # The foldable-constexpr value set minus what the branch above
            # already lifted or rejected: a str, a DataType, an enum member,
            # a list — the ``pl.constexpr`` contract's non-scalar members,
            # which can never become a runtime argument.
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared {hint!r} "
                "(a runtime parameter) but received a host "
                f"{type(arg).__name__} — only a 'pl.constexpr' parameter "
                "bakes a host value; pass a facade (or a torch tensor at a "
                "host call)"
            )
        if not staged_declared and is_staged(arg):
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared "
                f"{hint!r} (a host value baked at trace time) but received a "
                "staged value — a staged value cannot become host; annotate "
                "the parameter (e.g. pl.Scalar[dtype]) to make it runtime"
            )


def _lift_runtime_scalar_args(
    param_hints: dict[str, Any],
    param_names: Sequence[str],
    args: tuple[Any, ...],
    fn_name: str,
    *,
    in_trace: bool,
) -> tuple[Any, ...]:
    """Call-argument lifting: a bare host scalar crossing into a
    ``Scalar``-declared runtime slot lifts at the call-argument boundary —
    to a typed ``Const`` argument inside a trace, to the runtime-scalar
    dispatch marker on the host (dtype-keyed, value delivered at dispatch).

    The declaration owns the parameter's world, so the call site carries
    no marker. The scalar's *category* is validated against the declared
    dtype before the lane split (``_require_representable``), so a trace
    and a host call answer the same input identically. Tensor/array-shaped
    slots stay loud (nothing lifts a host scalar into them —
    ``_check_param_worlds``), and plain-declared parameters keep baking by
    value exactly as before.
    """
    from pypto.language.typing.const import const  # noqa: PLC0415 — cycle-free laziness

    lifted: list[Any] = []
    for name, arg in zip(param_names, args):
        hint = param_hints.get(name)
        if isinstance(arg, (bool, int, float)) and _scalar_declared(hint):
            dtype = getattr(hint, "dtype", None)
            _require_representable(arg, dtype, name, fn_name)
            if in_trace:
                lifted.append(const(arg, dtype or _host_scalar_dtype(arg)))
            else:
                lifted.append(RuntimeScalar(arg, dtype))
        else:
            lifted.append(arg)
    return tuple(lifted)


def _require_representable(value: bool | int | float, dtype: Any, name: str, fn_name: str) -> None:
    """Refuse a Python scalar the declared scalar slot cannot hold as-is.

    A Python scalar states a *category*, not a dtype, and the declaration is
    the only width statement in play, so each category is admitted only where
    the slot can hold it without silent reinterpretation: a ``bool`` only into
    ``BOOL`` (the same rule ``const()`` applies to a boolean constant), a
    ``float`` only into a float slot (a float states no integer width, so an
    integer slot could only hold it by truncation), and an ``int`` anywhere
    the declared bounds can hold it — the range check below. A *typed* value
    goes the other way — it is converted to the declared dtype by the admitted
    integer lattice, wrapping included — because the author already stated
    its width. The asymmetry is deliberate: width is the author's statement or
    it is nobody's.

    The verdict is computed before the lane split (``const()`` in a trace,
    the ``RuntimeScalar`` marker on the host), so both lanes answer the same
    input identically and a refusal names the parameter and the kernel rather
    than ``const()``'s raw message.

    This runs on every dispatch, not only when an artifact is built: a scalar
    value never splits the compiled artifact, so a warm cache hit is still a
    placement of a new value in the declared slot.
    """
    from pypto.pypto_core import DataType  # noqa: PLC0415 — keep the toplevel import surface unchanged

    if dtype is None:
        # A bare ``pl.Scalar`` slot promotes by the host defaults
        # (``_host_scalar_dtype``), which match the category by construction.
        return
    if isinstance(value, bool):
        if dtype != DataType.BOOL:
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared "
                f"pl.Scalar[pl.{str(dtype).upper()}] but received a host bool — "
                "a Python bool is a logical value and lifts only into "
                "pl.Scalar[pl.BOOL]; pass an int, or declare the slot "
                "pl.Scalar[pl.BOOL]."
            )
        return
    if isinstance(value, float):
        if not dtype.is_float():
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared "
                f"pl.Scalar[pl.{str(dtype).upper()}] but received a host float — "
                "a Python float states no integer width, so only a float-declared "
                "slot can hold it; pass an int in range, or spell the intended "
                "conversion with pl.const(value, dtype) and let the declared slot "
                "convert it."
            )
        return
    bounds = _dtype_bounds(dtype, DataType)
    if bounds is None:
        # A non-integer slot outside the category rules above (an opaque handle
        # dtype): the dtype-keyed lift already states the value's dtype.
        return
    low, high = bounds
    if low <= value <= high:
        return
    raise TypeError(
        f"call to {fn_name!r}: parameter {name!r} is declared "
        f"pl.Scalar[pl.{str(dtype).upper()}] but the value {value} is outside its "
        f"range [{low}, {high}]. A Python integer states no dtype, so the "
        "declaration must be able to hold it — pass a value in range, widen the "
        "declaration, or spell the intended conversion with pl.const(value, dtype) "
        "and let the declared slot convert it."
    )


def _dtype_bounds(dtype: Any, data_type: Any) -> tuple[int, int] | None:
    """The closed value range a Python integer must land in, or None.

    ``INDEX`` is the machine word, so it takes the signed 64-bit range. ``BOOL``
    takes 0/1 (a logical value is not a small integer). Floats and opaque handle
    dtypes return None: they are not integer placements.
    """
    if dtype == data_type.BOOL:
        return (0, 1)
    if dtype == data_type.INDEX:
        bits, signed = 64, True
    elif dtype.is_int():
        bits, signed = dtype.get_bit(), dtype.is_signed_int()
    else:
        return None
    if signed:
        return (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1)
    return (0, 2**bits - 1)


def _scalar_declared(hint: Any) -> bool:
    """Whether a staged-declared hint is a Scalar slot — the one slot kind
    a bare host scalar lifts into at the call-argument boundary."""
    if hint is None or not _staged_declared(hint):
        return False
    return hint is Scalar or isinstance(hint, Scalar)


def _host_scalar_dtype(arg: bool | int | float) -> Any:
    """The host promotion defaults for a bare scalar into an untyped
    (bare ``pl.Scalar``) slot."""
    from pypto.pypto_core import DataType  # noqa: PLC0415 — keep the toplevel import surface unchanged

    if isinstance(arg, bool):
        return DataType.BOOL
    if isinstance(arg, float):
        return DataType.FP32
    return DataType.INT32


def _renamed_function(fn: ir.Function, name: str) -> ir.Function:
    """A shallow rebuild of ``fn`` under a variant name (params, body and
    classification carried by reference — the rename is bookkeeping, not a
    re-trace)."""
    return ir.Function(
        name,
        fn.params,
        fn.return_types,
        fn.body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
    )


class _LazyGlobalFn(Marker):
    """A module-global standalone ``ir.Function`` made callable for one
    traced entry execution — a ``Marker`` whose call lane is lazy.

    The Marker surface (declared signature, classification, ``gvar``) comes
    from the frozen eager-traced function, so every Marker consumer
    (``pl.spmd_submit`` validation, program-method adaptation) accepts it
    unchanged.

    Without an eager record — or outside a trace — the call is the
    verbatim lane: register the function with the active program assembly at
    first call and emit the detached ``ir.Call`` (a global the body never
    calls stays out of the program).

    With an eager record (the global is an eagerly traced jit callee), an
    in-trace call is the per-call variant lane — the same mechanism as the
    lazy callee lane: the callee re-traces with the call-site argument types
    (a static-shaped caller derives a static-shaped variant, so the frozen
    annotation's ``pl.dynamic`` extents never leak into the caller's scope),
    dedupes in the active assembly, and the emitted ``Call`` drops the
    arguments baked into the variant's signature."""

    def __init__(self, fn: ir.Function, record: _EagerTraceRecord | None = None) -> None:
        self._init_from_ir_function(fn)
        self.kernel = None  # an adapted function, not a traceable kernel
        self._fn = fn
        self._record = record
        self.__name__ = fn.name

    def on_submit_reference(self) -> None:
        """The Submit-reference hook (``Marker.on_submit_reference``): this
        wrapper registers lazily (first call), so a Submit that references
        the frozen function without calling it registers it here — with the
        active program assembly, if one is running."""
        assembly = _ACTIVE_ASSEMBLY.get()
        if assembly is not None:
            assembly.register(self._fn)

    def resolve_submit(self, args: tuple[Any, ...]) -> tuple[Marker, tuple[Any, ...]] | None:
        """The Submit-reference resolution (``Marker.resolve_submit``): with a
        eager record inside a trace, a Submit reference instantiates
        through the same per-call lane a call takes (call and submit differ
        only in the emitted instruction); otherwise the verbatim reference —
        the frozen function, registered here."""
        if self._record is not None and active_frame() is not None:
            return self._specialized_resolve(args)
        assembly = _ACTIVE_ASSEMBLY.get()
        if assembly is not None:
            assembly.register(self._fn)
        return None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        in_trace = active_frame() is not None
        # An eagerly traced callee answers to the jit envelope's binding, so a
        # keyword naming one of its parameters folds into declaration order —
        # the record owns the declared names because the frozen function may
        # have baked host knobs away. Without a record the call is verbatim
        # (a ``pl.function`` global or an external declaration), and the parser
        # lane refuses keywords at those call sites too.
        if self._record is not None:
            args, overrides = bind_declared_keywords(
                self._record.specializer().param_names,
                args,
                kwargs,
                self._fn.name,
                reserved=CALL_SITE_KEYWORDS if in_trace else (),
            )
            kwargs = overrides
        elif kwargs and (not in_trace or set(kwargs) - CALL_SITE_KEYWORDS):
            raise TypeError(
                f"call to {self._fn.name!r}: keyword arguments are not accepted "
                f"on a sub-kernel call (parser-aligned — pass all arguments positionally)"
            )
        if self._record is not None and in_trace:
            return self._specialized_call(args, kwargs)
        assembly = _ACTIVE_ASSEMBLY.get()
        if assembly is not None:
            assembly.register(self._fn)
        return Marker.__call__(self, *args, **kwargs)

    def _specialized_call(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        target, runtime_args = self._specialized_resolve(args)
        return target(*runtime_args, **kwargs)

    def _specialized_resolve(self, args: tuple[Any, ...]) -> tuple[Marker, tuple[Any, ...]]:
        """The shared per-call instantiation of a call or a Submit reference:
        re-trace the callee with the call-site argument types, dedupe in the
        active program assembly under the variant key, and return the
        specialized target with the surviving runtime args (positions whose
        values were baked into the variant's signature are dropped). Call
        and submit consume this identically; only the emitted instruction
        differs."""
        record = self._record
        assert record is not None
        spec = record.specializer()
        param_names = spec.param_names
        if len(args) != len(param_names):
            raise TypeError(
                f"call to {self._fn.name!r} expects {len(param_names)} argument(s), got {len(args)}"
            )
        args = _lift_runtime_scalar_args(spec._param_hints, param_names, args, self._fn.name, in_trace=True)
        _check_param_worlds(spec._param_hints, param_names, args, self._fn.name)  # noqa: SLF001 — the in-module lane
        args = coerce_declared_arguments(spec._param_hints, param_names, args, self._fn.name)  # noqa: SLF001
        annotations = {
            name: _in_trace_annotation(name, arg, self._fn.name, hint=spec._param_hints.get(name))  # noqa: SLF001 — the in-module lane
            for name, arg in zip(param_names, args)
        }
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        callee: ir.Function | None = None
        if assembly is not None:
            key = _variant_key(
                record.kernel,
                annotations,
                spec._function_type,  # noqa: SLF001 — the in-module lane
                spec._level,  # noqa: SLF001 — the in-module lane
                spec._role,  # noqa: SLF001 — the in-module lane
                spec._attrs,  # noqa: SLF001 — the in-module lane
            )
            callee = assembly.by_key.get(key)
        if callee is None:
            callee = spec._trace_with(annotations, derive_return=True)  # noqa: SLF001 — the in-module lane
            if assembly is not None and key is not None:
                variant_name = assembly.unique_name(callee.name)
                if variant_name != callee.name:
                    callee = _renamed_function(callee, variant_name)
                assembly.register(callee, key=key)
        surviving = {p.name_hint for p in callee.params}
        runtime_args = tuple(a for a, n in zip(args, param_names) if n in surviving)
        return Marker._from_ir_function(callee), runtime_args


class _CompiledImpl:
    """``jit.compile()``'s return: one frozen materialization
    whose calling signature is the runtime ABI.

    Host knobs were consumed at compile — they are no longer parameters —
    so arguments beyond the runtime positions are an arity error naming the
    consumed knob, not a silent freeze. Fewer arguments are forwarded
    unchanged and ``CompiledProgram.__call__`` owns the return-style
    convention. Everything else (``isinstance``, attributes,
    equality, ``prepare``/``warmup`` surfaces) delegates to the wrapped
    program via the ``__class__`` projection.
    """

    def __init__(
        self,
        compiled: Any,
        runtime_names: tuple[str, ...],
        knob_names: tuple[str, ...],
        kernel_name: str,
    ) -> None:
        self._wrapped = compiled
        self._runtime_names = runtime_names
        self._knob_names = knob_names
        self._kernel_name = kernel_name

    @property
    def __class__(self) -> type[Any]:  # noqa: N805 — the isinstance-delegation idiom  # type: ignore[reportIncompatibleVariableOverride]
        # ``type()`` of the wrapped object: ``type[Any]`` is the only spelling
        # that describes "whatever the wrapped impl is" without widening the
        # result to an unknown type argument.
        return typing.cast("type[Any]", type(self._wrapped))

    def __call__(self, *args: Any, config: RunConfig | None = None) -> Any:
        if len(args) > len(self._runtime_names):
            knobs = ", ".join(self._knob_names) or "none"
            names = ", ".join(self._runtime_names) or "(no arguments)"
            raise TypeError(
                f"impl of {self._kernel_name!r} takes at most "
                f"{len(self._runtime_names)} runtime argument(s) ({names}); got "
                f"{len(args)}. Host knob(s) {knobs} were consumed at compile — "
                "request a new compilation for a different value."
            )
        if config is not None:
            return self._wrapped(*args, config=config)
        return self._wrapped(*args)

    def __eq__(self, other: object) -> bool:
        # The __class__ projection makes the impl *claim to be* the wrapped
        # program, so equality must agree with that claim: identity-keyed
        # lookups against mappings built from the inner object (e.g.
        # DistributedWorker.register's _states, keyed by the programs the
        # worker was constructed from) would otherwise miss — the worker is
        # built from the wrapped program while the caller holds the impl.
        if self is other:
            return True
        return self._wrapped == other

    def __hash__(self) -> int:
        return hash(self._wrapped)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped, name)

    def __getitem__(self, key: Any) -> Any:
        # The ``__class__`` projection covers ``isinstance`` and attribute
        # reads, but operators look up dunders on the type, so warmup's
        # ``compiled[name]`` (per-orchestration ``load``) would miss it.
        return self._wrapped[key]


def _compile_program(program: ir.Program, config: RunConfig | None = None) -> Any:
    from pypto.ir import compile as ir_compile  # noqa: PLC0415 — cycle-free laziness
    from pypto.runtime.jit_cache import resolve_compile_request  # noqa: PLC0415 — torch is a runner dep

    kwargs: dict[str, Any] = {}
    if config is not None:
        # The shared compile-request resolver: ``RunConfig.compile_kwargs()``
        # (the one mapping onto ``ir.compile()``'s parameters, the same one the
        # parsing-frontend jit and direct callers use) plus the dump-level
        # coercion, the source-location default, and the pass-context conflict
        # validation ``ir.compile()`` would perform anyway. With no config, no
        # kwargs: the call defers to ``ir.compile()``'s own defaults.
        kwargs, _ = resolve_compile_request(config)
    return ir_compile(program, **kwargs)


def _checked_run_config(config: object) -> RunConfig | None:
    """Validate a caller-supplied ``config=``: ``None`` passes, anything but
    a ``RunConfig`` is a loud TypeError (duck-typed access would surface as a
    confusing AttributeError deep in the field mapping)."""
    if config is None:
        return None
    from pypto.runtime.runner import RunConfig  # noqa: PLC0415 — torch is a runner dep

    if not isinstance(config, RunConfig):
        raise TypeError(f"config= must be a pypto.runtime.RunConfig, got {type(config).__name__!r}")
    return config


# —— the ``pl.jit`` binding: the parsing-frontend jit's per-func_type
# mapping (specializer.py's generated ``@pl.function`` form), traced ——


# The parsing frontend's "kwarg not passed" sentinel: a default value cannot
# tell ``auto_scope=True`` apart from an omitted argument, and only one of those
# is legal on every family spelling.
_AUTO_SCOPE_UNSET: Any = object()


class _JitAlias:
    """One ``@pl.jit.<alias>`` — the specializer's ``@pl.function`` form for
    that function class, collected through tracing."""

    def __init__(
        self,
        label: str,
        *,
        function_type: ir.FunctionType = ir.FunctionType.Opaque,
        level: ir.Level | None = None,
        role: ir.Role | None = None,
        allow_level: bool = False,
        allow_auto_scope: bool = True,
    ) -> None:
        self._label = label
        self._function_type = function_type
        self._level = level
        self._role = role
        self._allow_level = allow_level
        self._allow_auto_scope = allow_auto_scope

    def __call__(
        self,
        kernel: types.FunctionType | None = None,
        *,
        level: ir.Level | None = None,
        auto_scope: Any = _AUTO_SCOPE_UNSET,
    ) -> Any:
        # The kwarg surface mirrors the parsing frontend's: only ``incore``
        # takes ``level=``, and only the entries whose body lands in the caller
        # (``host``, ``inline``, and the bare ``@pl.jit_trace``) take
        # ``auto_scope=``. Accepting a kwarg that cannot take effect would
        # advertise a policy the pass never reads.
        if level is not None and not self._allow_level:
            raise TypeError(f"@pl.jit.{self._label} does not accept level=")
        if auto_scope is not _AUTO_SCOPE_UNSET and not self._allow_auto_scope:
            raise TypeError(
                f"@pl.jit.{self._label} does not accept an auto_scope= argument "
                "(auto_scope is only meaningful for the Orchestration entry, the "
                "HOST orchestrator, and inline sub-functions)"
            )
        resolved_auto_scope = True if auto_scope is _AUTO_SCOPE_UNSET else auto_scope
        if kernel is None:  # the parenthesized form: @pl.jit.<alias>(...)
            return functools.partial(self, level=level, auto_scope=resolved_auto_scope)
        return jit_trace(
            kernel,
            function_type=self._function_type,
            level=level if level is not None else self._level,
            role=self._role,
            attrs=fold_auto_scope(None, resolved_auto_scope),
        )


def _unsupported_family_form(name: str) -> TypeError:
    """The rejection for the tracing jit's unverified family spellings.

    Raised by the property access itself, so neither the bare
    (``@pl.jit_trace.host``) nor the parenthesized
    (``@pl.jit_trace.host(...)``) form can start tracing a lane this wave
    does not verify — an accepted-but-unverified spelling would advertise
    a surface without acceptance, so it fails loudly instead.
    """
    return TypeError(
        f"@pl.jit_trace.{name} is not part of the verified tracing surface "
        "(this wave: @pl.jit_trace, @pl.jit_trace.body, @pl.jit_trace.incore, "
        "@pl.jit_trace.inline, @pl.jit_trace.host, @pl.jit_trace.graph). "
        f"Declare the kernel on the parsing frontend (@pl.jit.{name}) instead."
    )


# ---- the reusable-statement-fragment spelling --------------------------------


class _TracedBody:
    """A reusable statement fragment, called like a function.

    The kernel-author spelling of the splice placement: ``@pl.jit_trace.body``
    prepares the fragment once, and each call expands its statements at the
    call site — no function, no declaration, no ``ir.Call``, nothing for a
    later pass to inline. A body therefore owns a *statement boundary*, not a
    signature: the arguments are the call's staged values, and a value the
    body returns crosses back into the caller's expression position.

    This is one policy over the authoring tier, not a private path into the
    tracer: the wrapper is ``@pl.authoring.recipe`` over ``ctx.splice_body``,
    so a framework outside this repository can build the same spelling from
    the public ``pl.authoring`` surface. ``@pl.authoring.body`` stays the
    framework-author spelling because it keeps the placement choice explicit
    (``.splice`` / ``.inline`` / ``.incore``); a kernel author declaring one
    meaning at the definition site is the narrower, readable form.

    Distinct from ``@pl.jit_trace.inline``, which declares a typed ``Inline``
    function *and* a call to it, for the inlining pass to erase.

    One body cannot call another: the tier admits one splice at a time (the
    spliced body's parameters are the caller's own names), so reuse between
    fragments is expressed by writing the statements in the body, or by
    declaring a ``@pl.jit_trace.inline`` callee.
    """

    def __init__(self, kernel: types.FunctionType, *, auto_scope: bool | None) -> None:
        # Deferred: ``pypto.language.authoring`` imports the tracing entry, so
        # importing it at module scope would close the cycle.
        from pypto.language.authoring import body as prepare_body  # noqa: PLC0415
        from pypto.language.authoring import recipe as authoring_recipe  # noqa: PLC0415

        prepared = prepare_body(kernel)

        @authoring_recipe
        def expand(ctx: Any, *args: Any, **kwargs: Any) -> Any:
            if "auto_scope" in kwargs:
                raise TypeError(
                    f"{kernel.__name__}: auto_scope is declared on the body, not passed at the call site"
                )
            return ctx.splice_body(prepared, *args, auto_scope=auto_scope, **kwargs)

        self._kernel = kernel
        self._expand = expand
        # The body's own metadata, not the wrapper's: a trace, a diagnostic and
        # a stack trace should all name the fragment the author wrote.
        self.__name__ = kernel.__name__
        self.__qualname__ = kernel.__qualname__
        self.__doc__ = kernel.__doc__
        self.__module__ = kernel.__module__
        self.__wrapped__ = kernel

    def __repr__(self) -> str:
        return f"<traceable body {self.__name__}>"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Expand the body's statements into the active trace."""
        frame = active_frame()
        if frame is None:
            raise TypeError(
                f"{self.__name__} is a body declared with @pl.jit_trace.body — it expands statements "
                "at a call site inside a traced kernel, so it cannot run as a host call"
            )
        # The tier admits one splice at a time (a splice's parameters are the
        # caller's own names), so a body cannot call another body. Say that in
        # the vocabulary the kernel author wrote, not in "splice" terms.
        if frame.sema.splice_parent is not None:
            raise TypeError(
                f"{self.__name__} is a @pl.jit_trace.body and cannot be called from inside another body — "
                "a body expands into its caller, so the statements it reuses have to be written "
                "in the body that calls them (or declared as a @pl.jit_trace.inline callee)"
            )
        return self._expand(*args, **kwargs)


class _BodyDecorator:
    """``pl.jit_trace.body`` — declare one reusable statement fragment.

    The fragment is placed when it is called, so the spelling reads as a
    function call (``scale_block(a, out, row0, k)``) rather than as a
    placement method. Deliberately not a ``@pl.jit.*`` alias: those declare a
    function of their own, while a body is expanded into its caller.
    """

    def __call__(
        self, kernel: types.FunctionType | None = None, *, auto_scope: Any = _AUTO_SCOPE_UNSET
    ) -> Any:
        # ``Any``, like ``_JitAlias.__call__``: a decorator's return type is the
        # decorated name's type, and a union would make every call site of the
        # body a static error while saying nothing about the staged values the
        # body actually takes.
        # ``auto_scope`` is the splice's own knob, so it belongs to the
        # declaration (one meaning per fragment), not to each call site.
        resolved = None if auto_scope is _AUTO_SCOPE_UNSET else bool(auto_scope)
        if kernel is None:  # the parenthesized form: @pl.jit_trace.body(auto_scope=False)
            return functools.partial(self, auto_scope=resolved)
        if not isinstance(kernel, types.FunctionType):
            raise TypeError(f"@pl.jit_trace.body expects a plain function, got {kernel!r}")
        return _TracedBody(kernel, auto_scope=resolved)


class _JitDecorator:
    """The ``pl.jit_trace`` object — the traced jit envelope.

    Every family spelling that the tracing engine verifies is an alias of the
    parsing frontend's, with the same function kind, level, role and kwarg
    surface: ``@pl.jit_trace`` is ``@pl.jit`` (an Orchestration entry),
    ``.incore`` / ``.inline`` / ``.host`` / ``.graph`` are their ``@pl.jit.*``
    twins. ``opaque`` and ``extern`` stay on the parsing frontend:
    ``@pl.jit.extern`` is a declaration with no Python body to trace, and
    ``opaque`` admits bodies the tracer's vocabulary does not cover.

    ``.body`` is the one member of this surface with no ``@pl.jit.*`` twin: it
    declares a statement fragment that expands at its call site, so it belongs
    to the tracing frontend alone (see :class:`_TracedBody`).
    """

    def __init__(self) -> None:
        self.incore = _JitAlias(
            "incore", function_type=ir.FunctionType.InCore, allow_level=True, allow_auto_scope=False
        )
        self.inline = _JitAlias("inline", function_type=ir.FunctionType.Inline, allow_auto_scope=True)
        # The parser lane's host entry: an Opaque function at HOST level. Its
        # body is the per-rank dispatch loop (``pld.system.world_size()``,
        # ``device=r``), so it lives at the host tier rather than as a chip
        # orchestration function.
        self.host = _JitAlias(
            "host",
            function_type=ir.FunctionType.Opaque,
            level=ir.Level.HOST,
            role=ir.Role.Orchestrator,
            allow_auto_scope=True,
        )
        # A recordable orchestration fragment: outline_graph_scopes and the
        # host_build_graph runtime consume it as a Graph function.
        self.graph = _JitAlias("graph", function_type=ir.FunctionType.Graph, allow_auto_scope=False)
        # The kernel-author spelling of a reusable statement fragment. Not an
        # alias: a body has no function of its own to declare.
        self.body = _BodyDecorator()

    @property
    def opaque(self) -> NoReturn:
        raise _unsupported_family_form("opaque")

    @property
    def extern(self) -> NoReturn:
        raise _unsupported_family_form("extern")

    def __call__(
        self,
        kernel: types.FunctionType | None = None,
        *,
        auto_scope: Any = _AUTO_SCOPE_UNSET,
    ) -> Any:
        resolved_auto_scope = True if auto_scope is _AUTO_SCOPE_UNSET else auto_scope
        if kernel is None:  # the parenthesized form: @pl.jit(auto_scope=False)
            return functools.partial(self, auto_scope=resolved_auto_scope)
        return jit_trace(
            kernel,
            function_type=ir.FunctionType.Orchestration,
            attrs=fold_auto_scope(None, resolved_auto_scope),
        )


jit = _JitDecorator()
