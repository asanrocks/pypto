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
traces at decoration; the decoration trace (``.fn``) and the callees its
body reached are retained in a record, and the jit's host ``compile`` /
dispatch reuse them — the annotation determines the artifact, so a call
argument supplies a dispatch value, never a re-specialization. **Lazy** —
an unannotated/bare/Callable param forces deferral: at call, the real
args specialize the kernel (tensor facades → static-shape params; host
scalars → passed through, lifted to ``Const`` by the ops at the operand
boundary), re-traced per call (no cache).

The jit object is the full jit surface:

- **eager lane** — the decoration record's program (the decoration trace
  plus its collected callees) serves every ``compile``/dispatch whose
  arguments are compatible with the declaration (a declared dynamic
  extent accepts any concrete shape; a static dim must match — a
  contradicting static call re-traces, as a static kernel always has);
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

``@pl.jit.extern`` — a hand-written C++ kernel: a single-core stub traces
as a header-only AIC/AIV ``ir.Function``; a ``mixed`` stub traces an AIC
and an AIV member and assembles the Group wrapper at the call site (the
``@pl.program`` group route).
"""

from __future__ import annotations

import functools
import inspect
import os
import types
import typing
import warnings
from collections.abc import Sequence
from contextvars import ContextVar
from typing import Any

from pypto._external_source import EXTERNAL_INCLUDE_DIRS_ATTR, encode_external_include_dirs
from pypto._function_attrs import DUAL_AIV_DISPATCH_ATTR
from pypto.ir.formatter import raw_print
from pypto.language.ambient import active_frame
from pypto.language.entrance.annotation import capture_caller_names, source_span_for_function
from pypto.language.entrance.declaration import fold_auto_scope
from pypto.language.op.marker import Marker
from pypto.language.protocol import is_staged
from pypto.language.tracer.decorators.function import (
    _adapt_captured_functions,
    _trace_with_annotations,
    function_trace,
)
from pypto.language.tracer.return_contract import pin_derived_return_types
from pypto.language.typing._runtime_scalar import RuntimeScalar
from pypto.language.typing.arg_direction import param_directions, unmangle_hints, unmangle_param_name
from pypto.language.typing.scalar import RUNTIME, Scalar
from pypto.language.typing.tensor import Tensor
from pypto.pypto_core import ir

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
) -> _EagerTraceRecord:
    """Run the one-time eager trace (decoration semantics) and register its
    record. Shared by ``jit_trace``'s eager lane and the Inline deferral —
    the same body, run at decoration or at first use."""
    from pypto.language.tracer.decorators.function import function_trace  # noqa: PLC0415 - avoid a cycle

    # The decoration trace owns an assembly: in-trace calls to eagerly
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
                    materialize_self=True,
                    specialize_global_callees=True,
                )
            ),
            function_type,
        )
    finally:
        _ACTIVE_ASSEMBLY.reset(token)
    record = _EagerTraceRecord(fn, kernel, None, assembly)
    _EAGER_TRACES[fn] = record
    return record


def jit_trace(
    kernel: types.FunctionType,
    *,
    function_type: ir.FunctionType = ir.FunctionType.Opaque,
    level: ir.Level | None = None,
    role: ir.Role | None = None,
    attrs: dict[str, Any] | None = None,
) -> _JitSpecializer:
    """The ``pl.jit`` envelope: eager ⊂ lazy, one public object.

    Eager (every param a facade annotation) → traces at decoration; the
    returned jit object's host ``compile``/dispatch reuse the decoration
    record (the traced ``ir.Function`` — ``.fn`` — plus the callees the
    decoration trace reached), so the annotation-determined artifact keeps
    its declared dynamic extents (the conformance gate compares ``.fn`` to
    ``function_parser``). Lazy (an unannotated/bare/Callable param — the IR
    depends on call-time info) → a ``_JitSpecializer`` callable that
    re-traces per call with the baked args and returns the IR.
    Both lanes materialize ``self`` in the traced body as the module's
    traced-kernel registry (``self.<sibling>`` — upstream's jit contract).
    """
    if _eagerly_traceable(kernel):
        if function_type in (ir.FunctionType.Inline, ir.FunctionType.InCore):
            # The callee-tier lanes (Inline, InCore) defer their decoration
            # trace to first use (01-language.md §10): a callee-tier
            # function has no dispatch arguments, so tracing at decoration
            # buys nothing, and eager execution before the module finishes
            # loading turns a definition-order choice into a raw NameError —
            # the AST-era jit decorated lazily, and def order was free (the
            # corpus's ``decode_indexer`` chain: an ``@pl.jit.incore``
            # caller referencing a later-defined sibling). The specializer
            # materializes the record (the identical eager body) at the
            # first ``.fn`` / host-compile access; in-trace calls take the
            # per-call variant lane and never need it.
            return _JitSpecializer(
                kernel,
                function_type=function_type,
                level=level,
                role=role,
                attrs=attrs,
                defer_eager=True,
            )
        record = _eager_trace_record_for(
            kernel, function_type=function_type, level=level, role=role, attrs=attrs
        )
        # The public object is the jit surface, not the bare ir.Function: the
        # host compile/dispatch half of the specializer's contract (the
        # golden runner's ``fn.compile(*args, config=...)`` and
        # ``fn(*tensors)``) routes through the record's specializer, which
        # serves the decoration trace's program (annotations determine the
        # artifact; call args are dispatch values). In-trace calls take the
        # per-call variant lane unchanged.
        return record.specializer()
    return _JitSpecializer(
        kernel,
        function_type=function_type,
        level=level,
        role=role,
        attrs=attrs,
        localns=capture_caller_names(),
    )


def _eagerly_traceable(kernel: types.FunctionType) -> bool:
    """Signature inspection: eager iff every positional param carries a
    facade annotation (concrete or DynVar-shaped — both annotation-
    determined). An unannotated / bare / Callable param forces lazy (the IR
    depends on the call-time argument)."""
    hints = unmangle_hints(kernel, typing.get_type_hints(kernel))
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
        # The eager lane's decoration record: when present, the annotation
        # determined the trace at decoration time, so host compile/dispatch
        # reuse the record's program instead of re-tracing with the bridged
        # call shapes (which would demote declared dynamic extents to
        # static — the exact silent specialization the declaration exists
        # to prevent).
        # ``defer_eager`` — the callee-tier lanes (Inline, InCore): the
        # record materializes at first use (01-language.md §10), running
        # the identical eager body then. A genuinely lazy jit (an
        # unannotated/bare/Callable param) keeps ``None`` forever — ``.fn``
        # stays loud there.
        self.__eager_record = eager_record
        self.__defer_eager = defer_eager
        # The parameter declarations pin each parameter's world (types flow
        # from values): a pl facade annotation is a runtime parameter, a
        # plain Python annotation (or none) is a host knob baked at trace
        # time. The world checks consult this map.
        self._param_hints: dict[str, Any] = typing.get_type_hints(kernel)
        self._param_names = [
            unmangle_param_name(kernel, p.name)
            for p in inspect.signature(kernel).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        self._return_type = typing.get_type_hints(kernel).get("return")
        self.__name__ = kernel.__name__
        self._cache: dict[Any, Any] = {}

    def param_names(self) -> list[str]:
        """The positional parameter names, in signature order — shared with
        the eager-record variant lane (``_LazyGlobalFn``)."""
        return list(self._param_names)

    @property
    def _eager_record(self) -> _EagerTraceRecord | None:
        """The decoration record — materialized on first access for the
        deferred Inline lane, ``None`` forever for a genuinely lazy jit."""
        if self.__eager_record is None and self.__defer_eager:
            self.__eager_record = _eager_trace_record_for(
                self.kernel,
                function_type=self._function_type,
                level=self._level,
                role=self._role,
                attrs=self._attrs,
            )
        return self.__eager_record

    @property
    def fn(self) -> ir.Function:
        """The decoration-time trace — the eager jit's annotation-determined
        ``ir.Function`` (directions already normalized, derived return
        already pinned). Structural consumers (the conformance gate, corpus
        pins, the printer) read the IR here. A lazily traced jit has no
        single pre-executed trace — the property is eager-only, loud
        otherwise."""
        if self._eager_record is None:
            raise AttributeError(
                f"{self.kernel.__name__!r} is lazily traced — there is no decoration "
                "trace; a call re-traces with the call-site arguments"
            )
        return self._eager_record.fn

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        # The parser-aligned kwargs rule: a jit call takes its arguments
        # positionally; the call-site-only ``device=``/``attrs=`` overrides
        # are accepted in-trace (the Marker emission carries them). One
        # top-level carve-out exists: ``config=RunConfig(...)`` selects the
        # compile knobs and runtime dispatch options for the host call.
        config = None
        if active_frame() is None and set(kwargs) == {"config"}:
            config = _checked_run_config(kwargs.pop("config"))
        if kwargs and (active_frame() is None or set(kwargs) - {"device", "attrs"}):
            raise TypeError(
                f"call to {self.kernel.__name__!r}: keyword arguments are not accepted "
                f"on a jit call (parser-aligned — pass all arguments positionally)"
            )
        if len(args) != len(self._param_names):
            raise TypeError(
                f"call to {self.kernel.__name__!r} expects {len(self._param_names)} "
                f"argument(s), got {len(args)}"
            )

        if active_frame() is not None:
            return self._emit_detached_call(args, kwargs)
        bridged, has_torch = self._bridge_args(args)
        for name, arg in zip(self._param_names, bridged):
            if is_staged(arg) and not _annotation_shaped(arg):
                raise TypeError(
                    f"call to {self.kernel.__name__!r}: argument {name!r} is a "
                    f"expr-wrapping facade (shape-less) — a jit call specializes "
                    f"with annotation-mode facades (shape + dtype)"
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
        lane — ignores the declared ``->`` and pins the return types on the
        actual trace: a per-call variant's declared return would carry the
        declaration's dynamic extents while its specialized params are
        concrete, and the two meeting at a type-sensitive rebind (the relaxed
        reassignment rule) would fork the caller's variable instead of
        assigning through — the declared form would lie about the produced
        values.
        """
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
        return _drop_inline_directions(fn, self._function_type)

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
        this identically; only the emitted instruction differs."""
        _check_param_worlds(self._param_hints, self._param_names, args, self.kernel.__name__)
        annotations: dict[str, object] = {}
        for name, arg in zip(self._param_names, args):
            annotations[name] = _in_trace_annotation(name, arg, self.kernel.__name__)
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        callee: ir.Function | None = None
        if assembly is not None:
            key = (self.kernel, *(_variant_key_part(a) for a in annotations.values()))
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
            if is_staged(arg) or isinstance(arg, (bool, int, float, ir.Expr)):
                bridged.append(arg)
                continue
            facade = _torch_facade(arg)
            if facade is not None:
                bridged.append(facade)
                has_torch = True
                continue
            raise TypeError(
                f"call to {self.kernel.__name__!r}: argument {name!r} "
                f"({type(arg).__name__}) is not a tensor, facade, or host scalar — "
                "a jit call specializes with torch tensors, annotation-mode "
                "facades, or Python scalars"
            )
        _check_param_worlds(self._param_hints, self._param_names, bridged, self.kernel.__name__)
        return bridged, has_torch

    def _dispatch(self, args: tuple[Any, ...], bridged: list[Any], config: Any = None) -> Any:
        key = self._cache_key(bridged, config)
        compiled = self._cache.get(key)
        if compiled is None:
            compiled = _compile_program(self._build_program(bridged), config)
            self._cache[key] = compiled
        # pl.runtime unwraps to the raw host value — CompiledProgram takes
        # scalars at dispatch.
        dispatch_args = tuple(a.value if isinstance(a, RuntimeScalar) else a for a in args)
        if config is None:
            return compiled(*dispatch_args)
        return compiled(*dispatch_args, config=config)

    def _cache_key(self, bridged: list[Any], config: Any = None) -> tuple[Any, ...]:
        # The compile-identity suffix (platform, strategy, distributed config,
        # planner, dbC flag — resolved from the RunConfig and any active
        # PassContext) splits the cache so two compile configurations never
        # share one artifact (03-entrance.md §5.4 ArtifactKey).
        from pypto.runtime.runner import run_config_compile_identity  # noqa: PLC0415 — torch is a runner dep

        parts: list[Any] = []
        for arg in bridged:
            if isinstance(arg, RuntimeScalar):
                # The entry runtime-scalar marker: type-only — one artifact
                # serves every value (the whole point of the spelling).
                parts.append(("runtime_scalar", str(arg.dtype)))
            elif is_staged(arg) and isinstance(arg, Tensor):
                parts.append((tuple(map(str, arg.shape or ())), str(arg.dtype), str(arg.layout)))
            else:
                parts.append((type(arg).__name__, arg))
        parts.append(run_config_compile_identity(config))
        return tuple(parts)

    # —— the program build shared by compile / compile_for_test ———

    def _build_program(self, bridged: list[Any]) -> ir.Program:
        """Trace the entry with the bridged annotations and assemble the
        multi-function program: the entry plus every callee its body
        reached (lazy jit callees traced at their call sites, captured
        standalone functions adapted from the closure and the module
        globals).

        The eager lane short-circuits to the decoration record when the
        bridged call is compatible with the declaration: the decoration
        trace plus its collected callees IS the artifact, and re-tracing
        would bridge the concrete arg shapes, demoting declared dynamic
        extents to static."""
        if self._eager_record is not None and self._declaration_serves(bridged):
            rec = self._eager_record
            return ir.Program(
                [rec.fn, *rec.assembly.functions],
                rec.fn.name,
                source_span_for_function(self.kernel),
            )
        assembly = _ProgramAssembly()
        token = _ACTIVE_ASSEMBLY.set(assembly)
        try:
            kernel = self._adapted_kernel(assembly)
            entry = self._trace_with(dict(zip(self._param_names, bridged)), kernel=kernel)
        finally:
            _ACTIVE_ASSEMBLY.reset(token)
        return ir.Program(
            [entry, *assembly.functions], self.kernel.__name__, source_span_for_function(self.kernel)
        )

    def _declaration_serves(self, bridged: list[Any]) -> bool:
        """Can the decoration trace's program serve this bridged call? Each
        declared dynamic extent accepts any concrete shape; a declared
        static dim (or dtype) must match. A contradicting static call falls
        through to the per-call re-trace — a statically annotated kernel
        re-specializes at the new shape, as it always has — while a
        dynamic-annotated kernel is always served by its record."""
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
        kernel, captured = _adapt_captured_functions(self.kernel, specialize=True)
        for fn in captured:
            # Only the verbatim world registers eagerly: a jit-eager callee
            # (decoration record present) joins the program as its per-call
            # variant at each call or submit-reference site — eagerly
            # registering the frozen decoration would ship it as dead code
            # alongside the variants.
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

    def compile(self, *args: Any, config: Any = None, **kwargs: Any) -> Any:
        """Build + compile for the signature implied by *args*; returns the
        ``CompiledProgram`` (device dispatch is its ``__call__``).

        For an eagerly traced kernel (annotation-determined), *args* carry
        no specialization power: the decoration trace's program — with its
        declared dynamic extents — is the artifact, and the args only need
        to be compatible with the declaration (the golden runner's dummy
        tensors satisfy this by construction). A statically annotated kernel
        called at a different shape re-traces at that shape.

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

        Each call compiles a fresh artifact (no cache): explicit
        compile-then-call freezes one materialization (03-entrance.md §5.5).
        """
        config = _checked_run_config(config)
        self._check_compile_kwargs(kwargs)
        bridged, _ = self._bridge_args(args)
        return _compile_program(self._build_program(bridged), config)

    def _check_compile_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Validate the name-keyed scalar vocabulary against the declaration.

        Every parameter's world was pinned at decoration, so the check can
        only confirm that each keyword speaks the declaration's language —
        it can never retype or re-specialize (the specializer's literal-
        specialization is deliberately not adopted: the jit's own call
        boundary rejects a host scalar into a runtime parameter, and compile
        must not be looser than the call it serves).
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
    """Is ``arg`` an annotation-mode facade (shape/extent + dtype present, as
    opposed to an expr-wrapping facade whose shape is None)? A
    Ptr has no shape — always annotation-ok."""
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


def _torch_facade(arg: Any) -> Tensor | None:
    """A ``torch.Tensor`` → an annotation-mode facade (shape + dtype), or
    ``None`` when ``arg`` is not one. The specializer's TensorMeta
    extraction, re-skinned as the tracing annotation."""
    torch = _torch()
    if torch is None or not isinstance(arg, torch.Tensor):
        return None
    if not _TORCH_DTYPES:
        _TORCH_DTYPES.update(_load_torch_dtypes())
    dtype = _TORCH_DTYPES.get(arg.dtype)
    if dtype is None:
        raise TypeError(f"jit argument of torch dtype {arg.dtype!r} has no PyPTO DataType mapping")
    return Tensor(list(arg.shape), dtype, _annotation_only=True)


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
    # guarded table the quarantined specializer maps (``pypto/jit/decorator.py``);
    # uint8 maps to UINT8 there too, and the bridge must preserve that behavior.
    for _torch_name, _pto_dt in (
        ("float8_e4m3fn", DataType.FP8E4M3FN),
        ("float8_e5m2", DataType.FP8E5M2),
        ("float8_e8m0fnu", DataType.FP8E8M0),
        ("float4_e2m1fn_x2", DataType.FP4),
    ):
        _td = getattr(torch, _torch_name, None)
        if _td is not None:
            table[_td] = _pto_dt
    return table


_TORCH_DTYPES: dict[Any, Any] = {}


def _in_trace_annotation(name: str, arg: Any, kernel_name: str) -> Any:
    """Bridge one call-site argument (inside a trace) to the callee's
    annotation: a runtime facade's wrapped expr carries the full type
    (shape + dtype + layout), which becomes the static-shape annotation the
    callee trace specializes with — no retype at the call boundary. Host
    scalars pass through."""
    if is_staged(arg):
        ty = arg.unwrap().type
        # The DistributedTensorType check must precede TensorType: the pybind
        # flavour IS-A TensorType, so the plain branch would silently
        # downgrade the window to a plain Tensor annotation (the retype the
        # type-soundness contract forbids).
        if isinstance(ty, ir.DistributedTensorType):
            from pypto.language.distributed.typing import (  # noqa: PLC0415 — lazy: distributed pulls the op tier
                DistributedTensor,
            )

            return DistributedTensor(
                list(ty.shape),  # type: ignore[arg-type]
                ty.dtype,
                layout=ty.tensor_view,  # type: ignore[arg-type]  # slot takes a TensorView too
                memref=ty.memref,
                _annotation_only=True,
            )
        if isinstance(ty, ir.TensorType):
            # shape entries carry over as-is (int or extent Expr — the
            # annotation accepts both; ``_shape_dim`` normalizes at the
            # type boundary), and so do the memref / tensor_view slots —
            # dropping them would retype a layout-carrying argument.
            return Tensor(
                list(ty.shape),  # type: ignore[arg-type]
                ty.dtype,
                layout=ty.tensor_view,  # type: ignore[arg-type]  # slot takes a TensorView too
                memref=ty.memref,
                _annotation_only=True,
            )
        if isinstance(ty, ir.ScalarType):
            return Scalar(ty.dtype, _annotation_only=True)
        if isinstance(ty, ir.ArrayType):
            from pypto.language.typing import Array  # noqa: PLC0415

            return Array(ty.extent, ty.dtype, _annotation_only=True)  # type: ignore[arg-type]  # Expr extent
        raise TypeError(
            f"call to {kernel_name!r} inside a trace: argument {name!r} has "
            f"type {ty}, which has no annotation form (Tensor/DistributedTensor/"
            "Scalar/Array only)"
        )
    if _is_host_scalar(arg):
        return arg
    raise TypeError(
        f"call to {kernel_name!r} inside a trace: argument {name!r} "
        f"({type(arg).__name__}) is not a facade or host scalar"
    )


# —— the program assembly context (one entry build) ——


class _ProgramAssembly:
    """The callees collected during one entry build: appended in first-use
    order, deduped on object identity (adapted functions) and on the
    variant key (kernel identity + derived annotations + baked host values
    — a helper called twice with the same shapes and constants stays one
    function; the specializer's ScalarCacheInfo semantics, one tier below
    the compiled-artifact cache)."""

    def __init__(self) -> None:
        self.functions: list[ir.Function] = []
        self.by_key: dict[Any, ir.Function] = {}
        self._ids: set[int] = set()
        self._name_counts: dict[str, int] = {}

    def register(self, fn: ir.Function, key: Any | None = None) -> None:
        if id(fn) in self._ids:
            return
        self._ids.add(id(fn))
        # Reserve the base name: a frozen function (a Submit reference, an
        # eager capture) and a later per-call variant of the same kernel are
        # distinct functions and must not collide on ir.Program's
        # program-unique name check. unique_name suffixes on this count.
        self._name_counts[fn.name] = max(self._name_counts.get(fn.name, 0), 1)
        self.functions.append(fn)
        if key is not None:
            self.by_key[key] = fn

    def unique_name(self, base: str) -> str:
        """Mint a program-unique name for one variant of ``base``: the first
        variant keeps the kernel's name (the caller's Call and the
        decoration-time name agree), each further distinct variant gets a
        numeric suffix — one helper instantiated at multiple shapes never
        produces duplicate function names in one program."""
        count = self._name_counts.get(base, 0)
        self._name_counts[base] = count + 1
        return base if count == 0 else f"{base}_{count + 1}"


_ACTIVE_ASSEMBLY: ContextVar[_ProgramAssembly | None] = ContextVar("pto_jit_assembly", default=None)


# —— the eager-trace record: what per-call specialization needs ———


class _EagerTraceRecord:
    """Decoration-time facts of one eagerly traced jit kernel.

    The eager decoration replaces the module global with the jit object
    (the record's specializer); this record (keyed by the decoration trace's
    ``ir.Function``) retains what that object serves its lanes from: the
    trace (``fn``), the kernel (the re-trace source), the decoration
    assembly (the callees the decoration trace reached), and the lazily
    built specializer. The specializer's AST-era counterpart held all of
    this by construction — it *is* the function object."""

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
        decoration-time function's classification facts. Carries the record
        so the host compile/dispatch lanes serve the decoration trace's
        program; the per-call variant lane re-traces through the kernel as
        before."""
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


def _parser_aligned_callee(value: Any) -> ir.Function | None:
    """The frozen function a parser-aligned lane (the standalone
    ``pl.function`` build, the program methods) should see for *value*: an
    ``ir.Function`` passes through; an eagerly traced jit object contributes
    its decoration trace — the same callee the bare ``ir.Function`` module
    global was before the jit object became the public decoration result.
    Without this, the jit object's own ``__call__`` would leak the jit
    envelope's world checks into a lane the conformance oracle pins as
    parser-aligned. Anything else (a lazy jit, a plain callable) is None —
    not this lane's callee."""
    if isinstance(value, ir.Function):
        return value
    if isinstance(value, _JitSpecializer) and value._eager_record is not None:
        return value._eager_record.fn
    return None


def _eager_trace_record(fn: ir.Function) -> _EagerTraceRecord | None:
    """The decoration-time record for an eagerly traced jit kernel's
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
    expression (``ir.Expr``) or an unmapped staged kind falls back to its
    raw printed text: the printer is the lossless round-trip serializer, so
    raw text equality is structural equality."""
    if isinstance(annotation, (bool, int, float)):
        return ("host", type(annotation).__name__, annotation)
    if isinstance(annotation, ir.Expr):
        with raw_print():
            return ("expr", str(annotation))
    dtype = getattr(annotation, "dtype", None)
    if dtype is not None:
        shape = getattr(annotation, "shape", None)
        extents = tuple(str(e) for e in shape) if shape is not None else ()
        return ("facade", type(annotation).__name__, dtype, extents)
    from pypto.language.typing import annotation_to_type  # noqa: PLC0415 — cycle-free laziness

    with raw_print():
        return ("raw", str(annotation_to_type(annotation)))


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
    param_names: list[str],
    args: tuple[Any, ...] | list[Any],
    fn_name: str,
) -> None:
    """The declaration-pinned parameter worlds (types flow from values).

    A parameter declared with a pl facade annotation is a **runtime
    parameter** (always in the signature, always a runtime argument); a plain
    Python annotation (or none) is a **host knob** (baked at trace time,
    never IR). A call site contradicting the declaration is a type lie and is
    loud in both directions: a host scalar into a runtime parameter must be
    spelled ``pl.runtime`` (the entry) or re-declared, and a staged value
    cannot become host. This is the migration contract's no-context-retyping
    rule applied to call arguments; the ``Marker`` lane has always been loud
    on host args, so this converges the jit lanes to it.
    """
    for name, arg in zip(param_names, args):
        hint = param_hints.get(name)
        staged_declared = _staged_declared(hint)
        if staged_declared and isinstance(arg, (bool, int, float)):
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared {hint!r} "
                "(a runtime parameter) but received a host "
                f"{type(arg).__name__} — pass pl.runtime(value, dtype) for a "
                "runtime scalar input, or drop the pl annotation to make it "
                "a host knob baked at trace time"
            )
        if not staged_declared and is_staged(arg):
            raise TypeError(
                f"call to {fn_name!r}: parameter {name!r} is declared "
                f"{hint!r} (a host value baked at trace time) but received a "
                "staged value — a staged value cannot become host; annotate "
                "the parameter (e.g. pl.Scalar[dtype]) to make it runtime"
            )


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
    from the frozen decoration-time function, so every Marker consumer
    (``pl.spmd_submit`` validation, program-method adaptation) accepts it
    unchanged.

    Without a decoration record — or outside a trace — the call is the
    verbatim lane: register the function with the active program assembly at
    first call and emit the detached ``ir.Call`` (a global the body never
    calls stays out of the program).

    With a decoration record (the global is an eagerly traced jit callee), an
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
        decoration record inside a trace, a Submit reference instantiates
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
        if kwargs and (active_frame() is None or set(kwargs) - {"device", "attrs"}):
            raise TypeError(
                f"call to {self._fn.name!r}: keyword arguments are not accepted "
                f"on a jit call (parser-aligned — pass all arguments positionally)"
            )
        if self._record is not None and active_frame() is not None:
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
        param_names = spec.param_names()
        if len(args) != len(param_names):
            raise TypeError(
                f"call to {self._fn.name!r} expects {len(param_names)} argument(s), got {len(args)}"
            )
        _check_param_worlds(spec._param_hints, param_names, args, self._fn.name)  # noqa: SLF001 — the in-module lane
        annotations = {
            name: _in_trace_annotation(name, arg, self._fn.name) for name, arg in zip(param_names, args)
        }
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        callee: ir.Function | None = None
        if assembly is not None:
            key = (record.kernel, *(_variant_key_part(a) for a in annotations.values()))
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


def _compile_program(program: ir.Program, config: Any = None) -> Any:
    from pypto.ir import compile as ir_compile  # noqa: PLC0415 — cycle-free laziness

    kwargs: dict[str, Any] = {}
    if config is not None:
        from pypto.runtime.runner import run_config_compile_kwargs  # noqa: PLC0415 — torch is a runner dep

        # ``platform`` is forwarded alongside the mapped fields: ir.compile()
        # derives the codegen backend from it (``backend_type`` stays implicit,
        # mirroring the RunConfig field-mapping rationale).
        kwargs["platform"] = config.platform
        kwargs.update(run_config_compile_kwargs(config))
    return ir_compile(program, **kwargs)


def _checked_run_config(config: Any) -> Any:
    """Validate a caller-supplied ``config=``: ``None`` passes, anything but
    a ``RunConfig`` is a loud TypeError (duck-typed access would surface as a
    confusing AttributeError deep in the field mapping)."""
    if config is None:
        return None
    from pypto.runtime.runner import RunConfig  # noqa: PLC0415 — torch is a runner dep

    if not isinstance(config, RunConfig):
        raise TypeError(f"config= must be a pypto.runtime.RunConfig, got {type(config).__name__!r}")
    return config


# —— the ``pl.jit`` binding: the quarantined specializer's per-func_type
# mapping (specializer.py's generated ``@pl.function`` form), traced ——


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
    ) -> None:
        self._label = label
        self._function_type = function_type
        self._level = level
        self._role = role
        self._allow_level = allow_level

    def __call__(
        self,
        kernel: types.FunctionType | None = None,
        *,
        level: ir.Level | None = None,
        auto_scope: bool = True,
    ) -> Any:
        if level is not None and not self._allow_level:
            raise TypeError(f"@pl.jit.{self._label} does not accept level=")
        if kernel is None:  # the parenthesized form: @pl.jit.<alias>(...)
            return functools.partial(self, level=level, auto_scope=auto_scope)
        return jit_trace(
            kernel,
            function_type=self._function_type,
            level=level if level is not None else self._level,
            role=self._role,
            attrs=fold_auto_scope(None, auto_scope),
        )


def _resolve_extern_path(path: str, kernel: types.FunctionType, *, what: str) -> str:
    """Resolve an extern source/include path relative to the stub's file
    (the ``@pl.program`` external_source convention)."""
    if not os.path.isabs(path):
        src_file = inspect.getsourcefile(kernel)
        if src_file is not None and not src_file.startswith("<"):
            path = os.path.join(os.path.dirname(src_file), path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"@pl.jit.extern {what} file not found: {path}")
    return path


def _resolve_extern_include_dirs(
    include_dirs: Sequence[str | os.PathLike[str]] | None, kernel: types.FunctionType
) -> tuple[str, ...]:
    """Resolve extern include directories relative to the stub's file."""
    if include_dirs is None:
        return ()
    if isinstance(include_dirs, (str, bytes, os.PathLike)):
        raise TypeError(
            f"@pl.jit.extern include_dirs must be a sequence of paths, got {type(include_dirs).__name__}"
        )
    src_file = inspect.getsourcefile(kernel)
    base_dir = os.path.dirname(src_file) if src_file is not None and not src_file.startswith("<") else None
    resolved: list[str] = []
    for include_dir in include_dirs:
        path = os.fspath(include_dir)
        if not os.path.isabs(path) and base_dir is not None:
            path = os.path.join(base_dir, path)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            raise ValueError(f"@pl.jit.extern include directory not found: {path}")
        resolved.append(path)
    return tuple(resolved)


def _extern_member(
    kernel: types.FunctionType,
    *,
    member_name: str,
    function_type: ir.FunctionType,
    external_source: str,
    include_dirs: tuple[str, ...],
    dual_aiv_dispatch: bool = False,
) -> ir.Function:
    """Trace one header-only extern member (the signature-only stub is the
    declared contract; the referenced ``.cpp`` is the implementation). The
    member carries the ``<name>_<core>`` name — the specializer's member
    naming — so a mixed pair never collides."""
    attrs: dict[str, Any] | None = None
    if include_dirs:
        attrs = {EXTERNAL_INCLUDE_DIRS_ATTR: encode_external_include_dirs(include_dirs)}
    if dual_aiv_dispatch:
        attrs = {**(attrs or {}), DUAL_AIV_DISPATCH_ATTR: True}
    fn = function_trace(
        kernel,
        function_type=function_type,
        external_source=external_source,
        attrs=attrs,
        localns=capture_caller_names(),
    )
    return ir.Function(
        member_name,
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


class _MixedExternJit:
    """A mixed (AIC+AIV) extern kernel on the tracing jit: two traced
    members plus a Group wrapper named after the stub, assembled at the
    call site — the ``@pl.program`` group route the specializer rendered
    into generated source. A value-returning kernel threads the return
    through the AIC member (the AIC lane echoes the outputs; AIV lanes
    never capture one)."""

    def __init__(
        self,
        kernel: types.FunctionType,
        *,
        aic: ir.Function,
        aiv: ir.Function,
    ) -> None:
        self.kernel = kernel
        self._aic = aic
        self._aiv = aiv
        self._aic_marker = Marker._from_ir_function(aic)
        self._aiv_marker = Marker._from_ir_function(aiv)
        self._param_names = [
            p.name
            for p in inspect.signature(kernel).parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        self.__name__ = kernel.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs:
            raise TypeError(f"call to extern {self.kernel.__name__!r}: keyword arguments are not accepted")
        if len(args) != len(self._param_names):
            raise TypeError(
                f"call to extern {self.kernel.__name__!r} expects {len(self._param_names)} "
                f"argument(s), got {len(args)}"
            )
        frame = active_frame()
        if frame is None:
            raise TypeError(
                f"call to extern {self.kernel.__name__!r} outside a trace — an extern "
                "kernel is a callee; call it inside a @pl.jit entry"
            )
        annotations = {
            name: _in_trace_annotation(name, arg, self.kernel.__name__)
            for name, arg in zip(self._param_names, args)
        }
        assembly = _ACTIVE_ASSEMBLY.get()
        key: tuple[Any, ...] | None = None
        group: ir.Function | None = None
        if assembly is not None:
            from pypto.language.typing import annotation_to_type  # noqa: PLC0415 — cycle-free laziness

            key = (id(self), *(str(annotation_to_type(a)) for a in annotations.values()))
            group = assembly.by_key.get(key)
        if group is None:
            group = self._build_group(annotations)
            if assembly is not None and key is not None:
                assembly.register(self._aic)
                assembly.register(self._aiv)
                assembly.register(group, key=key)
        return Marker._from_ir_function(group)(*args)

    def _build_group(self, annotations: dict[str, object]) -> ir.Function:
        """Trace the Group wrapper: the stub's signature, the body the
        specializer rendered (``_r = <aic>(args); <aiv>(args); return _r``).
        The member Markers emit the member Calls into this frame directly
        (no instrumented kernel — the wrapper is synthetic)."""
        from pypto.language.ambient import (  # noqa: PLC0415 — cycle-free laziness
            reset_active_frame,
            set_active_frame,
        )
        from pypto.language.tracer.semantics import PtoSema  # noqa: PLC0415
        from pypto.language.typing import annotation_to_type  # noqa: PLC0415

        return_type = typing.get_type_hints(self.kernel, localns=capture_caller_names()).get("return")
        sema = PtoSema(
            name=self.kernel.__name__,
            return_type=annotation_to_type(return_type) if is_staged(return_type) else None,
            function_type=ir.FunctionType.Group,
            source_span=source_span_for_function(self.kernel),
        )
        facades = sema.bind_parameters(annotations, directions=param_directions(self.kernel))
        sema.begin_frame()  # lock the one-shot trace (the seeded frame handout)
        frame = sema.frame
        assert frame is not None
        token = set_active_frame(frame)
        try:
            ib = sema._builder  # noqa: SLF001 — the synthetic-wrapper driver (no kernel to instrument)
            aic_result = self._aic_marker(*facades)
            aiv_result = self._aiv_marker(*facades)
            if aic_result is not None and is_staged(aic_result):
                r_var = ib.var("_r", aic_result.unwrap().type, ir.Span.unknown())
                ib.assign(r_var, aic_result.unwrap(), ir.Span.unknown())
                ib.eval_stmt(aiv_result.unwrap(), ir.Span.unknown())
                ib.return_stmt(r_var, ir.Span.unknown())
            else:
                ib.eval_stmt(aic_result.unwrap(), ir.Span.unknown())
                ib.eval_stmt(aiv_result.unwrap(), ir.Span.unknown())
        finally:
            reset_active_frame(token)
        frame.finish()
        return sema.result()


class _JitExtern:
    """``@pl.jit.extern`` — a hand-written C++ kernel on the tracing jit.

    The decorated function is a signature-only stub (``...`` body); its
    implementation is the referenced ``.cpp``. Single core traces one
    header-only AIC/AIV member (an ``ir.Function`` — the entry's globals
    adaptation makes it callable in-trace). ``mixed`` traces an AIC member
    + an AIV member and returns a callable that assembles the Group
    wrapper at the call site."""

    def __call__(
        self,
        kernel: types.FunctionType | None = None,
        *,
        core_type: str = "aiv",
        source: str | Any = None,
        aic_source: str | Any = None,
        aiv_source: str | Any = None,
        dual_aiv_dispatch: bool = False,
        include_dirs: Sequence[str | os.PathLike[str]] | None = None,
    ) -> Any:
        if kernel is None:  # the parenthesized form: @pl.jit.extern(...)
            return functools.partial(
                self,
                core_type=core_type,
                source=source,
                aic_source=aic_source,
                aiv_source=aiv_source,
                dual_aiv_dispatch=dual_aiv_dispatch,
                include_dirs=include_dirs,
            )
        if core_type not in ("aic", "aiv", "mixed"):
            raise ValueError(f"@pl.jit.extern core_type must be 'aic', 'aiv', or 'mixed', got {core_type!r}")
        if dual_aiv_dispatch and core_type != "mixed":
            raise ValueError("@pl.jit.extern dual_aiv_dispatch=True requires core_type='mixed'")
        dirs = _resolve_extern_include_dirs(include_dirs, kernel)
        if core_type == "mixed":
            if aic_source is None or aiv_source is None:
                raise ValueError(
                    "@pl.jit.extern(core_type='mixed') requires both aic_source= and aiv_source="
                )
            aic = _extern_member(
                kernel,
                member_name=f"{kernel.__name__}_aic",
                function_type=ir.FunctionType.AIC,
                external_source=_resolve_extern_path(os.fspath(aic_source), kernel, what="source"),
                include_dirs=dirs,
            )
            aiv = _extern_member(
                kernel,
                member_name=f"{kernel.__name__}_aiv",
                function_type=ir.FunctionType.AIV,
                external_source=_resolve_extern_path(os.fspath(aiv_source), kernel, what="source"),
                include_dirs=dirs,
                dual_aiv_dispatch=dual_aiv_dispatch,
            )
            return _MixedExternJit(kernel, aic=aic, aiv=aiv)
        single = source if source is not None else (aic_source or aiv_source)
        if single is None:
            raise ValueError(f"@pl.jit.extern(core_type={core_type!r}) requires source=")
        return _extern_member(
            kernel,
            member_name=kernel.__name__,
            function_type=ir.FunctionType.AIC if core_type == "aic" else ir.FunctionType.AIV,
            external_source=_resolve_extern_path(os.fspath(single), kernel, what="source"),
            include_dirs=dirs,
        )


class _JitDecorator:
    """The ``pl.jit`` object — the traced jit envelope. The quarantined
    specializer's function-class mapping:

    ==================  =============================================
    ``@pl.jit``         ``@pl.function(type=Orchestration)``
    ``@pl.jit.host``    ``@pl.function(level=HOST, role=Orchestrator)``
    ``@pl.jit.incore``  ``@pl.function(type=InCore[, level=…])``
    ``@pl.jit.inline``  ``@pl.function(type=Inline)``
    ``@pl.jit.opaque``  ``@pl.function(type=Opaque)``
    ``@pl.jit.graph``   ``@pl.function(type=Graph)`` — a recordable
                        orchestration fragment (``host_build_graph``)
    ``@pl.jit.extern``  the extern tier (AIC/AIV/mixed members)
    ==================  =============================================
    """

    def __init__(self) -> None:
        self.host = _JitAlias("host", level=ir.Level.HOST, role=ir.Role.Orchestrator)
        self.incore = _JitAlias("incore", function_type=ir.FunctionType.InCore, allow_level=True)
        self.inline = _JitAlias("inline", function_type=ir.FunctionType.Inline)
        self.opaque = _JitAlias("opaque", function_type=ir.FunctionType.Opaque)
        self.graph = _JitAlias("graph", function_type=ir.FunctionType.Graph)
        self.extern = _JitExtern()

    def __call__(
        self,
        kernel: types.FunctionType | None = None,
        *,
        auto_scope: bool = True,
    ) -> Any:
        if kernel is None:  # the parenthesized form: @pl.jit(auto_scope=False)
            return functools.partial(self, auto_scope=auto_scope)
        return jit_trace(
            kernel,
            function_type=ir.FunctionType.Orchestration,
            attrs=fold_auto_scope(None, auto_scope),
        )


jit = _JitDecorator()
