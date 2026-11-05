# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Where a test case's IR comes from.

The compile pipeline needs exactly one thing from a test case: an
``ir.Program`` that **no pass has run on yet**, because ``ir.compile()`` runs
the pass pipeline *and* code generation itself.  Everything else about how the
kernel was authored is irrelevant to it.

``KernelSource`` is that one thing, so the three authoring surfaces stop being
three execution paths:

======================  =========================================
Authoring surface       KernelSource
======================  =========================================
``@pl.jit`` entry        :class:`JitKernel`
``@pl.jit_trace`` entry  :class:`TracingJitKernel` (either construction mode)
``@pl.program`` class   :class:`ProgramKernel`
an ``ir.Program``       :class:`IRKernel`
======================  =========================================

A source may also derive its own :class:`~harness.core.harness.TensorSpec`
list.  ``JitKernel`` can — parameter directions come from the kernel's
``pl.Out`` / ``pl.InOut`` annotations and shapes come from the sample tensors —
so a JIT-authored case never writes its shapes a second time.  The other two
return ``None`` and let the case declare them.
"""

import threading
from typing import Any, Protocol, runtime_checkable

import torch
from pypto.jit import JITFunction
from pypto.language.tracing.decorators.jit_trace import _eager_trace_record, _JitSpecializer
from pypto.language.typing.arg_direction import param_directions
from pypto.language.typing.tensor import Tensor as _TensorFacade
from pypto.pypto_core import DataType as _PyptoDType
from pypto.pypto_core import ir as _ir

from harness.core.harness import DataType, TensorSpec

# Serialises every program build across the compile pool.
#
# Neither authoring surface is thread-safe: the ``@pl.program`` decorator
# mutates module-level builder state, and ``JITFunction``'s specializer walks
# and memoises a shared dep graph.  The compile pool runs ``build_program()``
# under this lock and everything after it — ``ir.compile()``, golden
# materialisation, the .so build — concurrently, which is where the parallelism
# actually is.
#
# One lock for all sources, on purpose: a per-source lock would let a
# ``@pl.program`` build race a JIT specialization, and the JIT specializer
# emits ``@pl.program`` source.
#
# Reentrant, because the acquisition nests: ``_compile_for_cache`` takes the
# lock around ``get_program()``, and for a ``Case`` that call lands right back
# here in ``build_program()``. Re-entering from the owning thread is safe by
# construction — the thread already has exclusive access and the nested build
# finishes before it releases — whereas a plain Lock self-deadlocks.
program_build_lock = threading.RLock()


@runtime_checkable
class KernelSource(Protocol):
    """Supplies the pre-pass IR for one test case."""

    def build_program(self) -> Any:
        """Return the ``ir.Program`` to compile, before any pass has run.

        Called on a compile-pool worker. Implementations MUST hold
        :data:`program_build_lock` for the duration of the build.
        """
        ...

    def tensor_specs(self) -> list[TensorSpec] | None:
        """Return the case's tensors, or ``None`` to let the case declare them."""
        ...

    def cache_id(self) -> str:
        """Stable identity for the compiled artifact of this kernel."""
        ...


# ---------------------------------------------------------------------------
# torch dtype -> harness DataType
# ---------------------------------------------------------------------------

# ``DataType.torch_dtype`` is many-to-one (UINT32 and INT32 both map to
# torch.int32, UINT16 and INT16 both to torch.int16), so the reverse map is
# built explicitly and resolves each collision to the signed member. A test
# that needs the unsigned reading declares that TensorSpec itself.
_TORCH_TO_DATATYPE: "dict[torch.dtype, DataType]" = {}
for _member in DataType:
    try:
        _torch_dtype = _member.torch_dtype
    except (ValueError, KeyError):
        continue  # optional MX dtype this torch build does not provide
    _TORCH_TO_DATATYPE.setdefault(_torch_dtype, _member)
del _member


# The IR dtype enum → harness DataType member name, keyed on enum identity.
# The two enums spell their values differently ("bfloat16" vs "bf16"), so the
# member name is the only shared spelling.
_PYPTO_DTYPE_NAMES = {getattr(_PyptoDType, m.name): m.name for m in DataType if hasattr(_PyptoDType, m.name)}


def datatype_from_torch(dtype: torch.dtype) -> DataType:
    """Return the harness :class:`DataType` matching a torch dtype.

    Raises:
        ValueError: The dtype has no harness equivalent, naming what was
            received and what is available.
    """
    try:
        return _TORCH_TO_DATATYPE[dtype]
    except KeyError:
        known = ", ".join(sorted(str(d) for d in _TORCH_TO_DATATYPE))
        raise ValueError(f"No harness DataType for torch dtype {dtype}. Known dtypes: {known}") from None


# ---------------------------------------------------------------------------
# Sample-argument binding (shared by the two tracing jit lanes)
# ---------------------------------------------------------------------------


def _bind_samples(
    names: "list[str]", label: str, args: tuple[Any, ...], kwargs: dict[str, Any], why: str
) -> tuple[Any, ...]:
    """Bind positional + by-name sample arguments to signature order.

    The jit call itself is parser-aligned positional; the by-name form is
    ``st.case(inputs=...)``, resolved here once. Every parameter must be bound
    — *why* names the lane's reason.
    """
    unknown = sorted(set(kwargs) - set(names))
    if unknown:
        raise TypeError(f"{label}: not a parameter of the kernel: {unknown}")
    bound: dict[str, Any] = dict(zip(names, args))
    clashes = sorted(set(bound) & set(kwargs))
    if clashes:
        raise TypeError(
            f"{label}: parameter(s) {clashes} got both a positional and a keyword sample argument"
        )
    bound.update(kwargs)
    missing = [n for n in names if n not in bound]
    if missing:
        raise TypeError(f"{label}: no sample argument for parameter(s) {missing} — {why}")
    return tuple(bound[n] for n in names)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


class JitKernel:
    """A parsing-lane ``@pl.jit`` entry plus the sample arguments that specialize it.

    The sample arguments carry the shape/dtype contract, exactly as they do
    when a test calls the kernel directly.  They are the case's real inputs, so
    :meth:`tensor_specs` derives the whole tensor list from them and the test
    never restates a shape.

    Sample arguments may be omitted only when every tensor parameter carries a
    fully-shaped annotation (``pl.Tensor[[128, 128], pl.FP32]``); a bare
    ``pl.Tensor`` has no shape to read and ``specialize()`` says so.
    """

    def __init__(self, entry: Any, *args: Any, **kwargs: Any):
        # ``entry`` is typed Any because the guard below is what test authors
        # actually hit: passing a @pl.program class here is the natural mistake,
        # and it must fail with a message that names the right source.
        if not isinstance(entry, JITFunction):
            raise TypeError(
                f"JitKernel expects a parsing-frontend @pl.jit function, got {type(entry).__name__}. "
                "Use ProgramKernel for a @pl.program class, TracingJitKernel for a "
                "@pl.jit_trace entry, or IRKernel for an ir.Program."
            )
        if "config" in kwargs:
            raise ValueError(
                f"JitKernel({entry.__name__}) does not take config=: compile knobs "
                "(strategy, memory_planner, platform) belong on the Case, which applies "
                "them to ir.compile() for every source alike."
            )
        self.entry: JITFunction = entry
        self.args = args
        self.kwargs = kwargs
        self._tensor_params: set[str] | None = None

    def build_program(self) -> Any:
        with program_build_lock:
            return self.entry.specialize(*self.args, **self.kwargs)

    def tensor_specs(self) -> list[TensorSpec] | None:
        """Derive the tensor list from the kernel signature and sample args.

        A parameter is an output when the kernel annotates it ``pl.Out[...]``
        or ``pl.InOut[...]``.

        **Every** tensor is seeded from its sample argument, outputs included.
        The sample argument is the buffer the test itself prepared, so its
        contents are the test's statement of what the kernel starts from —
        usually zeros, but not always. An atomic-add kernel accumulates onto
        the destination it is handed, so zeroing a ``pl.Out`` buffer here
        because the annotation says "output" would silently discard the
        baseline the test set up and compare against a golden that assumed it.
        Seeding costs one small host-to-device copy for a buffer the kernel
        overwrites anyway.

        Returns ``None`` when the sample arguments do not cover every tensor
        parameter (annotation-only specialization) — the case declares its
        tensors in that case.
        """
        bound = self._bound_tensors()
        if bound is None:
            return None
        outputs = set(self.entry.output_param_names)
        specs: list[TensorSpec] = []
        for name, value in bound.items():
            specs.append(
                TensorSpec(
                    name=name,
                    shape=list(value.shape),
                    dtype=datatype_from_torch(value.dtype),
                    init_value=value,
                    is_output=name in outputs,
                )
            )
        return specs

    def cache_id(self) -> str:
        """Stable identity for this kernel's specialization.

        Covers **every** bound argument, in declaration order, because every one
        of them shapes the specialized program: a tensor through its shape and
        dtype, anything else through its type and value.

        A scalar passed *positionally* used to be omitted entirely -- only
        keyword scalars were recorded -- so two declarations differing solely in
        such a scalar produced the same id. That id is the default ``Case``
        name, and the name is the artifact cache key, so the second declaration
        silently reused the first one's compiled artifact and could pass against
        a specialization that was never built for it.

        The type is part of the id rather than the value alone: ``1`` and
        ``1.0`` specialize to different dtypes and must not collide.
        """
        parts = [self.entry.__name__]
        bound = self._bound_arguments()
        for name in self.entry.param_names:
            if name not in bound:
                continue
            value = bound[name]
            if isinstance(value, torch.Tensor):
                shape = "x".join(str(d) for d in value.shape)
                parts.append(f"{name}_{shape}_{datatype_from_torch(value.dtype).value}")
            else:
                parts.append(f"{name}_{type(value).__name__}_{value}")
        return "__".join(parts)

    def _tensor_param_names(self) -> set[str]:
        """Names of parameters that own a tensor slot, parsed once.

        A scalar parameter is specialized as a literal and owns no slot, so it
        is not "missing" when no sample tensor is supplied for it.
        ``param_names`` alone cannot tell the two apart.
        """
        if self._tensor_params is None:
            from pypto.jit._ast_analysis import get_func_def  # noqa: PLC0415
            from pypto.jit.specializer import _classify_params  # noqa: PLC0415

            self._tensor_params = set(_classify_params(get_func_def(self.entry._func))[2])
        return self._tensor_params

    def _bound_arguments(self) -> "dict[str, Any]":
        """Every supplied argument keyed by parameter name, positional and keyword.

        :meth:`_bound_tensors` keeps only tensors, which is the right question
        for the tensor list and the wrong one for identity: a scalar
        specializes the kernel into different IR whichever way it was passed.
        """
        bound: dict[str, Any] = dict(zip(self.entry.param_names, self.args))
        bound.update(self.kwargs)
        return bound

    def _bound_tensors(self) -> dict[str, torch.Tensor] | None:
        """Sample tensors keyed by parameter name, or ``None`` if incomplete."""
        names = self.entry.param_names
        bound: dict[str, torch.Tensor] = {}
        for name, value in zip(names, self.args):
            if isinstance(value, torch.Tensor):
                bound[name] = value
        for name, value in self.kwargs.items():
            if isinstance(value, torch.Tensor):
                bound[name] = value
        # Every tensor parameter must be covered; see _tensor_param_names.
        tensor_params = self._tensor_param_names()
        missing = [n for n in names if n not in bound and n in tensor_params]
        return None if missing else {n: bound[n] for n in names if n in bound}

    def __repr__(self) -> str:
        return f"JitKernel({self.entry.__name__})"


class TracingJitKernel:
    """A tracing ``@pl.jit_trace`` entry plus the buffers a case runs it with.

    One source, two constructions — chosen by what determines the program:

    * **Declaration-driven** — every parameter carries a fully-shaped facade
      annotation, so the jit traced the kernel at first use and exposes that
      one trace as ``.fn``; the baked trace is the artifact. The samples may
      be omitted entirely (a fully-shaped declaration is a self-sufficient
      contract — the case states its buffers through ``tensors=`` instead,
      the same rule the parsing lane's :class:`JitKernel` follows) or
      supplied as the case's real inputs; either way they specialize
      nothing and are cross-checked against the baked signature, because
      a baked signature cannot be resized by a sample.
    * **Sample-driven** — a bare or unannotated parameter forces re-tracing
      from the call arguments, so every parameter needs a sample; the
      samples carry the shape/dtype contract exactly as they do when a test
      calls the kernel directly.

    A bare ``ir.Function`` (a caller that already resolved ``.fn``, or a
    ``pl.function``-traced kernel) constructs declaration-driven too: its
    signature is baked, and when the jit produced it its record keeps the
    callees the trace collected.

    The boundary every mode shares: :meth:`build_program` hands the harness
    the complete pre-pass program — the entry plus every callee its trace
    reached — through the jit's own assembly route; the case owns the
    compile knobs (``config=`` is refused here) and runs ``ir.compile()``
    once, so no source runs a pass pipeline of its own.
    """

    def __init__(self, entry: Any, *args: Any, **kwargs: Any):
        # ``entry`` is typed Any because the guard below is what test authors
        # actually hit: passing a @pl.program class or a parsing-lane
        # ``@pl.jit`` function must fail with a message that names the right
        # source, not deep inside the trace.
        jit: _JitSpecializer | None = None
        baked: _ir.Function | None = None
        if isinstance(entry, _JitSpecializer):
            # One public jit object covers both modes: a decoration trace
            # (``.fn``) exists exactly when the declaration determined the
            # trace — the property is loud on the sample-driven mode by
            # design, so the probe is the mode test. The jit object is the
            # assembly route either way.
            jit = entry
            try:
                baked = entry.fn
            except AttributeError:
                baked = None
        elif isinstance(entry, _ir.Function):
            # A bare ir.Function that *is* a decoration trace keeps its record,
            # and with it the callees ``.fn`` alone does not name.
            record = _eager_trace_record(entry)
            jit = record.specializer() if record is not None else None
            baked = entry
        else:
            raise TypeError(
                f"TracingJitKernel expects a tracing @pl.jit_trace function or its "
                f"traced-out ir.Function, got {type(entry).__name__}. "
                "Use JitKernel for the parsing-lane @pl.jit, ProgramKernel for a "
                "@pl.program class or IRKernel for an ir.Program."
            )
        label = baked.name if baked is not None else entry.__name__
        if "config" in kwargs:
            raise ValueError(
                f"TracingJitKernel({label}) does not take config=: compile knobs "
                "(strategy, memory_planner, platform) belong on the Case, which applies "
                "them to ir.compile() for every source alike."
            )
        self._jit = jit
        self._baked = baked
        self.entry: Any = baked if baked is not None else entry
        if baked is not None:
            self._names: list[str] = [p.name_hint for p in baked.params]
            self._directions: set[Any] = set(baked.param_directions)
            # All-or-none, mirroring the jit's own sample binding: either
            # every parameter's buffer is bound, or none is and the baked
            # declaration alone determines the program (annotation-only
            # construction).
            self._annotation_only = not args and not kwargs
            self.args: tuple[Any, ...] = (
                ()
                if self._annotation_only
                else _bind_samples(
                    self._names,
                    f"TracingJitKernel({label})",
                    args,
                    kwargs,
                    "the sample is the parameter's run-time buffer — pass every "
                    "buffer, or none and declare the case's tensors= instead",
                )
            )
        else:
            self._names = list(entry.param_names)
            self._directions = set(param_directions(entry.kernel))
            # The declaration leaves shapes unknown, so the samples
            # specialize the kernel: every parameter needs one.
            self._annotation_only = False
            self.args = _bind_samples(
                self._names,
                f"TracingJitKernel({label})",
                args,
                kwargs,
                "the tracing jit specializes from the call arguments",
            )

    def build_program(self) -> Any:
        with program_build_lock:
            # The jit's own assembly route: ``specialize`` returns the trace's
            # program — the entry plus every callee its body reached
            # (``.fn`` alone would leave those calls dangling) — before any
            # pass runs, which is exactly the pass-less ``ir.Program`` the
            # harness's compile step requires. A bare ``ir.Function`` with
            # no record has no callees to lose: the one function is the
            # whole program.
            if self._jit is not None:
                return self._jit.specialize(*self.args)
            baked = self._baked
            assert baked is not None
            return _ir.Program([baked], baked.name, baked.span)

    def tensor_specs(self) -> list[TensorSpec] | None:
        """Derive the tensor list, in whichever mode the declaration left.

        Sample-driven, the bridged argument for a parameter is its
        annotation-mode facade, so it says which parameters own a tensor slot
        without an AST scan: a host knob (a Python scalar annotation) bridges
        to itself and owns no slot. Declaration-driven, the traced parameter
        types already carry each tensor's dtype and its constant extents, so
        the samples are cross-checked against those — a baked signature
        cannot be resized by a sample, and a mismatched buffer would
        otherwise only fail on the device. A dynamic extent is bound at
        dispatch from the real buffer, so it is exempt and the sample's
        concrete shape becomes the spec's.

        **Every** tensor is seeded from its sample argument, outputs included,
        for the same reason :meth:`JitKernel.tensor_specs` gives: the sample is
        the buffer the test itself prepared, so its contents are the test's
        statement of what the kernel starts from.

        Returns ``None`` when a tensor parameter's sample is not a torch
        tensor, and outright for annotation-only construction (no buffer was
        bound, so the case's ``tensors=`` declaration is the whole tensor
        list) — the case declares its tensors in those cases.
        """
        if self._baked is not None:
            return self._declaration_specs()
        return self._sample_specs()

    def _sample_specs(self) -> list[TensorSpec] | None:
        jit = self._jit
        assert jit is not None
        bridged, _ = jit._bridge_args(self.args)
        specs: list[TensorSpec] = []
        for name, arg, annotation in zip(self._names, self.args, bridged):
            if not isinstance(annotation, _TensorFacade):
                continue
            if not isinstance(arg, torch.Tensor):
                return None
            specs.append(
                TensorSpec(
                    name=name,
                    shape=list(arg.shape),
                    dtype=datatype_from_torch(arg.dtype),
                    init_value=arg,
                    is_output=name in self._directions,
                )
            )
        return specs

    def _declaration_specs(self) -> list[TensorSpec] | None:
        baked = self._baked
        assert baked is not None
        if self._annotation_only:
            return None
        specs: list[TensorSpec] = []
        for name, arg, param, direction in zip(self._names, self.args, baked.params, baked.param_directions):
            if not isinstance(arg, torch.Tensor):
                return None
            dtype = datatype_from_torch(arg.dtype)
            # A dynamic extent traces as a Var, not a ConstInt — it is bound at
            # dispatch from the real buffer, so only constant extents are
            # cross-checkable and the sample's concrete shape is the spec's.
            traced_shape = [getattr(d, "value", None) for d in param.type.shape]
            traced_dtype = _PYPTO_DTYPE_NAMES.get(param.type.dtype)
            unsigned_twin = _UNSIGNED_TWINS.get(dtype.name)
            mismatch = traced_dtype not in (dtype.name, unsigned_twin) or any(
                expected is not None and got != expected for expected, got in zip(traced_shape, arg.shape)
            )
            if len(arg.shape) != len(traced_shape) or mismatch:
                raise TypeError(
                    f"TracingJitKernel({baked.name}): sample for {name!r} is "
                    f"{list(arg.shape)}x{dtype.name}, but the traced parameter is "
                    f"{traced_shape}x{traced_dtype} — a declaration-driven kernel baked "
                    "its signature at decoration and cannot be resized by a sample"
                )
            specs.append(
                TensorSpec(
                    name=name,
                    shape=list(arg.shape),
                    dtype=dtype,
                    init_value=arg,
                    is_output=direction in (_ir.ParamDirection.Out, _ir.ParamDirection.InOut),
                )
            )
        return specs

    def cache_id(self) -> str:
        """Stable identity for this kernel's compiled artifact.

        Sample-driven, every bound argument shapes the specialized program —
        same contract as :meth:`JitKernel.cache_id`, type included so ``1``
        and ``1.0`` never collide — so every one is recorded. Declaration-
        driven, the samples only pick the buffers; the id records their
        shapes/dtypes (never values) to stay descriptive without implying
        they specialize anything. The annotation-only form has no buffers to
        record, so the baked signature stands in: the same shape/dtype
        facts, read from the traced parameter types instead of the samples.
        """
        if self._baked is not None:
            return self._declaration_cache_id()
        parts = [self.entry.__name__]
        for name, value in zip(self._names, self.args):
            if isinstance(value, torch.Tensor):
                shape = "x".join(str(d) for d in value.shape)
                parts.append(f"{name}_{shape}_{datatype_from_torch(value.dtype).value}")
            else:
                parts.append(f"{name}_{type(value).__name__}_{value}")
        return "__".join(parts)

    def _declaration_cache_id(self) -> str:
        baked = self._baked
        assert baked is not None
        parts = [baked.name]
        if self._annotation_only:
            for name, param in zip(self._names, baked.params):
                if isinstance(param.type, _ir.TensorType):
                    shape = "x".join(_extent_text(d) for d in param.type.shape)
                    dtype = _PYPTO_DTYPE_NAMES.get(param.type.dtype) or str(param.type.dtype)
                    parts.append(f"{name}_{shape}_{dtype}")
                else:
                    parts.append(f"{name}_scalar")
            return "__".join(parts)
        for name, value in zip(self._names, self.args):
            if isinstance(value, torch.Tensor):
                shape = "x".join(str(d) for d in value.shape)
                parts.append(f"{name}_{shape}_{datatype_from_torch(value.dtype).value}")
            else:
                parts.append(f"{name}_{type(value).__name__}")
        return "__".join(parts)

    def __repr__(self) -> str:
        name = self.entry.name if isinstance(self.entry, _ir.Function) else self.entry.__name__
        return f"TracingJitKernel({name})"


# The torch→harness dtype map is many-to-one (torch.int32 is both INT32 and
# UINT32), so the unsigned twin of the torch-resolved member is an equally
# valid reading of the same bits for the declaration-driven sample cross-check.
_UNSIGNED_TWINS = {"INT32": "UINT32", "INT16": "UINT16", "INT8": "UINT8"}


def _extent_text(dim: Any) -> str:
    """A traced extent as cache-id text: its constant, or ``?`` when dynamic.

    A dynamic extent is bound at dispatch, so it is deliberately not part of
    the identity — every concrete dispatch shape shares one artifact.
    """
    value = getattr(dim, "value", None)
    return str(value) if isinstance(value, int) else "?"


class ProgramKernel:
    """A ``@pl.program`` class (or a zero-argument factory returning one)."""

    def __init__(self, program: Any, *, name: str | None = None):
        if program is None:
            raise ValueError("ProgramKernel requires a @pl.program class or a factory returning one")
        self.program = program
        self._name = name or getattr(program, "__name__", type(program).__name__)

    def build_program(self) -> Any:
        # A @pl.program class is itself the program, so only a *plain* callable
        # (``PTOTestCase.get_program``, a factory) is invoked here.
        is_factory = callable(self.program) and not isinstance(self.program, type)
        with program_build_lock:
            built = self.program() if is_factory else self.program
            if built is None:
                raise ValueError(f"ProgramKernel({self._name}) factory returned None")
            return built

    def tensor_specs(self) -> list[TensorSpec] | None:
        # A @pl.program's parameter shapes live in its annotations, but the
        # initial values do not; the case declares the tensors.
        return None

    def cache_id(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"ProgramKernel({self._name})"


class IRKernel:
    """An already-built ``ir.Program`` — parser round-trips, hand-built IR."""

    def __init__(self, program: Any, *, name: str):
        if program is None:
            raise ValueError("IRKernel requires an ir.Program")
        self.program = program
        self._name = name

    def build_program(self) -> Any:
        return self.program

    def tensor_specs(self) -> list[TensorSpec] | None:
        return None

    def cache_id(self) -> str:
        return self._name

    def __repr__(self) -> str:
        return f"IRKernel({self._name})"


__all__ = [
    "IRKernel",
    "JitKernel",
    "KernelSource",
    "ProgramKernel",
    "TracingJitKernel",
    "datatype_from_torch",
    "program_build_lock",
]
