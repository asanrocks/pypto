# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared submit constructs for the PyPTO language."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pypto.language.protocol import TraceContext

from pypto.language.ambient import active_frame
from pypto.language.diagnostics import ParserTypeError
from pypto.language.facade import Facade, is_facade
from pypto.language.op.marker import Marker
from pypto.language.typing import Tuple, wrap_expr
from pypto.language.typing._emit import _EmitState, _structural_projection
from pypto.language.typing.tensor import NoDepMarker
from pypto.pypto_core import DataType
from pypto.pypto_core import ir as _ir

#: The closed submit-operand vocabulary — exactly what :func:`_submit_expr`
#: accepts at the operand boundary: the five unwrap facades, a raw
#: ``ir.Expr``, and the host scalars lifted to typed constants. Anything
#: else is a loud TypeError there.
SubmitOperand = Facade | _ir.Expr | int | float | bool

#: A submit kernel argument: an operand, or a ``pl.no_dep`` marker wrapping one.
SubmitArg = SubmitOperand | NoDepMarker

#: The ``deps=``/``dumps=`` entry-list form (a ``None`` entry is the
#: conditional-edge sentinel — skipped, never coerced).
_ARRAY_GET_ELEMENT = _ir.get_op("array.get_element").name
SubmitEntries = list[SubmitOperand | None] | tuple[SubmitOperand | None, ...]


def _submit_expr(value: Any, what: str) -> _ir.Expr:
    """Normalize a submit operand to an ``ir.Expr`` at the operand boundary
    (the one automatic host-literal lift): facades unwrap,
    raw ``ir.Expr`` pass through, host ``int``/``bool`` literals lift to
    typed constants. Anything else is loud (the AST's parse discipline:
    every submit operand is an IR expression)."""
    if is_facade(value):
        return value.unwrap()
    if isinstance(value, _ir.Expr):
        return value
    if isinstance(value, bool):
        return _ir.ConstInt(int(value), DataType.INT32, _ir.Span.unknown())
    if isinstance(value, int):
        return _ir.ConstInt(value, DataType.INT32, _ir.Span.unknown())
    if isinstance(value, float):
        return _ir.ConstFloat(value, DataType.DEFAULT_CONST_FLOAT, _ir.Span.unknown())
    raise TypeError(f"{what}: expected a staged value (facade or ir.Expr), got {value!r}")


#: The callee surface as the type checker sees it. At runtime the argument
#: is always a ``Marker`` (validated below), but ``@pl.function`` inside a
#: ``@pl.program`` class *replaces* the method with a Marker — a class
#: decorator effect invisible to static checkers — so call sites hold a
#: plain callable. The union admits that static view; the runtime check
#: keeps the real contract loud.
SubmitCallee = Marker | Callable[..., object]


def _resolve_submit_callee(
    callee: SubmitCallee, kernel_args: tuple[SubmitArg, ...], construct: str
) -> tuple[SubmitCallee, tuple[SubmitArg, ...]]:
    """Run the callee's Submit-reference resolution before validation.

    Call and submit share one instantiation semantics: a callee that
    specializes per call (the jit envelope) resolves the reference through
    the same lane a call takes, so validation, arg coercion, and the lowered
    ``ir.Submit`` all see the specialized target — never the frozen
    declaration with a stale signature. A plain Marker declines (``None``)
    and the submit lowers verbatim.
    """
    resolve = getattr(callee, "resolve_submit", None)
    if resolve is None:
        return callee, kernel_args  # _check_submit_callee raises the loud marked-callee error
    resolved = resolve(kernel_args)
    if resolved is None:
        return callee, kernel_args  # the verbatim reference: the frozen callee
    target, runtime_args = resolved
    if not isinstance(target, Marker):
        raise TypeError(f"{construct} callee resolution returned {target!r}, not a marked sub-kernel")
    return target, runtime_args


def _check_submit_callee(callee: object, kernel_args: Sequence[SubmitArg], construct: str) -> Marker:
    """Validate the marked callee and its arity (the collect-tier conformance
    contract — the Submit's Tuple[<ret>, TASK_ID] type is derived from
    the callee's *declared* return, never its implicit return). Lifted from the
    old ``_submit_collect`` body so the call sites own validation.

    Every failure raises; the return is the narrowed ``Marker`` for the
    lowering path (``_lower_submit`` wants the real type, not the widened
    :data:`SubmitCallee` union).
    """
    if not isinstance(callee, Marker):
        raise TypeError(
            f"{construct} first argument must be a marked sub-kernel (a @pl.function Marker), got {callee!r}"
        )
    declared = list(callee.declared_return_types)
    if not declared:
        if callee.return_types:
            # The lie guard: a callee that *implicitly* returns must declare the
            # ``->`` — the Submit's Tuple[<ret>, TASK_ID] type is derived from the
            # declared return, never the implicit one. A callee with no return at
            # all is unambiguous: the Submit yields only its TASK_ID.
            raise TypeError(
                f"{construct} callee {callee.name!r} must declare a -> return type — the "
                f"Submit's Tuple[<ret>, TASK_ID] type is derived from it"
            )
        if not callee.implicit_returns_pinned:
            # An annotation-less program method the decorator has not traced
            # yet (defined after the caller). Deriving TASK_ID-only from an
            # *unknown* return set would be a silent guess — the callee may
            # well return values — so stay loud.
            raise TypeError(
                f"{construct} callee {callee.name!r} must declare a -> return type — its "
                f"implicit return set is not pinned yet (annotation-less program methods "
                f"are pinned in class-body order, so define it before the caller)"
            )
    if len(kernel_args) > len(callee.params):
        raise TypeError(
            f"{construct} callee {callee.name!r} expects at most {len(callee.params)} "
            f"argument(s), got {len(kernel_args)}"
        )
    # The IR contract (the orchestration codegen's CountTrailingCommCtxParams
    # rule): CommCtx-typed params form a trailing suffix — the materialize
    # pass appends them and forwards them as the last positional args — so
    # the omitted window between the caller-supplied prefix and that suffix
    # may hold only Out parameters (runtime-allocated by the backend). A
    # mid-signature Out param is therefore omittable, exactly like the
    # pass-produced Submit shape the printer round-trips.
    params = callee.params
    directions = callee.param_directions
    ctx_count = 0
    while ctx_count < len(params) and isinstance(params[len(params) - 1 - ctx_count].type, _ir.CommCtxType):
        ctx_count += 1
    if len(kernel_args) < ctx_count:
        raise TypeError(
            f"{construct} callee {callee.name!r}: {len(kernel_args)} argument(s) "
            f"cannot cover the {ctx_count} materialized CommCtx parameter(s)"
        )
    original_arg_count = len(kernel_args) - ctx_count
    original_param_count = len(params) - ctx_count
    for index in range(original_arg_count, original_param_count):
        if directions[index] != _ir.ParamDirection.Out:
            raise TypeError(
                f"{construct} callee {callee.name!r} missing positional argument "
                f"{index}, which is declared {directions[index].name}; "
                "only Out parameters may be runtime-allocated"
            )
    return callee


def _reject_host_level_omitted_outputs(
    callee: Marker, kernel_args: Sequence[SubmitArg], construct: str
) -> None:
    """A distributed HOST submit dispatches to L3, which allocates no output
    tensors, so it cannot omit Out/InOut params the way an Orchestration-level
    submit can (mirrors the AST parser's ``host_submit`` arity window)."""
    frame = active_frame()
    if frame is None or frame.function_level is not _ir.Level.HOST:
        return
    if len(kernel_args) < len(callee.params):
        raise ParserTypeError(
            f"Distributed HOST {construct}(...) requires every callee argument, including "
            f"Out/InOut tensors; function '{callee.name}' expects {len(callee.params)} "
            f"argument(s), got {len(kernel_args)}"
        )


def _strip_no_dep_and_coerce(
    kernel_args: Sequence[SubmitArg], construct: str
) -> tuple[list[int], list[_ir.Expr]]:
    """Unwrap ``pl.no_dep`` markers and coerce each kernel arg to an
    ``ir.Expr`` at the operand boundary. Returns ``(no_dep_indices,
    arg_exprs)`` — the indices feed the Submit's ``arg_direction_overrides`` attr
    (the AST's ``_strip_call_arg_markers`` threads the same per-arg override)."""
    no_dep_indices: list[int] = []
    args_list = list(kernel_args)
    for i, a in enumerate(args_list):
        if isinstance(a, NoDepMarker):
            no_dep_indices.append(i)
            args_list[i] = a.unwrap()
    arg_exprs = [_submit_expr(a, f"{construct} argument") for a in args_list]
    return no_dep_indices, arg_exprs


def _is_dep_var_type(type_: _ir.Type | None) -> bool:
    """A ``deps=`` entry type the whole-Var path accepts: a TaskId scalar or
    a TASK_ID array (the parser's ``_is_dep_var_type``)."""
    if isinstance(type_, _ir.ScalarType) and type_.dtype == DataType.TASK_ID:
        return True
    return isinstance(type_, _ir.ArrayType) and type_.dtype == DataType.TASK_ID


def _is_per_element_task_id_read(expr: _ir.Expr) -> bool:
    """An ``array.get_element(arr, i)`` reading a TaskId from a TASK_ID
    array — the Form-1 per-element ``deps=`` entry."""
    if not isinstance(expr, _ir.Call) or expr.op.name != _ARRAY_GET_ELEMENT:
        return False
    if not isinstance(expr.type, _ir.ScalarType) or expr.type.dtype != DataType.TASK_ID:
        return False
    if not expr.args:
        return False
    return _is_dep_var_type(expr.args[0].type)


def _synthesize_deps_array(entries: list[_ir.Expr]) -> _ir.Var:
    """Fold per-element ``deps=`` entries into one fresh ``Array[K, TASK_ID]``
    (``array.create`` + ``array.update_element`` rebinds) so the existing
    whole-array dep codegen fires unchanged — the parser's
    ``_synthesize_deps_array`` on the tracing side."""
    frame = active_frame()
    if frame is None:
        raise TypeError("deps= per-element desugaring requires an active trace")
    size = _ir.ConstInt(len(entries), DataType.INDEX, _ir.Span.unknown())
    create = _ir.create_op_call("array.create", [size], {"dtype": DataType.TASK_ID}, _ir.Span.unknown())
    buffer = frame.let_temp(create, "submit_deps")
    for index, entry in enumerate(entries):
        slot = _ir.ConstInt(index, DataType.INDEX, _ir.Span.unknown())
        update = _ir.create_op_call("array.update_element", [buffer, slot, entry], {}, _ir.Span.unknown())
        buffer = frame.let_temp(update, "submit_deps")
    return buffer


def _coerce_deps(deps: Sequence[SubmitOperand | None] | None, construct: str) -> list[_ir.Expr]:
    """Coerce the ``deps=`` list to the IR's Var-based dependency form.

    The parser's ``deps=`` contract, ported to the operand boundary: a bare
    TaskId Var or a whole TASK_ID array Var passes direct; an ``arr[i]``
    per-element read (Form 1) folds the whole list into one fresh
    ``Array[K, TASK_ID]`` synthesized at the current insertion point, so
    ``manual_dep_edges`` / ``Submit.deps`` stay Var-only (a raw element Call
    is rejected by the builder and would leave invalid entries for
    ``SubmitToCallView``); a ``None`` entry is the conditional-edge sentinel
    — skipped, never coerced. Mixing a whole-array entry with per-element
    entries is loud: the synthesizer would feed the array Var into a scalar
    ``update_element`` slot.
    """
    if deps is None:
        return []
    if not isinstance(deps, (list, tuple)):
        raise TypeError(f"{construct} deps= must be a list of TaskId values, got {deps!r}")
    entries: list[_ir.Expr] = []
    needs_desugar = False
    saw_whole_array = False
    for d in deps:
        if d is None:
            continue
        expr = _submit_expr(d, f"{construct} deps entry")
        if isinstance(expr, _ir.Var) and _is_dep_var_type(expr.type):
            if isinstance(expr.type, _ir.ArrayType):
                if needs_desugar:
                    raise TypeError(
                        f"{construct} deps= cannot mix a whole TASK_ID array with "
                        "per-element entries (got an array entry after a per-element "
                        "entry) — pass the array alone or index it slot-by-slot"
                    )
                saw_whole_array = True
            entries.append(expr)
            continue
        if _is_per_element_task_id_read(expr):
            if saw_whole_array:
                raise TypeError(
                    f"{construct} deps= cannot mix a whole TASK_ID array with "
                    "per-element entries (got a per-element entry after a whole "
                    "array) — pass the array alone or index it slot-by-slot"
                )
            entries.append(expr)
            needs_desugar = True
            continue
        raise TypeError(
            f"{construct} deps= entries must be a TaskId variable, a TASK_ID "
            f"array, a TASK_ID array element (e.g. arr[i]), or None — got {d!r}"
        )
    if not entries:
        return []
    if needs_desugar:
        return [_synthesize_deps_array(entries)]
    return entries


def _coerce_dumps(dumps: Sequence[SubmitOperand | None] | None, construct: str) -> list[_ir.Expr] | None:
    """Coerce the ``dumps=`` list (tensor args) to ``ir.Expr``; ``None`` stays
    ``None`` (no dump attr)."""
    if dumps is None:
        return None
    if not isinstance(dumps, (list, tuple)):
        raise TypeError(f"{construct} dumps= must be a list of tensor args, got {dumps!r}")
    return [_submit_expr(d, f"{construct} dumps entry") for d in dumps]


def _opt_expr(value: Any, what: str) -> _ir.Expr | None:
    """Coerce an optional submit operand (predicate/device/core_num) to
    ``ir.Expr`` at the operand boundary; ``None`` passes through."""
    return _submit_expr(value, what) if value is not None else None


#: The runtime's fixed task-timing slot count (sixteen; the parser pins the
#: same number — the scheduler exposes no more).
_NUM_TASK_TIMING_SLOTS = 16


def _validated_timing_slot(timing_slot: object, construct: str) -> int | None:
    """Validate an optional ``timing_slot=0..15`` tag.

    The slot is a compile-time configuration value (a non-negative integer
    literal, never an IR expression): the runtime has sixteen fixed
    task-timing slots and a tagged task contributes its dispatch/finish
    window to the selected one."""
    if timing_slot is None:
        return None
    if isinstance(timing_slot, bool) or not isinstance(timing_slot, int):
        raise TypeError(
            f"{construct} timing_slot must be a non-negative integer literal in "
            f"0..{_NUM_TASK_TIMING_SLOTS - 1}, got {timing_slot!r}"
        )
    if not 0 <= timing_slot < _NUM_TASK_TIMING_SLOTS:
        raise ValueError(
            f"{construct} timing_slot must be in 0..{_NUM_TASK_TIMING_SLOTS - 1}, got {timing_slot}"
        )
    return timing_slot


class _SubmitOutputs:
    """Nested Python unpacking view over only the output projections."""

    def __init__(self, node: _ir.Submit, output_count: int, state: _EmitState) -> None:
        self._node = node
        self._output_count = output_count
        self._state = state

    def __iter__(self) -> Iterator[object]:
        for index in range(self._output_count):
            expr = _ir.TupleGetItemExpr(self._node, index, _ir.Span.unknown())
            output = wrap_expr(expr, expr.type)
            yield _structural_projection(output, self._state, index, "submit")


class _SubmitResult:
    """Nested Python unpacking view for a multi-return Submit.

    The IR result stays flat as ``(out0, out1, task_id)`` for codegen, while
    the ergonomic source form ``(out0, out1), task_id = pl.submit(...)`` needs
    one nested Python tuple boundary. Every projection shares the same emit
    state so the Submit is named exactly once in the traced IR.
    """

    def __init__(self, node: _ir.Submit, output_count: int, state: _EmitState) -> None:
        self._node = node
        self._output_count = output_count
        self._state = state

    def __iter__(self) -> Iterator[Any]:
        """Yield the (outputs-view, task-id) pair for ``res, tid = submit``.

        The element types are dynamic facades derived from the callee's
        *declared* return types — unknowable statically — so they cross as
        ``Any``; the bind hooks carry the real types.
        """
        yield _SubmitOutputs(self._node, self._output_count, self._state)
        task_expr = _ir.TupleGetItemExpr(self._node, self._output_count, _ir.Span.unknown())
        task = wrap_expr(task_expr, task_expr.type)
        yield _structural_projection(task, self._state, self._output_count, "submit")

    def __getitem__(self, index: int) -> Any:
        """Expose flat projections for the parser-compatible temporary form.

        The element types are the callee's declared-return-derived
        projections — unknowable statically (the same dynamic-facade
        contract as ``__iter__``)."""
        if index < 0 or index > self._output_count:
            raise IndexError(f"Submit result index {index!r} is out of range")
        expr = _ir.TupleGetItemExpr(self._node, index, _ir.Span.unknown())
        projection = wrap_expr(expr, expr.type)
        return _structural_projection(projection, self._state, index, "submit")


def _lower_submit(  # noqa: PLR0913 - one kwarg per ir.Submit field
    *,
    callee: Marker,
    arg_exprs: list[_ir.Expr],
    no_dep: list[int],
    deps: list[_ir.Expr],
    dumps: list[_ir.Expr] | None,
    allow_early_resolve: bool,
    predicate: _ir.Expr | None,
    device: _ir.Expr | None,
    extra_attrs: dict[str, Any] | None,
    core_num: _ir.Expr | None,
    sync_start: bool,
    spmd: bool,
    timing_slot: int | None = None,
) -> Any:
    """Pure lowering: assemble the Submit attrs, deduce the return type, build
    the ``ir.Submit``, and wrap the structural Tuple facade (the 2-target unpack
    / single-LHS bind machinery). No kwargs parsing, no isinstance-on-kwargs,
    no coercion — the call sites own that.

    Returns a staged ``Tuple`` facade over the Submit node: the 2-target
    unpack ``a, a_tid = pl.submit(...)`` iterates the element projections (each
    carrying ``__trace_bind__`` + a shared ``_EmitState``); the single-LHS
    ``res = pl.submit(...)`` binds the facade itself, whose ``__trace_bind__``
    names the shared Submit once via ``ctx.let_temp`` (the AST parser's
    ``_submit_tmp`` shape) and re-wraps over the temp so passes that walk for
    ``ir.Submit`` find it. Dedup is value-owned (``state.temp``)."""
    # A Submit references its callee without calling it; a callee whose
    # program-assembly registration is lazy (the jit envelope's global
    # wrapper) registers here — the op-layer hook, frontend-implemented.
    callee.on_submit_reference()
    attrs: dict[str, Any] | None = None
    if dumps:
        attrs = {"dump_vars": dumps}
    if no_dep:
        attrs = attrs or {}
        attrs["arg_direction_overrides"] = list(no_dep)
    if device is not None:
        attrs = attrs or {}
        attrs["device"] = device
    if extra_attrs:
        attrs = attrs or {}
        for k, v in extra_attrs.items():
            attrs.setdefault(k, v)
    if timing_slot is not None:
        # The dedicated kwarg wins over a round-trip attrs= dict (the AST's
        # setdefault discipline shared with device= / no_dep below).
        attrs = attrs or {}
        attrs.setdefault("task_timing_slot", timing_slot)

    declared = list(callee.declared_return_types)
    supplied_params = callee.params[: len(arg_exprs)]
    return_types = _ir.deduce_call_return_type(supplied_params, arg_exprs, declared)
    # A callee's structural ``pl.Tuple[...]`` return is one Function return
    # slot, but Submit exposes each output followed by its TaskId. Builtin
    # ``tuple[...]`` annotations already arrive as flat return slots.
    if len(return_types) == 1 and isinstance(return_types[0], _ir.TupleType):
        result_types = list(return_types[0].types)
    else:
        result_types = return_types
    ret_type = _ir.TupleType([*result_types, _ir.ScalarType(DataType.TASK_ID)])
    span = _ir.Span.unknown()
    if spmd or attrs is not None or allow_early_resolve or predicate is not None:
        submit_node = _ir.Submit(
            callee.gvar,
            arg_exprs,
            deps,
            {},
            attrs,
            ret_type,
            span,
            core_num=core_num,
            sync_start=sync_start,
            allow_early_resolve=allow_early_resolve,
            predicate=predicate,
        )
    else:
        submit_node = _ir.Submit(callee.gvar, arg_exprs, deps, ret_type, span)
    state = _EmitState(submit_node)

    def _emit_submit(ctx: TraceContext, name: str) -> Tuple:
        """The single-LHS bind hook: name the shared Submit once as a temp
        and return a fresh facade over the temp — the Sema's
        ``_default_bind`` then stores the whole flat tuple to the target."""
        del name
        if state.temp is None:
            state.temp = ctx.let_temp(state.node, "submit")
        assert state.temp is not None  # set above; pyright closure narrowing
        return Tuple._runtime(state.temp)

    if len(result_types) > 1:
        return _SubmitResult(submit_node, len(result_types), state)

    facade = Tuple._runtime(submit_node)
    facade._emit_state = state
    # The value-owned bind-hook slot declared on Tuple —
    # the same pattern as _structural_projection's projections.
    facade.__trace_bind__ = _emit_submit
    return facade


def submit(
    callee: SubmitCallee,
    *kernel_args: SubmitArg,
    deps: SubmitEntries | None = None,
    dumps: SubmitEntries | None = None,
    allow_early_resolve: bool = False,
    predicate: SubmitOperand | None = None,
    device: SubmitOperand | None = None,
    timing_slot: int | None = None,
    attrs: dict[str, Any] | None = None,
) -> _SubmitResult | Tuple:
    """Submit a kernel and capture its producer TaskId.

    The facade runs the call and returns a real Python tuple
    ``(result_facade, task_id_facade)`` whose elements the rewriter's
    tuple-target routing binds through the Frame (``rt.bind``);
    ``_lower_submit`` builds
    an :class:`ir.Submit` carrying the dep edges (``deps=``), dispatch
    predicate (``predicate=``), early-resolve hint
    (``allow_early_resolve=``), and dump marks (``dumps=``) — codegen
    packs ``deps`` into ``Arg::set_dependencies`` and the runtime honors
    the predicate at the dispatch point.

    Surface form (must be unpacked as a 2-tuple)::

        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid])
        (a, b), tid = pl.submit(self.multi_out_kernel, x)
        out, tid    = pl.submit(self.stage1, x, scratch, deps=[prev_tid], dumps=[x])
        out, tid    = pl.submit(self.stage1, x, scratch, allow_early_resolve=True)

    The kernel-side ``ir.Submit`` natively returns
    ``Tuple[<kernel return>, TASK_ID]``; element 0 is the tensor result(s),
    element 1 is the producer TaskId (``Scalar[TASK_ID]``). The optional
    ``deps=[...]`` kwarg lists TaskId scalars / arrays this submit must
    wait on. The single-LHS form ``res = pl.submit(...)`` is also accepted
    and binds the whole flat tuple to one variable.

    ``pl.submit`` and its ``deps=`` kwarg work in **both auto and manual
    scope**: ``Arg::set_dependencies`` is orthogonal to OverlapMap
    auto-tracking (final fanin = auto ∪ explicit). In auto scope, use
    ``deps=[...]`` as a precision tool to patch edges the runtime cannot
    infer; in ``pl.manual_scope()``, use it to declare every edge.

    The optional ``dumps=[...]`` kwarg is the submit-side selective tensor
    dump surface (symmetric with ``deps=``): it lists tensor arguments of
    this submit to mark for dump (simpler#844), so an enabled dump pipeline
    filters down to just those bindings. Each entry must be a tensor passed
    positionally to the submitted kernel. ``dumps=`` is the explicit dump
    surface on a submit; the declarative ``pl.dump_tag(t)`` statement feeds the
    same ``dump_vars`` set. These marks only take effect under partial dump
    (``RunConfig.enable_dump_args == 1``); they are a no-op when dump is off
    (``0``) and irrelevant under full dump (``2``), which captures every
    binding regardless.

    The optional ``allow_early_resolve=True`` kwarg (default ``False``) opts
    this task in as a speculative early-dispatch producer (simpler#1065): the
    scheduler may pre-stage this task's consumers onto idle cores before it
    completes, releasing them with a doorbell the instant it finishes. It is a
    producer-side hint — a consumer only pre-stages once *all* of its producers
    are flagged (or already complete). It lowers to
    ``Arg::set_allow_early_resolve(true)`` in orchestration codegen and is a
    pure scheduling optimisation (no effect on results). Pays off on critical
    paths built from many short tasks; harmless otherwise.

    The optional ``predicate=(...)`` kwarg attaches a **dispatch predicate** the
    scheduler evaluates at this task's dispatch point — after its dependencies
    are satisfied, so the value is current without an orchestration-time wait.
    When the comparison is false the task is retired inline (never dispatched to
    a core) while its fanin/fanout still settle, so downstream consumers still
    unlock. Use it to skip work whose need is only known at runtime (e.g. an MoE
    expert with an empty row count)::

        out, tid = pl.spmd_submit(self.expert_ffn, tokens, out, core_num=N,
                                  deps=[gather_tid],
                                  predicate=(row_count[e] > 0))

    The comparison is matched **syntactically, never evaluated**: in this
    position ``row_count[e] > 0`` is a declarative spec handed to the scheduler,
    not a ``tensor.read`` plus a compare. Only ``tensor[indices] OP
    int-literal`` is expressible (one comparison; ``==`` ``!=`` ``>`` ``<``
    ``>=`` ``<=``), mirroring the runtime's single-comparison predicate — no
    chained comparisons, arithmetic, or boolean combination. Reduce anything
    richer to a single gate value in a prior kernel and predicate on that.

    **Contract:** the operand tensor's producer must be one of this submit's
    ``deps=`` (the parser enforces this where statically provable), so the
    dispatch-point read observes the current value.

    The return annotation is ``Any`` (not ``NoReturn``) because the executed
    facade returns a real Python tuple ``(result, task_id)`` — the rewriter's
    tuple-target routing binds each element through the Frame
    (``rt.bind``) — and downstream
    code that does ``out, tid = pl.submit(...)`` would not type-check under
    ``NoReturn``.
    """
    callee, kernel_args = _resolve_submit_callee(callee, kernel_args, "pl.submit")
    marker = _check_submit_callee(callee, kernel_args, "pl.submit")
    _reject_host_level_omitted_outputs(marker, kernel_args, "pl.submit")
    no_dep, arg_exprs = _strip_no_dep_and_coerce(kernel_args, "pl.submit")
    return _lower_submit(
        callee=marker,
        arg_exprs=arg_exprs,
        no_dep=no_dep,
        deps=_coerce_deps(deps, "pl.submit"),
        dumps=_coerce_dumps(dumps, "pl.submit"),
        allow_early_resolve=allow_early_resolve,
        predicate=_opt_expr(predicate, "pl.submit predicate"),
        device=_opt_expr(device, "pl.submit device"),
        extra_attrs=attrs,
        core_num=None,
        timing_slot=_validated_timing_slot(timing_slot, "pl.submit"),
        sync_start=False,
        spmd=False,
    )


def spmd_submit(
    callee: SubmitCallee,
    *kernel_args: SubmitArg,
    core_num: SubmitOperand | None,
    sync_start: bool = False,
    deps: SubmitEntries | None = None,
    dumps: SubmitEntries | None = None,
    allow_early_resolve: bool = False,
    predicate: SubmitOperand | None = None,
    device: SubmitOperand | None = None,
    timing_slot: int | None = None,
    attrs: dict[str, Any] | None = None,
) -> _SubmitResult | Tuple:
    """Launch a kernel as an SPMD task and capture its producer TaskId.

    The facade returns a real Python tuple ``(result, task_id)`` whose
    elements the caller's tuple-target routing binds through the Frame
    (``rt.bind``); ``_lower_submit`` builds an :class:`ir.Submit` carrying the SPMD
    launch spec (``core_num`` / ``sync_start``) alongside the dep edges,
    dispatch predicate, and early-resolve hint. The runtime fans the task
    out across ``core_num`` logical blocks (each kernel reads its block
    index via ``pl.tile.get_block_idx()``).

    It is the SPMD sibling of :func:`submit`: a single orchestration task that
    the runtime fans out across ``core_num`` logical blocks (each kernel reads
    its block index via ``pl.tile.get_block_idx()``). Like :func:`submit` it
    returns one producer TaskId, so the whole dispatch can be named as a
    dependency of later tasks.

    Surface form (must be unpacked as a 2-tuple)::

        out, tid = pl.spmd_submit(self.incore_kernel, x, y, core_num=8)
        out, tid = pl.spmd_submit(self.kernel, x, core_num=8, sync_start=True,
                                  deps=[prev_tid])

    ``core_num`` is a **required keyword argument** (a positive integer
    expression) — the positional slots are the kernel's own arguments.
    ``sync_start`` (default ``False``) requires all blocks to launch atomically.
    ``deps=[...]``, ``allow_early_resolve=True`` and ``predicate=(...)`` work
    exactly as on :func:`submit` (note: a ``sync_start`` task cannot itself be
    block-by-block pre-staged, but it can still be flagged to let its consumers
    pre-stage). The callee may be an InCore / AIC / AIV kernel or a co-scheduled
    Group.

    Like :func:`submit`, it works in both auto and manual scope; its primary use
    is explicit dependency wiring inside ``pl.manual_scope()``.
    """
    if core_num is None:
        raise TypeError(
            "pl.spmd_submit requires the core_num= keyword (a positive integer "
            "expression — the SPMD block count)"
        )
    callee, kernel_args = _resolve_submit_callee(callee, kernel_args, "pl.spmd_submit")
    marker = _check_submit_callee(callee, kernel_args, "pl.spmd_submit")
    _reject_host_level_omitted_outputs(marker, kernel_args, "pl.spmd_submit")
    no_dep, arg_exprs = _strip_no_dep_and_coerce(kernel_args, "pl.spmd_submit")
    return _lower_submit(
        callee=marker,
        arg_exprs=arg_exprs,
        no_dep=no_dep,
        deps=_coerce_deps(deps, "pl.spmd_submit"),
        dumps=_coerce_dumps(dumps, "pl.spmd_submit"),
        allow_early_resolve=allow_early_resolve,
        predicate=_opt_expr(predicate, "pl.spmd_submit predicate"),
        device=_opt_expr(device, "pl.spmd_submit device"),
        extra_attrs=attrs,
        core_num=_opt_expr(core_num, "pl.spmd_submit core_num"),
        timing_slot=_validated_timing_slot(timing_slot, "pl.spmd_submit"),
        sync_start=sync_start,
        spmd=True,
    )


__all__ = ["spmd_submit", "submit"]
