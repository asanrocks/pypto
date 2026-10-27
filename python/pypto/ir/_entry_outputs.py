# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Adapt fresh entry results to the runtime's external-buffer calling convention."""

from dataclasses import dataclass

from pypto.pypto_core import ir, passes

from .utils import derived_return_types, return_value_types


@dataclass(frozen=True)
class EntryOutputs:
    program: ir.Program
    # Physical argument indices, in logical result order (duplicates retain aliasing).
    result_indices: tuple[int, ...] | None = None
    num_synthesized: int = 0


def _rebuild(fn: ir.Function, params: list[tuple[ir.Var, ir.ParamDirection]], body: ir.Stmt) -> ir.Function:
    return ir.Function(
        fn.name,
        params,
        fn.return_types,
        body,
        fn.span,
        fn.func_type,
        fn.level,
        fn.role,
        fn.attrs,
        fn.requires_runtime_binding,
        fn.ir_stage,
    )


def derive_entry_return_type(entry: ir.Function) -> ir.Function:
    """Recover an expressible signature for an entry that declares none.

    A declared signature passes through untouched: refusing a declaration that
    disagrees with the returned values is the frontends' job, not the compiler
    boundary's — each frontend checks its return contract before a program
    reaches ``ir.compile``. The values an *already lowered* program returns
    carry the layout and storage metadata the pipeline attached, so re-checking
    them against the declared source types would refuse lowered input, and
    ``ir.compile`` accepts lowered IR by contract
    (``docs/en/user/execution/01-run.md``: ``ir.compile(kernel.lower(...))``).
    """
    if entry.return_types:
        return entry
    actual = return_value_types(entry)
    if not actual:
        return entry
    derived = derived_return_types(entry)
    if not derived:
        # Nothing declared and nothing derivable is "no recovery to make", not an
        # invalid program: an entry may legally return values whose type lives only
        # in the caller's world. The frontend that owns the return contract refuses
        # such a kernel; a compiler entry point must leave the program alone.
        return entry
    return ir.Function(
        entry.name,
        list(zip(entry.params, entry.param_directions)),
        derived,
        entry.body,
        entry.span,
        entry.func_type,
        entry.level,
        entry.role,
        entry.attrs,
        entry.requires_runtime_binding,
        entry.ir_stage,
    )


def resolve_entry_return_type(entry: ir.Function) -> ir.Function:
    """Check a declaration against values, or recover an expressible signature.

    The frontend-facing half: a pre-pass program's declared return types are
    source contracts, so a declaration that disagrees with the values the body
    returns is refused here, before any pass can attach physical metadata that
    would make the comparison meaningless. Callers that hand a program to
    ``ir.compile`` must use :func:`derive_entry_return_type` instead — the
    compiler boundary accepts lowered input and must not re-check source
    contracts.
    """
    actual = return_value_types(entry)
    if not actual:
        return entry
    if entry.return_types:
        if len(actual) != len(entry.return_types) or any(
            not ir.structural_equal(a, b, enable_auto_mapping=True)
            for a, b in zip(actual, entry.return_types)
        ):
            raise TypeError(f"Entry {entry.name!r}: declared return types disagree with returned values")
        return entry
    return derive_entry_return_type(entry)


def materialize_entry_outputs(
    program: ir.Program, entry: ir.Function, *, defer_unlowered: bool = False
) -> EntryOutputs:
    """Export only proven whole-buffer results; never guess storage from a type.

    This is boundary ABI adaptation, before ordinary IR optimization. Temporarily
    exposing top-level creates as parameters lets the shared, type-sensitive
    lineage analysis prove which buffers the entry actually returns. Helpers keep
    their signatures and allocation semantics.
    """
    actual = return_value_types(entry)
    # A signature may describe an implicit Out result without a ReturnStmt.
    # That existing calling convention does not need a storage transformation.
    if not actual:
        return EntryOutputs(program)
    # Source contracts are checked before lowering. Physical tensor types may
    # subsequently acquire layout/storage metadata, so do not recheck them as
    # though they were source declarations at the late ABI boundary.
    if not entry.return_types:
        entry = resolve_entry_return_type(entry)
    if not entry.return_types:
        return EntryOutputs(program)
    delivered = ir.returned_param_indices(entry, program)
    if len(delivered) == len(entry.return_types) and all(i is not None for i in delivered):
        return EntryOutputs(program)

    statements = list(entry.body.stmts) if isinstance(entry.body, ir.SeqStmts) else [entry.body]
    creates = [
        stmt
        for stmt in statements
        if isinstance(stmt, ir.AssignStmt)
        and isinstance(stmt.value, ir.Call)
        and stmt.value.op.name == ir.get_op("tensor.create").name
    ]
    params = list(zip(entry.params, entry.param_directions))
    # Removing these definitions is essential: each candidate becomes a real
    # parameter in the analysis program, with no competing local definition.
    create_ids = {id(s) for s in creates}
    candidate_body = ir.SeqStmts([s for s in statements if id(s) not in create_ids], entry.span)
    candidate = _rebuild(entry, params + [(s.var, ir.ParamDirection.Out) for s in creates], candidate_body)
    functions = list(program.functions.values())
    candidate_program = ir.Program(
        [candidate if f.name == entry.name else f for f in functions],
        program.name,
        program.span,
    )
    bindings = ir.returned_param_indices(candidate, candidate_program)
    if len(bindings) != len(entry.return_types) or any(i is None for i in bindings):
        if defer_unlowered:
            # InCore results acquire backing tensor.create buffers during
            # ordinary lowering. The final compile boundary must check again.
            return EntryOutputs(program)
        raise TypeError(
            f"Entry {entry.name!r}: outputs created by the return value require a fresh "
            "top-level static-shape tensor allocation; use explicit pl.Out parameters"
        )
    indices = tuple(i for i in bindings if i is not None)
    used = sorted({i - len(params) for i in indices if i >= len(params)})
    for index in used:
        create = creates[index]
        call = create.value
        assert isinstance(call, ir.Call)
        ty = create.var.type
        if type(ty) is not ir.TensorType or any(
            not isinstance(d, ir.ConstInt) or d.value < 0 for d in ty.shape
        ):
            raise TypeError(f"Entry {entry.name!r}: created outputs require static tensor shapes")
        # The hidden Out ABI carries no per-parameter dependency contract: the
        # create statement — sole carrier of the lifetime-granularity OverlapMap
        # opt-out — is deleted on promotion. Refuse rather than drop the semantic.
        if bool(call.kwargs.get("manual_dep", False)):
            raise TypeError(
                f"Entry {entry.name!r}: created output {create.var.name_hint!r} opts out of "
                "auto-dependency tracking (manual_dep=True), which a hidden result parameter "
                "cannot preserve; use explicit pl.Out parameters or manual_dep=False"
            )

    remap = {len(params) + old: len(params) + new for new, old in enumerate(used)}
    names = {p.name_hint for p in entry.params}
    substitutions: list[tuple[ir.Var, ir.Var]] = []
    for index in used:
        name = f"entry_result_{index}"
        while name in names:
            name += "_"
        names.add(name)
        var = ir.Var(name, creates[index].var.type, entry.span)
        params.append((var, ir.ParamDirection.Out))
        substitutions.append((creates[index].var, var))
    removed = {id(creates[i]) for i in used}
    remaining = [s for s in statements if id(s) not in removed]
    body = remaining[0] if len(remaining) == 1 else ir.SeqStmts(remaining, entry.span)
    physical = _rebuild(entry, params, ir.substitute_stmt(body, substitutions))
    return EntryOutputs(
        ir.Program([physical if f.name == entry.name else f for f in functions], program.name, program.span),
        tuple(remap.get(i, i) for i in indices),
        len(used),
    )


def materialize_entry_returns() -> passes.Pass:
    """Legalize the single-chip entry ABI before scope and storage lowering."""

    def transform(program: ir.Program) -> ir.Program:
        from ._compile_contract import entry_function  # noqa: PLC0415

        if any(
            f.level is not None and ir.level_to_linqu_level(f.level) >= 3 for f in program.functions.values()
        ):
            return program
        try:
            entry = entry_function(program)
        except ValueError:
            return program
        # Only artifact compilation requests an external result ABI. Structural
        # lowering remains usable for dynamic allocations and view-valued IR.
        if not entry.attrs.get("entry_output_abi", False):
            return program
        adapted = materialize_entry_outputs(program, entry)
        if not adapted.num_synthesized:
            return program
        physical = adapted.program.get_function(entry.name)
        assert physical is not None
        attrs = dict(physical.attrs)
        for position, index in enumerate(adapted.result_indices or ()):
            attrs[f"entry_result_{position}"] = index
        attrs["entry_result_count"] = len(adapted.result_indices or ())
        attrs["entry_synthesized_count"] = adapted.num_synthesized
        physical = ir.Function(
            physical.name,
            list(zip(physical.params, physical.param_directions)),
            physical.return_types,
            physical.body,
            physical.span,
            physical.func_type,
            physical.level,
            physical.role,
            attrs,
            physical.requires_runtime_binding,
            physical.ir_stage,
        )
        return ir.Program(
            [physical if f.name == entry.name else f for f in program.functions.values()],
            program.name,
            program.span,
        )

    required = passes.IRPropertySet()
    for prop in (
        passes.IRProperty.SSAForm,
        passes.IRProperty.SplitIncoreOrch,
        passes.IRProperty.NormalizedStmtStructure,
        passes.IRProperty.IncoreTileOps,
    ):
        required.insert(prop)
    properties = passes.PassProperties(
        required,
        passes.IRPropertySet(),
        passes.IRPropertySet(),
    )
    return passes.create_program_pass(transform, "MaterializeEntryReturns", properties)


def prepare_program_outputs(program: ir.Program) -> ir.Program:
    """Check the source signature and request ABI adaptation where necessary."""
    from ._compile_contract import entry_function, return_aliases  # noqa: PLC0415

    if any(f.level is not None and ir.level_to_linqu_level(f.level) >= 3 for f in program.functions.values()):
        return program
    try:
        entry = derive_entry_return_type(entry_function(program))
    except ValueError:
        return program
    if not entry.return_types:
        # No expressible signature to adapt: leave the program untouched rather
        # than mark an entry-output ABI it has no declared result for.
        return program
    functions = list(program.functions.values())
    typed_program = ir.Program(
        [entry if f.name == entry.name else f for f in functions],
        program.name,
        program.span,
    )
    if return_aliases(typed_program, entry) is not None:
        return typed_program
    attrs = dict(entry.attrs)
    attrs["entry_output_abi"] = True
    marked = ir.Function(
        entry.name,
        list(zip(entry.params, entry.param_directions)),
        entry.return_types,
        entry.body,
        entry.span,
        entry.func_type,
        entry.level,
        entry.role,
        attrs,
        entry.requires_runtime_binding,
        entry.ir_stage,
    )
    return ir.Program(
        [marked if f.name == entry.name else f for f in functions],
        program.name,
        program.span,
    )
