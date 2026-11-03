# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Parsing-frontend bodies calling traced inline helpers, one way.

A parser caller imports a helper the tracing frontend collected, so shared
computational leaves can migrate before the callers that use them. The helper
joins the caller's program as an imported Inline function and the existing
inliner splices it; nothing about it survives into the artifact, and no traced
ABI callee is ever composed in.

The tests read the *pre-pass* program for the shape of the import (the helper
exists, its compile-time inputs are baked) and the *post-pass* program for the
claim that matters to a user (the helper is gone, its body is inlined).
"""

import ast
import importlib.util
import linecache

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
import torch
from pypto.jit import decorator as jit_decorator
from pypto.jit._ast_analysis import get_func_def
from pypto.language.diagnostics import ParserTypeError, UndefinedVariableError, UnsupportedFeatureError
from pypto.language.tracing.inline import traced_source_records
from pypto.pypto_core import DataType, ir
from pypto.runtime import RunConfig

F32 = torch.float32

# A dynamic extent the caller declares and the helper must keep symbolic.
BR_T_DYN = pl.dynamic("BR_T_DYN")

# A module-level dynamic whose label spells a minted binding name exactly: the
# specializer declares dynamic symbols in the same module namespace the bridge
# injects its bindings into, so the spelling is reserved (TestNameAllocation).
MINTED_NAME_DYN = pl.dynamic("_pto_traced_inline_1")

# A module-level constant the helper folds. Rebinding it must move the caller's
# source hash, because the helper's body is part of what the caller compiles.
GAIN = 2.0


@pl.jit_trace.inline(auto_scope=False)
def _scale_tile(a: pl.Tensor, out: pl.Tensor, factor: pl.constexpr):
    """Multiply one tile by a compile-time factor and write it through ``out``."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, factor), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _scaled_by_module_constant(a: pl.Tensor, out: pl.Tensor):
    """Same shape, but the factor is a module constant the helper folds itself."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, GAIN), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _pair(x: pl.Tensor, out: pl.Tensor):
    """Two results, so the caller's later call has to type a traced result."""
    return pl.Tuple(x, out)


@pl.jit_trace.inline(auto_scope=False)
def _copy_tile(a: pl.Tensor, out: pl.Tensor):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _nested(a: pl.Tensor, out: pl.Tensor):
    """A traced helper reached through another traced helper."""
    return _copy_tile(a, out)


@pl.jit_trace.inline(auto_scope=False)
def _self_recursive(a: pl.Tensor, out: pl.Tensor):
    """A helper that calls itself — no finite traced body exists."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return _self_recursive(a, out)


@pl.jit_trace.inline(auto_scope=False)
def _mutual_first(a: pl.Tensor, out: pl.Tensor):
    """One half of a two-helper cycle."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return _mutual_second(a, out)


@pl.jit_trace.inline(auto_scope=False)
def _mutual_second(a: pl.Tensor, out: pl.Tensor):
    """The other half: it calls back into ``_mutual_first``."""
    return _mutual_first(a, out)


@pl.jit_trace.inline(auto_scope=False)
def _interspersed(a: pl.Tensor, factor: pl.constexpr, out: pl.Tensor):
    """A compile-time knob BETWEEN two staged parameters (declared order
    a, factor, out — the runtime-then-baked insertion order is not it)."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, factor), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _self_recursive_interspersed(a: pl.Tensor, factor: pl.constexpr, out: pl.Tensor):
    """A self-cycle whose knob sits BETWEEN two staged parameters — both
    the entry and the re-entry bake it, so a lane whose key hashed the
    annotation map's insertion order would mint two keys for one helper.
    """
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, factor), [0, 0], out)
    return _self_recursive_interspersed(a, factor, out)


@pl.jit_trace.inline(auto_scope=False)
def _mutual_interspersed_first(a: pl.Tensor, factor: pl.constexpr, out: pl.Tensor):
    """One half of a two-helper cycle, knob between the staged parameters."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, factor), [0, 0], out)
    return _mutual_interspersed_second(a, factor, out)


@pl.jit_trace.inline(auto_scope=False)
def _mutual_interspersed_second(b: pl.Tensor, knob: pl.constexpr, sink: pl.Tensor):
    """The other half: a differently-named knob, calling back into the first."""
    return _mutual_interspersed_first(b, knob, sink)


@pl.jit_trace.inline(auto_scope=False)
def _cross_lane_first(a: pl.Tensor, factor: pl.constexpr, out: pl.Tensor):
    """One half of a cycle that leaves the bridge and comes back through
    the jit callee lane (an ABI callee that imports this helper again)."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, factor), [0, 0], out)
    return _cross_lane_abi(a, out)


@pl.jit_trace
def _cross_lane_abi(a: pl.Tensor, out: pl.Tensor):
    """The ABI half: a traced caller that re-imports the inline helper."""
    return _cross_lane_first(a, 2.0, out)


@pl.jit_trace.inline(auto_scope=False)
def _identity_tensor(a: pl.Tensor):
    """Return the argument unchanged, so its result is the argument's type."""
    return a


@pl.jit_trace.inline(auto_scope=False)
def _declares_static_input(a: pl.Tensor[[64, 64], pl.FP32]):
    """Declare the static shape the identity helper hands over."""
    return pl.mul(a, 3.0)


@pl.jit_trace.inline(auto_scope=False)
def _plus_one(a: pl.Tensor):
    """A traced leaf a parser dependency's own body reaches."""
    return pl.add(a, 1.0)


@pl.jit_trace.inline(auto_scope=False)
def _times_two(a: pl.Tensor):
    return pl.mul(a, 2.0)


@pl.jit.inline(auto_scope=False)
def _parser_relay(a: pl.Tensor):
    """A parser helper whose body calls the traced leaf."""
    return _plus_one(a)


@pl.jit.inline(auto_scope=False)
def _parser_doubles(a: pl.Tensor):
    """A parser helper fed by a traced helper's result."""
    return pl.mul(a, 2.0)


# An aliased binding: the call site names the alias, not the helper's own name.
_scale_alias = _scale_tile


@pl.jit_trace.incore
def _traced_incore(a: pl.Tensor, out: pl.Out[pl.Tensor]):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _reaches_incore(a: pl.Tensor, out: pl.Tensor):
    """Inline, but its own closure leaves the inline lane."""
    return _traced_incore(a, out)


@pl.jit_trace.inline(auto_scope=False)
def _declares_bf16(a: pl.Tensor[[64, 64], pl.BF16], out: pl.Tensor):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit
def _calls_traced(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """The parser caller: one imported helper, one keyword compile-time value.

    The call sits in a ``CORE_GROUP`` scope, which is where a tile-level inline
    helper belongs: it is outlined into an InCore function, so the spliced tile
    ops keep a device home.
    """
    with pl.at(level=pl.Level.CORE_GROUP):
        _scale_tile(a, out, factor=2.0)
    return out


@pl.jit
def _calls_two_factors(
    a: pl.Tensor[[64, 64], pl.FP32],
    b: pl.Tensor[[64, 64], pl.FP32],
    out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    out_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """One helper, two compile-time values: two variants, one per call site."""
    with pl.at(level=pl.Level.CORE_GROUP):
        _scale_tile(a, out_a, factor=2.0)
        _scale_tile(b, out_b, factor=3.0)
    return out_a, out_b


@pl.jit
def _calls_alias(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _scale_alias(a, out, factor=2.0)
    return out


@pl.jit
def _symbolic(a: pl.Tensor[[BR_T_DYN, 64], pl.FP32], out: pl.Out[pl.Tensor[[BR_T_DYN, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _copy_tile(a, out)
    return out


@pl.jit
def _calls_traced_with_minted_dynamic(
    a: pl.Tensor[[MINTED_NAME_DYN, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[MINTED_NAME_DYN, 64], pl.FP32]],
):
    """The dynamic's module-level declaration shares the injected bindings' namespace."""
    with pl.at(level=pl.Level.CORE_GROUP):
        _scale_tile(a, out, factor=2.0)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _counts(n: pl.Scalar[pl.INT32], out: pl.Tensor):
    """A staged scalar keeps its declaration; an in-lattice INDEX caller converts."""
    tile = pl.load(out, [0, 0], [64, 64])
    pl.store(pl.mul(tile, 1.0), [0, 0], out)
    return out


@pl.jit.incore
def _incore_calling_traced(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """A parser *dependency* body that imports the traced helper, not the entry."""
    _scale_tile(a, out, factor=2.0)
    return out


@pl.jit
def _calls_incore_dep(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    _incore_calling_traced(a, out)
    return out


@pl.jit
def _calls_index_scalar(n: pl.Scalar[pl.INDEX], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _counts(n, out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _nz_operand(a: pl.Tensor, out: pl.Tensor):
    """A bare annotation: whatever layout the caller's argument carries."""
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(tile, [0, 0], out)
    return out


@pl.jit
def _calls_nz(a: pl.Tensor[[64, 64], pl.FP32, pl.NZ], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _nz_operand(a, out)
    return out


@pl.jit
def _calls_nested(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _nested(a, out)
    return out


@pl.jit
def _calls_self_recursive(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _self_recursive(a, out)
    return out


@pl.jit
def _calls_interspersed(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _interspersed(a, 2.0, out)
    return out


@pl.jit
def _calls_cross_lane(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _cross_lane_first(a=a, out=out, factor=2.0)
    return out


@pl.jit
def _calls_self_interspersed(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _self_recursive_interspersed(a, 2.0, out)
    return out


@pl.jit
def _calls_mutual_interspersed(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _mutual_interspersed_first(a, 2.0, out)
    return out


@pl.jit
def _calls_mutual_cycle(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _mutual_first(a, out)
    return out


@pl.jit
def _calls_identity_then_static(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """A traced result feeds a helper that declares the same static shape."""
    with pl.at(level=pl.Level.CORE_GROUP):
        y = _identity_tensor(a)
        out[:] = _declares_static_input(y)
    return out


@pl.jit
def _calls_traced_below_parser_dep(
    a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
):
    """The traced leaf sits one parser hop below the entry."""
    with pl.at(level=pl.Level.CORE_GROUP):
        y = _parser_relay(a)
        out[:] = _times_two(y)
    return out


@pl.jit
def _calls_parser_dep_with_traced_result(
    a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]
):
    """A traced result feeds a parser dependency's parameter."""
    with pl.at(level=pl.Level.CORE_GROUP):
        y = _plus_one(a)
        out[:] = _parser_doubles(y)
    return out


@pl.jit
def _calls_pair(
    a: pl.Tensor[[64, 64], pl.FP32],
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
    out2: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    """A traced result feeds a later traced call, so its metadata must resolve."""
    with pl.at(level=pl.Level.CORE_GROUP):
        qa, qb = _pair(a, out)
        _copy_tile(qb, out2)
    return qa, out2


@pl.jit
def _calls_incore_helper(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _reaches_incore(a, out)
    return out


@pl.jit
def _calls_bf16_helper(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _declares_bf16(a, out)
    return out


@pl.jit
def _calls_module_constant(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _scaled_by_module_constant(a, out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _twice_tile(x: pl.Tile):
    """A tile parameter: its memory space and extent have no tensor metadata."""
    return pl.mul(x, 2.0)


def _leaf_factory(gain: float):
    """Two instances share source, qualname and first line; only ``gain`` differs."""

    @pl.jit_trace.inline(auto_scope=False)
    def leaf(a: pl.Tensor, out: pl.Tensor):
        tile = pl.load(a, [0, 0], [64, 64])
        pl.store(pl.mul(tile, gain), [0, 0], out)
        return out

    return leaf


_GAIN_2 = _leaf_factory(2.0)
_GAIN_3 = _leaf_factory(3.0)
_PICKED = _GAIN_2


class _NestedConfig:
    """A module-level knob a nested helper bakes from an attribute chain.

    An attribute chain is the shape no free-name fold renders, so it is the
    shape a source-only identity record cannot see: the knob's value reaches
    the compiled body only through the nested helper's own trace.
    """

    KNOB = 2.0


@pl.jit_trace.inline(auto_scope=False)
def _nested_scale_leaf(a: pl.Tensor, out: pl.Tensor, k: pl.constexpr):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, k), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _nested_scale_middle(a: pl.Tensor, out: pl.Tensor):
    return _nested_scale_leaf(a, out, _NestedConfig.KNOB)


@pl.jit
def _calls_nested_baked(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """The baked value lives one traced hop below the imported helper."""
    with pl.at(level=pl.Level.CORE_GROUP):
        _nested_scale_middle(a, out)
    return out


@pl.jit
def _picks_one_of_two(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    """Both factory instances are reached; the third name picks one of them."""
    with pl.at(level=pl.Level.CORE_GROUP):
        _GAIN_2(a, out)
        _GAIN_3(a, out)
        _PICKED(a, out)
    return out


def _sample(shape: tuple[int, int] = (64, 64), dtype: torch.dtype = F32) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype)


_SIM_CONFIG = RunConfig(platform="a2a3sim")


def _by_name(program: ir.Program) -> dict[str, ir.Function]:
    """The program's functions by name — ``Program.functions`` is keyed by GlobalVar."""
    return {gvar.name: fn for gvar, fn in program.functions.items()}


def _program(kernel, *samples: object) -> ir.Program:
    """The pre-pass program: no pipeline rule can hide the import."""
    return kernel.specialize(*samples)


class TestImportedHelper:
    """The helper reaches the caller's program as an imported Inline function."""

    def test_helper_joins_the_program_as_inline(self):
        program = _program(_calls_traced, _sample(), _sample())
        inline = [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline]
        assert [fn.name for fn in inline] == ["_scale_tile"]

    def test_the_call_site_names_the_imported_function(self):
        program = _program(_calls_traced, _sample(), _sample())
        entry = _by_name(program)["_calls_traced"]
        assert "_scale_tile(" in str(entry)

    def test_default_pipeline_inlines_the_helper_away(self):
        program = _calls_traced.lower(_sample(), _sample())
        assert "_scale_tile" not in _by_name(program)

    def test_nested_traced_helpers_import_transitively(self):
        program = _program(_calls_nested, _sample(), _sample())
        inline = {fn.name for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline}
        assert inline == {"_nested", "_copy_tile"}

    def test_nested_helpers_are_inlined_away(self):
        program = _calls_nested.lower(_sample(), _sample())
        assert "_nested" not in _by_name(program)
        assert "_copy_tile" not in _by_name(program)


class TestBindingResolution:
    """How the call site names a helper does not matter, only what it resolves to."""

    def test_an_aliased_binding_resolves_to_the_helper(self):
        program = _program(_calls_alias, _sample(), _sample())
        inline = {name for name in _by_name(program) if name.startswith("_scale_tile")}
        assert inline, _by_name(program)

    def test_a_factory_captured_helper_resolves_through_the_closure(self):
        def factory():
            @pl.jit_trace.inline(auto_scope=False)
            def closure_scale(a: pl.Tensor, out: pl.Tensor):
                tile = pl.load(a, [0, 0], [64, 64])
                pl.store(pl.mul(tile, 2.0), [0, 0], out)
                return out

            @pl.jit
            def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
                with pl.at(level=pl.Level.CORE_GROUP):
                    closure_scale(a, out)
                return out

            return entry

        program = _program(factory(), _sample(), _sample())
        assert "closure_scale" in _by_name(program)

    def test_the_declared_dynamic_extent_stays_symbolic(self):
        program = _program(_symbolic, _sample((8, 64)), _sample((8, 64)))
        helper = _by_name(program)["_copy_tile"]
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TensorType), param_type
        shape = list(param_type.shape)
        assert any(isinstance(dim, ir.Var) for dim in shape), shape


class TestCallerShapes:
    """A parser dependency body composes exactly like the entry does."""

    def test_an_incore_dependency_body_can_import_the_helper(self):
        program = _program(_calls_incore_dep, _sample(), _sample())
        names = _by_name(program)
        assert "in_core_calling_traced" in names or "_incore_calling_traced" in names
        assert any(name.startswith("_scale_tile") for name in names), sorted(names)

    def test_the_dependency_path_inlines_away_too(self):
        program = _calls_incore_dep.lower(_sample(), _sample())
        assert not [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline]

    def test_the_caller_declared_layout_reaches_the_helper(self):
        program = _program(_calls_nz, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        view = getattr(helper.params[0].type, "tensor_view", None)
        assert view is not None and view.layout == ir.TensorLayout.NZ

    def test_a_traced_leaf_below_a_parser_helper_is_reached(self):
        """A parser dep's body is typed against *its own* traced calls."""
        program = _program(_calls_traced_below_parser_dep, _sample(), _sample())
        inline = {fn.name for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline}
        assert "_plus_one" in inline, sorted(inline)
        lowered = _calls_traced_below_parser_dep.lower(_sample(), _sample())
        assert not [fn for fn in lowered.functions.values() if fn.func_type == ir.FunctionType.Inline]

    def test_a_traced_result_types_a_parser_helper_parameter(self):
        """The traced result's static extents reach the parser dep's parameter."""
        program = _program(_calls_parser_dep_with_traced_result, _sample(), _sample())
        relay = next(fn for fn in program.functions.values() if fn.name.startswith("_parser_doubles"))
        relay_type = relay.params[0].type
        assert isinstance(relay_type, ir.TensorType), relay_type
        assert all(isinstance(dim, ir.ConstInt) and dim.value == 64 for dim in relay_type.shape), relay_type


class TestCompileTimeArguments:
    """Compile-time parameters are folded at the call site and dropped."""

    def test_keyword_value_is_baked_and_dropped_from_the_call(self):
        program = _program(_calls_traced, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        assert len(helper.params) == 2, [p.name_hint for p in helper.params]
        # The call carries the two surviving parameters; a third argument would
        # have failed the parser's arity check against the imported signature.
        entry = _by_name(program)["_calls_traced"]
        assert f"{helper.name}(a, out)" in str(entry)

    def test_two_values_make_two_variants(self):
        program = _program(_calls_two_factors, _sample(), _sample(), _sample(), _sample())
        inline = [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline]
        assert len(inline) == 2, [fn.name for fn in inline]

    def test_matching_values_share_one_variant(self):
        @pl.jit
        def entry(
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP32],
            out_a: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            out_b: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out_a, factor=2.0)
                _scale_tile(b, out_b, factor=2.0)
            return out_a, out_b

        program = _program(entry, _sample(), _sample(), _sample(), _sample())
        inline = [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline]
        assert len(inline) == 1, [fn.name for fn in inline]

    def test_unbound_compile_time_parameter_is_loud(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out)
            return out

        with pytest.raises(TypeError, match=r"does not bind parameter\(s\) \['factor'\]"):
            entry.specialize(_sample(), _sample())

    def test_an_interspersed_compile_time_parameter_binds_correctly(self):
        """A knob between two staged parameters must not scramble the body.

        The bridge built its annotation map runtime-params-then-sorted-baked,
        but ``bind_parameters`` binds the instrumented kernel positionally in
        annotation order — so the staged parameter after the knob received
        the knob's slot and the trace died on a mixed-type arithmetic (or,
        had the types lined up, written the wrong buffer). Declared order is
        the binding contract.
        """
        a, out = _sample(), _sample()
        _calls_interspersed.compile(a, out, config=_SIM_CONFIG)(a, out, config=_SIM_CONFIG)
        assert torch.equal(out, a * 2.0)


class TestTracedResults:
    """A traced helper's results type the caller's later calls."""

    def test_a_traced_result_is_accepted_by_a_later_traced_call(self):
        program = _program(_calls_pair, _sample(), _sample(), _sample())
        inline = {fn.name for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline}
        assert inline == {"_pair", "_copy_tile"}

    def test_the_pair_program_survives_the_default_pipeline(self):
        program = _calls_pair.lower(_sample(), _sample(), _sample())
        assert not [fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline]

    def test_a_static_return_extent_stays_static(self):
        """IR shapes carry ``ir.ConstInt``, not ``int``: neither may be lost."""
        program = _program(_calls_identity_then_static, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.name.startswith("_declares_static_input"))
        helper_type = helper.params[0].type
        assert isinstance(helper_type, ir.TensorType), helper_type
        assert all(isinstance(dim, ir.ConstInt) and dim.value == 64 for dim in helper_type.shape), helper_type


class TestBoundaryRefusals:
    """The directions the composition must not silently cross."""

    def test_a_traced_abi_callee_is_not_imported_by_this_bridge(self):
        """Inline-only discovery leaves an ABI callee to the parser, which refuses it."""

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _traced_incore(a, out)
            return out

        with pytest.raises(UnsupportedFeatureError, match="Unsupported function call: _traced_incore"):
            entry.specialize(_sample(), _sample())

    def test_a_helper_reaching_a_traced_abi_callee_is_refused(self):
        with pytest.raises(ParserTypeError, match="reaches InCore function '_traced_incore'"):
            _calls_incore_helper.specialize(_sample(), _sample())

    def test_a_staged_type_contradiction_keeps_the_declaration(self):
        with pytest.raises(ParserTypeError, match="BF16"):
            _calls_bf16_helper.specialize(_sample(), _sample())


class TestRecursiveHelpers:
    """A helper cycle has no finite traced body, so it is refused loudly.

    The variant a cycle re-enters is still being traced — it cannot be
    registered yet — so without a guard the dedupe lookup misses on every
    level and the tracer re-enters the helper until the interpreter's
    recursion limit.
    """

    def test_a_self_recursive_helper_is_refused(self):
        with pytest.raises(ParserTypeError, match="cycle") as info:
            _calls_self_recursive.specialize(_sample(), _sample())
        assert str(info.value).count("_self_recursive") >= 2, info.value

    def test_a_mutual_helper_cycle_is_refused(self):
        with pytest.raises(ParserTypeError, match="cycle") as info:
            _calls_mutual_cycle.specialize(_sample(), _sample())
        message = str(info.value)
        assert "_mutual_first" in message, message
        assert "_mutual_second" in message, message

    def test_a_self_recursive_helper_with_an_interspersed_knob_is_refused(self):
        """A self-cycle whose baked parameter sits before a staged one.

        Both the entry and the re-entry bake the knob, but through the same
        map — so this pins the guard against any future lane that rebuilds
        the annotation map in a different order (the bridge once inserted
        runtime-then-sorted-baked): the re-entry must hit the in-flight key
        and name the cycle, not re-trace under a second key.
        """
        with pytest.raises(ParserTypeError, match="cycle") as info:
            _calls_self_interspersed.specialize(_sample(), _sample())
        message = str(info.value)
        assert message.count("_self_recursive_interspersed") >= 2, message

    def test_a_mutual_cycle_with_interspersed_knobs_is_refused(self):
        """A→B→A where both halves carry a knob between staged parameters
        (and spell it differently), so each helper's key parts must pair
        with their own parameter names to stay stable across the cycle."""
        with pytest.raises(ParserTypeError, match="cycle") as info:
            _calls_mutual_interspersed.specialize(_sample(), _sample())
        message = str(info.value)
        assert "_mutual_interspersed_first" in message, message
        assert "_mutual_interspersed_second" in message, message
        assert message.count("_mutual_interspersed_first") >= 2, message

    def test_a_cross_lane_cycle_is_refused_with_the_true_chain(self):
        """A cycle that leaves the import bridge and re-enters through the jit
        callee lane (an ABI callee importing the inline helper back).

        The guard's key is computed by whichever lane re-enters — the bridge
        inserts the annotation map runtime-then-sorted-baked, the callee lane
        in declared order — so an interspersed baked/staged helper had two
        keys and the re-entry missed the in-flight one, re-tracing until a
        *different*, misleading refusal. One name-canonical key names the
        true chain.
        """
        with pytest.raises(ParserTypeError, match="cycle") as info:
            _calls_cross_lane.specialize(_sample(), _sample())
        message = str(info.value)
        assert "_cross_lane_first" in message, message
        assert "_cross_lane_abi" in message, message
        assert message.count("_cross_lane_first") >= 2, message


class TestCallValidation:
    """A malformed call fails instead of quietly passing a different program.

    The rewrite that emits the call used to build its argument list through a
    dict, so a surplus positional argument was truncated, an unknown keyword was
    dropped, and a duplicate binding silently kept whichever value came last —
    three input errors that changed the compiled program with no diagnostic.
    """

    def test_a_surplus_positional_argument_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out, 2.0, 99)
            return out

        with pytest.raises(TypeError, match="surplus argument"):
            entry.specialize(_sample(), _sample())

    def test_an_unknown_keyword_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out, 2.0, typo=99)
            return out

        with pytest.raises(TypeError, match="no parameter 'typo'"):
            entry.specialize(_sample(), _sample())

    def test_a_duplicate_binding_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out, 2.0, factor=3.0)
            return out

        with pytest.raises(TypeError, match="both positionally and by keyword"):
            entry.specialize(_sample(), _sample())

    def test_a_missing_compile_time_argument_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out)
            return out

        with pytest.raises(TypeError, match="does not bind parameter"):
            entry.specialize(_sample(), _sample())

    def test_a_kwargs_splat_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _scale_tile(a, out, **{})
            return out

        with pytest.raises(TypeError, match="kwargs"):
            entry.specialize(_sample(), _sample())


class TestTileArguments:
    """A tile argument is typed from its parsed IR value, not from source."""

    def test_a_loaded_tile_reaches_the_helper_and_inlines_away(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64])
                pl.store(_twice_tile(tile), [0, 0], out)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        # The extent the caller loaded and the dtype of its source tensor — the
        # same type the parser gives the caller's own ``tile`` local.
        assert len(param_type.shape) == 2, param_type.shape
        assert param_type.dtype == DataType.FP32, param_type.dtype
        lowered = entry.lower(_sample(), _sample())
        assert not [fn for fn in lowered.functions.values() if fn.func_type == ir.FunctionType.Inline]

    def test_an_inline_computed_tile_argument_resolves(self):
        """A tile argument may be any expression, exactly like on the tracing lane.

        The argument is typed from the parsed expression's own result — the
        position the old source-level walk could not read, which used to force
        a refusal ("could not be resolved") for anything but a plain name.
        """

        @pl.jit_trace.inline(auto_scope=False)
        def consumes(x: pl.Tile):
            return pl.mul(x, 2.0)

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64])
                pl.store(consumes(pl.reshape(tile, [64, 64])), [0, 0], out)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.dtype == DataType.FP32, param_type
        lowered = entry.lower(_sample(), _sample())
        assert not [fn for fn in lowered.functions.values() if fn.func_type == ir.FunctionType.Inline]

    def test_an_alias_keeps_the_type_it_was_bound_to(self):
        """The alias is read where it is written, not where the helper is called.

        ``tile`` is reloaded at a different dtype after ``saved = tile``, so
        resolving ``saved`` against the call site's name would hand the helper
        the FP16 load. The IR then disagreed with the FP32 value it actually
        passed — visible here as the helper's declared parameter dtype.
        """

        @pl.jit
        def entry(
            a: pl.Tensor[[64, 64], pl.FP32],
            b: pl.Tensor[[64, 64], pl.FP16],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64])
                saved = tile
                tile = pl.load(b, [0, 0], [64, 64])
                pl.store(_twice_tile(saved), [0, 0], out)
            return out

        program = _program(entry, _sample(), _sample(dtype=torch.float16), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.dtype == DataType.FP32, param_type

    def test_an_alias_chain_that_reuses_a_name_still_resolves(self):
        """Reusing a name is rebinding, not a cycle.

        ``tile = saved`` is read before the original ``tile = pl.load(...)``,
        which is strictly earlier in the source, so the chain is finite even
        though the name ``tile`` appears twice. Tracking visited *names* instead
        of visited definitions made this legal chain decline with "could not be
        resolved".
        """

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64])
                saved = tile
                tile = saved
                pl.store(_twice_tile(tile), [0, 0], out)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.dtype == DataType.FP32, param_type

    def test_a_self_alias_reads_the_previous_binding(self):
        """``tile = tile`` is a no-op, not a cycle: the RHS reads the load."""

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64])
                tile = tile  # noqa: PLW0127 - the self-assignment is the subject under test
                pl.store(_twice_tile(tile), [0, 0], out)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        assert isinstance(helper.params[0].type, ir.TileType), helper.params[0].type
        assert helper.params[0].type.dtype == DataType.FP32, helper.params[0].type

    def test_an_alias_cycle_is_refused_by_name_resolution(self):
        """A name that only ever names itself never becomes a tile type.

        The malformed cycle is caught before the bridge is involved: the parser
        resolves names statement by statement, so ``first = second`` reads a
        name nothing defined, and the loud undefined-variable refusal replaces
        the source walk's declined answer.
        """

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                first = second  # noqa: F821  # pyright: ignore[reportUnboundVariable] -- the malformed cycle is the subject
                second = first
                pl.store(_twice_tile(second), [0, 0], out)
            return out

        with pytest.raises(UndefinedVariableError, match="second"):
            entry.specialize(_sample(), _sample())


class TestArgumentFidelity:
    """The parsed argument's own IR type crosses the boundary, field for field.

    A source-level reconstruction of the caller lost fields the parser's own
    metadata model does not carry — an explicit tile memory space, the valid
    region of a view, the distributed kind of a window. Each test pins one of
    those fields on the helper's parameter, because the traced body is
    collected against exactly that type.
    """

    def test_an_index_scalar_converts_to_the_declared_dtype(self):
        """The helper's declared scalar dtype is the ABI, not the caller's dtype.

        The specialized helper is typed from the parsed argument, so the
        in-lattice conversion must happen before it: the entry binds the
        shared conversion temp and the helper keeps one INT32 signature for
        every integer caller (B5a's admission rule, applied at this bridge).
        """
        program = _calls_index_scalar.specialize(3, _sample())
        text = str(program)
        assert "n: pl.Scalar[pl.INT32]" in text  # the helper's declaration stands
        assert "_n_int32: pl.Scalar[pl.INT32] = pl.cast(n, pl.INT32)" in text
        assert "self._counts(_n_int32, out)" in text

    def test_an_explicit_tile_memory_space_reaches_a_declaring_helper(self):
        """A Mat-resident tile into a helper declaring Mat specializes cleanly.

        This is the shape the source walk could not express: the walk modeled a
        tile as shape and dtype, so the declared ``pl.Mem.Mat`` contradicted a
        reconstructed ``None`` and the call was refused.
        """

        @pl.jit_trace.inline(auto_scope=False)
        def reads_mat(x: pl.Tile[[64, 64], pl.FP32, pl.Mem.Mat]):
            return x

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                reads_mat(tile)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.memory_space == ir.MemorySpace.Mat, param_type

    def test_an_explicit_tile_memory_space_crosses_to_a_bare_parameter(self):
        """The memory space is a property of the value, not of the declaration."""

        @pl.jit_trace.inline(auto_scope=False)
        def takes_any(x: pl.Tile):
            return x

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64], target_memory=pl.Mem.Mat)
                takes_any(tile)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.memory_space == ir.MemorySpace.Mat, param_type

    def test_a_tile_valid_shape_reaches_the_helper(self):
        """A partially valid tile keeps its valid region across the boundary."""

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(a, [0, 0], [64, 64], valid_shape=[8, 64])
                _twice_tile(tile)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.tile_view is not None, param_type
        assert "valid_shape=[8, 64]" in str(param_type), param_type

    def test_a_sliced_tensor_keeps_its_view(self):
        """A tensor slice's valid region survives the boundary, not just its shape."""

        @pl.jit_trace.inline(auto_scope=False)
        def identity(x: pl.Tensor):
            return x

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[8, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                view = pl.slice(a, [8, 64], [0, 0], valid_shape=[4, 64])
                identity(view)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        assert "TensorView(valid_shape=[4, 64]," in str(helper.params[0].type), helper.params[0].type

    def test_a_positionally_dtyped_created_tile_resolves(self):
        """``pl.create_tile`` states its dtype positionally, and that crosses."""

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.create_tile([64, 64], pl.FP32)
                _twice_tile(tile)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        param_type = helper.params[0].type
        assert isinstance(param_type, ir.TileType), param_type
        assert param_type.dtype == DataType.FP32, param_type

    def test_a_reshaped_tile_argument_resolves(self):
        """A reshaped tile keeps its new shape without a source-level model."""

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                initial = pl.load(a, [0, 0], [64, 64])
                tile = pl.tile.reshape(initial, [32, 128])
                _twice_tile(tile)
            return out

        program = _program(entry, _sample(), _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        assert "[32, 128]" in str(helper.params[0].type), helper.params[0].type

    def test_a_distributed_window_keeps_its_kind(self):
        """A window argument is a distributed tensor, not a downgraded plain one."""

        @pl.jit_trace.inline(auto_scope=False)
        def window_identity(x: pld.DistributedTensor):
            return x

        @pl.jit.host
        def entry(out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            buffer = pld.alloc_window_buffer([64, 64], dtype=pl.FP32)
            window = pld.window(buffer, [64, 64], dtype=pl.FP32)
            window_identity(window)
            return out

        program = _program(entry, _sample())
        helper = next(fn for fn in program.functions.values() if fn.func_type == ir.FunctionType.Inline)
        assert isinstance(helper.params[0].type, ir.DistributedTensorType), helper.params[0].type


class TestNameAllocation:
    """A minted variant name cannot land on a name the program already holds."""

    def test_a_variant_suffix_does_not_collide_with_an_entry_name(self):
        @pl.jit_trace.inline(auto_scope=False)
        def leaf(a: pl.Tensor, out: pl.Tensor, gain: pl.constexpr):
            tile = pl.load(a, [0, 0], [64, 64])
            pl.store(pl.mul(tile, gain), [0, 0], out)
            return out

        @pl.jit
        def leaf_2(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            """Two variants of ``leaf`` want ``leaf`` and ``leaf_2``."""
            with pl.at(level=pl.Level.CORE_GROUP):
                leaf(a, out, 2.0)
                leaf(a, out, 3.0)
            return out

        names = sorted(_by_name(_program(leaf_2, _sample(), _sample())))
        assert "leaf_2" in names, names
        assert len(names) == len(set(names)), names
        assert len([name for name in names if name.startswith("leaf")]) == 3, names

    def test_a_minted_binding_name_skips_a_caller_local_of_that_name(self):
        """A local literally named `_pto_traced_inline_1` plus a helper call.

        The parser's bare-name call dispatch resolves the parse-namespace
        binding even where the caller's own local of that name should win, so a
        mint that landed on the local would run the helper at the caller's own
        call to it. The mint skips past the collision — both the local's reads
        and the helper's call keep their own meanings, and the composed program
        compiles and runs correctly.
        """

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _pto_traced_inline_1 = pl.load(a, [0, 0], [64, 64])
                pl.store(pl.mul(_pto_traced_inline_1, 3.0), [0, 0], out)
                _scale_tile(a, out, factor=2.0)
            return out

        a = torch.randn(64, 64, dtype=F32)
        out = torch.empty(64, 64, dtype=F32)
        entry.compile(a, out, config=_SIM_CONFIG)(a, out, config=_SIM_CONFIG)
        # The helper's store ran last, on the same buffer the local's store wrote.
        assert torch.equal(out, a * 2.0)

    def test_a_caller_call_to_a_local_of_the_minted_name_stays_its_own(self):
        """The caller's own call to its local is not redirected to the helper.

        The user's Python raises on this call (a tensor is not callable), so the
        compiled program must refuse it naming the user's own source — which
        the minted name moving past the collision is what makes possible. A
        mint that landed on the local silently ran the traced helper instead.
        """

        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _pto_traced_inline_1 = a
                _pto_traced_inline_1(a, out)
                _scale_tile(a, out, factor=2.0)
            return out

        with pytest.raises(UnsupportedFeatureError, match="Unsupported function call"):
            entry.specialize(_sample(), _sample())

    def test_a_dynamic_symbol_cannot_take_a_minted_binding_name(self):
        """A module-level dynamic spelled like a minted binding is refused.

        The generated program declares its dynamic symbols at module level —
        the same namespace the bridge injects its bindings into before exec —
        so a dynamic spelled ``_pto_traced_inline_1`` would overwrite the
        injected resolver and the rewritten call would resolve to the dynamic
        (observed as ``Unsupported function call: pl.tensor.dim(a, 0)(a,
        out)``). The reservation refuses the dynamic loudly at specialization
        instead, naming the reserved namespace.
        """
        with pytest.raises(ValueError, match="reserved for the traced-inline import bindings"):
            _calls_traced_with_minted_dynamic.specialize(_sample((128, 64)), _sample((128, 64)))


class TestCacheReuse:
    """A warm caller reuses its artifact instead of rebuilding the program."""

    def test_a_warm_call_does_not_rebuild_the_program(self, monkeypatch):
        built: list[str] = []
        original = jit_decorator._parse_generated

        def counting(*args, **kwargs):
            built.append("parse")
            return original(*args, **kwargs)

        monkeypatch.setattr(jit_decorator, "_parse_generated", counting)
        a, out = _sample(), _sample()
        first = _calls_traced.compile(a, out, config=_SIM_CONFIG)
        second = _calls_traced.compile(a, out, config=_SIM_CONFIG)
        assert first is second, "an identical request must reuse its compiled artifact"
        assert built == ["parse"], "the warm request must not re-parse or re-specialize"


class TestArtifact:
    """The composed program reaches code generation, not just the pass pipeline."""

    def test_the_mixed_program_compiles_to_an_artifact(self):
        pytest.importorskip("pypto.runtime")
        compiled = _calls_traced.compile(_sample(), _sample(), config=_SIM_CONFIG)
        assert type(compiled).__name__ == "CompiledProgram"

    def test_the_inlined_helper_computes_on_the_simulator(self):
        """The composed program's output is the helper's arithmetic, not a stub.

        The only thing that can go wrong silently here is the splice: an imported
        function that the inliner left as a call, or whose body lost the folded
        constant, still compiles. Running it is what distinguishes that from a
        correct composition.
        """
        a = torch.randn(64, 64, dtype=F32)
        out = torch.empty(64, 64, dtype=F32)
        _calls_traced.compile(a, out, config=_SIM_CONFIG)(a, out, config=_SIM_CONFIG)
        assert torch.equal(out, a * 2.0)

    def test_a_multi_result_helper_compiles(self):
        compiled = _calls_pair.compile(_sample(), _sample(), _sample(), config=_SIM_CONFIG)
        assert type(compiled).__name__ == "CompiledProgram"


class TestCacheIdentity:
    """The imported helper is part of what the caller's key identifies."""

    def test_the_helper_source_is_in_the_caller_key(self):
        records = traced_source_records([_calls_traced._func])
        assert any("_scale_tile" in repr(record) for record in records)

    def test_a_folded_module_constant_moves_the_caller_key(self, monkeypatch):
        before = _calls_module_constant._get_source_hash()
        # The helper folds the constant itself, so the caller's key has to move
        # even though the caller's own source never names it.
        globals_ = _scaled_by_module_constant.kernel.__globals__
        monkeypatch.setitem(globals_, "GAIN", 4.0)
        after = _calls_module_constant._get_source_hash()
        assert before != after

    def test_rebinding_the_helper_name_refreshes_the_graph(self, monkeypatch):
        before = _calls_module_constant._get_source_hash()
        monkeypatch.setitem(
            _calls_module_constant._func.__globals__, "_scaled_by_module_constant", _copy_tile
        )
        after = _calls_module_constant._get_source_hash()
        assert before != after

    def test_rebinding_one_of_two_reached_helpers_moves_the_key(self, monkeypatch):
        """Which helper a call *name* reaches is part of the identity.

        Two factory instances share their source, qualname and first line, and
        differ only in a captured constant. Both are already reached by the body
        through two other names, so rebinding the third name leaves the *set* of
        reached helpers — and any record that dedupes by helper — untouched,
        while the generated call moves from one variant to the other. The key has
        to follow the wiring, not just the reached set.
        """
        globals_ = _picks_one_of_two._func.__globals__
        monkeypatch.setitem(globals_, "_PICKED", _GAIN_2)
        before = _picks_one_of_two._get_source_hash()
        before_entry = str(_by_name(_program(_picks_one_of_two, _sample(), _sample()))["_picks_one_of_two"])

        monkeypatch.setitem(globals_, "_PICKED", _GAIN_3)
        after = _picks_one_of_two._get_source_hash()
        after_entry = str(_by_name(_program(_picks_one_of_two, _sample(), _sample()))["_picks_one_of_two"])

        assert before_entry != after_entry, "the rebinding must change the emitted calls"
        assert before != after, "the caller's key must move with the emitted calls"

    def test_a_helper_configuration_change_moves_the_caller_key(self, monkeypatch):
        """``auto_scope`` changes the helper's IR, so it must change the key.

        The kernel, its source and its folded constants are identical; only the
        decoration differs. Both the ordinary and the persistent key have to
        follow, because the materialized variant is not the same function.
        """
        a, out = _sample(), _sample()
        before_hash = _calls_traced._get_source_hash()
        before_digest = _calls_traced._persistent_source_digest()
        before_attrs = [
            dict(fn.attrs)
            for fn in _program(_calls_traced, a, out).functions.values()
            if fn.func_type == ir.FunctionType.Inline
        ]

        rescoped = pl.jit_trace.inline(auto_scope=True)(_scale_tile.kernel)
        monkeypatch.setitem(_calls_traced._func.__globals__, "_scale_tile", rescoped)

        after_attrs = [
            dict(fn.attrs)
            for fn in _program(_calls_traced, a, out).functions.values()
            if fn.func_type == ir.FunctionType.Inline
        ]
        assert before_attrs != after_attrs, "the decoration must change the materialized variant"
        assert before_hash != _calls_traced._get_source_hash(), "the caller's key must move"
        assert before_digest != _calls_traced._persistent_source_digest(), (
            "the persisted artifact must not serve the other decoration"
        )

    def test_a_nested_baked_value_moves_the_caller_key(self, monkeypatch):
        """A value a *nested* helper bakes is part of what the caller compiles.

        The plan-level records fold a parser caller's own call sites only; a
        helper body's baked call argument — ``_NestedConfig.KNOB``, an
        attribute chain no free-name fold renders — reaches the compiled body
        through the helper's own trace. Without a record for it the caller's key
        and digest stood still while the compiled body moved, and the second
        compile was served the first artifact.
        """
        a = torch.randn(64, 64, dtype=F32)
        out = torch.empty(64, 64, dtype=F32)
        before_hash = _calls_nested_baked._get_source_hash()
        before_digest = _calls_nested_baked._persistent_source_digest()
        _calls_nested_baked.compile(a, out, config=_SIM_CONFIG)(a, out, config=_SIM_CONFIG)
        assert torch.equal(out, a * 2.0), "the first compile bakes KNOB=2.0"

        monkeypatch.setattr(_NestedConfig, "KNOB", 3.0)
        assert _calls_nested_baked._get_source_hash() != before_hash, (
            "the caller's key must move with the nested baked value"
        )
        assert _calls_nested_baked._persistent_source_digest() != before_digest, (
            "the persisted artifact must not serve the stale bake"
        )

        out_again = torch.empty(64, 64, dtype=F32)
        _calls_nested_baked.compile(a, out_again, config=_SIM_CONFIG)(a, out_again, config=_SIM_CONFIG)
        assert torch.equal(out_again, a * 3.0), "the recompile must bake the new value"


class TestLexicalShadowing:
    """A local named like a traced helper is refused, not silently rewritten.

    Discovery resolves a bare call name through globals and closures only —
    the parser dependency rule — so a parameter or local assignment of the same
    name would let the rewrite import the helper at a call the function's own
    Python gives to its local binding: a silent miscompile. Failing loudly with
    the name and the remediation is the contract; a module-level alias is not a
    local, so the aliased-import path (``TestBindingResolution``) stays open.
    """

    def test_a_parameter_named_like_the_helper_is_refused(self):
        @pl.jit
        def entry(
            a: pl.Tensor[[64, 64], pl.FP32],
            out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
            _copy_tile: pl.Scalar[pl.INT32],
        ):
            with pl.at(level=pl.Level.CORE_GROUP):
                _copy_tile(a, out)  # the user's Python calls the INT32 scalar
            return out

        with pytest.raises(TypeError, match=r"binds the name '_copy_tile' locally"):
            entry.specialize(_sample(), _sample(), 3)

    def test_a_local_assignment_named_like_the_helper_is_refused(self):
        @pl.jit
        def entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                _copy_tile = a
                _copy_tile(a, out)
            return out

        with pytest.raises(TypeError, match="Rename the local binding or the helper"):
            entry.specialize(_sample(), _sample())


class TestSourceEdits:
    """A body edited under a live function object must not keep a stale AST.

    ``traced_source_records`` reads fresh ``inspect.getsource`` while discovery
    once read a tree cached on the function object, so a helper edited between
    two records mixed the new source text with the old call wiring. The AST
    caches key on the parsed tree, so an edit — a fresh parse — refreshes every
    derived answer.
    """

    _MODULE = """\\
import pypto.language as pl


@pl.jit_trace.inline(auto_scope=False)
def _edit_leaf_alpha(a: pl.Tensor, out: pl.Tensor):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, 1.0), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _edit_leaf_beta(a: pl.Tensor, out: pl.Tensor):
    tile = pl.load(a, [0, 0], [64, 64])
    pl.store(pl.mul(tile, 2.0), [0, 0], out)
    return out


@pl.jit_trace.inline(auto_scope=False)
def _edit_helper(a: pl.Tensor, out: pl.Tensor):
    return {leaf}(a, out)


@pl.jit
def _edit_entry(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        _edit_helper(a, out)
    return out
"""

    def _write_and_load(self, tmp_path, leaf):
        path = tmp_path / "edited_traced_inline.py"
        path.write_text(self._MODULE.format(leaf=leaf))
        spec = importlib.util.spec_from_file_location(path.stem, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_an_edited_helper_body_refreshes_the_identity_records(self, tmp_path):
        module = self._write_and_load(tmp_path, "_edit_leaf_alpha")
        before = traced_source_records([module._edit_entry._func])
        assert any("_edit_leaf_alpha" in repr(record) for record in before)

        (tmp_path / "edited_traced_inline.py").write_text(self._MODULE.format(leaf="_edit_leaf_beta"))
        linecache.checkcache()
        after = traced_source_records([module._edit_entry._func])
        assert any("_edit_leaf_beta" in repr(record) for record in after)
        assert not any("_edit_leaf_alpha" in repr(record) for record in after), (
            "a stale tree must not keep describing the edited body"
        )

    def test_get_func_def_parses_the_edited_source(self, tmp_path):
        module = self._write_and_load(tmp_path, "_edit_leaf_alpha")
        kernel = module._edit_helper.kernel
        assert "_edit_leaf_alpha" in ast.unparse(get_func_def(kernel))

        (tmp_path / "edited_traced_inline.py").write_text(self._MODULE.format(leaf="_edit_leaf_beta"))
        linecache.checkcache()
        rendered = ast.unparse(get_func_def(kernel))
        assert "_edit_leaf_beta" in rendered
        assert "_edit_leaf_alpha" not in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
