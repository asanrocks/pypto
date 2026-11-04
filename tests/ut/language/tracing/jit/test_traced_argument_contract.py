# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The declared type of a parameter owns the call boundary.

The tracing lane derives a callee's parameter from the call site, so a call is
a specialization — but only inside the declaration. A ``pl.Scalar[pl.INT32]``
parameter therefore keeps INT32 whatever the caller holds: an INDEX argument
inside the declared slot's integer lattice is *converted* to the declared dtype
and bound to a temporary, so the callee's signature stops depending on which
lane (or which loop variable) reached it. Everything outside that lattice keeps
its refusal, naming the parameter and the one-word fix.

The two spellings that must keep deferring to the call site are asserted here
too, since the guard is only useful if it is not over-broad: a declared dynamic
extent accepts any concrete extent, and a host literal lifts into the declared
dtype — and, because a Python integer states no dtype, only a literal the
declaration can hold lifts at all.
"""

import pypto.language as pl
import pytest
import torch
from pypto.pypto_core import DataType, ir

F32 = torch.float32
B = pl.dynamic("B")


@pl.jit_trace
def per_rank(
    x: pl.Tensor[[8, 8], pl.FP32],
    rank: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
):
    """The callee of a ``device=`` submit: its scalar declares INT32."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, rank), [0, 0], out)
    return out


@pl.jit_trace
def submit_index(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """``r`` is an INDEX value: the declared slot converts it."""
    for r in pl.parallel(0, 2):
        per_rank(x, r, out, device=r)
    return out


@pl.jit_trace
def submit_cast(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """The same submit with the conversion written out: legal, and not doubled."""
    for r in pl.parallel(0, 2):
        per_rank(x, pl.cast(r, pl.INT32), out, device=r)
    return out


@pl.jit_trace
def submit_unrepresentable(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """A Python integer states no dtype, so the declaration must hold it."""
    for r in pl.parallel(0, 2):
        per_rank(x, 2**31, out, device=r)
    return out


@pl.jit_trace
def submit_typed_out_of_range(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """A *typed* value converts: the author already stated its width."""
    for r in pl.parallel(0, 2):
        per_rank(x, pl.const(2**31, pl.INDEX), out, device=r)
    return out


@pl.jit_trace
def submit_uint_out(
    x: pl.Tensor[[8, 8], pl.FP32],
    rank: pl.Scalar[pl.UINT32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
):
    """Two fixed-width dtypes: outside the admitted lattice."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, rank), [0, 0], out)
    return out


@pl.jit_trace
def submit_fixed_width(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    for r in pl.parallel(0, 2):
        submit_uint_out(x, pl.cast(r, pl.INT32), out, device=r)
    return out


@pl.jit_trace
def submit_literal(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """A host literal into a Scalar slot lifts *by declaration*, not by value."""
    for r in pl.parallel(0, 2):
        per_rank(x, 3, out, device=r)
    return out


@pl.jit_trace
def lazy_scale(scale, rank: pl.Scalar[pl.INT32]):
    """A lazy entry: ``scale`` is a host knob, so the trace follows the samples."""
    return pl.cast(rank, pl.FP32)


@pl.jit_trace
def lazy_index(scale, offset: pl.Scalar[pl.INDEX]):
    """The symmetric lattice direction, on the same lazy lane."""
    return pl.cast(offset, pl.FP32)


@pl.jit_trace
def index_tensor(
    x: pl.Tensor[[1], pl.INDEX],
    out: pl.Out[pl.Tensor[[1], pl.INDEX]],
):
    """INDEX has no torch dtype; the host ABI carries it as torch.int64."""
    return out


@pl.jit_trace
def bare_tensor(x: pl.Tensor, out: pl.Out[pl.Tensor]):
    """A bare slot declares no dtype at all: the sample's world stands."""
    return out


@pl.jit_trace
def narrow(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], out)
    return out


@pl.jit_trace
def nz_layout(
    w: pl.Tensor[[128, 256], pl.FP16, pl.NZ],
    out: pl.Out[pl.Tensor[[64, 128], pl.FP32]],
):
    """A declared NZ layout: the sample says shape and dtype, the ``->`` says packing."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(out, [0, 0], [64, 128])
        pl.store(tile, [0, 0], out)
    return out


@pl.jit_trace
def wider_argument(x: pl.Tensor[[16, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    narrow(x, out)
    return out


@pl.jit_trace
def narrow_bf16(x: pl.Tensor[[8, 8], pl.BF16], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    narrow(x, out)
    return out


@pl.jit_trace
def row_block(x: pl.Tensor[[B, 8], pl.FP32], out: pl.Out[pl.Tensor[[B, 8], pl.FP32]]):
    """A declared dynamic extent: the call site supplies the concrete one."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, 1.0), [0, 0], out)
    return out


@pl.jit_trace
def shared_rows(
    x: pl.Tensor[[B, 8], pl.FP32],
    y: pl.Tensor[[B, 8], pl.FP32],
    out: pl.Out[pl.Tensor[[B, 8], pl.FP32]],
):
    """One symbol, three parameters: every occurrence names the same extent."""
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(x, [0, 0], [8, 8]), [0, 0], out)
    return out


@pl.jit_trace
def call_row_block(x: pl.Tensor[[16, 8], pl.FP32], out: pl.Out[pl.Tensor[[16, 8], pl.FP32]]):
    row_block(x, out)
    return out


MX_ROWS, MX_COLS = 8, 64


@pl.jit_trace
def mx_scale_blank(
    s: pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
):
    """A blank-stride MX declaration: the packed-canonical abbreviation."""
    return out


@pl.jit_trace
def mx_scale_explicit(
    s: pl.Tensor[
        [MX_ROWS, MX_COLS], pl.FP8E8M0, pl.TensorView(stride=[MX_COLS, 1], layout=pl.TensorLayout.MX_B_NN)
    ],
    out: pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
):
    """The same view with the canonical strides spelled out."""
    return out


@pl.jit_trace
def calls_blank_from_a_view(
    s: pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0],
    out: pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
):
    """``tensor.view`` hands the callee explicitly derived canonical strides."""
    mx_scale_blank(pl.tensor.view(s, [MX_ROWS, MX_COLS], layout=pl.MX_B_NN), out)
    return out


@pl.jit_trace
def calls_explicit_from_a_tag(
    s: pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
):
    """A blank-tagged argument reaches an explicitly canonical declaration."""
    mx_scale_explicit(s, out)
    return out


@pl.jit_trace
def calls_blank_from_a_transpose(
    s: pl.Tensor[
        [MX_ROWS, MX_COLS], pl.FP8E8M0, pl.TensorView(stride=[1, MX_COLS], layout=pl.TensorLayout.MX_B_NN)
    ],
    out: pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
):
    """The same layout, a stride outside the canonical packing."""
    mx_scale_blank(s, out)
    return out


@pl.jit_trace
def mx_scale_dyn(
    s: pl.Tensor[[B, MX_COLS], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[B, MX_COLS], pl.FP8E8M0]],
):
    """A blank-stride MX declaration over a dynamic extent."""
    return out


@pl.jit_trace
def calls_dyn_blank(
    s: pl.Tensor[[B, MX_COLS], pl.FP8E8M0, pl.MX_B_NN],
    out: pl.Out[pl.Tensor[[B, MX_COLS], pl.FP8E8M0]],
):
    mx_scale_dyn(s, out)
    return out


def _tensor(shape: tuple[int, int] = (8, 8), dtype: torch.dtype = F32) -> torch.Tensor:
    return torch.empty(*shape, dtype=dtype)


def _function(program: ir.Program, name: str) -> ir.Function:
    fn = next((f for f in program.functions.values() if f.name == name), None)
    assert fn is not None, f"missing function {name!r}; have {[f.name for f in program.functions.values()]}"
    return fn


class TestTheSubmitBoundary:
    """A submitted value must reach the callee as the declared type."""

    def test_a_declared_int32_slot_converts_an_index_value(self):
        """The two lanes now agree: the callee keeps its declared INT32."""
        param = _function(submit_index.specialize(), "per_rank").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_the_conversion_is_bound_to_the_shared_temporary(self):
        """One statement, one name, on both lanes: `_rank_int32`."""
        program = submit_index.specialize()
        caller = _function(program, "submit_index")
        printed = program.as_python()
        assert "_rank_int32: pl.Scalar[pl.INT32] = pl.cast(r, pl.INT32)" in printed, printed
        assert "_rank_int32" in str(caller.body) or "_rank_int32" in printed

    def test_an_author_written_cast_is_not_doubled(self):
        """Writing the conversion out stays legal and idempotent."""
        program = submit_cast.specialize()
        printed = program.as_python()
        assert printed.count("pl.cast(r, pl.INT32)") == 1, printed
        param = _function(program, "per_rank").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_a_host_literal_lifts_into_the_declared_dtype(self):
        param = _function(submit_literal.specialize(), "per_rank").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_a_python_integer_the_slot_cannot_hold_is_refused(self):
        with pytest.raises(TypeError, match=r"parameter 'rank' is declared pl\.Scalar\[pl\.INT32\]"):
            submit_unrepresentable.specialize()

    def test_the_range_error_names_the_slot_and_the_range(self):
        with pytest.raises(TypeError) as excinfo:
            submit_unrepresentable.specialize()
        message = str(excinfo.value)
        assert "outside its range [-2147483648, 2147483647]" in message, message

    def test_a_typed_value_outside_the_range_converts(self):
        """A typed value goes the other way: the author stated its width."""
        program = submit_typed_out_of_range.specialize()
        printed = program.as_python()
        assert "pl.cast(" in printed, printed
        param = _function(program, "per_rank").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_a_fixed_width_pair_stays_refused(self):
        """Two fixed-width dtypes are outside the lattice: the language states
        no signedness policy between them."""
        with pytest.raises(TypeError, match=r"dtype int32 against the declared uint32") as excinfo:
            submit_fixed_width.specialize()
        message = str(excinfo.value)
        assert "one end is pl.INDEX" in message, message
        assert "pl.cast" in message, message


class TestTheCallBoundary:
    """A plain call is checked exactly like a submit reference."""

    def test_a_declared_static_extent_refuses_a_wider_argument(self):
        with pytest.raises(TypeError, match=r"declared static extent 8"):
            wider_argument.specialize()

    def test_a_declared_dtype_refuses_a_different_one(self):
        with pytest.raises(TypeError, match=r"dtype bfloat16 against the declared fp32"):
            narrow_bf16.specialize()

    def test_a_declared_dynamic_extent_still_specializes(self):
        program = call_row_block.specialize()
        param = _function(program, "row_block").params[0]
        assert isinstance(param.type, ir.TensorType)
        # A literal extent is an ``int``; a symbolic one carries its value.
        assert [getattr(dim, "value", dim) for dim in param.type.shape] == [16, 8]

    def test_an_explicit_facade_call_is_checked_too(self):
        """``specialize``/``lower`` take facades — the author's own type statement."""
        with pytest.raises(TypeError, match=r"parameter 'x' is declared"):
            per_rank.specialize(
                pl.Tensor[[8, 8], pl.BF16],
                pl.Scalar[pl.INT32],
                pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            )

    def test_a_matching_facade_call_still_traces(self):
        program = per_rank.specialize(
            pl.Tensor[[8, 8], pl.FP32],
            pl.Scalar[pl.INT32],
            pl.Out[pl.Tensor[[8, 8], pl.FP32]],
        )
        param = _function(program, "per_rank").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_one_symbol_cannot_receive_two_extents(self):
        with pytest.raises(TypeError, match=r"declared extent 'B' is 16 .* but 32"):
            shared_rows.specialize(_tensor((16, 8)), _tensor((32, 8)), _tensor((16, 8)))

    def test_matching_symbol_extents_still_specialize(self):
        program = shared_rows.specialize(_tensor((16, 8)), _tensor((16, 8)), _tensor((16, 8)))
        assert [str(p.type) for p in _function(program, "shared_rows").params] == [
            "pl.Tensor[[B, 8], pl.FP32]",
            "pl.Tensor[[B, 8], pl.FP32]",
            "pl.Tensor[[B, 8], pl.FP32]",
        ], "the declaration keeps the symbol; the concrete sample specializes it"


class TestTheRootSampleLane:
    """A lazily traced entry mints its parameters from the samples, so an
    admitted pair must not leak the sample's dtype into the signature: the
    declaration is the ABI on the host lane, exactly as the eager record and
    the in-trace conversion already state it."""

    def test_an_index_sample_facade_reaches_a_lazy_entry_as_the_declared_dtype(self):
        program = lazy_scale.specialize(2.0, pl.Scalar[pl.INDEX])
        param = _function(program, "lazy_scale").params[0]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INT32

    def test_the_symmetric_direction_reads_the_declaration_too(self):
        program = lazy_index.specialize(2.0, pl.Scalar[pl.INT32])
        param = _function(program, "lazy_index").params[0]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.INDEX


class TestTheHostBoundary:
    """A torch sample answers to the declaration exactly like a facade."""

    def test_a_torch_sample_is_checked_against_a_static_extent(self):
        with pytest.raises(TypeError, match=r"parameter 'x' is declared pl\.Tensor\[\[8, 8\]"):
            narrow.specialize(_tensor((16, 8)), _tensor((8, 8)))

    def test_a_torch_sample_is_checked_against_the_dtype(self):
        with pytest.raises(TypeError, match=r"dtype bfloat16 against the declared fp32"):
            narrow.specialize(_tensor(dtype=torch.bfloat16), _tensor())

    def test_a_matching_torch_sample_still_traces(self):
        program = narrow.specialize(_tensor(), _tensor())
        assert [p.name_hint for p in _function(program, "narrow").params] == ["x", "out"]

    def test_a_torch_int64_sample_bridges_to_a_declared_index_slot(self):
        sample = torch.empty(1, dtype=torch.int64)
        program = index_tensor.specialize(sample, sample)
        first, second = _function(program, "index_tensor").params
        assert isinstance(first.type, ir.TensorType)
        assert isinstance(second.type, ir.TensorType)
        assert [first.type.dtype, second.type.dtype] == [DataType.INDEX, DataType.INDEX]

    def test_an_int64_sample_into_a_bare_slot_stays_int64(self):
        """An absent declaration is not a contradiction.

        The INDEX bridge reads the declared dtype, but ``DataType.__eq__`` is a
        nanobind binding that raises on a non-DataType operand instead of
        answering False, so a bare ``pl.Tensor`` slot turned every int64 sample
        into a TypeError from inside the bridge rather than a traced parameter.
        """
        sample = torch.empty(1, dtype=torch.int64)
        program = bare_tensor.specialize(sample, sample)
        assert [str(p.type) for p in _function(program, "bare_tensor").params] == [
            "pl.Tensor[[1], pl.INT64]",
            "pl.Tensor[[1], pl.INT64]",
        ]

    def test_an_explicit_int64_facade_is_still_outside_index(self):
        with pytest.raises(TypeError, match=r"dtype int64 against the declared index"):
            index_tensor.specialize(
                pl.Tensor[[1], pl.INT64],
                pl.Out[pl.Tensor[[1], pl.INDEX]],
            )

    def test_a_declared_layout_bridges_from_a_torch_sample(self):
        """A torch sample states no layout, so the declaration's layout wins.

        A declaration carrying ``pl.NZ`` describes how the buffer is packed --
        the one fact a host tensor cannot express. Deriving the layout from the
        sample instead made every host call of a layout-declared parameter fail
        with "a different tensor layout" on a kernel whose declaration and
        buffer were both correct.
        """
        program = nz_layout.specialize(_tensor((128, 256), dtype=torch.float16), _tensor((64, 128)))
        param = _function(program, "nz_layout").params[0]
        assert isinstance(param.type, ir.TensorType)
        assert str(param.type) == (
            "pl.Tensor[[128, 256], pl.FP16, pl.TensorView(stride=[], layout=pl.TensorLayout.NZ)]"
        )
        assert param.type.tensor_view is not None
        assert param.type.tensor_view.layout == pl.TensorLayout.NZ

    def test_a_sample_with_the_wrong_shape_still_fails_a_layout_parameter(self):
        """Bridging the layout must not turn the declaration into a rubber stamp."""
        with pytest.raises(TypeError, match=r"dim 0 = 64 against the declared static extent 128"):
            nz_layout.specialize(_tensor((64, 256), dtype=torch.float16), _tensor((64, 128)))


@pl.jit_trace
def int_slot(x: pl.Tensor[[8, 8], pl.FP32], flag: pl.Scalar[pl.INT32]):
    """An integer slot: the category refusals aim here."""
    return pl.cast(flag, pl.FP32)


@pl.jit_trace
def float_slot(x: pl.Tensor[[8, 8], pl.FP32], ratio: pl.Scalar[pl.FP32]):
    """A float slot: admits a host float, refuses a host bool."""
    return pl.cast(ratio, pl.FP32)


@pl.jit_trace
def bool_slot(x: pl.Tensor[[8, 8], pl.FP32], flag: pl.Scalar[pl.BOOL]):
    """A BOOL slot: the one slot a host bool lifts into."""
    return x


@pl.jit_trace
def uint_slot(x: pl.Tensor[[8, 8], pl.FP32], u: pl.Scalar[pl.UINT32]):
    """An unsigned slot: the range check's low end is the refusal."""
    return pl.cast(u, pl.FP32)


@pl.jit_trace
def calls_int_slot_with_bool(x: pl.Tensor[[8, 8], pl.FP32]):
    return int_slot(x, True)


@pl.jit_trace
def calls_int_slot_with_float(x: pl.Tensor[[8, 8], pl.FP32]):
    return int_slot(x, 1.5)


@pl.jit_trace
def calls_float_slot_with_bool(x: pl.Tensor[[8, 8], pl.FP32]):
    return float_slot(x, True)


@pl.jit_trace
def calls_uint_slot_negative(x: pl.Tensor[[8, 8], pl.FP32]):
    return uint_slot(x, -1)


@pl.jit_trace
def dispatch_int_slot(
    x: pl.Tensor[[8, 8], pl.FP32],
    flag: pl.Scalar[pl.INT32],
    out: pl.Out[pl.Tensor[[8, 8], pl.FP32]],
):
    """A torch-dispatchable entry: the same INT32 slot, reached per dispatch."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(x, [0, 0], [8, 8])
        pl.store(pl.add(tile, flag), [0, 0], out)
    return out


class TestTheScalarCategoryLattice:
    """A Python scalar states a category, not a dtype.

    ``True`` is a logical value and ``1.5`` states no integer width, so the
    declared slot admits the category or refuses it — and the verdict is
    computed before the lane split (``const()`` in a trace, the
    ``RuntimeScalar`` marker on the host), so both lanes answer the same
    input identically. Before the check ran before the split, the host lane
    wrapped ``True``/``1.5`` into an INT32 marker while the in-trace lane
    refused the bool with ``const()``'s raw message (no parameter, no kernel)
    and built a float constant under an integer dtype.
    """

    def test_a_host_bool_is_refused_by_an_integer_slot(self):
        with pytest.raises(
            TypeError, match=r"parameter 'flag' is declared pl\.Scalar\[pl\.INT32\] but received a host bool"
        ):
            int_slot(pl.Tensor[[8, 8], pl.FP32], True)

    def test_a_host_float_is_refused_by_an_integer_slot(self):
        with pytest.raises(
            TypeError, match=r"parameter 'flag' is declared pl\.Scalar\[pl\.INT32\] but received a host float"
        ):
            int_slot(pl.Tensor[[8, 8], pl.FP32], 1.5)

    @pytest.mark.parametrize(
        ("caller", "word"),
        [
            (calls_int_slot_with_bool, "bool"),
            (calls_int_slot_with_float, "float"),
        ],
    )
    def test_the_in_trace_lane_refuses_the_same_value_with_the_same_words(self, caller, word):
        with pytest.raises(
            TypeError,
            match=rf"parameter 'flag' is declared pl\.Scalar\[pl\.INT32\] but received a host {word}",
        ):
            caller(pl.Tensor[[8, 8], pl.FP32])

    def test_a_bool_is_refused_by_a_float_slot(self):
        """``const()``'s rule decides the remaining edge: a boolean constant
        requires BOOL, so the float slot refuses a host bool too."""
        with pytest.raises(
            TypeError, match=r"parameter 'ratio' is declared pl\.Scalar\[pl\.FP32\] but received a host bool"
        ):
            float_slot(pl.Tensor[[8, 8], pl.FP32], True)
        with pytest.raises(
            TypeError,
            match=r"parameter 'ratio' is declared pl\.Scalar\[pl\.FP32\] but received a host bool",
        ):
            calls_float_slot_with_bool(pl.Tensor[[8, 8], pl.FP32])

    def test_a_host_float_is_admitted_by_a_float_slot(self):
        program = float_slot.specialize(pl.Tensor[[8, 8], pl.FP32], 1.5)
        param = _function(program, "float_slot").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.FP32

    def test_a_host_bool_is_admitted_by_a_bool_slot(self):
        program = bool_slot.specialize(pl.Tensor[[8, 8], pl.FP32], True)
        param = _function(program, "bool_slot").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.BOOL

    def test_an_integer_is_admitted_by_a_float_slot(self):
        """An int into a float slot keeps the promotion lane: an integer
        states an exact value a float slot holds, not a category refusal."""
        program = float_slot.specialize(pl.Tensor[[8, 8], pl.FP32], 2)
        param = _function(program, "float_slot").params[1]
        assert isinstance(param.type, ir.ScalarType)
        assert param.type.dtype == DataType.FP32

    def test_a_negative_integer_stays_outside_an_unsigned_slot(self):
        """The range check refuses by the low end too, not only by overflow."""
        with pytest.raises(TypeError, match=r"outside its range \[0, 4294967295\]"):
            uint_slot(pl.Tensor[[8, 8], pl.FP32], -1)

    def test_the_in_trace_lane_refuses_the_negative_integer_too(self):
        with pytest.raises(TypeError, match=r"outside its range \[0, 4294967295\]"):
            calls_uint_slot_negative(pl.Tensor[[8, 8], pl.FP32])


class TestTheCheckRunsOnEveryDispatch:
    """A scalar placement never splits the artifact, so the check is per
    dispatch, not per artifact build: a warm cache must not hand an
    unadmitted value a dispatch the declaration refuses."""

    def test_a_warm_cache_still_refuses_an_unadmitted_value(self, monkeypatch):
        from pypto.language.tracing.decorators import jit_trace as jit_module  # noqa: PLC0415

        monkeypatch.setattr(
            jit_module,
            "_compile_program",
            lambda program, config: lambda *args, **kwargs: ("dispatched", args),
        )
        x = _tensor()
        out = _tensor()
        dispatch_int_slot(x, 3, out)  # the valid placement warms the cache
        with pytest.raises(TypeError, match=r"but received a host bool"):
            dispatch_int_slot(x, True, out)


class TestLayoutStrideSpelling:
    """A blank stride and its derived canonical strides state one view.

    RFC #1300 keeps two spellings of one canonical TensorView: a bare layout
    tag abbreviates the packed strides, and ``tensor.view`` derives them
    explicitly with ``BuildLogicalStridesFromLayout`` — the same helper the
    boundary now consults. Comparing spellings instead of canonical forms made
    the derived form refuse the abbreviated declaration, which is the shape
    every per-rank MX scale handoff in a host dispatch loop writes: slice the
    stack, re-view with the layout, call the callee that declares the bare tag.
    """

    def test_a_derived_canonical_stride_matches_a_blank_declaration(self):
        program = calls_blank_from_a_view.specialize()
        assert "mx_scale_blank(" in program.as_python()

    def test_a_blank_tag_matches_an_explicitly_canonical_declaration(self):
        """The equivalence is symmetric: either side may carry the abbreviation."""
        program = calls_explicit_from_a_tag.specialize()
        assert "mx_scale_explicit(" in program.as_python()

    def test_a_stride_outside_the_canonical_packing_still_refuses(self):
        """Canonicalization admits the packing, not the layout tag alone."""
        with pytest.raises(TypeError, match=r"a different tensor layout"):
            calls_blank_from_a_transpose.specialize()

    def test_an_untagged_argument_still_refuses_a_layout_declaration(self):
        """No tensor_view at all is not a spelling of the canonical form."""
        with pytest.raises(TypeError, match=r"a different tensor layout"):
            mx_scale_blank.specialize(
                pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0],
                pl.Out[pl.Tensor[[MX_ROWS, MX_COLS], pl.FP8E8M0]],
            )

    def test_two_blank_declarations_over_dynamic_extents_still_agree(self):
        """A symbolic extent cannot derive the packing, so the comparison keeps
        its pre-canonical fallback: two blanks stay two blanks, not a mismatch
        between a derivation and an abbreviation."""
        program = calls_dyn_blank.specialize(
            pl.Tensor[[16, MX_COLS], pl.FP8E8M0, pl.MX_B_NN],
            pl.Out[pl.Tensor[[16, MX_COLS], pl.FP8E8M0]],
        )
        assert "mx_scale_dyn(" in program.as_python()


class TestBareDtypeMarkers:
    """A marker states a dtype but no world, so it is refused, not guessed."""

    def test_a_bare_dtype_annotation_is_refused_at_decoration(self):
        with pytest.raises(TypeError, match=r"annotated pl\.INDEX"):
            # A bare dtype marker is not a world: the fixture states exactly the
            # annotation the guard must refuse, which Python's checker rejects.
            @pl.jit_trace
            def bare_marker(vlen: pl.INDEX, x: pl.Tensor[[8, 8], pl.FP32]):  # pyright: ignore[reportInvalidTypeForm]
                return x

    def test_the_message_names_the_runtime_scalar_spelling(self):
        with pytest.raises(TypeError) as excinfo:

            @pl.jit_trace
            def bare_marker(vlen: pl.FP32, x: pl.Tensor[[8, 8], pl.FP32]):  # pyright: ignore[reportInvalidTypeForm]
                return x

        assert "pl.Scalar[pl.FP32]" in str(excinfo.value), str(excinfo.value)

    def test_a_host_knob_annotation_still_bakes(self):
        """The guard is about facades' dtype markers, not about host knobs."""

        @pl.jit_trace
        def baked(scale: float, x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
            with pl.at(level=pl.Level.CORE_GROUP):
                tile = pl.load(x, [0, 0], [8, 8])
                pl.store(pl.mul(tile, scale), [0, 0], out)
            return out

        program = baked.specialize(2.0, pl.Tensor[[8, 8], pl.FP32], pl.Out[pl.Tensor[[8, 8], pl.FP32]])
        assert [p.name_hint for p in _function(program, "baked").params] == ["x", "out"]


class TestTheDeclaredSignature:
    """A jit kernel declares its parameters; a variadic one has no lane.

    ``*args``/``**kwargs`` would be silently omitted from the declared
    parameter order — no IR parameter, no baked host knob, and no keyword
    the caller could ever bind (the binding reads the declared names, so a
    variadic name is an unknown keyword) — so the signature is refused at
    decoration instead of specializing without them. A keyword-only
    parameter is the same silent omission: arguments bind in declaration
    order (a keyword names a declared parameter, it does not declare one),
    so it never enters ``_param_names`` and the caller's keyword is an
    unknown name.
    """

    def test_a_var_positional_parameter_is_refused_at_decoration(self):
        with pytest.raises(TypeError, match=r"parameter '\*values' is variadic"):

            @pl.jit_trace
            def var_args(x: pl.Tensor[[8, 8], pl.FP32], *values):
                return x

    def test_a_var_keyword_parameter_is_refused_at_decoration(self):
        with pytest.raises(TypeError, match=r"parameter '\*\*options' is variadic"):

            @pl.jit_trace
            def var_kwargs(x: pl.Tensor[[8, 8], pl.FP32], **options):
                return x

    def test_a_keyword_only_parameter_is_refused_at_decoration(self):
        with pytest.raises(TypeError, match=r"parameter 'x' is keyword-only"):

            @pl.jit_trace
            def kw_only(*, x: pl.Tensor[[8, 8], pl.FP32]):
                return x


class TestConstexprHostValues:
    """A ``pl.constexpr`` declaration bakes the documented host-value set.

    The marker's contract admits ``int``/``float``/``bool``/``str``, a
    ``DataType``, an enum member and lists of those. The tracer used to admit
    only the scalar trio at both call boundaries, so a kernel naming its own
    ``pl.spmd`` regions from a parameter (B10) or selecting a reinterpret
    dtype from one (B9) could not serve any traced or host caller.
    """

    def test_a_str_bakes_into_the_trace_and_reaches_the_region(self):
        program = named_entry.specialize()
        assert "entry_hint" in str(program)
        assert "other_hint" not in str(program)

    def test_two_baked_strings_never_share_a_program(self):
        """The baked constant joins the variant, so the hint is not ignored."""
        assert str(named_entry.specialize()) != str(other_named_entry.specialize())

    def test_a_baked_knob_parameter_vanishes_from_the_entry_abi(self):
        program = dtype_knob.specialize(pl.Tensor[[8], pl.FP32], pl.Out[pl.Tensor[[8], pl.FP32]], pl.BF16)
        entry = next(iter(program.functions.values()))
        assert [p.name_hint for p in entry.params] == ["source", "out"]

    def test_a_str_into_a_runtime_slot_stays_loud(self):
        with pytest.raises(TypeError, match=r"parameter 'tag' is declared .* but received a host str"):
            staged_caller.specialize()

    def test_an_unfoldable_value_names_the_foldable_set(self):
        with pytest.raises(TypeError, match=r"constexpr parameter 'knob' .* no form the tracer can fold"):
            junk_caller.specialize()


@pl.jit_trace.inline(auto_scope=False)
def named_leaf(
    a: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Tensor[[8, 8], pl.FP32],
    name_hint: pl.constexpr,
):
    """The B10 shape: the region name comes from the parameter."""
    with pl.spmd(1, name_hint=name_hint):
        pl.store(pl.load(a, [0, 0], [8, 8]), [0, 0], out)
    return out


@pl.jit_trace
def named_entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        named_leaf(a, out, "entry_hint")
    return out


@pl.jit_trace
def other_named_entry(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        named_leaf(a, out, "other_hint")
    return out


@pl.jit_trace
def dtype_knob(
    source: pl.Tensor[[8], pl.FP32],
    out: pl.Out[pl.Tensor[[8], pl.FP32]],
    byte_dtype: pl.constexpr,
):
    """The B9 shape: a dtype-parameterised entry."""
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.load(source, [0], [8]), [0], out)
    return out


@pl.jit_trace
def runtime_tag(x: pl.Tensor[[8], pl.FP32], out: pl.Tensor[[8], pl.FP32], tag: pl.Scalar[pl.INT32]):
    return out


@pl.jit_trace
def staged_caller(x: pl.Tensor[[8], pl.FP32], out: pl.Out[pl.Tensor[[8], pl.FP32]]):
    runtime_tag(x, out, "oops")
    return out


@pl.jit_trace
def junk_caller(x: pl.Tensor[[8], pl.FP32], out: pl.Out[pl.Tensor[[8], pl.FP32]]):
    junk_leaf(x, out, object())
    return out


@pl.jit_trace.inline(auto_scope=False)
def junk_leaf(x: pl.Tensor[[8], pl.FP32], out: pl.Tensor[[8], pl.FP32], knob: pl.constexpr):
    return out


@pl.jit_trace.inline(auto_scope=False)
def scale_leaf(
    a: pl.Tensor[[8, 8], pl.FP32],
    out: pl.Tensor[[8, 8], pl.FP32],
    factor: pl.constexpr,
):
    """A leaf whose compile-time factor is a declared parameter."""
    with pl.at(level=pl.Level.CORE_GROUP):
        tile = pl.load(a, [0, 0], [8, 8])
        pl.store(pl.mul(tile, factor), [0, 0], out)
    return out


@pl.jit_trace
def named_call(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    scale_leaf(a, out=out, factor=2.0)
    return out


@pl.jit_trace
def positional_call(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    scale_leaf(a, out, 2.0)
    return out


@pl.jit_trace
def doubled_binding(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """``out`` arrives twice: positionally and by name."""
    scale_leaf(a, out, out=out, factor=2.0)
    return out


@pl.jit_trace
def unknown_keyword(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    scale_leaf(a, out, 2.0, gain=1.0)  # pyright: ignore[reportCallIssue]  # the refusal is the assertion
    return out


@pl.jit_trace
def unbound_parameter(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    scale_leaf(a, out=out)  # pyright: ignore[reportCallIssue]  # the refusal is the assertion
    return out


@pl.jit_trace
def surplus_positional(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    scale_leaf(a, out, 2.0, 3.0)  # pyright: ignore[reportCallIssue]  # the refusal is the assertion
    return out


@pl.jit_trace
def submit_by_name(x: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    """A named parameter beside the call-site ``device=`` metadata."""
    for r in pl.parallel(0, 2):
        per_rank(x, rank=r, out=out, device=r)
    return out


@pl.function(type=pl.FunctionType.Inline)
def global_leaf(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Tensor[[8, 8], pl.FP32]):
    """A ``pl.function`` global: the parser lane's external-call surface."""
    return out


@pl.jit_trace
def calls_global_by_name(a: pl.Tensor[[8, 8], pl.FP32], out: pl.Out[pl.Tensor[[8, 8], pl.FP32]]):
    global_leaf(a, out=out)  # pyright: ignore[reportCallIssue]  # the refusal is the assertion
    return out


class TestTheNameBoundary:
    """A keyword names a declared parameter; every lane below reads positions.

    The parser lane always read keywords at a jit call, so the tracing lane
    refusing them made a spelling (not a contract) decide whether shared code
    compiled. Both now bind a name to its slot and hand the lanes one positional
    list, which is why the two spellings cannot specialize differently.
    """

    def test_a_keyword_binds_to_its_declared_slot(self):
        printed = named_call.specialize().as_python()
        assert "scale_leaf(a, out)" in printed, printed

    def test_the_two_spellings_specialize_to_one_program(self):
        """The call's own spelling never reaches the program."""
        named = _function(named_call.specialize(), "named_call")
        positional = _function(positional_call.specialize(), "positional_call")
        assert str(named.body) == str(positional.body), str(named.body)

    def test_two_names_for_one_parameter_are_refused(self):
        with pytest.raises(TypeError, match=r"binds parameter 'out' both positionally and by keyword"):
            doubled_binding.specialize()

    def test_an_unknown_keyword_names_the_declared_parameters(self):
        with pytest.raises(TypeError, match=r"passes no parameter 'gain'; the callee declares"):
            unknown_keyword.specialize()

    def test_an_unbound_parameter_is_named(self):
        """A keyword spelling must still supply every parameter — no defaults."""
        with pytest.raises(TypeError, match=r"does not bind parameter\(s\) \['factor'\]"):
            unbound_parameter.specialize()

    def test_a_surplus_positional_argument_is_refused(self):
        with pytest.raises(TypeError, match=r"passes 4 positional argument\(s\) but the callee declares 3"):
            surplus_positional.specialize()

    def test_the_device_override_stays_call_site_metadata(self):
        """``device=`` still dispatches; it is not a parameter of the callee."""
        printed = submit_by_name.specialize().as_python()
        assert "self.per_rank(x, _rank_int32, out, device=r)" in printed, printed

    def test_a_pl_function_global_keeps_the_parser_lanes_refusal(self):
        """Verbatim callees are not the jit envelope: the AST lane refuses too."""
        with pytest.raises(TypeError, match=r"keyword arguments are not accepted on a sub-kernel call"):
            calls_global_by_name.specialize()

    def test_a_host_call_binds_a_keyword_too(self):
        """The specialize lane reads the same binding as an in-trace call."""
        program = named_leaf.specialize(
            pl.Tensor[[8, 8], pl.FP32],
            out=pl.Out[pl.Tensor[[8, 8], pl.FP32]],
            name_hint="keyword_bound",
        )
        assert "keyword_bound" in program.as_python()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
