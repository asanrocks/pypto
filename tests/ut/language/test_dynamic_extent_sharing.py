# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Shared extent Vars: ``pl.dynamic`` unwraps name-keyed across modules.

The corpus idiom is one ``T = pl.dynamic("T")`` per module; the extent Var
must be shared by name so cross-function structural equality and variant
dedup treat same-named extents as one dimension — the tracing counterpart
of the parser frontend's shared ``dyn_var_cache``.
"""

import pypto.language as pl
from pypto.pypto_core import ir


def test_same_name_shares_the_extent_var():
    a = pl.dynamic("T_DYN")
    b = pl.dynamic("T_DYN")
    assert a.unwrap() is b.unwrap()


def test_distinct_names_stay_distinct():
    a = pl.dynamic("T_DYN")
    b = pl.dynamic("OTHER_DYN")
    assert a.unwrap() is not b.unwrap()


def test_same_named_dynvars_shape_functions_with_one_extent():
    """Two functions shaped by two same-named DynVar objects (the per-module
    corpus idiom) share one extent Var in their param types."""

    t_a = pl.dynamic("SHARED_T")
    t_b = pl.dynamic("SHARED_T")

    @pl.function
    def f_a(x: pl.Tensor[[t_a, 8], pl.FP32]) -> pl.Tensor[[t_a, 8], pl.FP32]:
        return x

    @pl.function
    def f_b(x: pl.Tensor[[t_b, 8], pl.FP32]) -> pl.Tensor[[t_b, 8], pl.FP32]:
        return x

    assert f_a.params[0].type.shape[0] is f_b.params[0].type.shape[0]
    ir.assert_structural_equal(f_a, f_b, enable_auto_mapping=False)


def test_parser_and_tracing_dynvars_agree():
    """A DynVar resolved by the parser's dyn_var_cache and one resolved by
    tracing's unwrap share the name-keyed Var."""

    t = pl.dynamic("PARITY_T")

    @pl.function
    def traced(x: pl.Tensor[[t, 4], pl.FP32]) -> pl.Tensor[[t, 4], pl.FP32]:
        return x

    assert traced.params[0].type.shape[0] is t.unwrap()


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
