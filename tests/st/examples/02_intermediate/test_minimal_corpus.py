# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Device acceptance for the tracing-jit corpus.

``tile_softmax`` and the fused-linear pair already run under
``tests/st/examples``; this covers the remaining example entries (a staged
elementwise loop and inline composition) and the two focused fixtures: runtime
control flow with a host knob, and an annotated entry with a runtime scalar.
Every corpus kernel thus has a numerical golden on hardware.

The samples cover the declared contract, not one path through it: F1 runs both
branches, the zero-iteration edge, both host-knob values, and the leading-tile
shape variant; F2 runs two runtime-scalar values against one artifact; and E1
runs an FP16 case for the dtype specialization. The fixtures themselves are
imported from the same module the CPU suite pins, so IR and hardware cannot
drift onto different kernels.
"""

import pytest
import torch
from examples.beginner.elementwise import chunked_add
from harness import st
from harness.core.harness import TensorSpec
from harness.core.kernel_source import datatype_from_torch
from pypto.runtime.tensor_spec import ScalarSpec

from examples.utils.cross_function_calls import main_kernel
from tests.ut.language.tracing.jit._traced_corpus_fixtures import annotated_add, controlled_accumulate


def _chunked_add_case():
    """E1: c = a + b over a staged 512x128 loop."""
    torch.manual_seed(0)
    a = torch.randn(512, 128, dtype=torch.float32)
    b = torch.randn(512, 128, dtype=torch.float32)
    c = torch.zeros(512, 128, dtype=torch.float32)
    return st.case(chunked_add, a, b, c, name="chunked_add", golden=lambda _: a + b)


def _chunked_add_fp16_case():
    """E1 (FP16): the dtype is a specialization, not a reshape of the same run."""
    torch.manual_seed(0)
    a = torch.randn(512, 128, dtype=torch.float16)
    b = torch.randn(512, 128, dtype=torch.float16)
    c = torch.zeros(512, 128, dtype=torch.float16)
    return st.case(
        chunked_add,
        a,
        b,
        c,
        name="chunked_add_fp16",
        golden=lambda _: a + b,
        rtol=1e-3,
        atol=1e-3,
    )


def _main_kernel_case():
    """E5: c = (a + 1.0) * 2.0, composed from two inline helpers."""
    torch.manual_seed(0)
    a = torch.randn(128, 128, dtype=torch.float32)
    c = torch.zeros(128, 128, dtype=torch.float32)
    return st.case(main_kernel, a, c, name="main_kernel", golden=lambda _: (a + 1.0) * 2.0)


def _controlled_accumulate_case(n: int, scale: float, shape: tuple[int, int] = (64, 64)):
    """F1: acc = (n+1)*x; then n > 1 multiplies by the baked knob, else subtracts it."""
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=torch.float32)

    def golden(_):
        acc = (n + 1) * x[:64, :64]
        return acc * scale if n > 1 else acc - scale

    return st.case(
        controlled_accumulate,
        x,
        n,
        scale,
        torch.zeros(64, 64, dtype=torch.float32),
        name=f"controlled_accumulate_n{n}_scale{scale:g}_x{shape[0]}",
        scalars=[ScalarSpec("n", n, "int64")],
        golden=golden,
    )


def _annotated_add_case(value: float):
    """F2: a runtime scalar; the artifact is not specialized on its value."""
    torch.manual_seed(0)
    x = torch.randn(64, 64, dtype=torch.float32)
    out = torch.zeros(64, 64, dtype=torch.float32)
    return st.case(
        annotated_add,
        x,
        value,
        out,
        name=f"annotated_add_{value:g}",
        tensors=[
            TensorSpec("x", [64, 64], datatype_from_torch(x.dtype), init_value=x),
            TensorSpec("out", [64, 64], datatype_from_torch(out.dtype), init_value=out, is_output=True),
        ],
        scalars=[ScalarSpec("n", value, "float")],
        golden=lambda _: x + value,
    )


@st.cases(
    _chunked_add_case(),
    _chunked_add_fp16_case(),
    _main_kernel_case(),
    _controlled_accumulate_case(3, 2.0),
    _controlled_accumulate_case(0, 2.0),
    _controlled_accumulate_case(1, 2.0),
    _controlled_accumulate_case(3, 3.0),
    _controlled_accumulate_case(3, 2.0, shape=(128, 64)),
    _annotated_add_case(1.0),
    _annotated_add_case(-2.0),
)
def test_minimal_corpus_device(case_run):
    """Every case matches its torch reference on hardware."""
    case_run.assert_passed()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
