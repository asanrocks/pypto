# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You should not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The jit torch-argument dtype bridge must cover every DataType with a torch
spelling.

Found by the pypto-lib corpus probe (2026-09, ``models/deepseek_v4_pro/gate.py``):
the tracing jit's bridge table dropped the MX FP8/FP4 entries and mistyped
``uint8`` as ``INT8`` — the quarantined specializer (``pypto/jit/decorator.py``)
maps all of them, and the compatibility envelope requires the tracing jit to
preserve that behavior. The golden harness passes torch tensors (its
``TensorSpec`` dtype *is* a torch dtype), so a kernel author cannot avoid the
bridge.
"""

import pytest

torch = pytest.importorskip("torch")

from pypto.language.tracer.decorators.jit import _load_torch_dtypes  # noqa: E402
from pypto.pypto_core import DataType  # noqa: E402

# The unconditional core: every torch dtype that always exists and has a
# DataType counterpart. uint8 is UINT8 — the specializer's mapping, not INT8.
_CORE = {
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

# Optional MX dtypes — present per torch version, mapped whenever present.
_OPTIONAL = {
    "float8_e4m3fn": DataType.FP8E4M3FN,
    "float8_e5m2": DataType.FP8E5M2,
    "float8_e8m0fnu": DataType.FP8E8M0,
    "float4_e2m1fn_x2": DataType.FP4,
}


def test_core_torch_dtypes_map() -> None:
    table = _load_torch_dtypes()
    for torch_dtype, pto_dtype in _CORE.items():
        assert table[torch_dtype] is pto_dtype, f"{torch_dtype} mis-mapped"


@pytest.mark.parametrize("torch_name,pto_dtype", sorted(_OPTIONAL.items()))
def test_optional_mx_dtypes_map_when_torch_provides_them(torch_name: str, pto_dtype: DataType) -> None:
    torch_dtype = getattr(torch, torch_name, None)
    table = _load_torch_dtypes()
    if torch_dtype is None:
        pytest.skip(f"torch {torch.__version__} has no {torch_name}")
    assert table[torch_dtype] is pto_dtype, f"{torch_dtype} mis-mapped"
