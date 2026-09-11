# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""The callee-lane variant key: structural identity, formatter-free.

``_variant_key_part`` builds the dedup key the jit variant lane looks
callees up by. Its identity contract: two annotations that produce the same
key must produce an identical trace. The key is built from facade
primitives — kind, dtype, extent strings — the same fields the dispatch
cache keys on; building it must never spawn the ruff formatter.
"""

from typing import Any

import pypto.language as pl
import pypto.language.distributed as pld
import pytest
from pypto.ir import formatter
from pypto.language.tracer.decorators.jit import _variant_key_part
from pypto.language.typing.dynamic import DynVar


@pytest.fixture()
def spawn_counter(monkeypatch: pytest.MonkeyPatch):
    calls: list[Any] = []

    class _FakeResult:
        stdout = ""
        returncode = 1  # fail closed: any spawn means the test sees it

    def fake_run(*args: Any, **kwargs: Any) -> _FakeResult:
        calls.append(args)
        return _FakeResult()

    monkeypatch.setattr(formatter.subprocess, "run", fake_run)
    formatter._FORMAT_CACHE.clear()
    return calls


def test_structurally_equal_annotations_share_one_key(spawn_counter: list[Any]) -> None:
    a = _variant_key_part(pl.Tensor[[8, 2048], pl.FP32])
    b = _variant_key_part(pl.Tensor[[8, 2048], pl.FP32])
    assert a == b
    assert hash(a) == hash(b)
    assert {a: 1}.get(b) == 1  # dict-based dedup hits
    assert not spawn_counter


def test_distinct_variants_differ(spawn_counter: list[Any]) -> None:
    base = _variant_key_part(pl.Tensor[[8, 2048], pl.FP32])
    assert _variant_key_part(pl.Tensor[[9, 2048], pl.FP32]) != base  # extent
    assert _variant_key_part(pl.Tensor[[8, 2048], pl.BF16]) != base  # dtype
    assert _variant_key_part(pl.Scalar[pl.INT32]) != base  # kind


def test_dynamic_extents_key_on_the_extent(spawn_counter: list[Any]) -> None:
    a = _variant_key_part(pl.Tensor[[DynVar("tokens")], pl.FP32])
    b = _variant_key_part(pl.Tensor[[DynVar("tokens")], pl.FP32])
    c = _variant_key_part(pl.Tensor[[DynVar("other")], pl.FP32])
    assert a == b
    assert a != c


def test_distributed_tensor_type_keys(spawn_counter: list[Any]) -> None:
    """``DistributedTensorType`` is outside the C++ structural hash — the
    primitive route must handle it where hashing the type object raises."""
    a = _variant_key_part(pld.DistributedTensor[[4, 8], pl.INT32])
    b = _variant_key_part(pld.DistributedTensor[[4, 8], pl.INT32])
    assert a == b
    assert _variant_key_part(pld.DistributedTensor[[4, 9], pl.INT32]) != a
    # kind is part of the key: a distributed and a plain tensor never merge
    assert _variant_key_part(pl.Tensor[[4, 8], pl.INT32]) != a
    assert not spawn_counter


def test_runtime_scalar_keys_on_dtype_not_value(spawn_counter: list[Any]) -> None:
    """One variant serves every runtime-scalar value (the dispatch cache's
    type-only rule); only the dtype splits entries."""
    assert _variant_key_part(pl.runtime(3, pl.INT32)) == _variant_key_part(pl.runtime(4, pl.INT32))
    assert _variant_key_part(pl.runtime(3, pl.INT32)) != _variant_key_part(pl.runtime(3, pl.FP32))
    assert not spawn_counter


def test_tile_and_out_annotate_as_facade_primitives(spawn_counter: list[Any]) -> None:
    a = _variant_key_part(pl.Tile[[64], pl.FP32])
    assert a == _variant_key_part(pl.Tile[[64], pl.FP32])
    assert _variant_key_part(pl.Tile[[32], pl.FP32]) != a
    # ``pl.Out[T]`` erases to T's facade — direction was never part of the
    # key (it rides the callee's own declaration, not the call site).
    assert _variant_key_part(pl.Out[pl.Tensor[[8], pl.BF16]]) == _variant_key_part(pl.Tensor[[8], pl.BF16])
    assert not spawn_counter


def test_host_scalars_key_on_value(spawn_counter: list[Any]) -> None:
    assert _variant_key_part(3) == _variant_key_part(3)
    assert _variant_key_part(3) != _variant_key_part(4)
    assert _variant_key_part(1) != _variant_key_part(True)  # 1 == True in Python; the key must not merge them
    assert not spawn_counter
