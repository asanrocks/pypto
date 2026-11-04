# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Tests that ``pl.at(...)`` carries its keyword arguments onto the construct.

``pl.at`` is a construct recipe, so the constructor surface is the bound
arguments of the recipe call — readable on the instance before it is entered.
Regression (PR #1819 review): ``pl.at(..., allow_early_resolve=True)`` was
accepted but silently dropped because the flag was not forwarded; every
accepted keyword must reach the recipe body.
"""

import pypto.language as pl
import pytest
from pypto import ir


def test_at_forwards_allow_early_resolve():
    """pl.at(..., allow_early_resolve=True) carries the flag onto the construct."""
    ctx = pl.at(ir.Level.CORE_GROUP, allow_early_resolve=True)
    assert getattr(ctx, "allow_early_resolve") is True


def test_at_allow_early_resolve_defaults_false():
    """allow_early_resolve defaults to False on the construct when omitted."""
    ctx = pl.at(ir.Level.CORE_GROUP)
    assert getattr(ctx, "allow_early_resolve") is False


def test_at_forwards_name_hint():
    """pl.at(..., name_hint=...) reaches the construct (sibling-kwarg forwarding guard)."""
    ctx = pl.at(ir.Level.CORE_GROUP, name_hint="fused_scope")
    assert getattr(ctx, "name_hint") == "fused_scope"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
