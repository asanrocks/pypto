# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Behavioral corpus: every construct shipped so far, run
end-to-end through the trace driver against its native twin. The kernels live in
_corpus.py — shared with the portability litmus so the relocated-copy
test drives the *full* corpus, not a single kernel."""

import collector
import pytest
from _corpus import CASES, _Cm, k_with_loop_exit


@pytest.mark.parametrize("name,kernel,argsets", CASES, ids=[c[0] for c in CASES])
def test_corpus_behavioral_equivalence(name, kernel, argsets):
    for args in argsets:
        assert collector.instrument(kernel)(*args) == kernel(*args)


def test_with_loop_exit_log_order():
    """the host ``with`` keeps the host context-manager protocol:
    __enter__/__exit__ order identical to native; cleanup fires on normal exit."""
    log = []
    assert collector.instrument(k_with_loop_exit)(_Cm(log), 5) == k_with_loop_exit(_Cm([]), 5)
    assert log == ["enter", "exit"]
