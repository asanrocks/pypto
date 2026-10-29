# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Flash attention with loop-level carries, spelled the ergonomic way.

Demonstrates advanced DSL control flow:
  - ``pl.range`` loop-carried state: bind the state before the loop and rebind it
    at the end of the body — the frontend derives the carried channels, so there
    is no ``init_values=`` tuple and no trailing ``pl.yield_``
  - Nested ``if``/``else`` whose value is the branch's rebinding (the SSA phi)
  - Online softmax with running max/sum accumulators

Algorithm:
  For each KV block j in [0, 16):
    sij = Q @ Kj^T (scaled)
    pij = softmax(sij) with online max/sum correction
    Oi = update(Oi, pij @ Vj)
  Return Oi / li (final normalization)

Run:  python examples/models/03_flash_attention.py
Next: examples/models/04_paged_attention.py
"""

import pypto.language as pl


@pl.jit_trace
def flash_attention(q_13: pl.Tensor, k_16: pl.Tensor, v_19: pl.Tensor, out: pl.Out[pl.Tensor]):
    with pl.at(level=pl.Level.CORE_GROUP):
        mi_update = pl.create_tensor([64, 1], dtype=pl.FP32)
        li_update = pl.create_tensor([64, 1], dtype=pl.FP32)
        attn_update = pl.create_tensor([64, 128], dtype=pl.FP32)
        oi_update = pl.create_tensor([64, 128], dtype=pl.FP32)

        for i in pl.range(16):
            # Inner statement.block
            kj = pl.slice(k_16, [64, 128], [i * 64, 0])
            vj = pl.slice(v_19, [64, 128], [i * 64, 0])
            sij = pl.matmul(q_13, kj, out_dtype=pl.FP16, a_trans=False, b_trans=True, c_matrix_nz=False)
            sij_1 = pl.mul(sij, 0.0883883)
            row_max = pl.row_max(sij_1)
            sub = pl.sub(sij_1, row_max)
            p_ij = pl.exp(sub)
            l_ij = pl.row_sum(p_ij)
            tildaPij_83 = pl.cast(p_ij, target_type=pl.FP16, mode="round")

            # Nested if: each branch rebinds ``attn_next`` (the SSA phi node)
            if i == 0:
                # Inner statement.block
                oiUpdate_87 = pl.matmul(tildaPij_83, vj, out_dtype=pl.FP16)
                oiUpdate_90 = pl.assemble(oi_update, oiUpdate_87, offset=[0, 0])

                # Nested if inside first branch
                if i == 15:
                    attn_next = pl.div(oiUpdate_90, l_ij)
                else:
                    attn_next = attn_update

                # More statements in first branch
                li_update = pl.assemble(li_update, l_ij, offset=[0, 0])
                mi_update = pl.assemble(mi_update, row_max, offset=[0, 0])
                attn_update = attn_next
                oi_update = oiUpdate_90
            else:
                # Else branch
                mi_102 = pl.create_tensor(shape=[64, 1], dtype=pl.FP32)
                miUpdate_103 = pl.maximum(mi_102, row_max)
                t1_104 = pl.sub(mi_102, miUpdate_103)
                t2_105 = pl.exp(t1_104)
                t3_106 = pl.sub(row_max, miUpdate_103)
                t4_107 = pl.exp(t3_106)
                t5_108 = pl.mul(t4_107, l_ij)
                t6_109 = pl.mul(t2_105, li_update)
                liUpdate_110 = pl.add(t6_109, t5_108)
                liUpdate_113 = pl.assemble(li_update, liUpdate_110, offset=[0, 0])
                q3_114 = pl.mul(oi_update, t2_105)
                q1_115 = pl.matmul(
                    tildaPij_83, vj, out_dtype=pl.FP16, a_trans=False, b_trans=False, c_matrix_nz=False
                )
                q2_116 = pl.mul(q1_115, t4_107)
                oiUpdate_117 = pl.add(q3_114, q2_116)
                oiUpdate_120 = pl.assemble(oi_update, oiUpdate_117, offset=[0, 0])

                # Nested if in else branch
                if i == 15:
                    attn_next = pl.div(oiUpdate_120, liUpdate_113)
                else:
                    attn_next = attn_update

                mi_update = miUpdate_103
                li_update = liUpdate_113
                attn_update = attn_next
                oi_update = oiUpdate_120
        out = pl.assemble(out, attn_update, offset=[0, 0])
    return out


if __name__ == "__main__":
    # The body traces, but the pass pipeline rejects the online-softmax working
    # set: its Vec usage (197632 B) exceeds the 188416 B limit on a2a3 (65536 B
    # of which is the cross-core pipe ring). ``__main__`` therefore exercises the
    # trace and reports the skip instead of executing.
    import torch

    q = torch.randn(64, 128, dtype=torch.float16)
    k = torch.randn(1024, 128, dtype=torch.float16)
    v = torch.randn(1024, 128, dtype=torch.float16)
    out = torch.zeros(64, 128, dtype=torch.float32)
    try:
        flash_attention.lower(q, k, v, out)
    except ValueError as e:
        print(f"SKIPPED: {e}".split(". ")[0])
    else:
        print("OK")
