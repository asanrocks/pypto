# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
Tracing examples — one idea per file, each needed before the next appears.

The main ladder is a single story, and it is the kernel author's story: every
step starts from a kernel that works and ends at the point where the next
abstraction is the obvious thing to reach for.

  01_two_domains.py     — a host value bakes, a staged value stays: why anything else follows
  02_control_flow.py    — the same rule applied to a loop: range() expands, pl.range() records
  03_host_objects.py    — Python objects organize staged computation, leave no trace, and
                          expand into a kernel's parameter list
  04_body_placement.py  — statements need a boundary: @pl.jit_trace.body, called like a function
  05_block_recipe.py    — a tiling pattern stated once, used by every kernel that tiles

The first three teach what a traced kernel *is*; the last two teach how to stop
repeating yourself. Nothing in the main ladder needs `pl.authoring`: a kernel
author uses a construct someone else wrote, and declares reusable statements
with `@pl.jit_trace.body`.

``advanced/`` is the framework-author track — read it when you are writing
constructs for other people rather than kernels:

  advanced/06_recipe_policy.py — one body, three placements; what .inline()/.incore()
                                 are made of; and `@pl.jit_trace.body` rebuilt from the
                                 public authoring surface
  advanced/07_diagnostics.py   — the two ways a kernel goes wrong, and which one raises

Every example runs on the simulator (no device) and asserts what it claims: the
numerics of each kernel and the structural property its concept predicts.
"""
