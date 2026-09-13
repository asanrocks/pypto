# 控制流

循环与条件、它们跨迭代携带的值，以及塑造二者的那条 SSA 规则。

> **前置**：[函数与程序](01-functions.md)。

## Concept

PyPTO 的 IR 是 **SSA**：每个绑定只被写一次。在直线代码里这是不可见的 —— 你重新绑定多少次 `x`，tracer 就替你重命名多少次 —— 但在循环边界或分支汇合处它就显形了，因为编译器没有地方安放第二次写入。

在 tracing 前端下，你从不声明这些数据流。kernel 体被执行一次，其余由**pass** 推导：

- 在循环里，函数体重新绑定的名字（`acc = pl.add(acc, x)`）*就是*携带值。循环之后读它，读到的是最后一次迭代的结果。
- 在 staged `if` / `else` 里，两个分支直接给同一个名字赋值；SSA pass 负责合成汇合（phi 节点）。

解析器时代的声明都已消失：`init_values=`（及随之而来的迭代元组解包）、逐分支的 `pl.yield_`、以及 `pl.while_` / `pl.cond()` 拼写都不复存在。你仍会*读到*它们 —— printed IR 会显示 `init_values=` 和 `pl.yield_` —— 但那是打印器的 round-trip 方言，是 `pl.parse` 消费的独立语法，而不是 tracing 的输入。不要把它抄回 kernel。

本页一切规则背后只有一条：**staged 状态始于算子或 `pl.const`。**宿主字面量（`acc = 0.0`）不是 staged 的 —— 把它喂给算子是一个响亮的报错，而不是静默的零。

四种循环构造共用一套语法，区别只在于告诉了编译器什么：

```text
pl.range     sequential — the default
pl.parallel  iterations are independent and may overlap
pl.unroll    fully unrolled at compile time; bounds must be literals
pl.pipeline  body replicated `stage` times for ping-pong buffering
```

## Quickstart：一个累加的循环

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
torch.manual_seed(0)
A = torch.randn(128, 128, dtype=torch.float32)
```

<!-- doctest: run -->
```python
@pl.jit
def accumulate(
    a: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        t = pl.add(pl.load(a, [0, 0], [128, 128]), pl.load(a, [0, 0], [128, 128]))
        for i in pl.range(3):
            t = pl.add(t, pl.load(a, [0, 0], [128, 128]))   # carried across iterations
        pl.store(pl.mul(t, t), [0, 0], out)
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
accumulate(A, out, config=CFG)
torch.testing.assert_close(out, (5 * A) ** 2, rtol=1e-4, atol=1e-4)
```

| 行 | 作用 |
| -- | ---- |
| `t = pl.add(pl.load(...), pl.load(...))` | 在循环之前建立该值 —— 算子结果，因此是 staged 的 |
| `for i in pl.range(3)` | 三次迭代；`i` 是循环变量 |
| `t = pl.add(t, ...)` | 看起来像修改，实为携带值 |
| `pl.store(pl.mul(t, t), ...)` | 循环之后读 `t` 读到的是最后一次迭代的结果 |

在循环里重新绑定 `t` 是携带值的惯用写法 —— 也是唯一的声明方式。IR 是 SSA 的，循环 pass 把每次迭代的 `t` 作为携带值穿过循环；这套管线你从不需要亲自动手。

## Mechanics

### 四种循环形式

四者都接受 `(stop)`、`(start, stop)` 或 `(start, stop, step)`。除注明外，参数可以是 `int` 字面量或 `pl.Scalar`。

| 形式 | 产生 | 携带值 | 边界 |
| ---- | ---- | ------ | ---- |
| `pl.range(...)` | `ForKind.Sequential` | 由重新绑定推导 | int 或 `Scalar` |
| `pl.parallel(...)` | `ForKind.Parallel` | 由重新绑定推导 | int 或 `Scalar` |
| `pl.unroll(...)` | `ForKind.Unroll` | 仅推导 —— 显式 carry 通道被拒绝 | 仅字面量 |
| `pl.pipeline(..., stage=N)` | Sequential + 软流水 | 由重新绑定推导 | int 或 `Scalar` |

```python
for i in pl.range(10): ...            # 0..9
for i in pl.range(2, 10): ...         # 2..9
for i in pl.range(0, 100, 4): ...     # 0, 4, ..., 96
for i in pl.parallel(0, nblocks): ...
for i in pl.unroll(4): ...            # no explicit carry channel here
```

`pl.parallel` 是一个断言而非请求：你在告诉编译器这些迭代彼此独立。如果它们其实不独立，结果就是竞态。

`pl.pipeline(N, stage=F)` 把循环体在每个外层迭代里复制 `F` 份，使缓冲区可以乒乓。外层循环以 `stage * step` 为步长推进，当行程数不能整除时由一个尾部派发覆盖余数。`stage` 是必填的正整数（通常 2–4）。它在 tile 层由 [LowerPipelineLoops](../../dev/passes/31-lower_pipeline_loops.md) 降级；在 `memory_planner=PTOAS` 下则由 [LowerPipelineToSlots](../../dev/passes/30-lower_pipeline_to_slots.md) 接手，改为让单份循环体轮转同一块分配的多个 slot 而不复制。

```python
for i in pl.pipeline(64, stage=4):
    t = pl.load(a, [i * 64, 0], [64, 64])
    pl.store(t, [i * 64, 0], out)
```

### 携带值如何推导

循环的携带值恰是函数体自引用地重新绑定的那些名字：

```python
acc = pl.load(a, [0, 0], [64, 64])     # staged 种子 —— 算子结果
for i in pl.range(4):
    acc = pl.add(acc, tile_i)          # `acc` 被携带；`tile_i` 不被
```

同一个循环可以重新绑定多个名字；每个自引用的名字都成为携带值。循环体内单分支定义是合法的 —— 缺失的分支读到上一次迭代的值；这就是循环携带语义，由 pass 负责。

种子必须是 staged 的。宿主字面量种子（`acc = 0.0`）不会静默变成携带值：第一个消费它的算子就会拒绝（`pl.add: expected Tensor or Tile operands, got (float, Tensor)`）。用算子结果做种子，或用带类型的常量 —— `pl.const(0.0, pl.FP32)`，它也是计数器和布尔标志的拼写。

### while 循环

staged 的 while 循环就是一个作用在**staged** 谓词上的裸 `while`。`pl.while_` 和 `pl.cond()` 已不存在。用 `pl.const` 初始化计数器，携带值由重新绑定推导，与上面完全相同 —— 函数体追踪一次代表性迭代：

<!-- doctest: run -->
```python
@pl.jit
def chunked_square(
    a: pl.Tensor,
    out: pl.Out[pl.Tensor],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        pos = pl.const(0, pl.INDEX)    # typed staged constant — a host `pos = 0` stays host
        while pos < 2:                 # staged predicate -> one representative body iteration
            tile = pl.load(a, [pos * 64, 0], [64, 128])
            pl.store(pl.mul(tile, tile), [pos * 64, 0], out)
            pos = pos + 1              # the rebind is the carry
    return out


out = torch.zeros(128, 128, dtype=torch.float32)
chunked_square(A, out, config=CFG)
torch.testing.assert_close(out, A * A, rtol=1e-3, atol=1e-3)
```

staged 还是 host 由*值*决定，而不是关键字：如果谓词（以及它读到的名字）是宿主值 —— 纯 Python 整数、张量形状、任何不是算子产出的东西 —— 这个 `while` 在追踪期执行，函数体按宿主迭代次数逐次发射。这是真正的元编程（形状驱动的展开），而不是什么回退路径。

### 条件分支

staged 分支通过**赋值**发布它的值。两个分支给同一个名字赋值；汇合是 pass 的事：

```python
if i == last:
    valid_len = last_valid_len             # staged 值：直接赋值
else:
    valid_len = pl.const(block_size, pl.INDEX)   # 宿主值跨汇合：pl.const
```

staged `if` 会追踪**两个**分支，无论谓词在追踪期的值是什么。宿主 `if`（宿主谓词）在追踪期只走一条路径，与上面的宿主 `while` 相同。

两个报错守护着汇合：一个名字在某分支是 staged 的、在另一分支是宿主字面量，会被拒绝（`... is staged on one branch of this if and a host value on the other`）；谓词本身必须是 BOOL 类型 —— 整数 `Scalar` 上写 `if k:` 会被拒绝，要写 `if k != 0:`。

### 汇编通道：`pl.intrin.carry` / `yield_` / `leak`

人体工学方言从不显式声明携带值。显式通道唯一出现的地方是汇编方言 —— 编译器测试，以及必须把推导会掩盖的数据流写明的 kernel 代码（任务图依赖循环就是面向用户的例子；见[编译期指令](06-syntax.md)）。三个拼写：

```python
(acc,) = pl.intrin.carry(c)        # open the iter args the loop consumes
for i in pl.range(4):
    acc = pl.store(tile_i, [i * 64, 0], acc)
    pl.intrin.yield_(acc)          # publish the branch/iteration value
(result,) = pl.intrin.leak()       # bind the loop's result after it closes
```

规则：carry 必须被紧随其后的循环开环消费；`pl.unroll` 直接拒绝它；循环之后要读的是 leak 绑定的名字（`result`，不是 `acc`）；并且每个携带值在整个循环中必须保持同一种类型。**混级**正是它咬人的地方 —— 一个由张量级表达式播种、随后在 InCore 作用域里与 tile 级值结合的携带值不会通过类型检查，而报错指向的是算子而不是携带值。

`pl.yield_` 仍作为 `pl.intrin.yield_` 的弃用别名存在；它会告警。printed IR 会用 `init_values=` 渲染同一个循环 —— 再说一次，那是打印器的方言，不是输入拼写。

### SSA，以及你何时需要在意它

照常写 Python —— 名字随便重绑。tracer 会重命名：

```python
result = pl.mul(x, 2.0)
result = pl.add(result, 1.0)      # fine; the tracer produces two bindings
```

`@pl.function(strict_ssa=True)` 会改为拒绝重绑，偶尔可用于捕捉无意的变量遮蔽。它是 round-trip 方言的纪律 —— 报错的帮助文本指向 `@pl.function_parser` —— 它同时会禁用 `dst[...] = src` 下标写语法糖，那个糖正是靠重绑实现的，见[编译期指令 § 下标语法糖](06-syntax.md#下标语法糖)。

流水线很早就会跑 [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md)，所以非 SSA 源码是正常输入，不是兼容模式。

## 边界情况

> **致命陷阱（解析器时代，现在很响亮）：**宿主字面量种子曾经会丢掉携带值 —— 循环照常编译，而每次迭代读到的都是初值。tracing 会在第一个消费种子的算子处拒绝它。如果下面的报错在你期望张量的位置点名了 Python 类型，说明你的状态从来就没被 staged。

| 症状 | 可能原因 | 修复 |
| ---- | -------- | ---- |
| **`pl.add: expected Tensor or Tile operands, got (float, Tensor)`** | 宿主字面量播种 staged 状态（`acc = 0.0`） | 用算子结果或 `pl.const(value, dtype)` 做种子 |
| **`range() got an unexpected keyword argument 'init_values'`** | 解析器时代的携带值声明 | 删掉它 —— 重新绑定就是携带 |
| **`... is staged on one branch of this if and a host value on the other`** | 宿主字面量跨 staged 汇合 | 该分支用 `pl.const(...)`，或两个分支都 staged |
| **`a staged if predicate must be BOOL-typed`** | 整数 `Scalar` 上写 `if k:` | 写显式判断（`if k != 0:`） |
| **`a staged for loop cannot have an 'else' suite`** | `else` 会被无条件收集 | 重构，或让谓词保持宿主值 |
| **`pl.unroll() cannot be combined with init_values`** | 显式 `pl.intrin.carry` 喂给展开循环 | 由重新绑定推导携带值，或改用 `pl.range` |
| **`tensor.add requires ... but got TileType`** | 携带值中途换了级别 —— 在 InCore 作用域里 `pl.slice` 产出的是 Tile | 让携带值全程保持同一级别；在作用域内显式使用 `pl.load` / `pl.store` |
| **`unroll()` 拒绝某个边界** | 边界必须是编译期字面量 | `Scalar` 边界请用 `pl.range` |
| **`pipeline()` 报 stage 错误** | `stage` 缺失或不是正整数 | 传 `stage=N`，N ≥ 1 |
| **`pl.parallel` 下出现竞态** | 迭代其实并不独立 | 改用 `pl.range`，或消除跨迭代依赖 |
| **循环体在 IR 里被展开** | 谓词或边界是宿主值，循环在追踪期就执行了 | 这是合法的元编程；想要设备侧控制流就把值 staged（`pl.const`） |
| **`SSAViolationError`** | 在 `strict_ssa=True` 下重绑 | 换用不同名字，或去掉 `strict_ssa` |

## 配套示例

| 示例 | 展示 |
| ---- | ---- |
| `examples/beginner/02_elementwise.py`（`chunked_add`） | 对 tile 分块的朴素 `pl.range` |
| `examples/intermediate/04_matmul_acc.py` | 无循环的累加 —— 宿主排程的 K 分块走 `pl.matmul_acc` |
| `examples/models/03_flash_attention.py` | 推导出的多名字携带 + 嵌套 staged `if` —— 完整形态 |

## See Also

- [类型](00-types.md) —— 携带值到底是什么。
- [作用域与放置](04-scopes.md) —— 包含这些循环的放置作用域。
- [迁移指南](../../../design/04-migration-guide.md) —— 每个解析器时代拼写及其 tracing 替代。
- [ConvertToSSA](../../dev/passes/04-convert_to_ssa.md) —— 本页规则的来源。
- [UnrollLoops](../../dev/passes/02-unroll_loops.md) —— `pl.unroll` 变成什么。
- [LowerPipelineToSlots](../../dev/passes/30-lower_pipeline_to_slots.md) —— `memory_planner=PTOAS` 下 `pl.pipeline` 变成什么。
- [LowerPipelineLoops](../../dev/passes/31-lower_pipeline_loops.md) —— 其余情况下 `pl.pipeline` 变成什么。
