# 元编程一个 Kernel

一个 kernel，五步：从一个值的两种世界，经过 host 循环与对象，到一个三个 kernel 共用的分块策略。
每一步都可运行，每一步都正好是“下一个抽象变得显而易见”的位置。

> **前置：** [值与暂存](../language/07-staging.md) 与
> [函数与程序 § 复用代码](../language/01-functions.md)。
> **配套文件：** `examples/tracing/01_two_domains.py` … `05_block_recipe.py` —— 每一步就是那个示例，
> 削到只剩它的想法。

## 你要构建什么

一个按行缩放的 kernel，它在变得**更易维护**的同时并没有变得更聪明：算术从未改变，每一步都换来一个
模型规模 kernel 需要的性质 —— 一个设备可变的旋钮、一个编译器可展开的计划、一个源码只声明一次的形态。

这一切能在一门语言里完成，原因只有一个：`@pl.jit_trace` kernel 在收集 IR 的同时被**执行**，所以普通
Python —— 遍历 host 数据的循环、对象、函数 —— 在构建期运行，并在程序里不留痕迹。决定一个构造是被执行
还是被记录的，是它**主体的域**，而不是它的语法。

<!-- doctest: setup -->
```python
import pypto.language as pl
import torch
from pypto import ir
from pypto.runtime import RunConfig

CFG = RunConfig(platform="__PLATFORM__")
ROWS = COLS = 64
a = torch.randn(ROWS, COLS, dtype=torch.float32)


def program_text(kernel, *args) -> str:
    """The lowered program a call would ship, without compiling for a device."""
    return ir.python_print(kernel.compile_for_test(*args))


def loads(kernel, *args) -> int:
    """Load ops in the program: the evidence that a statement landed."""
    return sum(1 for line in program_text(kernel, *args).splitlines() if ".load" in line)
```

后面每一块都不打印：它断言本步声称的性质，所以“本页说得对”和“代码能跑”是同一项检查。每一步也会点名
它的配套示例，那些示例把同一批主张推得更远。

## 第 1 步：一个值，两个世界

从最小的区分开始 —— 缩放因子：

<!-- doctest: run -->
```python
@pl.jit_trace
def baked(a: pl.Tensor[[64, 64], pl.FP32], k: float, out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        pl.store(pl.mul(pl.load(a, [0, 0], [64, 64]), k), [0, 0], out)
    return out


out = torch.zeros_like(a)
baked(a, 2.0, out, config=CFG)
assert torch.allclose(out, a * 2.0)
```

`k: float` 是 **host** 值：在 trace 运行时读取，被折叠进程序。改它就得到另一个程序 —— 编译出的
artifact 以该取值为键。改成声明 `s: pl.Scalar[pl.FP32]`，同一份源码就产出一个服务所有取值的程序，
因为 `s` 是 **staged** 的运行期参数。

这就是全部想法；本页剩下的内容把它花在控制流、对象与复用上。判断的方法是：*设备必须能选择它吗？*
→ stage 它。*编译器必须看见它吗？* → 留作 host。

配套：`examples/tracing/01_two_domains.py`。

## 第 2 步：同一条规则用于控制流

这个区分同样适用于语句，判据是主体。两个 kernel，只差一处 —— `range` 与 `pl.range`：

<!-- doctest: run -->
```python
@pl.jit_trace
def unrolled(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(4):                      # host subject: the body runs four times
            row0 = chunk * 16
            pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def recorded(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):                   # staged subject: one loop, traced once
            row0 = chunk * 16
            pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), 2.0), [row0, 0], out)
    return out


out = torch.zeros_like(a)
recorded(a, out, config=CFG)
assert torch.allclose(out, a * 2.0)

empty = torch.zeros_like(a)
assert loads(unrolled, a, empty) == 4          # one copy of the body per source iteration
assert loads(recorded, a, empty) == 1          # one body, repeated by the loop
assert "for " not in program_text(unrolled, a, empty)
assert "for " in program_text(recorded, a, empty)
```

两者算出相同结果，但交付的程序不同。host 循环持有函数体的四份拷贝，所以**一个分块决策会改变程序的
规模** —— tile 减半，粘贴式 kernel 的规模翻倍。staged 循环只有一份函数体，从循环里读取边界，这正是
同一个决策在记录式里不付出代码量代价的原因。

`if` 遵循同一规则：host 谓词在 trace 时只运行一条分支，不在程序里留下分支；staged 谓词记录一个分支，
两条分支各被 trace 一次，汇合处把各自的值串起来。让猜错变响亮的守护 —— 循环试图推进 host 状态、
host 名字只在一条分支里绑定 —— 见[值与暂存](../language/07-staging.md)。

配套：`examples/tracing/02_control_flow.py`。

## 第 3 步：让 Python 组织 staged 工作

因为函数体会运行，“常量与参数太多”就可以用普通 Python 的答案解决。一个 frozen dataclass 承载分块
计划，一个调用点把它展开成 kernel 的扁平参数表：

<!-- doctest: run -->
```python
from dataclasses import dataclass, fields


@dataclass(frozen=True)
class Plan:
    """Host knobs for one tiling decision."""

    tile_rows: int = 16
    k: float = 2.0


def expand(plan: Plan) -> tuple[object, ...]:
    """The expansion idiom: the fields, in declaration order, are the parameters."""
    return tuple(getattr(plan, field.name) for field in fields(plan))


@pl.jit_trace
def scaled(
    a: pl.Tensor[[64, 64], pl.FP32],
    tile_rows: int,                                # host: the tiling is part of the program
    k: float,                                      # host: so is the factor
    out: pl.Out[pl.Tensor[[64, 64], pl.FP32]],
):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in range(64 // tile_rows):
            row0 = chunk * tile_rows
            pl.store(pl.mul(pl.load(a, [row0, 0], [tile_rows, 64]), k), [row0, 0], out)
    return out


out = torch.zeros_like(a)
scaled(a, *expand(Plan(tile_rows=16, k=2.0)), out=out, config=CFG)
assert torch.allclose(out, a * 2.0)

empty = torch.zeros_like(a)
narrow = program_text(scaled, a, *expand(Plan(tile_rows=16, k=2.0)), empty)
wide = program_text(scaled, a, *expand(Plan(tile_rows=32, k=3.0)), empty)
assert narrow != wide                  # the host fields bake: two plans, two programs
assert "Plan" not in narrow            # ... and the bundle is never a parameter
```

三种形态覆盖这块地，这里是第三种：

```python
SHAPE.chunks                        # configuration: a dataclass of host data
Block(pl.load(a, ...), row0)        # a record carrying a staged tile
entry(a, *expand(plan), out=out)    # a bundle expanded into the parameter list
```

它们都不会进入程序：对象属于 Python，它*携带*的 staged 值才是 IR。记录的方法是用来构造操作数的写法 ——
`Block(...).scaled(k)` 构造一个 `pl.mul`，自己不留痕迹。对象**作为**实参传入会被拒绝，所以 bundle
要先展开。若某个字段应当留在签名里而不是被烘焙，就把它声明成 `pl.Scalar[...]`，展开方式完全一样。

配套：`examples/tracing/03_host_objects.py`（三种形态，以及对象写法与展开写法之间的程序相等）。

## 第 4 步：语句需要边界

值辅助函数是免费的 —— 返回 `pl.mul(...)` 的函数就是表达式，会消失。语句不是：调用 `pl.store` 的普通
函数不是 body，它的语句无处落地，kernel 会编译、会运行，却什么也不写，而且不会报错。

<!-- doctest: run -->
```python
@pl.jit_trace.body
def scale_block(a, out, row0, k):     # a declared fragment: expanded at the call site
    pl.store(pl.mul(pl.load(a, [row0, 0], [16, 64]), k), [row0, 0], out)


@pl.jit_trace
def entry(a: pl.Tensor[[64, 64], pl.FP32], k: pl.Scalar[pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with pl.at(level=pl.Level.CORE_GROUP):
        for chunk in pl.range(4):
            scale_block(a, out, chunk * 16, k)   # expanded here: no callee, no call
    return out


out = torch.zeros_like(a)
entry(a, 2.0, out, config=CFG)
assert torch.allclose(out, a * 2.0)
```

声明这个边界就是全部修复。body 只预备一次，在调用处展开，背后不留函数或调用 —— 它拥有的是语句边界，
而不是签名。如果片段需要自己的放置（做 tile 工作且必须被抹进调用方），那是 `.inline` 被调函数；
如果它必须作为设备函数保留，那是 `.incore`。

配套：`examples/tracing/04_body_placement.py`（body 与粘贴语句、逐字节相同的程序）与
`examples/tracing/advanced/07_diagnostics.py`（那个静默陷阱）。

## 第 5 步：把策略声明一次

现在两个 kernel 重复同一套仲裁故事 —— 把行分成块、每个 core-group 派发一块、从 block index 解出
本块的行。这种重复不是风格问题：它是*策略*，归拥有分块方式的人所有。下面把它声明一次，作为一个构造，
由两个 kernel 消费，外加一个事后才写出的第三个：

<!-- doctest: run -->
```python
class RowBlock(pl.authoring.ConstructHandle):
    """The `as` target: this block's row offset and its tile capacity."""

    def __init__(self, tile_rows: int) -> None:
        self._tile_rows = tile_rows

    @property
    def row_offset(self):
        """Staged offset of this block's first row; each read re-reads the block index."""
        return pl.get_block_idx() * self._tile_rows

    @property
    def tile_rows(self) -> int:
        return self._tile_rows


@pl.authoring.construct
def row_blocks(ctx, rows: int, tile_rows: int, *, name_hint: str = ""):
    """One block of `tile_rows` rows per core-group: host checks, then a body boundary."""
    if rows % tile_rows:
        raise ValueError(f"row_blocks: {rows} rows do not divide into {tile_rows}-row blocks")
    with ctx.region(
        pl.spmd(rows // tile_rows, name_hint=name_hint),
        single_call_body=ctx.single_call_body,
    ):
        yield RowBlock(tile_rows)


@pl.jit_trace
def scale_blocks(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 16, name_hint="scale") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.mul(tile, 2.0), [row0, 0], out)
    return out


@pl.jit_trace
def relu_blocks(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 16, name_hint="relu") as grid:
        row0 = grid.row_offset
        tile = pl.load(a, [row0, 0], [grid.tile_rows, COLS])
        pl.store(pl.relu(tile), [row0, 0], out)
    return out


out = torch.zeros_like(a)
scale_blocks(a, out, config=CFG)
assert torch.allclose(out, a * 2.0)

out = torch.zeros_like(a)                      # a third consumer, written after the policy
relu_blocks(a, out, config=CFG)
assert torch.allclose(out, torch.relu(a))


@pl.jit_trace
def bad_grid(a: pl.Tensor[[64, 64], pl.FP32], out: pl.Out[pl.Tensor[[64, 64], pl.FP32]]):
    with row_blocks(ROWS, 20) as grid:         # 20 does not divide 64
        row0 = grid.row_offset
        pl.store(pl.load(a, [row0, 0], [grid.tile_rows, COLS]), [row0, 0], out)
    return out


try:
    bad_grid.compile_for_test(a, torch.zeros_like(a))
except ValueError as exc:
    assert "do not divide" in str(exc)         # the policy's own check, at the call site
else:  # pragma: no cover - the refusal is the point
    raise AssertionError("a grid that does not divide the rows must be refused")
```

消费者只声明一次 grid，再也不提 block index。其背后的 recipe 是 `pl.authoring` 上的一个生成器：
`yield` 之前的代码打开区域，`yield` 把普通 Python handle 交给主体，之后的代码负责收尾。它不增加
自己的降低逻辑 —— 它构建的程序与手写 grid 构建的逐字节相同，而这正是“编译器没有魔法”这一主张可检验
的唯一理由。

回报在于改进落在哪里：落在 recipe 里，一次，惠及所有 kernel。策略自带的校验也保护了在它之后才写出的
kernel —— 这就是 `bad_grid` 展示的东西，也是为什么一次策略改进会同时惠及所有消费者，而不是逐个修复。

配套：`examples/tracing/05_block_recipe.py`（手写 grid、消费者、recipe，以及 IR 相等），其生产对应物是
`pypto-lib/models/deepseek_v4_flash_mtp/blocked_spmd.py` —— 同一形态，作用于 shape/tile 轴并携带
派发依赖，承载 DeepSeek 的分块网格。

## 这一页到哪里结束，接下来读什么

| 你想要 | 去读 |
| ------ | ---- |
| 每一步背后的规则与守卫 | [值与暂存](../language/07-staging.md) |
| 四种复用形式及其陷阱 | [函数与程序](../language/01-functions.md) |
| 为别人编写构造 | [作用域 § 编写你自己的构造](../language/04-scopes.md)，然后 [构造配方](../../dev/language/06-construct-recipes.md) |
| 把已有 kernel 迁到这个前端 | [函数 § 迁移 kernel](../language/01-functions.md) 与 `examples/tracing/advanced/08_migration.py` |
| 在一个小 kernel 上看穿整条流水线 | `examples/tracing/advanced/09_compiler_view.py` |

`examples/tracing/` 是本页的可执行形式；进阶部分进入框架作者领域
（`examples/tracing/advanced/06_recipe_policy.py` 用公开的 authoring 表面重建了 kernel 侧装饰器）。
