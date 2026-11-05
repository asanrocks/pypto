# 构造配方（Construct Recipes）

`@pl.authoring.construct` —— 无需触碰 tracer 即可编写新的 `with` 与 `for` 构造。

大多数 kernel 只是**消费**构造：写出 `with` 块、读取它的 handle，从不 import `pl.authoring`。
使用一个构造并不需要本页的任何内容 —— 值域一侧见 [函数与程序](../../user/language/01-functions.md)，
`examples/tracing/05_block_recipe.py` 先给出三个 kernel 共用同一个 grid，再给出 recipe 本身。

**构造配方（construct recipe）**声明一个构造如何下降：一个生成器函数，
`yield` 之前的代码打开构造的区域（region），`yield` 把值交给主体（body），
之后的代码负责收尾。其余一切都归 tracer —— 执行、栈纪律、IR 构造。

```python
import pypto.language as pl


@pl.authoring.construct
def blocked_spmd(ctx, shape, tile, *, name_hint="", deps=None):
    blocks = tuple((s + t - 1) // t for s, t in zip(shape, tile))
    core_num = 1
    for count in blocks:
        core_num *= count
    with ctx.region(pl.spmd(core_num, name_hint=name_hint, deps=deps)) as tid:
        tid_var = ctx.resolve_binding(tid) if tid is not None else None
        yield BlockGrid(blocks, tile, tid_var)


with blocked_spmd((tokens, heads), (1, HEAD_TILE)) as block:
    token, head = block.axes
    q = block.region(query).load()
```

这里的 `BlockGrid` 是这个 recipe 自己的 handle —— 一个 `pl.authoring.ConstructHandle` 子类，负责解码本
block 的坐标并提供其区域。同一 recipe 的可运行版本见 `examples/tracing/05_block_recipe.py`
（`row_blocks`，单轴，先给消费者的视角）与 `tests/ut/language/tracing/test_recipe.py`
（`blocked_spmd`，shape/tile 轴，并附 IR 等价断言）。

## 分层

```text
kernel 作者       配方（blocked_spmd）、pl.* 使用者
    │ 组合
构造层            @pl.authoring.construct + ConstructContext   ← 本页
    │ 驱动
tracer           rewriter、interpreter、Sema/Frame
    │
IR builder        封闭的语句/作用域词汇表
```

构造层是 tracer 内部实现唯一被许可的客户端（与其下的授权运行时一起）；
配方代码只能看到
`ConstructContext`。trace 协议（`__trace_enter__` / `__trace_bind__` /
`__trace_exit__`）仍是该层与 tracer 之间的 ABI —— 但配方从不自己实现它：
驱动器替配方实现，因此手写构造的那些私有约定（延迟 carrier 标志、绑定
decline、关闭顺序）不可能写错。

构造层与通用授权层
（[07-authoring-recipes.md](07-authoring-recipes.md)）共享同一个**授权
会话**——每次配方调用所拥有的、作用域栈与函数单元的一次性归属。
`ConstructContext` 是该会话上的 region 视图；`@pl.authoring.recipe` 的上下文是同一
视图外加函数、body 与控制效果，因此通用配方可以像构造配方一样组合构造
region（`ctx.region(pl.spmd(...))`）。

词汇表在必须封闭的地方保持封闭：配方只配置与组合**已有的** IR 构造。需要
新下降形态的配方先是一次框架改动 —— 新的作用域种类、新的语句、新的能力 ——
注册之后配方才能使用。`if`/`else` 与 Python `while` 保持原生；可编写的糖是
`with` 形态与 trace-iterable 的 `for` 形态（普通的 `pl.range` 式循环描述符仍是
框架事务）。

## 两种拼写

一个生成器服务两种语句拼写，区别只在生成器何时启动与 `yield` 值是什么：

| | `with construct(...) as x:` | `for i in construct(...):` |
| --- | --- | --- |
| 生成器启动 | 构造进入（`__trace_enter__`） | 循环目标绑定处 —— 目标源码名字可知的那一刻 |
| `yield` 值 | `as` 目标 —— `ConstructHandle`、捕获 binding 或 `None` | 循环变量 —— `ctx.loop_var()` 返回的 `ir.Var` |
| 主体 | 在 with 内运行 | 作为代表迭代运行 |
| `ctx.binding_name` | `as` 名字（或 `None`） | 循环目标名字 |
| 关闭 | `__trace_exit__` 处的 post-yield 代码 | 同一段 post-yield 代码 |

服务两种拼写的配方以 `ctx.spelling` 分支。`pl.spmd` 是参照：一个
`_spmd_scope` 配方同时下降 `with pl.spmd(...) as tid:` 与
`for i in pl.spmd(...)` —— for 臂铸造循环变量、总是 wrap carrier，并把 carrier
的延迟 prologue（`i = pl.tile.get_block_idx()`）设为其首条语句。

for 拼写带一条硬性契约：配方留下的每个待定（pending）区域都必须由其结构面
暴露 —— 即经 `ctx.open_carrier(..., primary=True)` 打开的 carrier。循环变量的
prologue 与主体的语句都落在该 carrier 内；主区域不是 carrier 的配方不会注册
任何东西，并把主体静默下降到其外，因此驱动器在目标绑定处响亮拒绝。全部
区域立即打开（`defer=False`）的配方没有待定区域，根本不需要结构面 ——
`pl.split_aiv` 就是这个形态。

## 配方契约

- **生成器函数，恰好一个 `yield`。** `yield` 之前的代码在构造进入时运行；
  `yield` 是主体边界；之后的代码在构造退出时运行（仅正常路径 —— 异常丢弃
  整条 trace，不运行任何 staged 收尾）。for 拼写中“进入”是循环目标绑定处 ——
  生成器在那里启动，目标源码名字已知时。
- **`yield` 值是 `as` 目标（`with`）或循环变量（`for`）。** 富句柄应继承
  `pl.authoring.ConstructHandle`：它按 host 绑定（主体把它当普通 Python 对象读取），
  不进入任何 staged 值路径。yield 出句柄的配方要求 `as` 绑定；不带 `as`
  使用的配方必须 yield `None`。for 拼写必须 yield `ctx.loop_var()` 返回的
  `ir.Var` —— 其他任何值都被拒绝（yield 句柄的配方只服务 `with` 拼写）。
- **每个实例单次使用。** `blocked_spmd(...)` 返回一次性构造，与
  `pl.spmd(...)` 相同；工厂本身无状态、可复用。
- **区域按打开的逆序关闭。** `ctx.region(cm)` 在生成器结束时自行关闭；
  显式 `ctx.enter(cm)` 必须在生成器结束前配对 `ctx.exit(cm)`。

## `ConstructContext`

作为第一个参数交给配方的能力对象。它的表面封闭且经过筛选 —— 没有
builder 访问，没有发射辅助。

| 能力 | 用途 |
| ---- | ---- |
| `ctx.region(cm, as_name=..., single_call_body=...)` | 在配方内组合一个构造（`pl.spmd`、`pl.at`、另一个配方）；其 with 体内放着 `yield`。 |
| `ctx.enter(cm)` / `ctx.exit(cm)` | `region` 的显式形式。 |
| `ctx.open_scope(kind, attrs=..., defer=True, primary=False, prologue=..., **begin_kwargs)` | 直接打开一个作用域 —— 下降原语。`kind` 取自框架封闭的作用域词汇表；attr 顺序归配方所有，因为配方就是下降策略。`defer=True`（declarable-scope 契约）时 Frame 在主体第一条发射 IR 的语句处打开它（折叠开头的 `pl.set_cache_policy(...)` 声明），或在 `exit` 处；`defer=False` 立即打开，适用于 attr 在进入时即固定、不接受 cache-policy 声明的构造。`prologue` 语句是该区域自有的首批发射（for 形式的循环变量绑定）—— 立即打开时紧跟 begin 发射，延迟打开时先于任何主体语句。`primary=True` 使该区域成为配方的结构面。 |
| `ctx.open_carrier(kind, attrs=..., flush_on_emit=..., prologue=..., primary=...)` | 打开决策归**主体**所有的延迟作用域。`flush_on_emit=True` 在第一次发射处打开（wrapped 主体）；`False` 等待 block-index 触发 —— 主体自证 inline，dispatch 主体永不打开它 —— `prologue` 语句（把 block index 绑定到循环变量）是 carrier 的首批发射。 |
| `ctx.capture_task_id(name)` | 为 `as tid` 形式铸造 TaskId 捕获：创建 Var、发射占位赋值，返回 `(binding, task_id_var_attr)` —— 配方按其规范位置追加该 attr 并 yield 出 binding。 |
| `ctx.lower_deps(deps, owner)` | 把 `deps=[...]` 列表下降为 `manual_dep_edges` 的 Var —— 追踪侧唯一的 `deps=` 下降实现（`coerce_deps`），与 dispatch 效果共享，作用域与 submit 拼写不会漂移。 |
| `ctx.resolve_binding(tid)` | 把 `as tid` 捕获代理解析为其 `ir.Var` —— 配方在句柄上暴露调度 TaskId 的方式（用于后续 `deps=` 边）。 |
| `ctx.binding_name` | 用户绑定配方所用的名字：`with` 的 `as` 名字（不带 `as` 时为 `None`），`for` 的循环目标名字。 |
| `ctx.spelling` | 本次使用进入时经过的语句拼写：`"with"` 或 `"for"` —— 服务两种拼写的配方以此为分支。 |
| `ctx.loop_var()` | 铸造 for 拼写的循环变量 —— 配方的 `yield` 值。Var 携带循环目标名字与 `INDEX` 类型；Frame 以目标名字注册它，不产生 `LetStmt`。 |
| `ctx.open_loop(start, stop, step, kind=..., attrs=..., var_name=...)` | 直接打开一个 `ForStmt` region —— 循环层的降低原语（`pl.range` 的 `begin` 作为能力）。循环立即打开并在配方的退出处关闭（`with ctx.open_loop(...) as i:` 括起 `yield`）；`kind` 是 `pl.range` 家族的词汇表并携带其转移规则（`range` 接受 `break`/`continue`，`parallel`/`unroll` 拒绝它们）。循环携带通道是 SSA pass 的推理：在打开前用 `ctx.bind` 绑定初值，在 body 中重绑名字，`ConvertToSSA` 创建 `iter_args` / `return_vars`。 |
| `ctx.bind(name, value)` | 在调用者 trace 中以 `name` 绑定一个 staged 值 —— Frame 自身的绑定路径。配方的通道/`as` 名字纪律：被绑定的名字是调用者 trace 名字，不得与配方自身的 `as` 目标或用户的局部变量冲突。 |
| `ctx.single_call_body` | rewriter 的主体形状事实；组合 IR 形状依赖它的构造（`pl.spmd` 的 direct-dispatch 与 InCore-wrapped 形态）时应转发。 |
| `ctx.sticky_dump_vars()` | 函数的前向 sticky dump 标记张量 Var（`dump_vars` attr 合并）。 |
| `ctx.is_inside_incore()` / `ctx.is_inside_scope(kind)` / `ctx.function_type()` | 只读的 trace 事实。 |
| `ctx.enclosing_user_split()` | 最内层打开的 InCore 作用域的用户声明 `pl.split(...)` 模式 —— AIV-split 互斥规则背后的事实（`pl.split_aiv` 读取它）。 |

`as_name` 默认为配方自身的 `as` 名字 —— 被组合构造的 TaskId 捕获在用户
绑定的名字下铸造其 Var。

配方绑定的构造参数在进入之前即可在实例上读取（**构造面**，
constructor surface）：`pl.at` 的 `level` 是 Frame 的 InCore 预提示读取的
pre-enter 事实，用于决定外层延迟 carrier 是吸收还是打开。错误调用在工厂处
失败 —— 与手写构造 `__init__` 抛出的构造面 `TypeError` 一致。

## 当前在该层上编写的内容

所有 stdlib 的 `with` 构造都是配方：`pl.at`、`pl.scope`、
`pl.manual_scope`、`pl.graph`、`pl.cluster`，以及 `pl.spmd`。每个构造的
pre-yield 代码都是其原 `__trace_enter__` 主体按原顺序的等价物
（attr 收集、TaskId 捕获、parser 一致性校验），没有 builder 访问。转换的
验收是与手写形态的 **IR 等价** —— 对一个极简 `core_group` 配方直接断言其与
`pl.at` 等价，并由完整的构造语料（tracer 与 parser 的 lane parity 套件）
保证。

三种形态覆盖全部：

- **叶子作用域**（`pl.at`、`pl.scope`、`pl.manual_scope`、`pl.graph`、
  `pl.cluster`）—— 每个构造一次 `ctx.open_scope`。`pl.at` 延迟（它接受
  cache-policy 声明）；其余四个立即打开（`defer=False`），与原先完全一致，
  因此它们对 cache-policy 的拒绝行为不变。
- **Carrier**（`pl.spmd`）—— `ctx.open_carrier` 打开嵌在 dispatch 作用域内的
  自动 outline InCore 区域，并标记 `primary=True`，因为 carrier —— 而非其外层
  dispatch 作用域 —— 拥有由主体决定的打开时机。普通形式等待 per-block-index
  读取；split 提示、cross-core slot，或带捕获的多语句主体在第一次发射处打开。
- **立即打开的 for 区域**（`pl.split_aiv`）—— `ctx.open_scope(...,
  defer=False, prologue=[(loop_var, lane_idx)])` 打开 SplitAiv 区域本身；
  裸顶层形态另合成一个 InCore wrapper（前向 sticky dump 标记合并在它上面；
  已在 InCore 内 —— 或在 InCore 类型函数体内 —— 的形态原位发射）。没有
  任何延迟，因此该配方不需要结构面。

两个 `for` 拼写都是同一层上的配方。`pl.spmd` 是**两种拼写共用**的一个配方：`with pl.spmd(...)` 与
`for i in pl.spmd(...)`。with 臂保持原手写下降（捕获、wrap 决策、名字拆分决策）；
for 臂不捕获、总是 wrap，并铸造绑定到 `pl.tile.get_block_idx()` 的循环变量
作为 carrier 的延迟 prologue。此次转换从 `constructs/scopes.py` 删除了
for 形式的独立描述符路径（其 `begin`/`__trace_exit__` 分支与 carrier 状态）——
该类现在只是创建配方实例并转发的薄门面。验收是 IR 等价：for 形式与手写
前任逐字节一致（plain、空主体、`deps=`、cache-policy、split/slot 与 cluster
拒绝诸变体），tracer 与 parser 的 lane parity 通过 `spmd_loop_kernel` 参照覆盖它。
`pl.split_aiv` 是立即打开的对应物：一个配方（`_split_aiv_scope`）只为裸顶层
形态合成 InCore wrapper，RFC #1820 的函数级 split 拒绝读取
`ctx.enclosing_user_split()` —— lane parity 通过 `split_aiv_loop_kernel` 参照覆盖它。

`pl.range` / `parallel` / `unroll` / `pipeline` 有意保持手写：它们已共享一个
干净的 `RangeIterator`，并携带 host 执行等价路径（同一源码的 host 迭代）——
一种配方按设计拒绝的语义。该层的 `ForStmt` 原语是 `ctx.open_loop`：配方可以
授权真实的循环 region（`counted` / `accumulate` 验收配方与它们替换的
`pl.range` 形式追踪到相同 IR，含携带通道），但标准库条目本身保持框架拥有，
使 `pl.range(n)` 保持其 host 行为。

第一个外部编写的配方是 `blocked_spmd` —— 基于 `pl.spmd` 的块网格分发：一个代表
性块，其坐标轴投影到张量上，并暴露调度 TaskId 用于 `deps=` 接线。它的验收
测试断言 **IR 等价** —— 配方与它所替代的手写 `pl.spmd` 网格 trace 出结构
完全一致的 IR —— 见 `tests/ut/language/tracing/test_recipe.py`；同处还有一个
`grid_loop` 配方，证明 kernel 作者能自己编写 for 拼写（与
`for i in pl.spmd(n)` IR 等价），且不 import `pypto.language` 以下的任何内容。

## 另请参阅

- [授权配方](07-authoring-recipes.md) —— 之下的通用层：`pypto.language.authoring` 是两个协议的规范导入根。
- [语句与控制流](01-statements.md) —— 配方所组合的作用域上下文管理器。
- [手工依赖原语](02-manual_dependencies.md) —— `deps=` 边与 TaskId 捕获通道。
- `examples/tracing/05_block_recipe.py` —— 一个策略、三个 kernel、以及 recipe 本身；本页在 kernel 作者一侧的另一半。
- `examples/tracing/advanced/06_recipe_policy.py` —— 同一层的下方：一个 body 的三种放置，以及从该层重建出的 `@pl.jit_trace.body`。
