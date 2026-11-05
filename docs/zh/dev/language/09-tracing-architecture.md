# Tracing 架构

执行一个 kernel 如何变成 IR：组件、它们的归属、让 trace 保持类型健全的不变量，以及失败模型。

> **读者：** `python/pypto/language/tracing/` 的贡献者。面向用户的契约见
> [函数与程序](../../user/language/01-functions.md)；值模型见
> [值与暂存](../../user/language/07-staging.md)；一条语句如何变成一条 IR 语句见
> [Tracing Collector](10-tracing-collector.md)。

## 模型

traced kernel 就是普通 Python。收集 IR 是**把函数体运行一次所产生的副作用**：`pl.*` 返回包装 IR 表达式的
带类型 facade，Frame 在源码绑定或使用它们的位置插入。任何值的类型都不取决于“我们是否在 trace” ——
它来自产生它的那个操作。

两条推论决定了整个组件布局：

- **值 IR 是分离的，结构 IR 是有状态的。** `ir.Call` 或常量是封闭表达式：构造它、按引用持有、稍后插入。
  `if`、循环或作用域则是*随时间打开*的：进入、在宿主执行函数体的过程中向其中发射语句、再关闭。
  因此值算子从不自行发射，而打开构造的机制归 Frame 一侧。
- **collector 与语言无关。** 重写、驱动与栈纪律不认识 tile、tensor 或作用域；语言前端提供语义。
  换掉任一侧都不影响另一侧。

## 组件

```text
kernel source ─► rewriter ─► generated function ─► interpreter ─► Frame / Sema ─► IR
   (python)      (type-blind)  (rt.* calls)      (host semantics)  (PTO meaning)
```

| 组件 | 拥有 | 不拥有 |
| ---- | ---- | ------ |
| rewriter (`collector/rewriter.py`) | 语句结构、求值顺序、`rt.*` 路由、`break`/`continue` 的 unwind 守卫 | 值类型、IR、host/staged 判定 |
| interpreter (`collector/interpreter.py`) | 插桩局部变量、各分支的 locals 映射、活动构造栈、分支选择、host 循环、一次代表性 staged 迭代、转移目标 | 语言类型、builder 操作 |
| Sema (`tracing/semantics.py`) | 一次函数 trace 编译：`IRBuilder`、参数、storage 驻留、trace 结果、一个全新的 Frame | 活动构造顺序 |
| Frame | 源码名 → 可变 `ir.Var`、绑定、表达式语句、return、待处理的 `iter_args`、staged 构造的创建 | 任何顺序镜像 —— 它向 interpreter 查询 |
| 构造 (`tracing/pto_constructs.py`) | staged 区域数据：延迟的 `for` 构造、惰性的 `while` staging、yield 跟踪、结果绑定 | 栈顺序与生命周期 |
| execution（`tracing/execution.py`） | 每个生产者共享的唯一运行路径：active frame 的安装与恢复、单次执行的 callee 适配、声明返回值的固定 | 单元生命周期及其守护（session 自己的）、前端策略 |
| entry (`tracing/entry/`、`decorators/`) | 注解与世界、通道、assembly、派发；见 [Tracing 前端边界](08-tracing-frontends.md) | 收集到的 `ir.Function` 以下的一切 |

两套栈并存且刻意互不镜像：interpreter 的构造栈（逻辑宿主顺序）与 builder 的栈（物理 IR 插入）。
一个逻辑构造可以拥有零个、一个或多个物理作用域；interpreter 从不预测这个形状，Frame 也从不重建逻辑栈。

## 不变量

1. **一个事实一个归属者。** Sema = 一次编译，Frame = 一次函数执行，构造 = 一个活动构造，
   interpreter = 宿主顺序。关于活动结构的问题一律问 interpreter，绝不从别处保存的状态重建。
2. **不观察类型。** 构造的行为只取决于它收到的值的*类别*（facade 还是 host 值），不取决于检查 builder
   或执行状态。发射时修改 builder 是正常的；*读取*它来决定类型才是 bug。
3. **Sema 是一次性的。** 成功或失败之后都不复用、也不重置 —— 失败的 trace 没有产出 IR，没有可重置的目标。
4. **代表性执行。** staged 循环的函数体只运行一次。任何会随迭代变化的东西必须 staged，否则被固化
   （于是有了下面的两处守卫）。
5. **产生结果的关闭处于 pending 状态。** 关闭返回 `AwaitingResultBindings`，构造留在栈上直到结果绑定完成它；
   普通操作在修改前拒绝 pending 的结果构造。
6. **失败是原子的。** 逃逸的异常跳过所有收尾 —— interpreter 关闭、Frame finish、staged 构造关闭、
   `builder.end_*` —— 调用方丢弃那个 Sema 并原样传播**原始**异常。

## 失败 trace 模型

trace 无法半成功，因此拒绝必须在 trace 时大声报错，而不是在程序里静默。两处守卫承担这件事，
它们都在函数体运行期间抛出：

| 守卫 | 拒绝什么 | 位置 |
| ---- | -------- | ---- |
| 循环携带的 host 状态 | 在 staged 控制流内被推进的 host 值；那一次迭代的值会被固化进所有迭代 | Frame 绑定 |
| 单侧 host 绑定 | 在 staged `if` 的一个分支里绑定、却在汇合之后读取的 host 名字 | Frame 汇合检查 |

两者都以 `TypeError` 出现，并指明修复方式（见 [值与暂存](../../user/language/07-staging.md)）。
对 *staged* 值，同样的不对称是合法的，并且被委派处理：汇合会把另一条路径的值串过来，因此单侧重绑
降到一次归并而不是报错。

并非所有静默失败都能这样关闭。发出语句的普通辅助函数不是 body，它的 `pl.store` 无处落地且不会报错 ——
那是语言形状的边界（[函数与程序 § 复用代码](../../user/language/01-functions.md)），
不是本架构能在不发明“任意 Python 调用的语句捕获”的前提下补上的守卫。

## 缺陷该归到哪一层

按缺陷所涉及的事实判断，然后看该归属者的测试：

| 症状 | 层 |
| ---- | -- |
| 求值顺序错、名字被改写错、缺少 unwind | rewriter |
| 选错分支、host 循环多跑一次、转移命中错误目标 | interpreter |
| 参数的 extent 或 storage 出错、一次编译泄漏到下一次 | Sema |
| 绑定、携带值、结果归并、构造形状不对 | Frame / 构造 |
| 实参的世界错、被调函数未被并入、该命中的缓存未命中 | entry |

## 另请参阅

- [Tracing Collector](10-tracing-collector.md) —— 插桩、被跟踪绑定与 IR 发射的细节。
- [Tracing 前端边界](08-tracing-frontends.md) —— 通道、assembly，以及迁移等价性度量的是什么。
- `tests/ut/language/tracing/test_semantics.py` —— trace 层的事实：staged `if`/`for`/`while`、守卫、汇合、span。
- `examples/tracing/advanced/09_compiler_view.py` —— 同一个流水线在一个 kernel 上打印出来。
