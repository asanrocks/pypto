# Tracing Collector

一条源码语句从改写插桩到变成 IR 语句的全过程 —— 以及它变成错误语句时该去哪里看。

> **读者：** `python/pypto/language/tracing/collector/` 的贡献者。组件图与不变量见
> [Tracing 架构](09-tracing-architecture.md)；本页讲机制。

## 四层

```text
rewriter.py    kernel source ─► instrumented source (rt.* calls on the interpreter)
driver.py      prepare once ─► compile ─► restore closure ─► inject a fresh interpreter
interpreter.py run the instrumented function; own locals, branches, the construct stack
interface.py   the protocol to a language frontend: Sema, Frame, typed lifecycle results
```

`emit_kernel(source, name, freevars, future_annotations, first_line)` 是 rewriter 的入口，
也是示例可以调用以展示插桩的唯一函数（`examples/tracing/advanced/09_compiler_view.py`）。
rewriter 工作在 AST 而非文本上，因此字符串内容不会被误改；`ast.parse` 丢弃的注释从 token 流恢复，
并作为显式元数据事件挂到对应语句上。

## 改写发射什么

| 插桩调用 | 含义 |
| -------- | ---- |
| `rt.enter_frame(...)` | trace 开始；kernel 里的名字变成插桩局部变量 |
| `rt.vars.x` | 一个被跟踪的名字 —— staged facade 或 host 值；由 interpreter 判定 |
| `rt.bind("x", value)` | 一次绑定，经由 Frame 路由（赋值、海象、解包） |
| `rt.on_begin_with(...)` / `rt.begin_for(...)` | `with` / `for` 站点变成 interpreter 驱动的构造 |
| `rt._emit(...)` | 一条发射 IR 的语句 |
| `rt.return_(...)` | 函数的返回路径 |
| `rt.unwind_guard()` | 包裹函数体内含有指向外部的 `break` / `continue` 的构造 |

改写是类型无关的：它保留原生求值顺序、短路、分组与异常行为，也从不判定一个构造是 host 还是 staged。
嵌套函数、类、lambda 与推导式保留自己的绑定作用域；对 kernel 局部变量的读取仍然重定向到 `rt.vars`。

## 执行

每次调用时，driver 把一个新的 interpreter 注入函数 globals 的副本并运行插桩后的函数。interpreter 拥有：

- **插桩局部变量**与各分支的 locals 映射（一个 `if` 分支可能绑定另一个分支没有的名字 ——
  对 staged 值是归并，对 host 值就是
  [Tracing 架构](09-tracing-architecture.md) 里的那个错误）；
- **一个活动构造栈** —— 活动结构的唯一逻辑表示；
- **宿主语义** —— 哪个分支运行、host 循环运行几次、`break` / `continue` 命中哪个构造；
- **代表性 staged 迭代** —— 函数体只运行一次；循环本身是 IR。

产生结果的关闭把它的构造留在栈上，成为 *pending 结果构造*。普通操作拒绝它，只有结果绑定可以寻址它；
报告完成的那次绑定把它移除。

## IR 从哪来

两条路径，混淆它们是本地设计中最常见的错误：

| 路径 | 例子 | 机制 |
| ---- | ---- | ---- |
| 值 | `pl.mul(tile, k)` | 构造一个分离的 `ir.Call` 并包进 facade；Frame 在名字被绑定或该表达式成为语句时插入它 |
| 结构 | `with pl.at(...)`、`if staged:`、`for i in pl.range(n)` | 构造修改 builder：打开区域、在函数体运行时发射其中的语句、再关闭 |

标准库的 trace protocol 钩子（`__trace_enter__`、`__trace_bind__`、`__trace_iter__`、`__trace_exit__`、
`__trace_stmt__`）收到的是一个显式的 `TraceContext`，绝不是 builder、Sema 或 Frame。
那个对象正是 authoring 层所建立的接缝（[构造配方](06-construct-recipes.md)、
[授权配方](07-authoring-recipes.md)）。

## 失败路径

逃逸的异常会跳过**所有**常规收尾：interpreter 关闭、Frame finish、staged 构造关闭，以及 builder 的
`end_*` 调用。调用方丢弃那个一次性 Sema 并重新抛出原始异常，因此没有半成品函数可检查，也没有东西需要重置。
这正是 trace 内部的拒绝必须在 trace 时抛出的原因：另一种结果是程序编译通过并算出别的东西。

成功路径上还有两项 prompt 级清理义务：pending 结果构造必须由其绑定完成；pending `iter_args` 组必须由紧随
其后的循环打开者消费（Frame 强制这种相邻关系，因此没有为它单独设 interface tag）。

## 调试一条错误语句

1. **打印插桩后的源码**（`emit_kernel`，或 `09_compiler_view.py`），对着 kernel 读。没有走 `rt.*` 的语句
   不会被收集。
2. **打印收集到的 IR**（`kernel.fn`），确认语句在 pass 之前存在。之前在、之后没有，是 pass 的问题，
   不是 collector 的。
3. **定位归属者：** 求值顺序与路由属于 rewriter；分支选择、host 迭代与转移目标属于 interpreter；
   绑定、携带值与构造形状属于 Frame 与构造。
4. **先检查守卫一侧：** 当症状是“值随迭代次数而变化”，那是代表性迭代模型，而不是代码生成 bug。

## 另请参阅

- [Tracing 架构](09-tracing-architecture.md) —— 组件、不变量、失败模型。
- [Tracing 前端边界](08-tracing-frontends.md) —— 收集到的函数之后会发生什么。
- `tests/ut/language/tracing/collector/` —— rewriter、interpreter、driver 与可移植性测试。
- `examples/tracing/advanced/09_compiler_view.py` —— 一个 kernel 的源码、插桩源码、收集到的 IR 与降低后的程序。
