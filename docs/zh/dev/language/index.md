# 语言

编译器视角下的 Python DSL。这些页面按**你正在做什么**划分 —— 读编译器消费的那层表面、扩展它、
或维护产出它的前端 —— 下面三条路线彼此不依赖。

| 你在做什么 | 读 | 顺序 |
| ---------- | -- | ---- |
| 编写或调试 kernel | 下面的表面各页 | 规范 → 语句 → 依赖 → 函数 |
| 为其他人编写构造或放置策略 | [构造配方](06-construct-recipes.md) → [授权配方](07-authoring-recipes.md) | 授权层之上的配方 |
| 修改 tracing 前端 | [前端边界](08-tracing-frontends.md) → [架构](09-tracing-architecture.md) → [Collector](10-tracing-collector.md) | 先边界，再组件，最后语句机制 |

## 语言表面

| 页面 | 内容 |
| ---- | ---- |
| [Python IR 语法规范](00-python_syntax.md) | 模块结构、类型系统、表达式 —— 规范的入口页 |
| [语句与控制流](01-statements.md) | 赋值、if/for/while、作用域上下文管理器、yield、编译期指令、SSA phi 节点 |
| [手工依赖原语](02-manual_dependencies.md) | `pl.manual_scope`、显式 `deps=` 边、调度谓词、array-carry fence |
| [函数与程序结构](03-functions.md) | 函数类型、参数方向、跨模块复用、打印 IR |
| [集成手写 C++ Kernel](04-external-kernels.md) | 从 PyPTO 程序中调用已有的手写 C++ InCore kernel |
| [GM 缓存访问策略与 L2 Hint](05-cache-policy.md) | 作用域声明与单次搬运的 `cache=` 控制：统一使用 `CachePolicy` 选择 A2/A3 运行时别名或 A5 模式 |

[寄存器操作](06-register-operations.md)：P0 schema、隐式捕获、区域归属和 Builder 构造；
VF 后端发射需要 PR3。

## 扩展语言

两页都是**为他人编写**的代码所用的协议参考：一个构造可以做什么、可以触及哪些能力，以及授权层保证
了什么。若某个配方需要这两页之外的能力，那是一次框架改动，而不是局部扩展。

| 页面 | 内容 |
| ---- | ---- |
| [构造配方](06-construct-recipes.md) | `@pl.authoring.construct` —— 无需触碰 tracer 即可编写新的 `with` 构造 |
| [授权配方](07-authoring-recipes.md) | `@pl.authoring.recipe` —— 通用效果组合层：函数、body、调用、控制 |

## 实现 tracing 前端

当 kernel **源码**是对的、IR 不对时读这几页。面向用户的契约在
[函数与程序](../../user/language/01-functions.md) 与 [值与暂存](../../user/language/07-staging.md)；
这几页解释实现如何守住该契约，以及一个缺陷该归到哪一层。

| 页面 | 内容 |
| ---- | ---- |
| [Tracing 前端边界](08-tracing-frontends.md) | traced kernel 在哪里变成 `ir.Program`：四条通道、assembly、前端不拥有的东西，以及迁移等价性证据度量的是什么 |
| [Tracing 架构](09-tracing-architecture.md) | 组件与归属、让 trace 保持健全的不变量、失败 trace 模型，以及缺陷该归到哪一层 |
| [Tracing Collector](10-tracing-collector.md) | 插桩、执行、IR 发射、失败清理，以及如何调试“变成了错误语句”的语句 |

## 另请参阅

- [IR Parser](../ir/07-parser.md) —— 该语法如何变成 IR。
- [语言指南](../../user/language/index.md) —— 从用户视角看同一门语言。
