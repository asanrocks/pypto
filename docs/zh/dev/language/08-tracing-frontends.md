# Tracing 前端的边界

一个 traced kernel 在哪里不再是 Python、变成 `ir.Program`；每一道边界上什么会通过；
以及 tracing 前端刻意不拥有的东西。

> **读者：** `python/pypto/language/tracing/` 的贡献者，以及正在排查“IR 与源码对不上”的人。
> 面向用户侧的契约见 [函数与程序](../../user/language/01-functions.md)；值模型见
> [值与暂存](../../user/language/07-staging.md)。

## 五道边界

```text
declaration  ──► trace ──► collected IR ──► program ──► compilation ──► dispatch
   annotations     tracer     ir.Function    assembly     ir.compile      jit cache
```

| 边界 | 通过的是什么 | 归属 |
| ---- | ------------ | ---- |
| 声明 → trace | 注解声明的参数世界；裸 `pl.Tensor` 让 shape 未知，于是 trace 推迟到调用时 | `tracing/entry/` |
| trace → 收集到的 IR | 每个 kernel *变体* 一个 `ir.Function`；host 值已被折叠，staged 值是操作数 | collector |
| 收集到的 IR → program | 入口，以及它的函数体所到达的每个被调函数；去重并取程序内唯一的名字 | `_ProgramAssembly` |
| program → 编译 | 一个 `ir.Program` 加上 `RunConfig.compile_kwargs()`，交给与 parser 前端同一个 `ir.compile` | `ir.compile` |
| program → 派发 | 桥接后的签名：shape、dtype、layout、host 旋钮取值、编译身份 | jit cache |

由此得到两条性质，也是“换前端不必动编译器”的原因：**tracing 前端只拥有 trace 与 assembly**
—— `ir.Program` 之后的一切都是共享的；并且 **host *旋钮* 从不作为值穿过边界** ——
它被折叠进程序，于是这个取值成为 artifact 身份的一部分。

## 四条通道

一个对象（`_JitSpecializer`）服务四条通道，由调用点传入的东西选择。它们就是前端的全部公开表面：

| 通道 | 触发 | 结果 |
| ---- | ---- | ---- |
| dispatch | 以 `torch.Tensor` 实参调用 | 把每个 tensor 桥接为注解形态的 facade，组装、编译、派发、缓存 |
| specialize | 以 facade 或 host 标量调用 | trace 出的 `ir.Function`；不编译 |
| callee | 在另一条 trace 进行中调用 | 一个分离的 `ir.Call`；被调函数加入调用方的 assembly |
| 声明期 | `.fn`，或任何结构性使用 | 由注解决定的那次 trace，物化一次并复用 |

`compile_for_test(*args)` 是做了 pass 流水线、不做代码生成的组装路径；它存在的目的是让结构性断言
（通道等价、配方等价）可以比较程序。注解不足以固定 shape 的 kernel 没有声明期 trace，因此每次调用都会
重新 trace：通道由声明已经能回答的东西决定。

## 组装与身份

assembly 是按程序构建的，不是全局的。它按首次使用顺序追加，按对象身份与变体键去重 —— 变体键是
kernel 身份、包装器命名的入口类别、推导出的注解，以及被固化的 host 取值 —— 因此同一实参形状与常量下
调用两次的辅助函数仍是一个函数，而同名辅助函数的第二种 shape 是第二个变体，名字带 `_2` 后缀。两处拒绝
发生在这里而不是 tracer 里：

- **环。** 某 kernel 的变体仍在 trace 中时又从它自己的函数体可达，它就没有有限的 traced 形态；
  assembly 报出这个环，而不是继续递归。
- **重名。** 变体可能取到的每个名字都在注册前被预留，因此新铸出的变体不会落在程序已占用的名字上。

## 前端不拥有的东西

| 关注点 | 归属 | 为什么留在那里 |
| ------ | ---- | -------------- |
| SSA、循环携带通道、outlining | pass 流水线 | trace 产出的是 SSA 之前的 IR；`ConvertToSSA` 推导携带关系，与 parsed kernel 完全一致 |
| 内存规划、地址、缓冲区 | pass 流水线 | traced kernel 不允许另写一套分配故事 |
| 代码生成、PTOAS、artifact | `ir.compile` 与 backend | 前端的产出就是一个 `ir.Program`，仅此而已 |
| 缓存键、持久化、warmup | [Persistent JIT Cache](../10-jit-cache.md) | dispatch 通道调用它，不重新实现它 |
| ABI、launch、设备选择 | [Entry Points](../08-entry-points.md) | host 旋钮在编译期被消费；剩下的是运行期签名 |

`compile()` 把最后一道边界展示得很具体：它返回一个冻结的物化结果，其调用签名**就是**运行期 ABI，
因此为已被消费的 host 旋钮再传值会得到一个指名该旋钮的实参数量错误，而不是悄悄重编译。

## 迁移与等价性证据

迁移主张之所以可检验，是因为两个前端停在同一个地方：parser 通道的 `.lower(*args)` 与 tracing 通道的
`compile_for_test(*args)` 都返回 pass 之后、代码生成之前的 `ir.Program`。该边界上的相等意味着
“这次移植换的是前端，不是 kernel”。

阅读证据时要看清它度量的是什么：

- **范围内：** 同一个 kernel 在两个前端上降低后的程序结构相等，入口与派生的被调函数名被折叠
  （`examples/tracing/advanced/08_migration.py`）。
- **范围外：** 代码生成产物、运行期数值、artifact 身份。示例 08 也检查了那一个 kernel 的数值，
  但那里的相等不是普遍的数值主张；缓存复用与特化需要各自的证据。
- **等价在哪里结束：** host 域元编程没有 parser 拼写 —— 对 host plan 的 `for` 循环不是 IR ——
  因此这类 kernel 根本无法写在 parser 通道上。这是迁移的理由，而不是它的附加说明。

## 另请参阅

- [函数与程序结构](03-functions.md) —— 面向用户的契约：参数世界、拼写、支持的表面。
- [Tracing Collector](10-tracing-collector.md) —— rewriter 与 interpreter：一条源码语句如何变成一条 IR 语句。
- [Entry Points](../08-entry-points.md) —— 编译后的 artifact 暴露什么。
- [Persistent JIT Cache](../10-jit-cache.md) —— dispatch 通道的缓存、键与持久化。
