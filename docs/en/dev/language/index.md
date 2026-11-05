# Language

The Python DSL as the compiler sees it. These pages split by what you are doing — reading
the surface the compiler consumes, extending it, or maintaining the frontend that produces
it — and the three routes below do not depend on each other.

| You are | Read | In order |
| ------- | ---- | -------- |
| writing or debugging a kernel | the surface pages below | spec → statements → dependencies → functions |
| building constructs or placements for other authors | [Construct Recipes](06-construct-recipes.md) → [Authoring Recipes](07-authoring-recipes.md) | recipes over the authoring tier |
| changing the tracing frontend | [Frontend Boundaries](08-tracing-frontends.md) → [Architecture](09-tracing-architecture.md) → [Collector](10-tracing-collector.md) | boundaries, then components, then statement mechanics |

## The language surface

| Page | What it covers |
| ---- | -------------- |
| [Python IR Syntax Specification](00-python_syntax.md) | Module structure, type system, expressions — the entry point to the spec |
| [Statements and Control Flow](01-statements.md) | Assignment, if/for/while, scope context managers, yield, compile-time directives, SSA phi nodes |
| [Manual Dependency Primitives](02-manual_dependencies.md) | `pl.manual_scope`, explicit `deps=` edges, dispatch predicates, array-carry fences |
| [Functions and Program Structure](03-functions.md) | Function types, parameter directions, cross-module reuse, printing IR |
| [Integrating Hand-Written C++ Kernels](04-external-kernels.md) | Calling an existing hand-written C++ InCore kernel from a PyPTO program |
| [GM Cache-Access Policy](05-cache-policy.md) | `pl.set_cache_policy` / `pl.load(cache=...)` — the declared GM cache policy, its coherency contract, and how it reaches codegen |

[Register operations](06-register-operations.md): P0 schemas, implicit captures, region
ownership and Builder construction (VF backend emission requires PR3).

## Extending the language

Both pages are protocol references for code that authors *for* other authors: what a
construct may do, which capabilities it may reach, and what the tier guarantees. A recipe
that needs a capability these pages do not describe is a framework change, not a local
extension.

| Page | What it covers |
| ---- | -------------- |
| [Construct Recipes](06-construct-recipes.md) | `@pl.authoring.construct` — authoring new `with` constructs without touching the tracer |
| [Authoring Recipes](07-authoring-recipes.md) | `@pl.authoring.recipe` — the general effect-composition tier: functions, bodies, calls, control |

## Implementing the tracing frontend

Read these when the kernel *source* is right and the IR is not. The user-facing contract is
[Functions and Programs](../../user/language/01-functions.md) and
[Values and Staging](../../user/language/07-staging.md); these pages explain how the
implementation upholds it, and which layer a defect belongs to.

| Page | What it covers |
| ---- | -------------- |
| [Tracing Frontend Boundaries](08-tracing-frontends.md) | Where a traced kernel becomes an `ir.Program`: the lanes, the assembly, what the frontend does not own, and what migration parity evidence measures |
| [Tracing Architecture](09-tracing-architecture.md) | Components and owners, the invariants that keep a trace sound, the failed-trace model, and which layer a defect belongs to |
| [Tracing Collector](10-tracing-collector.md) | Instrumentation, execution, IR emission, failure cleanup, and how to debug a statement that became the wrong statement |

## See Also

- [IR Parser](../ir/07-parser.md) — how this syntax becomes IR.
- [Language Guide](../../user/language/index.md) — the same language from a user's perspective.
