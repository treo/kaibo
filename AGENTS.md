# AGENTS.md

Working instructions for anyone (human or agent) changing this repository.
Read it fully before your first edit; re-read the pitfalls before touching
reflection, flows, or the server.

## What this is

A Kotlin port of the `xaibo` agent framework (`../xaibo`, Python). It kept
the framework's core ideas — protocol-based dependency injection, transparent
module proxies with an event stream, agents defined declaratively (YAML files
**and** a Kotlin DSL over one shared config model) — and dropped the Python
periphery behind SDKs it doesn't need (see *Out of scope*).

## Commandments

1. **Code is a liability.** More code = more maintenance = more places for
   bugs to hide. Use as little as possible while staying maintainable.
2. **Abstractions must pay rent.** A single-use abstraction is pure waste
   unless it makes maintenance visibly easier. Prefer one more case statement
   over one more interface if nobody needs the second case.
3. **Tests are liberation, not straightjackets.** Test contracts and
   behaviour (a wired agent answers; events fire; a timeout releases the
   agent). Never test reflection internals or exact JSON bodies the model
   sees — that pins implementation.
4. **Budget: keep `scc .` estimated cost under $500,000.** Current: ~$100K (this file bills too — keep docs lean).
   Measure before *and* after big additions (`rm -rf build .gradle` first —
   artifacts inflate it). The Python original was $1.16M; the point of this
   port is not to re-grow it.
5. **Be honest in names.** `NumpyVectorIndex` was renamed `JsonVectorIndex`
   for a reason: if the name promises a library we don't use, fix the name,
   not the lie. Class docs must state the real limits (e.g. linear-scan
   search, non-suspend tool functions only).

## Build & verify

- JDK 21 toolchain at `~/opt/jdk21`, wired through
  `~/.gradle/gradle.properties` (`org.gradle.java.installations.paths`).
  Gradle 9.7 / Kotlin 2.2 come from mise. `gradle test` must run offline:
  no API keys required, the 3 `RealLLMTest` cases self-skip via assumptions.
- Full verification ladder before declaring work done:
  1. `gradle clean test` — all suites green, 0 failures
  2. `gradle run --args="--agents agents --port 8000"` then
     `curl -X POST localhost:8000/v1/chat/completions -d '{"model":"demo",
     "messages":[{"role":"user","content":"hi"}]}'}` → `"6 times 7 is 42."`
     (proves YAML loading + tool loop + server still work end to end)
  3. With a real key: `XAIBO_TEST_API_KEY=<alibaba key from
     ~/.pi/agent/models.json, providers.alibaba.apiKey> gradle test
     --tests xaibo.RealLLMTest --rerun-tasks`
  4. `scc .` cost check
- **A `gradle test --tests X` invocation that fails to compile runs STALE
  classes and looks deceptively consistent.** Always confirm
  `compileTestKotlin` succeeded before believing a filtered test result.

## Architecture map

```
src/main/kotlin/xaibo/
  Models.kt          plain data classes; EventType/Event; Response events
  Protocols.kt       the interfaces modules are wired by (LLM, Response, …)
  Config.kt          AgentConfig + populateImplicits (auto-wiring) + YAML (kaml)
  AgentDsl.kt        agentConfig { } DSL — produces the SAME config objects
  Exchange.kt        instantiation, injection, EventProxy (JDK proxy)
  Registry.kt        Registry, Xaibo facade, Agent
  Json.kt            JSON<->Kotlin bridging + internal accessors
  primitives/        LLMs, orchestrators, tools, memory, mock
  server/            OpenAICompatServer (JDK http) + main()
agents/              YAML agent configs served by gradle run
src/test/kotlin/     behavioural tests, key-free by design
```

Invariants worth preserving:

- **One config model.** DSL and YAML converge on the same `AgentConfig`;
  classes are referenced by FQN *strings* in YAML — renames of public module
  classes are config-breaking for external users; treat them as API changes.
- Implicit wiring: single provider binds automatically, `List<P>` parameters
  bind all providers, ambiguity raises. A module is never its own provider
  (self-exclusion is deliberate — it makes `ToolCollector` patterns work).
- **Runtime overrides replace config bindings** for singleton parameters
  (`overrideExchange` wins in `dependenciesFor`) — that's how the server
  injects per-request history and streaming responders.
- `EventProxy` observes suspend calls (wraps the caller's `Continuation`)
  and `Flow` streams (YIELD per chunk). Interface params get proxies,
  concrete-class params get raw instances (JDK proxies can't extend classes).

## Pitfalls already paid for

- **Flows:** never `emit` from inside `withContext(...)` in a `flow {}`
  builder — Flow invariant violation. Use `.flowOn(Dispatchers.IO)`.
- **Init order:** a property initialiser must not call a method that writes
  to that same property (the `load()`/`entries` NPE). Use `init { load() }`.
- **KClass scanning:** `KClass.functions` lists only *declared* members —
  inherited annotations vanish. Walk `java.superclass` + `declaredMemberFunctions`
  and rely on virtual dispatch (`memberFunctions` alone isn't enough either:
  the override without annotation hides the base's).
- **Parameter type extraction:** list element protocols come from
  `p.type.arguments.firstOrNull()?.type?.classifier` — `?.type` alone is a
  `KType`, not a `KClass`, and silently yields "Any" protocols.
- **Cancellation:** in suspend tool/exec paths, catch
  `TimeoutCancellationException` but rethrow plain
  `CancellationException` — swallowing caller cancellation turns "agent
  cancelled" into a fake tool error. The streaming server cancels its agent
  job when the client walks away; keep that `finally`.
- **Tool timeouts can't interrupt blocked JVM code.** `call_timeout_ms`
  releases the *agent*; the abandoned function finishes unseen. Documented;
  don't pretend otherwise.
- Inside test sources in `package xaibo`, `xaibo.examples.X` references are
  shadowed — import the class.
- `@XaiboTool` `description` is a required annotation argument. Keep it so:
  undescribed tools are silently worse agents.

## Out of scope — do not "helpfully" port these

LiveKit/Bedrock/MCP glue, HuggingFace/SentenceTransformer embedders, the
Svelte UI + GraphQL, `/v1/responses`, file-watching hot reload,
`no_function_calling_adapter`, `claude_thinking` extras beyond the shared
effort/budget mapping. These are SDK-bound or peripheral; the extension
points are the protocols (`LLMProtocol`, `ToolProviderProtocol`,
`EmbeddingProtocol`, …), each real provider ≈ 50–200 lines. Adding one
requires a user-visible need, not completeness.

## Style

- Kotlin idioms first; DSL sugar only when it removes repetition at a real
  call site (the `agentConfig` DSL earns its keep; a second tool-definition
  DSL would not).
- Comments explain *why* (the original's ReasoningEffort doc rules are good
  taste); never narrate what the code plainly does.
- Small files by responsibility, but a 300-line `Memory.kt` beats five
  one-class files with import ceremony.
- Run the verification ladder after every behavioural change, and re-verify
  with a real LLM (`RealLLMTest`) after changes to providers, tool scanning,
  or the server — mocks cannot feel wire-format pain.
