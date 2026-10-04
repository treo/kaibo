# Kaibo

A Kotlin port of the Python [xaibo](../xaibo) agent framework: protocol-based
dependency injection, transparent module proxies with an event stream, YAML-
and DSL-defined agents, tool orchestration, vector memory, and an
OpenAI-compatible server.

The port is deliberately partial: everything that *is* the framework plus the
modules that demonstrate it; SDK-bound periphery left out (see Scope).

## Quick start

```bash
gradle test                                  # 48 tests: 45 offline, 4 live ones self-skip without a key
gradle run --args="--agents agents --port 8000"
```

The suite is key-free by design; provider fidelity against a *real* LLM is
documented by opt-in tests that skip unless a key is present:

```bash
KAIBO_TEST_API_KEY=sk-... gradle test --tests kaibo.RealLLMTest
# defaults to Alibaba DashScope (OpenAI-compatible, qwen3.8-flash);
# override with KAIBO_TEST_BASE_URL / KAIBO_TEST_MODEL
```

`agents/demo.yml` is key-free (scripted MockLLM); `agents/chat.yml`, `agents/react.yml` and `agents/real-llm.yml` need `OPENAI_API_KEY` (any OpenAI-compatible gateway via `base_url`). Point any OpenAI client at the server and use the agent id as the model name:

```bash
curl localhost:8000/v1/chat/completions -d '{"model":"demo","messages":[{"role":"user","content":"what is 6*7?"}]}'
```

## Defining agents

Two roads to the same config object. A YAML file (`AgentConfig.loadDirectory`
/ `fromYaml`, like the Python one — modules referenced by class name):

```yaml
id: chat
modules:
  - module: kaibo.primitives.SimpleConversation
    id: history
  - module: kaibo.primitives.OpenAILLM
    id: llm
  - module: kaibo.primitives.SimpleToolOrchestrator
    id: orchestrator
```

…or the typed DSL (`agentConfig { }`), which adds compile-time checked
protocol references instead of strings:

```kotlin
val cfg = agentConfig("chat") {
    module<SimpleConversation>("history", "max_history" to 40)
    module<OpenAILLM>("llm", "model" to "gpt-4.1-nano")
    module<SimpleToolOrchestrator>("orchestrator", "system_prompt" to "You are helpful.")
}
```

Unambiguous wiring is inferred (the framework's `populate_implicits`): a
module's constructor parameters typed with a protocol interface get the single
provider of that protocol; `List<P>` parameters get every provider; the
`__response__` module and `__entry__` handlers are added implicitly. Ambiguity
either is resolved by an explicit binding (`bind<P>(...)` / an `exchange:`
entry) or raises. `ConfigOverrides` swaps instances and rebinds protocols per
request — exactly how the server injects per-conversation history and a
streaming responder.

## How the design maps over

| Python xaibo | This port |
|---|---|
| pydantic models + `typing.Protocol` | data classes + Kotlin interfaces (`kaibo.Protocols.kt`) |
| `inspect` annotations + `importlib` | `kotlin-reflect` primary constructors (`injectableParams`) |
| `Proxy` / `MethodProxy` duck typing | JDK dynamic proxy (`EventProxy`) emitting `CALL/RESULT/EXCEPTION/YIELD` events; suspend and `Flow` calls fully observed |
| async generators (`yield` streams) | `kotlinx.coroutines.flow.Flow<String>` |
| `BinaryIO` attachments | `ByteArray` attachments |
| YAML via pydantic-yaml | YAML via kaml (`AgentConfig.fromYaml/loadDirectory`) |
| docstring-parsed Python functions as tools | `@KaiboTool` annotated Kotlin functions (`FunctionToolProvider`) |
| FastAPI/uvicorn + OpenAI adapter | JDK `com.sun.net.httpserver` (`OpenAICompatServer`), incl. SSE streaming |
| tiktoken / numpy / pickle memory stack | word-window chunker / `JsonVectorIndex` (linear cosine over `DoubleArray`, JSON persistence) |

## Adding tools

One annotated function; the annotations are what the model reads:

```kotlin
class WeatherTools {
    @KaiboTool(description = "Gets the current weather for a city")
    fun weather(@KaiboParam("City name, e.g. 'Amsterdam'") city: String) =
        fetchWeather(city)   // any JSON-shaped return value works
}
```

- `description` is a required annotation argument — an undescribed tool is a
  compile error, not a silently worse agent.
- Parameter types, names and requiredness come from the Kotlin signature;
  Kotlin enums become JSON-schema enums; defaults/nullability infer optional.
- Register code-side (`"tools" to listOf(WeatherTools())`) or config-side
  (`tool_classes: [com.example.WeatherTools]`) so YAML agents can use them.
- No code available? `OneShotTools` defines a tool as a prompt template in
  YAML alone.
- Many sources? `ToolCollector` merges providers (bind it explicitly when the
  consumer takes a single provider — see `agents/chat.yml`).
- Long-running tools: functions run on `Dispatchers.IO`; set
  `call_timeout_ms` on the provider to cap how long the agent *waits* — a
  timeout becomes a failed `ToolResult` (the model can then apologize or
  retry) instead of a stalled turn. A blocked JVM function cannot be
  interrupted, so after a timeout it finishes unseen: keep tool side effects
  idempotent, or build the async-job pattern (return a task id, poll it with a
  second tool) in your own provider.

## Scope — what was *not* ported, on purpose

- **LiveKit, Bedrock, MCP**: SDK-bound integrations. The framework's
  extension point is a ~50-line module implementing `LLMProtocol` or
  `ToolProviderProtocol`; porting the SDK glue is a user-side decision, not
  core.
- **HuggingFace / SentenceTransformer embedders**: no JVM equivalents worth
  the dependency weight; bring any model behind `EmbeddingProtocol`.
- **Svelte UI, GraphQL adapter, `/v1/responses`, hot-reload file watching**:
  frontends and a second wire format; the OpenAI-compatible surface covers
  the main use (debug events are already on the `Event` stream for any UI
  to consume).
- **`no_function_calling_adapter`, `claude_thinking` extras, `oneshot`
  images-from-disk**: niche shims; `ReasoningEffort` itself is honoured by
  all three provider modules.

Included beyond the core: `SimpleToolOrchestrator`, `ReActOrchestrator`,
`LLMCombinator`, `OpenAILLM`/`AnthropicLLM`/`GoogleLLM`/`MockLLM` (with the
reasoning-effort ladder rules), `OneShotTools`, `ToolCollector`,
`MemoryToolProvider`, vector memory, `DebugEventListener`,
server-module injection, agent-scoped modules, event listeners.

## Layout

```
src/main/kotlin/kaibo/
  Models.kt Protocols.kt        # shared vocabulary
  Config.kt AgentDsl.kt         # YAML model + implicit wiring + DSL
  Exchange.kt                   # instantiation, injection, EventProxy
  Registry.kt                   # Registry, Kaibo, Agent
  Json.kt                       # JSON <-> Kotlin bridging
  primitives/                   # LLMs, orchestrators, tools, memory
  server/OpenAICompatServer.kt  # HTTP façade + main()
  examples/DemoTools.kt
agents/                         # ready-to-serve agent configs
src/test/kotlin/kaibo/          # 48 tests documenting the framework's contracts
```

## Notes for maintainers

- `KAIBO_DEBUG=1` logs every module call/result event.
- Tests stay behavioural: they wire real agents through the exchange and
  assert on responses, events, and tool round-trips — never on internals.
- Kotlin 2.2 / JVM 21 toolchain, deps: coroutines, kotlinx-serialization,
  kaml, kotlin-reflect. No framework lock-in at runtime.

## Harness support (streaming frontends)

The OpenAI-compatible server is one frontend; a harness (TUI, websocket, IDE
protocol) plugs in at the framework's native seams:

- **`generateStream` yields `StreamFrame`s** — `Text`, `Thinking`, `ToolCall`,
  `Usage`, `Done`. Every frame crosses the module proxy as a `YIELD` event,
  so a frontend renders the entire turn — tokens, reasoning, tool calls and
  results, usage — from the event bus alone. Thinking is *not* content: it
  never mixes into the answer; it surfaces as the first-class
  `LLMResponse.thinking` field (`StreamAssembler`) and, per round, as a
  `ThinkingEvent` on the response lane.
- **`StreamingToolOrchestrator`** (`stream: true`) runs every round as a
  stream, journals the full transcript (`persist: true` — history is the
  source of truth), asks a `CompactionProtocol` history to compact before
  each turn, applies the reasoning level to all rounds, and delivers the
  settled answer in `stream_chunk_chars` slices so response-consuming
  adapters (like the SSE server) see deltas too.
- **Steering**: `agent.steer("text")` queues a user message into the running
  turn; it lands at the earliest point the conversation grammar allows
  (after a complete tool batch). Never lost, never half-taken. Entries that
  cannot steer make it return `false`.
- **Response lane injection**: replace `__response__` via `ConfigOverrides`
  with any sink (channel/websocket pusher) — live per-turn delivery without
  touching agents, exactly how the SSE endpoint streams.

One hot-path rule for event listeners (the Python harness learned it the hard
way): YIELDs fire once per proxy layer of a delegation chain — e.g. a
`LLMCombinator` wrapping two LLMs frames twice. Dedupe by
`Event.callerId`/`moduleId`.
