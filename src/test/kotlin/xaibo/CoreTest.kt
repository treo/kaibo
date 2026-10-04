package xaibo

import xaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.runBlocking

/**
 * The core xaibo assumptions: protocol-based wiring (implicit and explicit),
 * file + DSL configuration, overrides, and the module event stream.
 */
class CoreTest {

    private fun echoConfig(id: String = "echo") = agentConfig(id) {
        module<EchoEntry>("entry")
        module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hello")))
        entry("entry")
    }

    @Test fun `dsl-built agent is wired by protocol and answers`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(echoConfig())
        val response = runBlocking { xaibo.getAgent("echo").handleText("hi") }
        assertEquals("echo:hello", response.text)
    }

    @Test fun `yaml config loads modules and infers the wiring`() {
        val cfg = AgentConfig.fromYaml(
            """
            id: from-file
            description: loaded agent
            modules:
              - module: xaibo.EchoEntry
                id: entry
              - module: xaibo.primitives.MockLLM
                id: llm
                config:
                  responses:
                    - content: from-yaml
            exchange:
              - module: entry
                protocol: LLMProtocol
                provider: llm
            """.trimIndent()
        )
        assertTrue(cfg.modules.any { it.id == "__response__" }, "response module is implicit")
        assertTrue(
            cfg.exchange.any { it.module == "__entry__" && it.protocol == "TextMessageHandlerProtocol" },
            "entry handler is inferred",
        )

        val xaibo = Xaibo().apply { registerAgent(cfg) }
        assertEquals("echo:from-yaml", runBlocking { xaibo.getAgent("from-file").handleText("x").text })
    }

    @Test fun `single provider is auto-bound without exchange config`() {
        val cfg = agentConfig("auto") {
            module<EchoEntry>("entry") // uses LLMProtocol + ResponseProtocol
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "auto")))
        }
        // no explicit bindings at all: llm and __response__ resolved by uniqueness
        val xaibo = Xaibo().apply { registerAgent(cfg) }
        assertEquals("echo:auto", runBlocking { xaibo.getAgent("auto").handleText("x").text })
    }

    @Test fun `multiple providers demand an explicit bind`() {
        assertFailsWith<IllegalArgumentException> {
            agentConfig("bad") {
                module<EchoEntry>("entry")
                module<MockLLM>("llm1", "responses" to listOf(mapOf("content" to "1")))
                module<MockLLM>("llm2", "responses" to listOf(mapOf("content" to "2")))
            }
        }
    }

    @Test fun `explicit bind selects among providers and overrides replace instances`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("pick") {
            module<EchoEntry>("entry")
            module<MockLLM>("llm1", "responses" to listOf(mapOf("content" to "1")))
            module<MockLLM>("llm2", "responses" to listOf(mapOf("content" to "2")))
            bind<LLMProtocol>("llm2", module = "entry")
        })
        assertEquals("echo:2", runBlocking { xaibo.getAgent("pick").handleText("x").text })

        val replaced = ConfigOverrides(
            instances = mapOf("llm2" to MockLLM(mapOf("responses" to listOf(mapOf("content" to "override"))))),
        )
        assertEquals(
            "echo:override",
            runBlocking { xaibo.getAgentWith("pick", replaced).handleText("x").text },
        )
    }

    @Test fun `entry point ids come from the exchange`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(echoConfig())
        assertEquals(listOf("entry"), xaibo.getAgent("echo").getEntryPointIds())
    }

    @Test fun `listeners see call, result and exception events of every module hop`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(echoConfig())
        val events = mutableListOf<Event>()
        xaibo.registerEventListener("", { events.add(it) })

        runBlocking { xaibo.getAgent("echo").handleText("hi") }

        fun names() = events.map { it.eventName }
        assertContains(names(), "xaibo.EchoEntry.handleText.call")
        assertContains(names(), "xaibo.primitives.MockLLM.generate.call")
        assertContains(names(), "xaibo.primitives.MockLLM.generate.result")
        // the LLM call event records who asked and for what
        val llmCall = events.first { it.eventName.endsWith("generate.call") }
        assertEquals("entry", llmCall.callerId)
        assertEquals("llm", llmCall.moduleId)
        assertEquals(listOf("messages", "options"), llmCall.arguments?.keys?.toList())
        // results come back through the same proxy chain
        assertContains(names(), "xaibo.EchoEntry.handleText.result")
    }

    @Test fun `exception events carry the failure`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("boom") {
            module<EchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "x")))
        })
        val events = mutableListOf<Event>()
        xaibo.registerEventListener("", { events.add(it) })
        val broken = ConfigOverrides(
            instances = mapOf("llm" to object : LLMProtocol {
                override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?) =
                    error("nope")
                override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?) =
                    kotlinx.coroutines.flow.flow<String> { error("nope") }
            }),
        )
        assertFailsWith<IllegalStateException> {
            runBlocking { xaibo.getAgentWith("boom", broken).handleText("hi") }
        }
        assertTrue(
            events.any { it.eventType == EventType.EXCEPTION && it.exception?.contains("nope") == true },
            "the failure surfaced as an EXCEPTION event",
        )
    }

    @Test fun `prefix filters events`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(echoConfig())
        val seen = mutableListOf<Event>()
        xaibo.registerEventListener("xaibo.primitives.MockLLM", { seen.add(it) })
        runBlocking { xaibo.getAgent("echo").handleText("hi") }
        assertTrue(seen.isNotEmpty())
        assertTrue(seen.all { it.eventName.startsWith("xaibo.primitives.MockLLM") })
    }

    @Test fun `streaming generates yield events per chunk`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("stream") {
            module<StreamEchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "abcdefghi")))
            entry("entry")
        })
        val events = mutableListOf<Event>()
        xaibo.registerEventListener("xaibo.primitives.MockLLM.generateStream", { events.add(it) })
        val response = runBlocking { xaibo.getAgent("stream").handleText("hi") }
        assertEquals("abcdefghi", response.text)
        val yields = events.filter { it.eventType == EventType.YIELD }
        assertEquals(3, yields.size) // chunk size default 3
        assertEquals("abc", yields.first().result)
        assertEquals(3L, (events.last().result as Map<*, *>)["chunks"])
    }

    @Test fun `agent scoped modules live beyond one agent instance`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("shared") {
            module<SimpleConversation>("history", scope = Scope.AGENT)
            module<EchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hello")))
        })
        runBlocking {
            val first = xaibo.getAgent("shared")
            (first.exchange.getModule("history") as ConversationHistoryProtocol)
                .addMessage(LLMMessage.user("remembered"))
            val second = xaibo.getAgent("shared")
            val history = second.exchange.getModule("history") as ConversationHistoryProtocol
            assertEquals(1, history.getHistory().size)
        }
    }

    @Test fun `tool collector gathers every provider through list wiring`() {
        val cfg = AgentConfig.fromYaml(
            """
            id: collector
            modules:
              - module: xaibo.primitives.SimpleConversation
                id: history
              - module: xaibo.primitives.MockLLM
                id: llm
                config:
                  responses:
                    - content: done
              - module: xaibo.primitives.FunctionToolProvider
                id: kotlin-tools
                config:
                  tool_classes: [xaibo.examples.DemoTools]
              - module: xaibo.FakeTools
                id: fake-tools
              - module: xaibo.primitives.ToolCollector
                id: tools
              - module: xaibo.primitives.SimpleToolOrchestrator
                id: orchestrator
            exchange:
              - module: orchestrator
                protocol: ToolProviderProtocol
                provider: tools
            """.trimIndent()
        )
        val collectorBinding = cfg.exchange.single { it.module == "tools" && it.protocol == "ToolProviderProtocol" }
        assertEquals(setOf("kotlin-tools", "fake-tools"), collectorBinding.providers.toSet())

        val xaibo = Xaibo().apply { registerAgent(cfg) }
        assertEquals("done", runBlocking { xaibo.getAgent("collector").handleText("hi").text })
    }

    @Test fun `server modules are injectable by class`() {
        val xaibo = Xaibo()
        xaibo.registerServerModule("tools", FakeTools())
        xaibo.registerAgent(agentConfig("srv") {
            module<NeedsServerTools>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hi")))
        })
        // NeedsServerTools takes a FakeTools (concrete class) -> served by the registered instance
        val response = runBlocking { xaibo.getAgent("srv").handleText("x") }
        assertEquals("tool:42", response.text)
    }
}

/** Consumes a concrete server-provided class rather than a protocol. */
class NeedsServerTools(
    val tools: FakeTools,
    val llm: LLMProtocol,
    val response: ResponseProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol {
    override suspend fun handleText(text: String) {
        val result = tools.executeTool("add", mapOf("a" to 40, "b" to 2)).result
        response.respondText("tool:$result")
    }
}
