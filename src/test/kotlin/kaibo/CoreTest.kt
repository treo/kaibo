package kaibo

import kaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.runBlocking

/**
 * The core kaibo assumptions: protocol-based wiring (implicit and explicit),
 * file + DSL configuration, overrides, and the module event stream.
 */
class CoreTest {

    private fun echoConfig(id: String = "echo") = agentConfig(id) {
        module<EchoEntry>("entry")
        module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hello")))
        entry("entry")
    }

    @Test fun `dsl-built agent is wired by protocol and answers`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(echoConfig())
        val response = runBlocking { kaibo.getAgent("echo").handleText("hi") }
        assertEquals("echo:hello", response.text)
    }

    @Test fun `yaml config loads modules and infers the wiring`() {
        val cfg = AgentConfig.fromYaml(
            """
            id: from-file
            description: loaded agent
            modules:
              - module: kaibo.EchoEntry
                id: entry
              - module: kaibo.primitives.MockLLM
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

        val kaibo = Kaibo().apply { registerAgent(cfg) }
        assertEquals("echo:from-yaml", runBlocking { kaibo.getAgent("from-file").handleText("x").text })
    }

    @Test fun `single provider is auto-bound without exchange config`() {
        val cfg = agentConfig("auto") {
            module<EchoEntry>("entry") // uses LLMProtocol + ResponseProtocol
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "auto")))
        }
        // no explicit bindings at all: llm and __response__ resolved by uniqueness
        val kaibo = Kaibo().apply { registerAgent(cfg) }
        assertEquals("echo:auto", runBlocking { kaibo.getAgent("auto").handleText("x").text })
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
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("pick") {
            module<EchoEntry>("entry")
            module<MockLLM>("llm1", "responses" to listOf(mapOf("content" to "1")))
            module<MockLLM>("llm2", "responses" to listOf(mapOf("content" to "2")))
            bind<LLMProtocol>("llm2", module = "entry")
        })
        assertEquals("echo:2", runBlocking { kaibo.getAgent("pick").handleText("x").text })

        val replaced = ConfigOverrides(
            instances = mapOf("llm2" to MockLLM(mapOf("responses" to listOf(mapOf("content" to "override"))))),
        )
        assertEquals(
            "echo:override",
            runBlocking { kaibo.getAgentWith("pick", replaced).handleText("x").text },
        )
    }

    @Test fun `entry point ids come from the exchange`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(echoConfig())
        assertEquals(listOf("entry"), kaibo.getAgent("echo").getEntryPointIds())
    }

    @Test fun `listeners see call, result and exception events of every module hop`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(echoConfig())
        val events = mutableListOf<Event>()
        kaibo.registerEventListener("", { events.add(it) })

        runBlocking { kaibo.getAgent("echo").handleText("hi") }

        fun names() = events.map { it.eventName }
        assertContains(names(), "kaibo.EchoEntry.handleText.call")
        assertContains(names(), "kaibo.primitives.MockLLM.generate.call")
        assertContains(names(), "kaibo.primitives.MockLLM.generate.result")
        // the LLM call event records who asked and for what
        val llmCall = events.first { it.eventName.endsWith("generate.call") }
        assertEquals("entry", llmCall.callerId)
        assertEquals("llm", llmCall.moduleId)
        assertEquals(listOf("messages", "options"), llmCall.arguments?.keys?.toList())
        // results come back through the same proxy chain
        assertContains(names(), "kaibo.EchoEntry.handleText.result")
    }

    @Test fun `exception events carry the failure`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("boom") {
            module<EchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "x")))
        })
        val events = mutableListOf<Event>()
        kaibo.registerEventListener("", { events.add(it) })
        val broken = ConfigOverrides(
            instances = mapOf("llm" to object : LLMProtocol {
                override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?) =
                    error("nope")
                override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?) =
                    kotlinx.coroutines.flow.flow<StreamFrame> { error("nope") }
            }),
        )
        assertFailsWith<IllegalStateException> {
            runBlocking { kaibo.getAgentWith("boom", broken).handleText("hi") }
        }
        assertTrue(
            events.any { it.eventType == EventType.EXCEPTION && it.exception?.contains("nope") == true },
            "the failure surfaced as an EXCEPTION event",
        )
    }

    @Test fun `prefix filters events`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(echoConfig())
        val seen = mutableListOf<Event>()
        kaibo.registerEventListener("kaibo.primitives.MockLLM", { seen.add(it) })
        runBlocking { kaibo.getAgent("echo").handleText("hi") }
        assertTrue(seen.isNotEmpty())
        assertTrue(seen.all { it.eventName.startsWith("kaibo.primitives.MockLLM") })
    }

    @Test fun `streaming generates yield events per chunk`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("stream") {
            module<StreamEchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "abcdefghi")))
            entry("entry")
        })
        val events = mutableListOf<Event>()
        kaibo.registerEventListener("kaibo.primitives.MockLLM.generateStream", { events.add(it) })
        val response = runBlocking { kaibo.getAgent("stream").handleText("hi") }
        assertEquals("abcdefghi", response.text)
        val yields = events.filter { it.eventType == EventType.YIELD }
        assertEquals(3, yields.size) // chunk size default 3
        assertEquals(StreamFrame.Text("abc"), yields.first().result)
        assertEquals(3L, (events.last().result as Map<*, *>)["chunks"])
    }

    @Test fun `agent scoped modules live beyond one agent instance`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("shared") {
            module<SimpleConversation>("history", scope = Scope.AGENT)
            module<EchoEntry>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hello")))
        })
        runBlocking {
            val first = kaibo.getAgent("shared")
            (first.exchange.getModule("history") as ConversationHistoryProtocol)
                .addMessage(LLMMessage.user("remembered"))
            val second = kaibo.getAgent("shared")
            val history = second.exchange.getModule("history") as ConversationHistoryProtocol
            assertEquals(1, history.getHistory().size)
        }
    }

    @Test fun `tool collector gathers every provider through list wiring`() {
        val cfg = AgentConfig.fromYaml(
            """
            id: collector
            modules:
              - module: kaibo.primitives.SimpleConversation
                id: history
              - module: kaibo.primitives.MockLLM
                id: llm
                config:
                  responses:
                    - content: done
              - module: kaibo.primitives.FunctionToolProvider
                id: kotlin-tools
                config:
                  tool_classes: [kaibo.examples.DemoTools]
              - module: kaibo.FakeTools
                id: fake-tools
              - module: kaibo.primitives.ToolCollector
                id: tools
              - module: kaibo.primitives.SimpleToolOrchestrator
                id: orchestrator
            exchange:
              - module: orchestrator
                protocol: ToolProviderProtocol
                provider: tools
            """.trimIndent()
        )
        val collectorBinding = cfg.exchange.single { it.module == "tools" && it.protocol == "ToolProviderProtocol" }
        assertEquals(setOf("kotlin-tools", "fake-tools"), collectorBinding.providers.toSet())

        val kaibo = Kaibo().apply { registerAgent(cfg) }
        assertEquals("done", runBlocking { kaibo.getAgent("collector").handleText("hi").text })
    }

    @Test fun `server modules are injectable by class`() {
        val kaibo = Kaibo()
        kaibo.registerServerModule("tools", FakeTools())
        kaibo.registerAgent(agentConfig("srv") {
            module<NeedsServerTools>("entry")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hi")))
        })
        // NeedsServerTools takes a FakeTools (concrete class) -> served by the registered instance
        val response = runBlocking { kaibo.getAgent("srv").handleText("x") }
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
