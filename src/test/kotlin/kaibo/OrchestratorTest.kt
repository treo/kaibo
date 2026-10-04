package kaibo

import kaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.runBlocking

/**
 * Orchestrator behaviour through a fully wired agent: tool round-trips,
 * failure stress, thought limits, and the response event stream.
 */
class OrchestratorTest {

    private val toolCallResponse = mapOf(
        "content" to "",
        "tool_calls" to listOf(mapOf(
            "id" to "c1", "name" to "add",
            "arguments" to mapOf("a" to 2, "b" to 3),
        )),
        "usage" to mapOf("prompt_tokens" to 10, "completion_tokens" to 5, "total_tokens" to 15),
    )
    private val finalResponse = mapOf(
        "content" to "The sum is 5",
        "usage" to mapOf("prompt_tokens" to 12, "completion_tokens" to 4, "total_tokens" to 16),
    )

    /** registers a `math` agent: history + llm + tools + orchestrator */
    private fun registerMathAgent(llmConfig: Map<String, Any?>, toolsConfig: Map<String, Any?> = emptyMap()): Kaibo =
        Kaibo().apply {
            registerAgent(agentConfig("math") {
                module<SimpleConversation>("history")
                module<MockLLM>("llm", *llmConfig.toList().toTypedArray())
                module<FakeTools>("tools", *toolsConfig.toList().toTypedArray())
                module<SimpleToolOrchestrator>("entry", "max_thoughts" to 5)
                entry("entry")
            })
        }

    @Test fun `answers through a tool round trip`() {
        val kaibo = registerMathAgent(mapOf("responses" to listOf(toolCallResponse, finalResponse)))
        val response = runBlocking { kaibo.getAgent("math").handleText("add 2 and 3") }

        assertEquals("The sum is 5", response.text)
        val events = response.events
        assertIs<UsageEvent>(events[0])
        assertIs<ToolCallEvent>(events[1])
        val result = assertIs<ToolResultEvent>(events[2])
        assertTrue(result.success)
        assertEquals(5, result.result)
        assertEquals(2, events.count { it is UsageEvent })
    }

    @Test fun `tool failures raise the stress level`() {
        val kaibo = registerMathAgent(
            llmConfig = mapOf("responses" to listOf(finalResponse)),
            toolsConfig = mapOf("fail" to true),
        )
        val seen = mutableListOf<LLMOptions>()
        val scripted = ScriptedLLM(
            responses = listOf(LLMResponse.fromMap(toolCallResponse), LLMResponse.fromMap(finalResponse)),
            seen = seen,
        )
        val agent = kaibo.getAgentWith("math", ConfigOverrides(instances = mapOf("llm" to scripted)))
        runBlocking { agent.handleText("add 2 and 3") }

        assertEquals(2, seen.size)
        assertEquals(0.0, seen[0].temperature) // calm start
        assertEquals(0.1, seen[1].temperature) // stress after a failed tool
    }

    @Test fun `stops trying after max thoughts`() {
        val kaibo = registerMathAgent(mapOf("responses" to listOf(finalResponse)))
        val scripted = ScriptedLLM(responses = listOf(LLMResponse.fromMap(toolCallResponse)))
        val agent = kaibo.getAgentWith("math", ConfigOverrides(instances = mapOf("llm" to scripted)))
        runBlocking { agent.handleText("go") }

        assertEquals(5, scripted.seen.size) // max_thoughts from config
        assertNull(scripted.seen.last().functions) // tools disabled on the final thought
    }

    @Test fun `annotated kotlin functions work as tools`() {
        val provider = FunctionToolProvider(mapOf("tools" to listOf(CalcTools())))
        val tools = runBlocking { provider.listTools() }
        val multiply = tools.first { it.name == "multiply" }
        assertEquals("First factor", multiply.parameters["a"]?.description)
        assertEquals("integer", multiply.parameters["a"]?.type)
        assertTrue(multiply.parameters["a"]!!.required)
        runBlocking {
            assertEquals(42, provider.executeTool("multiply", mapOf("a" to 6L, "b" to 7L)).result)

            // a Kotlin enum parameter arrives as a JSON-schema enum...
            val pi = tools.first { it.name == "pi" }
            assertEquals("string", pi.parameters["precision"]?.type)
            assertEquals(listOf("LOW", "HIGH"), pi.parameters["precision"]?.enum)
            // ...and the string value is coerced back into the enum on execution
            assertEquals(3.14159, provider.executeTool("pi", mapOf("precision" to "HIGH")).result)
        }
    }

    @Test fun `tool collector routes to the right provider`() {
        val collector = ToolCollector(listOf(FunctionToolProvider(mapOf("tools" to listOf(CalcTools()))), FakeTools()))
        runBlocking {
            assertEquals(setOf("multiply", "pi", "add"), collector.listTools().map { it.name }.toSet())
            assertEquals(12, collector.executeTool("multiply", mapOf("a" to 3, "b" to 4)).result)
            assertEquals(7, collector.executeTool("add", mapOf("a" to 3, "b" to 4)).result)
            assertFalse(collector.executeTool("nope", emptyMap()).success)
        }
    }

    @Test fun `a long running tool can be timeboxed`() {
        val provider = FunctionToolProvider(mapOf("tools" to listOf(SlowTools()), "call_timeout_ms" to 100))
        val started = System.currentTimeMillis()
        val result = runBlocking { provider.executeTool("nap", mapOf("ms" to 2000)) }
        val waited = System.currentTimeMillis() - started
        assertFalse(result.success)
        assertTrue(result.error!!.contains("timed out"), result.error!!)
        assertTrue(waited < 1000, "the agent should be released at the deadline, not after the sleep (${waited}ms)")

        // unconfigured provider: no timebox, the slow result simply arrives
        val patient = FunctionToolProvider(mapOf("tools" to listOf(SlowTools())))
        assertEquals("woke after 10", runBlocking { patient.executeTool("nap", mapOf("ms" to 10)).result })
    }

    @Test fun `tools are inherited from base classes and dispatched virtually`() {
        val provider = FunctionToolProvider(mapOf("tools" to listOf(LoudGreeter())))
        val tools = runBlocking { provider.listTools() }
        // annotated on the base, declared nowhere in the subclass — still a tool;
        // the private annotated function is not
        assertEquals(listOf("greet"), tools.map { it.name })
        assertEquals("Name to greet", tools.first().parameters["name"]?.description)
        // and the call lands on the override, as any virtual call would
        assertEquals("HI zoe!!", runBlocking { provider.executeTool("greet", mapOf("name" to "zoe")).result })
    }

    @Test fun `history is trimmed to max history`() {
        val history = SimpleConversation(mapOf("max_history" to 2))
        runBlocking {
            history.addMessage(LLMMessage.user("one"))
            history.addMessage(LLMMessage.user("two"))
            history.addMessage(LLMMessage.user("three"))
            assertEquals(listOf("two", "three"), history.getHistory().map { it.content.first().text })
        }
    }

    @Test fun `conversation seeds from openai format`() {
        val history = SimpleConversation.fromOpenaiMessages(
            listOf(
                mapOf("role" to "system", "content" to "be nice"),
                mapOf("role" to "user", "content" to listOf(
                    mapOf("type" to "text", "text" to "look"),
                    mapOf("type" to "image_url", "image_url" to mapOf("url" to "data:image/png;base64,AA")),
                )),
            )
        )
        val msgs = runBlocking { history.getHistory() }
        assertEquals(LLMRole.SYSTEM, msgs[0].role)
        assertEquals("be nice", msgs[0].content.first().text)
        assertEquals(2, msgs[1].content.size)
        assertEquals("data:image/png;base64,AA", msgs[1].content[1].image)
    }

    @Test fun `one shot tools run prompt templates through the llm`() {
        val llm = MockLLM(mapOf("responses" to listOf(mapOf("content" to "a cat"))))
        val tools = OneShotTools(
            llm,
            mapOf(
                "tools" to listOf(mapOf(
                    "name" to "describe",
                    "description" to "describe a thing",
                    "parameters" to listOf(mapOf("name" to "thing", "type" to "string", "description" to "what")),
                    "conversation" to listOf(mapOf(
                        "role" to "user",
                        "message" to listOf(mapOf("type" to "text", "text" to "Describe: \$\$params.thing\$\$")),
                    )),
                ))
            ),
        )
        val result = runBlocking { tools.executeTool("describe", mapOf("thing" to "cat")) }
        assertTrue(result.success)
        assertEquals("a cat", result.result)
    }
}
