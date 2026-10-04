package xaibo

import xaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.toList

/**
 * The harness affordances, explicitly: stream frames as a live contract,
 * thinking kept out of content, steering mid-turn, compaction hooks, and
 * chunked delivery to response-consuming adapters.
 */
class StreamHarnessTest {

    private fun textResponse(vararg frames: StreamFrame) = listOf(frames.toList())

    private fun agent(
        rounds: List<List<StreamFrame>>,
        orchestratorConfig: Map<String, Any?> = mapOf("stream" to true),
        extraInstances: Map<String, Any> = emptyMap(),
        listener: ((Event) -> Unit)? = null,
        tools: Map<String, Any?> = mapOf("tools" to listOf(FakeTools())),
    ): Pair<Agent, FrameLLM> {
        val xaibo = Xaibo()
        val llm = FrameLLM(rounds)
        xaibo.registerAgent(agentConfig("h") {
            module<SimpleConversation>("history")
            module<FrameLLM>("llm")
            module<FunctionToolProvider>("tools", *tools.toList().toTypedArray())
            module<StreamingToolOrchestrator>("entry", *orchestratorConfig.toList().toTypedArray())
            entry("entry")
        })
        listener?.let { xaibo.registerEventListener("", it) }
        val agent = xaibo.getAgentWith("h", ConfigOverrides(instances = extraInstances + mapOf("llm" to llm)))
        return agent to llm
    }

    // -- assembler ------------------------------------------------------------

    @Test fun `assembler rebuilds the settled response from frames`() {
        val assembler = StreamAssembler()
        listOf(
            StreamFrame.Thinking("hm, 2+3"),
            StreamFrame.Text("the "),
            StreamFrame.Text("answer"),
            StreamFrame.ToolCall(LLMFunctionCall("c1", "add", mapOf("a" to 2, "b" to 3))),
            StreamFrame.Usage(LLMUsage(10, 5, 15)),
            StreamFrame.Done(mapOf("finish_reason" to "tool_calls")),
        ).forEach(assembler::feed)
        val settled = assembler.response()
        assertEquals("the answer", settled.content)
        assertEquals("add", settled.toolCalls!!.first().name)
        assertEquals(15, settled.usage!!.totalTokens)
        // thinking is NOT content: it is a field of its own, never mixed in
        assertEquals("hm, 2+3", settled.thinking)
        assertEquals("tool_calls", settled.vendorSpecific["finish_reason"])
        assertEquals(true, settled.vendorSpecific["streamed"])
    }

    // -- streaming through the agent loop ---------------------------------------

    @Test fun `streamed tool calls and usage drive the full turn`() {
        val (agent, _) = agent(
            rounds = listOf(
                listOf(StreamFrame.Text("let me add"),
                    StreamFrame.ToolCall(LLMFunctionCall("c1", "add", mapOf("a" to 2, "b" to 3))),
                    StreamFrame.Usage(LLMUsage(8, 2, 10))),
                listOf(StreamFrame.Text("sum is 5"), StreamFrame.Usage(LLMUsage(9, 4, 13))),
            ),
        )
        val response = runBlocking { agent.handleText("add 2 and 3") }
        assertEquals("sum is 5", response.text)
        assertTrue(response.events.any { it is ToolCallEvent && it.name == "add" })
        assertEquals(2, response.events.count { it is UsageEvent })
    }

    @Test fun `thinking frames reach the event bus but never the answer`() {
        val events = mutableListOf<Event>()
        val (agent, _) = agent(
            rounds = textResponse(
                StreamFrame.Thinking("counting..."),
                StreamFrame.Text("done counting"),
            ),
            listener = { events.add(it) },
        )
        val response = runBlocking { agent.handleText("go") }
        assertEquals("done counting", response.text)
        // live: frames on the event bus...
        val yields = events.filter { it.eventType == EventType.YIELD }.map { it.result }
        assertTrue(yields.contains(StreamFrame.Thinking("counting...")), "frames on the bus: $yields")
        // ...and settled: a first-class ThinkingEvent on the response lane
        assertTrue(response.events.any { it is ThinkingEvent && it.text == "counting..." },
            "response events: ${response.events}")
    }

    @Test fun `chunked delivery lets response-consuming adapters see deltas`() {
        val lane = RecordingResponse()
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("c") {
            module<SimpleConversation>("history")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "abcdefghijklmnop")))
            module<FunctionToolProvider>("tools", "tools" to listOf(FakeTools()))
            module<StreamingToolOrchestrator>("entry", "stream" to true, "stream_chunk_chars" to 4)
            entry("entry")
        })
        val agent = xaibo.getAgentWith("c", ConfigOverrides(
            instances = mapOf("__response__" to lane),
            exchange = listOf(ExchangeConfig(protocol = "ResponseProtocol", providers = listOf("__response__"))),
        ))
        runBlocking { agent.handleText("go") }
        assertEquals(listOf("abcd", "efgh", "ijkl", "mnop"), lane.texts)
    }

    // -- steering ----------------------------------------------------------------

    @Test fun `steered messages land after the tool batch, mid-turn`() {
        val toolStarted = CompletableDeferred<Unit>()
        val events = mutableListOf<Event>()
        val (agent, llm) = agent(
            rounds = listOf(
                listOf(StreamFrame.ToolCall(LLMFunctionCall("c1", "nap", mapOf("ms" to 300)))),
                listOf(StreamFrame.Text("final")),
            ),
            tools = mapOf("tool_classes" to listOf("xaibo.SlowTools")),
            listener = { e ->
                events += e
                if (e.eventType == EventType.CALL && e.methodName == "executeTool") toolStarted.complete(Unit)
            },
        )
        runBlocking {
            val turn = async { agent.handleText("take a nap") }
            withTimeout(5000) { toolStarted.await() } // nap is running
            assertTrue(agent.steer("new direction"), "entry module accepts steering")
            assertEquals("final", turn.await().text)
        }
        // the steered message reached the model: present in the second round's conversation
        assertEquals(2, llm.seenConversations.size)
        assertTrue(llm.seenConversations[1].any {
            it.role == LLMRole.USER && it.content.firstOrNull()?.text == "new direction"
        }, "second round did not see the steered message")
    }

    @Test fun `agents without a steerable entry report refusal`() {
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("plain") {
            module<SimpleConversation>("history")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "hi")))
            module<FunctionToolProvider>("tools", "tools" to listOf(FakeTools()))
            module<SimpleToolOrchestrator>("entry")
            entry("entry")
        })
        assertFalse(xaibo.getAgent("plain").steer("x"))
    }

    // -- compaction ----------------------------------------------------------------

    @Test fun `compaction is asked of the history module before each turn`() {
        val history = CompactingHistory()
        val (agent, _) = agent(
            rounds = textResponse(StreamFrame.Text("ok")),
            extraInstances = mapOf("history" to history),
        )
        runBlocking { agent.handleText("go") }
        assertEquals(1, history.compactions)
    }

    // -- persistence ---------------------------------------------------------------

    @Test fun `persist mode writes the transcript to history in order`() {
        val history = RecordingHistory()
        val (agent, _) = agent(
            rounds = listOf(
                listOf(StreamFrame.ToolCall(LLMFunctionCall("c1", "add", mapOf("a" to 2, "b" to 3)))),
                listOf(StreamFrame.Text("sum is 5")),
            ),
            orchestratorConfig = mapOf("stream" to true, "persist" to true),
            extraInstances = mapOf("history" to history),
        )
        runBlocking { agent.handleText("add 2 and 3") }
        val roles = history.added.map { it.role }
        assertEquals(listOf(LLMRole.USER, LLMRole.FUNCTION, LLMRole.FUNCTION, LLMRole.ASSISTANT), roles)
        assertEquals("add 2 and 3", history.added[0].content.first().text)
    }
}
