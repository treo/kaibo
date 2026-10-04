package xaibo

import xaibo.examples.DemoTools
import xaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * Real-provider smoke tests. Skipped unless XAIBO_TEST_API_KEY is set:
 *
 *   XAIBO_TEST_API_KEY=sk-... gradle test --tests xaibo.RealLLMTest
 *
 * Defaults to Alibaba's DashScope token-plan endpoint (OpenAI-compatible);
 * override with XAIBO_TEST_BASE_URL / XAIBO_TEST_MODEL.
 */
class RealLLMTest {

    private val apiKey = System.getenv("XAIBO_TEST_API_KEY")
    private val baseUrl = System.getenv("XAIBO_TEST_BASE_URL")
        ?: "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    private val model = System.getenv("XAIBO_TEST_MODEL") ?: "qwen3.8-flash"

    private fun llm() = OpenAILLM(mapOf("api_key" to apiKey, "base_url" to baseUrl, "model" to model))

    private fun requireKey() = assumeTrue(apiKey != null, "no XAIBO_TEST_API_KEY")
    @Test fun `a real model answers`() {
        requireKey()
        val response = runBlocking {
            llm().generate(listOf(LLMMessage.user("Reply with exactly: OK")))
        }
        assertTrue("OK" in response.content.uppercase(), "got: ${response.content}")
        assertNotNull(response.usage, "real providers report token usage")
    }

    @Test fun `a real model streams`() {
        requireKey()
        val frames = runBlocking {
            llm().generateStream(listOf(LLMMessage.user("Count from 1 to 5 separated by spaces."))).toList()
        }
        val text = frames.filterIsInstance<StreamFrame.Text>().joinToString("") { it.text }
        assertTrue("3" in text, "no meaningful text chunks: $frames")
        assertTrue(frames.any { it is StreamFrame.Done }, "stream must terminate with a Done frame")
        // usage requires the gateway to honor stream_options.include_usage; when
        // it does, the frame must carry real numbers
        frames.filterIsInstance<StreamFrame.Usage>().firstOrNull()?.let {
            assertTrue(it.usage.totalTokens > 0, "usage frame with zero tokens")
        }
    }

    @Test fun `a real model uses tools through a fully wired agent`() {
        requireKey()
        val xaibo = Xaibo()
        xaibo.registerAgent(agentConfig("real") {
            module<SimpleConversation>("history")
            module<OpenAILLM>("llm", "api_key" to apiKey, "base_url" to baseUrl, "model" to model)
            module<FunctionToolProvider>("tools", "tools" to listOf(DemoTools()))
            module<SimpleToolOrchestrator>("entry", "system_prompt" to
                "You have tools (multiply, reverse, current_time). You cannot know the current time from memory: " +
                "before answering any time question you MUST call the current_time tool and use exactly what it returned.")
            entry("entry")
        })

        // the event proxy must observe the genuine API calls too
        val events = mutableListOf<Event>()
        xaibo.registerEventListener("xaibo.primitives.OpenAILLM", { events.add(it) })

        // LLM compliance is probabilistic; two genuine attempts, any complete
        // tool round trip proves the framework carried a real model through it
        val year = java.time.Year.now().toString()
        var response: Response? = null
        for (attempt in 1..2) {
            response = runBlocking {
                xaibo.getAgent("real").handleText("What is the current date and time?")
            }
            val usedTool = response.events.any { it is ToolCallEvent && it.name == "current_time" }
            if (usedTool && (response.text ?: "").contains(year)) break
        }
        assertTrue(
            response!!.events.any { it is ToolCallEvent && it.name == "current_time" } &&
                (response.text ?: "").contains(year),
            "no real-tool round trip in 2 attempts; text=${response.text} events=${response.events}",
        )
        assertTrue(events.any { it.eventName.endsWith("generate.result") }, "event proxy observed the real API calls")
    }
}
