package kaibo

import kaibo.primitives.*
import kotlin.test.*
import kotlinx.coroutines.runBlocking

/** ReAct cycles, the LLM combinator chain, and the reasoning-effort ladder. */
class AdvancedTest {

    private fun reactAgent(llm: LLMProtocol, vararg cfg: Pair<String, Any?>): Agent {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("react") {
            module<SimpleConversation>("history")
            module<MockLLM>("llm", "responses" to listOf(mapOf("content" to "x")))
            module<FunctionToolProvider>("tools", "tools" to listOf(CalcTools()))
            module<ReActOrchestrator>("entry", *cfg)
            entry("entry")
        })
        return kaibo.getAgentWith("react", ConfigOverrides(instances = mapOf("llm" to llm)))
    }

    @Test fun `react reaches a final answer through thought action observation`() {
        val scripted = ScriptedLLM(
            responses = listOf(
                LLMResponse("THOUGHT: I need the product of 6 and 7"),
                LLMResponse("", toolCalls = listOf(LLMFunctionCall("c1", "multiply", mapOf("a" to 6, "b" to 7)))),
                LLMResponse("OBSERVATION: the tool returned 42"),
                LLMResponse("THOUGHT: I can answer now"),
                LLMResponse("FINAL_ANSWER: It is 42."),
            ),
        )
        val response = runBlocking { reactAgent(scripted, "show_reasoning" to false).handleText("what is 6*7?") }
        assertEquals("FINAL_ANSWER: It is 42.", response.text)
        assertEquals(5, scripted.seen.size)
    }

    @Test fun `react with reasoning shown streams its phases`() {
        val scripted = ScriptedLLM(responses = listOf(LLMResponse("FINAL_ANSWER: 42")))
        val response = runBlocking { reactAgent(scripted).handleText("q") }
        assertTrue((response.text ?: "").contains("THINKING"))
        assertTrue((response.text ?: "").contains("FINAL ANSWER"))
    }

    @Test fun `combinator chains llms and merges results`() {
        val kaibo = Kaibo()
        kaibo.registerAgent(agentConfig("chain") {
            module<MockLLM>("writer", "responses" to listOf(mapOf("content" to "draft")))
            module<MockLLM>("checker", "responses" to listOf(mapOf("content" to "checked")))
            module<EchoEntry>("entry")
            module<LLMCombinator>("combo", "prompts" to listOf("write it", "verify it"))
            bind<LLMProtocol>("combo", module = "entry")
            entry("entry")
        })
        val response = runBlocking { kaibo.getAgent("chain").handleText("go") }
        assertEquals("echo:draft\nchecked", response.text)
    }

    @Test fun `combinator demands one prompt per llm`() {
        assertFailsWith<IllegalArgumentException> {
            runBlocking {
                LLMCombinator(listOf(MockLLM(mapOf("responses" to listOf(mapOf("content" to "a"))))),
                    mapOf("prompts" to emptyList<String>())).generate(emptyList())
            }
        }
    }

    @Test fun `reasoning effort travels in the model's own shape`() {
        // unset sends nothing
        assertTrue(thinkingKwargs(null, "effort", null).isEmpty())
        // Claude can say no
        assertEquals(mapOf("thinking" to mapOf("type" to "disabled")), thinkingKwargs(ReasoningEffort.NONE, "effort", null))
        // a level the model cannot express is lowered, never dropped
        val adaptive = thinkingKwargs(ReasoningEffort.MINIMAL, "effort", null)
        assertEquals("adaptive", (adaptive["thinking"] as Map<*, *>)["type"])
        assertEquals("low", ((adaptive["output_config"] as Map<*, *>)["effort"]))
        // budget shape raises the ceiling so thinking never truncates the answer
        val budget = thinkingKwargs(ReasoningEffort.HIGH, "budget", 100)
        assertEquals(16384, (budget["thinking"] as Map<*, *>)["budget_tokens"])
        assertEquals(17408, budget["max_tokens"])
    }
}
