package kaibo.primitives

import kaibo.*

/** Phases of the ReAct cycle */
enum class ReActPhase { THOUGHT, ACTION, OBSERVATION, FINAL_ANSWER }

/**
 * ReAct (Reason and Act) orchestrator: explicit Thought → Action →
 * Observation cycles, ending in a FINAL_ANSWER or at `max_iterations`.
 *
 * Config: `system_prompt`, `thought_prompt`, `action_prompt`,
 * `observation_prompt`, `error_prompt` ({error}), `max_iterations_prompt`
 * ({max_iterations}), `max_iterations` (10), `show_reasoning` (true),
 * `reasoning_temperature` (0.7).
 */
class ReActOrchestrator(
    val response: ResponseProtocol,
    val llm: LLMProtocol,
    val toolProvider: ToolProviderProtocol,
    val history: ConversationHistoryProtocol,
    val config: Map<String, Any?> = emptyMap(),
) : TextMessageHandlerProtocol {

    private inner class Phase { var current = ReActPhase.THOUGHT }

    private val defaultSystemPrompt = """You are an AI assistant that follows the ReAct (Reasoning and Acting) pattern. You must think step by step and follow this exact format:

THOUGHT: [Your reasoning about what to do next, what information you need, or what action to take]

ACTION: [Either call a tool with specific parameters, or provide FINAL_ANSWER if you have enough information]

OBSERVATION: [After receiving tool results, analyze what you learned and decide next steps]

Rules:
1. Always start with THOUGHT to reason about the situation
2. Use ACTION to either call tools or provide FINAL_ANSWER
3. After tool execution, use OBSERVATION to analyze results
4. Continue the cycle until you can provide a FINAL_ANSWER
5. Be explicit about your reasoning process
6. If you need to use multiple tools, do them one at a time
7. When you have sufficient information, use ACTION: FINAL_ANSWER: [your complete answer]

Available tools will be provided to you. Use them when you need additional information or capabilities."""

    private val systemPrompt = config["system_prompt"] as? String ?: defaultSystemPrompt
    private val thoughtPrompt = config["thought_prompt"] as? String
        ?: "Generate your THOUGHT about what to do next. Consider the user's request and what information or actions you need."
    private val actionPrompt = config["action_prompt"] as? String
        ?: """Now take ACTION based on your thought. You have two options:
1. Call a tool if you need more information or capabilities
2. Provide FINAL_ANSWER: [your complete answer] if you have sufficient information

Choose the appropriate action."""
    private val observationPrompt = config["observation_prompt"] as? String
        ?: """Analyze the OBSERVATION from the tool results. What did you learn?
Do you have enough information to provide a final answer, or do you need to take more actions?
Provide your OBSERVATION and reasoning."""
    private val errorPrompt = config["error_prompt"] as? String
        ?: """An error occurred: {error}

Please provide a FINAL_ANSWER based on the information you have gathered so far,
or explain what went wrong and what you were trying to accomplish."""
    private val maxIterationsPrompt = config["max_iterations_prompt"] as? String
        ?: """You have reached the maximum number of iterations ({max_iterations}).
Please provide a FINAL_ANSWER based on the information and reasoning you have gathered so far."""
    private val maxIterations = (config["max_iterations"] as? Number)?.toInt() ?: 10
    private val showReasoning = config["show_reasoning"] as? Boolean ?: true
    private val reasoningTemperature = (config["reasoning_temperature"] as? Number)?.toDouble() ?: 0.7

    override suspend fun handleText(text: String) {
        val conversation = history.getHistory().toMutableList()
        if (systemPrompt.isNotEmpty()) conversation.add(0, LLMMessage.system(systemPrompt))
        conversation.add(LLMMessage.user(text))
        val tools = toolProvider.listTools()
        var phase = ReActPhase.THOUGHT
        var iteration = 0
        var answered = false

        while (iteration < maxIterations && !answered) {
            iteration++
            try {
                when (phase) {
                    ReActPhase.THOUGHT -> phase = generateThought(conversation, tools)
                    ReActPhase.ACTION -> {
                        val outcome = executeAction(conversation, tools)
                        if (outcome.first) { answered = true } else phase = outcome.second
                    }
                    ReActPhase.OBSERVATION -> phase = processObservation(conversation)
                    ReActPhase.FINAL_ANSWER -> answered = true
                }
            } catch (e: Exception) {
                handleError(conversation, e.message ?: e.toString())
                answered = true
            }
        }
        if (!answered) handleMaxIterations(conversation)

        conversation.last().content.firstOrNull()?.text?.let { response.respondText(it) }
    }

    private suspend fun say(text: String) { if (showReasoning) response.respondText(text) }

    private suspend fun generateThought(conversation: MutableList<LLMMessage>, tools: List<Tool>): ReActPhase {
        say("🤔 **THINKING...**")
        conversation.add(LLMMessage.system(thoughtPrompt))
        val llmResponse = llm.generate(conversation, LLMOptions(temperature = reasoningTemperature))
        conversation.add(LLMMessage.assistant(llmResponse.content))
        say("💭 **THOUGHT:** ${llmResponse.content}")
        return ReActPhase.ACTION
    }

    /** Returns (final answer reached?, next phase) */
    private suspend fun executeAction(conversation: MutableList<LLMMessage>, tools: List<Tool>): Pair<Boolean, ReActPhase> {
        say("⚡ **TAKING ACTION...**")
        conversation.add(LLMMessage.system(actionPrompt))
        val llmResponse = llm.generate(conversation, LLMOptions(temperature = 0.3, functions = tools))

        if ("FINAL_ANSWER:" in llmResponse.content.uppercase()) {
            say("✅ **FINAL ANSWER:** ${llmResponse.content}")
            conversation.add(LLMMessage.assistant(llmResponse.content))
            return true to ReActPhase.FINAL_ANSWER
        }

        conversation.add(
            LLMMessage(
                role = if (llmResponse.toolCalls != null) LLMRole.FUNCTION else LLMRole.ASSISTANT,
                content = listOf(LLMMessageContent(LLMMessageContentType.TEXT, text = llmResponse.content)),
                toolCalls = llmResponse.toolCalls,
            )
        )
        say("🔧 **ACTION:** ${llmResponse.content}")

        val calls = llmResponse.toolCalls.orEmpty()
        return if (calls.isEmpty()) false to ReActPhase.THOUGHT
        else {
            executeTools(conversation, calls)
            false to ReActPhase.OBSERVATION
        }
    }

    private suspend fun executeTools(conversation: MutableList<LLMMessage>, toolCalls: List<LLMFunctionCall>) {
        val results = toolCalls.map { call ->
            say("🛠️ **EXECUTING TOOL:** ${call.name} with args: ${call.arguments.jsonString()}")
            try {
                val result = toolProvider.executeTool(call.name, call.arguments)
                if (result.success) {
                    val content = result.result.jsonString()
                    say("✅ **TOOL SUCCESS:** ${call.name} returned: ${content.take(200)}...")
                    mapOf("id" to call.id, "name" to call.name, "result" to content)
                } else {
                    val content = "Error: ${result.error}"
                    say("❌ **TOOL ERROR:** ${call.name} failed: $content")
                    mapOf("id" to call.id, "name" to call.name, "error" to content)
                }
            } catch (e: Exception) {
                val content = "Exception: ${e.message}"
                say("💥 **TOOL EXCEPTION:** ${call.name} threw: $content")
                mapOf("id" to call.id, "name" to call.name, "error" to content)
            }
        }
        conversation.add(
            LLMMessage(
                role = LLMRole.FUNCTION,
                toolResults = results.map {
                    LLMFunctionResult(
                        id = it["id"] as String? ?: "",
                        name = it["name"] as String? ?: "",
                        content = (it["result"] ?: it["error"]) as String? ?: "",
                    )
                },
            )
        )
    }

    private suspend fun processObservation(conversation: MutableList<LLMMessage>): ReActPhase {
        say("👁️ **OBSERVING RESULTS...**")
        conversation.add(LLMMessage.system(observationPrompt))
        val llmResponse = llm.generate(conversation, LLMOptions(temperature = reasoningTemperature))
        conversation.add(LLMMessage.assistant(llmResponse.content))
        say("🔍 **OBSERVATION:** ${llmResponse.content}")
        return ReActPhase.THOUGHT
    }

    private suspend fun handleError(conversation: MutableList<LLMMessage>, error: String) {
        say("⚠️ **ERROR OCCURRED:** $error")
        conversation.add(LLMMessage.system(errorPrompt.replace("{error}", error)))
        val llmResponse = llm.generate(conversation, LLMOptions(temperature = 0.3))
        conversation.add(LLMMessage.assistant(llmResponse.content))
    }

    private suspend fun handleMaxIterations(conversation: MutableList<LLMMessage>) {
        say("⏰ **MAX ITERATIONS REACHED:** Providing best answer with available information...")
        conversation.add(LLMMessage.system(maxIterationsPrompt.replace("{max_iterations}", "$maxIterations")))
        val llmResponse = llm.generate(conversation, LLMOptions(temperature = 0.3))
        conversation.add(LLMMessage.assistant(llmResponse.content))
    }
}

/**
 * Chains several LLMs: each receives a specialized prompt appended to the
 * system message plus the previous models' outputs; results are merged.
 *
 * Config: `prompts` — one per bound llm, in order.
 */
class LLMCombinator(
    val llms: List<LLMProtocol>,
    val config: Map<String, Any?> = emptyMap(),
) : LLMProtocol {

    private val prompts = (config["prompts"] as? List<String>).orEmpty()
        .also { require(it.size == llms.size) {
            "Number of prompts (${it.size}) does not match number of LLMs (${llms.size})"
        } }

    override suspend fun generate(messages: List<LLMMessage>, options: LLMOptions?): LLMResponse {
        val results = mutableListOf<LLMResponse>()
        var current = messages
        prompts.forEachIndexed { i, prompt ->
            current = adaptPrompt(current, prompt, results)
            results += llms[i].generate(current, options)
        }
        return LLMResponse.merge(results)
    }

    override fun generateStream(messages: List<LLMMessage>, options: LLMOptions?) =
        kotlinx.coroutines.flow.flow<StreamFrame> {
            val results = mutableListOf<LLMResponse>()
            var current = messages
            prompts.forEachIndexed { i, prompt ->
                current = adaptPrompt(current, prompt, results)
                val chunks = StringBuilder()
                llms[i].generateStream(current, options).collect { frame ->
                    (frame as? StreamFrame.Text)?.let { chunks.append(it.text) }
                    emit(frame) // frames travel outward through this proxy too
                }
                results += LLMResponse(chunks.toString())
            }
        }

    private fun adaptPrompt(messages: List<LLMMessage>, prompt: String, intermediate: List<LLMResponse>): List<LLMMessage> {
        val found = messages.any { it.role == LLMRole.SYSTEM }
        val new = messages.map {
            if (it.role == LLMRole.SYSTEM)
                it.copy(content = it.content + LLMMessageContent(LLMMessageContentType.TEXT, text = prompt))
            else it
        }.toMutableList()
        if (!found) new.add(0, LLMMessage.system(prompt))
        intermediate.forEach { new.add(LLMMessage.assistant(it.content)) }
        return new
    }
}
